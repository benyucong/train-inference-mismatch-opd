#!/usr/bin/env python3
"""Bit-for-bit proof of the emulated INT8 SmoothQuant / INT4 AWQ samplers against NVIDIA Model Optimizer's own fake
quantization, on the GPU in the TRAINING venv (venv-specopd).

    $PYTHON_BIN bench/mo_smooth_check.py --capture D/x_capture.pt --static D/x_static.json --json D/x_check.json
    python3 bench/mo_smooth_check.py --selftest

On the IDENTICAL captured tensors (bench/mo_export.py smooth, venv-modelopt), per decoder layer 0 / 13 / 27 x the seven
linear layers:
  A  the activation: specopd_smooth's op (x * pqs; SmoothQuant + INT8 per tensor with the fixed amax) on X == Model
     Optimizer's input quantizer on X, bit for bit; the static file's pqs / amax == the quantizer's;
  W  the weight: the student's weight (checkpoint) smoothed with the file's inv (sha256) == Model Optimizer's smoothed
     weight; its fresh INT8-per-channel / INT4-per-block fake quantization == Model Optimizer's weight quantizer on it
     (whole-tensor sha256 and the first rows); a perturbed smoothed weight with fresh scales == Model Optimizer's weight
     quantizer re-calibrated on it; and SamplerQuant.apply on the HF weight (the sync code path) == the same, whole tensor;
  C  the op at the boundary of a compiled graph: an RMSNorm (Inductor-fused, in float) then the op, compiled; the op's
     output inside the compiled graph == the eager op on the very bf16 tensor the graph fed it (returned by the graph),
     the op's Python body ran (opaque to Inductor: nothing of the smoothing or the INT8 arithmetic was fused), and the
     same inside a CUDA graph; the op alone in a CUDA graph == eager. On layer 0's down_proj and q_proj inputs.
     Descriptive, not in the verdict: compiled vs eager end to end, because Inductor's RMSNorm (a fused reduction, FMA
     accumulation, split over the channels of the input, e.g. down_proj's 18,944) can differ from eager's by one bf16 ulp
     on some rows, before the op -- a property of the compiled bf16 model, not of the quantization.
Verdict BIT-EXACT or not; descriptive: the whole linear layer vs Model Optimizer's module output.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mo_common as C  # noqa: E402

A_CHECKS = ("act", "static_pqs", "static_amax")
W_CHECKS = ("wp_sha", "wq_sha", "wq_rows", "w_fresh_rows", "w_sampler_sync")
C_CHECKS = ("compiled_op_eq_eager_op", "compiled_op_body_ran", "compiled_op_input_bf16", "graph_op_eq_eager",
            "graph_compiled_op_eq_eager_op")


def verdict(mods, comp):
    """comp: {module: the C items} (one or more modules)."""
    bad = [f"{n}:{k}" for n, r in sorted(mods.items()) for k in A_CHECKS + W_CHECKS if r.get(k) is not True]
    bad += [f"compiled:{m}:{k}" for m, c in sorted(comp.items()) for k in C_CHECKS if c.get(k) is not True]
    return ("BIT-EXACT" if not bad and mods and comp else "NOT BIT-EXACT"), bad


def _sha(t):
    import torch
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _eq(a, b):
    import torch
    a, b = a.detach(), b.detach().to(a.device)
    return bool(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8),
                                                                           b.contiguous().view(torch.uint8)))


def _ndiff(a, b):
    a, b = a.detach().float(), b.detach().float().to(a.device)
    return dict(n_diff=int((a != b).sum()), n=a.numel(), max_abs=float((a - b).abs().max()),
                rel=float((a - b).norm() / b.norm().clamp_min(1e-30)))


def _student_tensor(name, device):
    return C.student_tensor(name, device)


def check_module(n, rec, static, tens, grid, sm, sq, dev):
    import torch
    import torch.nn.functional as F
    r = {}
    X = rec["x"].to(dev)
    H = X.shape[-1]
    pqs, inv = tens[n]
    mul, q8 = sm.define_ops()
    r["static_pqs"] = _eq(pqs, rec["pqs"].reshape(-1).to(torch.bfloat16))
    if grid == "int8sq":
        amax = torch.tensor(float(static["act_amax"][n]), dtype=torch.float32, device=dev)
        r["static_amax"] = float(static["act_amax"][n]) == float(rec["act_amax"])
        xa = q8(X.reshape(-1, H), pqs, amax)
    else:
        r["static_amax"] = True
        xa = mul(X.reshape(-1, H), pqs)
    r["act"] = _eq(xa, rec["x_mo"].reshape(-1, H))
    W = _student_tensor(n + ".weight", dev)
    Wp = sm.smooth_weight(W, inv)
    r["wp_sha"] = _sha(Wp) == rec["wp_sha256"]
    Wq = sm.qdq_weight_int8_rows(Wp) if grid == "int8sq" else sm.qdq_weight_int4_blocks(Wp)
    rows = rec["wp_rows"].shape[0]
    r["wq_sha"] = _sha(Wq) == rec["w_mo_sha256"]
    r["wq_rows"] = _eq(Wq[:rows], rec["w_mo_rows"])
    W2 = rec["w2p_rows"].to(dev)
    W2q = sm.qdq_weight_int8_rows(W2) if grid == "int8sq" else sm.qdq_weight_int4_blocks(W2)
    r["w_fresh_rows"] = _eq(W2q, rec["w2_mo_rows"])
    h = sq.SamplerQuant(sq.QuantSpec.parse("w=int8sq,a=int8sq" if grid == "int8sq" else "w=int4awq"))
    out = h.apply([(n + ".weight", W)])
    h.finish_sync()
    r["w_sampler_sync"] = _sha(out[0][1]) == rec["w_mo_sha256"]
    b = rec["bias"].to(dev) if rec["bias"] is not None else None
    y = F.linear(xa.reshape(X.shape), Wq, b)
    r["linear_vs_modelopt"] = _ndiff(y, rec["y_mo"])
    r["linear_bitwise"] = _eq(y, rec["y_mo"])
    del W, Wp, Wq, out
    torch.cuda.empty_cache()
    return r


def check_compiled(sm, rec, pqs, amax, grid, dev):
    import torch
    X = rec["x"].to(dev)
    H = X.shape[-1]
    h = X.reshape(-1, H)
    mul, q8 = sm.define_ops()
    g = torch.Generator(device=dev).manual_seed(7)
    wn = (1.0 + 0.1 * torch.randn(H, generator=g, device=dev)).to(torch.bfloat16)

    def op(v):
        return q8(v, pqs, amax) if grid == "int8sq" else mul(v, pqs)

    def f(t):   # an RMSNorm in float (as the engine's), rounded to bf16, then the op; the graph returns the op's input too
        v = t.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)
        v = (v * wn.float()).to(torch.bfloat16)
        return v, op(v)

    res = {}
    v_e, eager = f(h)
    comp = torch.compile(f, fullgraph=True, dynamic=False)
    seen0 = sum(sm.SEEN.values())
    v_c, out_c = comp(h)
    res["compiled_op_body_ran"] = sum(sm.SEEN.values()) > seen0     # an opaque call: Inductor fused nothing of it
    res["compiled_op_input_bf16"] = v_c.dtype == torch.bfloat16
    res["compiled_op_eq_eager_op"] = _eq(out_c, op(v_c))
    res["compiled_eq_eager"] = _eq(out_c, eager)                    # descriptive (the RMSNorm in front may differ)
    res["norm_compiled_vs_eager"] = _ndiff(v_c, v_e)
    res["out_compiled_vs_eager"] = _ndiff(out_c, eager)
    h2 = (h.float() * 1.5 + 0.01).to(torch.bfloat16)
    sin = h.clone()
    o = op(sin)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        o = op(sin)
    sin.copy_(h2)
    gr.replay()
    res["graph_op_eq_eager"] = _eq(o, op(h2))
    sin2 = h.clone()
    comp(sin2)
    gr2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr2):
        v2, o2 = comp(sin2)
    sin2.copy_(h2)
    gr2.replay()
    torch.cuda.synchronize()
    res["graph_compiled_op_eq_eager_op"] = _eq(o2, op(v2.clone()))
    res["graph_compiled_eq_eager"] = _eq(o2, f(h2)[1])              # descriptive
    torch.cuda.synchronize()
    return res


def run(a):
    import torch

    from verl.utils import specopd_quant as sq
    from verl.utils import specopd_smooth as sm
    dev = "cuda"
    os.environ[sq.ENV_STATIC] = os.path.abspath(a.static)     # SamplerQuant reads the fixed smoothing from here
    static = sq.load_static(a.static)
    grid = static["grid"]
    tens = sq.static_tensors(static, dev)
    cap = torch.load(a.capture, map_location="cpu", weights_only=False)
    mods = {}
    for n, rec in cap["modules"].items():
        mods[n] = r = check_module(n, rec, static, tens, grid, sm, sq, dev)
        print(f"[mo_smooth_check] {n}: " + " ".join(f"{k}={int(bool(r[k]))}" for k in A_CHECKS + W_CHECKS)
              + f" | linear rel {r['linear_vs_modelopt']['rel']:.2e}", flush=True)
    paths = {}
    for n, rec in cap["modules"].items():                     # Model Optimizer's CUDA extension vs its torch path
        for side, r in (rec.get("paths") or {}).items():
            for k, v in r.items():
                paths.setdefault(f"{side}.{k}", {}).setdefault(str(v), []).append(n)
    paths = {k: {v: len(ns) for v, ns in d.items()} for k, d in sorted(paths.items())}
    print(f"[mo_smooth_check] Model Optimizer's integer paths (value: modules): {paths}")
    comp = {}
    for m in ("model.layers.0.mlp.down_proj", "model.layers.0.self_attn.q_proj"):
        if m not in cap["modules"]:
            continue
        amax = torch.tensor(float((static.get("act_amax") or {}).get(m, 1.0)), dtype=torch.float32, device=dev)
        comp[m] = check_compiled(sm, cap["modules"][m], tens[m][0], amax, grid, dev)
        print(f"[mo_smooth_check] compiled {m}: {comp[m]}", flush=True)
    v, bad = verdict(mods, comp)
    out = dict(verdict=v, failures=bad, compiled=comp, modules=mods, grid=grid,
               modelopt_int_path=static.get("modelopt_int_path"), modelopt_paths=paths,
               meta=dict(capture_meta=cap.get("meta"), static_sha256=static["_sha256"], torch=torch.__version__,
                         gpu=torch.cuda.get_device_name(), specopd_smooth_md5=C.md5_file(sm.__file__)))
    if a.json:
        C.write_json(a.json, out)
    print(f"[mo_smooth_check] VERDICT {v} ({len(mods)} modules, grid {grid}, Model Optimizer's integer path "
          f"{static.get('modelopt_int_path')}; failures {bad[:6]})")
    return 0 if v == "BIT-EXACT" else 5


def selftest():
    good = {k: True for k in A_CHECKS + W_CHECKS}
    c1 = dict({k: True for k in C_CHECKS}, compiled_eq_eager=False, graph_compiled_eq_eager=False)   # descriptive only
    comp = {"a": c1, "b": dict(c1)}
    assert verdict({"m": dict(good)}, comp) == ("BIT-EXACT", [])
    assert verdict({}, comp)[0] == "NOT BIT-EXACT" and verdict({"m": dict(good)}, {})[0] == "NOT BIT-EXACT"
    assert verdict({"m": dict(good, wq_sha=False)}, comp)[1] == ["m:wq_sha"]
    assert verdict({"m": dict(good)}, dict(comp, b=dict(c1, graph_op_eq_eager=False)))[1] == ["compiled:b:graph_op_eq_eager"]
    assert verdict({"m": dict(good)}, dict(comp, a=dict(c1, compiled_op_body_ran=False)))[0] == "NOT BIT-EXACT"
    print("[mo_smooth_check] SELFTEST PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capture", default="")
    ap.add_argument("--static", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.capture and a.static):
        ap.error("--capture and --static are required")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
