#!/usr/bin/env python3
"""Bit-for-bit proof of the emulated NVFP4 sampler against NVIDIA Model Optimizer's own fake quantization, and the
diagnosis of vLLM 0.19.1's NVFP4 emulation backend, on the GPU in the TRAINING venv (venv-specopd, vLLM's stack).

    $PYTHON_BIN bench/mo_nvfp4_check.py --capture D/nvfp4_capture.pt --static D/nvfp4_static.json \\
        --export-dir FORMATS/ckpt/nvfp4_kv --json D/nvfp4_check.json            -> exit 0 = BIT-EXACT
    python3 bench/mo_nvfp4_check.py --selftest                                   (stdlib: the verdict logic)

The capture (bench/mo_export.py amax, venv-modelopt) holds, for decoder layers 0, 13, 27 x their seven linear layers,
the input X Model Optimizer's calibrated model fed each quantizer (bf16) and Model Optimizer's outputs on it. Here, on
the IDENTICAL tensors (same dtype, same device type):
  A  activations: the vendored kernel (verl/utils/specopd_nvfp4.py) on X (bf16) and on X in float32 == Model Optimizer's
     input quantizer on the same, bit for bit; the custom op on X as vLLM lays it out (2-D) == the same; the static
     file's amax == the amax Model Optimizer used;
  W  weights: the student's weight (read from its checkpoint, sha256 == the capture's) through nvfp4_qdq_weight (amax =
     max |W|, recomputed) == Model Optimizer's weight quantizer on W (whole-tensor sha256 and the first 256 rows); float32
     rows; a perturbed W' with Model Optimizer's re-calibrated amax (a training step's fresh weights); and the sampler's
     sync code path (SamplerQuant.apply, the HF tensor) == Model Optimizer, whole tensor;
  C  compiled: an RMSNorm (float32, rounded to bf16) followed by the op, torch.compile'd (Inductor) == eager, bit for
     bit -- the op is fed the bf16-ROUNDED activation inside a compiled graph, as Model Optimizer's eager forward feeds
     it -- and the op and the compiled function captured in a CUDA graph and replayed on new inputs == eager;
descriptive (never the verdict): the whole linear layer F.linear(op(X), Wq, b) vs Model Optimizer's module output; the
pure-torch reference vs the kernel; and vLLM 0.19.1's NVFP4 emulation (run_nvfp4_emulations on the formats job's
export of these layers, as the ModelOpt NVFP4 path calls it) vs Model Optimizer's output, as shipped and with its two
defects fixed one at a time: (1) the weight's block scales are DIVIDED by weight_global_scale, which for a Model
Optimizer checkpoint is weight_scale_2 (amax / 2688, far below 1) -- the compressed-tensors convention stores the
reciprocal -- so the weights come out ~1/weight_scale_2^2 too large; (2) the block scales are un-swizzled although
the ModelOpt path never swizzles them for this backend.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mo_common as C  # noqa: E402

A_CHECKS = ("act_bf16", "act_fp32", "op_2d", "static_amax")
W_CHECKS = ("w_sha_input", "w_amax", "w_sha", "w_rows", "w_fp32_rows", "w_fresh_rows", "w_sampler_sync")
C_CHECKS = ("compiled_eq_eager", "graph_op_eq_eager", "graph_compiled_eq_eager")


def verdict(mods, comp):
    """BIT-EXACT iff every A / W check of every module and every C check holds. Returns (verdict, failures)."""
    bad = [f"{n}:{k}" for n, r in sorted(mods.items()) for k in A_CHECKS + W_CHECKS if r.get(k) is not True]
    bad += [f"compiled:{k}" for k in C_CHECKS if comp.get(k) is not True]
    return ("BIT-EXACT" if not bad and mods else "NOT BIT-EXACT"), bad


def _sha(t):
    import torch
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _eq(a, b):
    """Bitwise equality (same dtype, same bytes; -0.0 != +0.0)."""
    import torch
    a, b = a.detach(), b.detach().to(a.device)
    return bool(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8),
                                                                           b.contiguous().view(torch.uint8)))


def _ndiff(a, b):
    a, b = a.detach().float(), b.detach().float().to(a.device)
    d = (a - b).abs()
    return dict(n_diff=int((a != b).sum()), n=a.numel(), max_abs=float(d.max()),
                rel=float((a - b).norm() / b.norm().clamp_min(1e-30)))


def _student_tensor(name, device):
    return C.student_tensor(name, device)


def _export_tensors(export_dir, names, device):
    import glob
    from safetensors import safe_open
    out = {}
    for f in sorted(glob.glob(os.path.join(export_dir, "*.safetensors"))):
        with safe_open(f, framework="pt", device=device) as fh:
            for k in fh.keys():
                if k in names:
                    out[k] = fh.get_tensor(k)
    return out


def check_module(n, rec, static, nv, sq, export, dev):
    import torch
    import torch.nn.functional as F
    r = {}
    X = rec["x"].to(dev)
    amax_in = rec["amax_in"].to(dev)
    H = X.shape[-1]
    r["x_dtype"] = str(X.dtype)
    r["act_bf16"] = _eq(nv.fp4_fake_quant_block(X, amax_in), rec["x_mo"])
    r["act_fp32"] = _eq(nv.fp4_fake_quant_block(X.float(), amax_in), rec["x32_mo"])
    xq = nv.nvfp4_qdq(X.reshape(-1, H), float(amax_in))
    r["op_2d"] = _eq(xq, rec["x_mo"].reshape(-1, H))
    r["static_amax"] = float(static["act_amax"].get(n, float("nan"))) == float(amax_in)
    r["ref_vs_kernel_act"] = _ndiff(nv.fp4_fake_quant_ref(X, amax_in), rec["x_mo"])
    W = _student_tensor(n + ".weight", dev)
    r["w_sha_input"] = _sha(W) == rec["w_sha256"]
    r["w_amax"] = float(W.abs().amax().float()) == float(rec["amax_w"])
    Wq = nv.nvfp4_qdq_weight(W)
    r["w_sha"] = _sha(Wq) == rec["w_mo_sha256"]
    rows = rec["w_rows"].shape[0]
    r["w_rows"] = _eq(Wq[:rows], rec["w_mo_rows"])
    r["w_fp32_rows"] = _eq(nv.fp4_fake_quant_block(rec["w_rows"].to(dev).float(), rec["amax_w"].to(dev)),
                           rec["w32_mo_rows"])
    r["w_fresh_rows"] = _eq(nv.fp4_fake_quant_block(rec["w2_rows"].to(dev), rec["amax_w2"].to(dev)), rec["w2_mo_rows"])
    h = sq.SamplerQuant(sq.QuantSpec.parse("w=nvfp4,a=nvfp4"))
    out = h.apply([(n + ".weight", W)])
    h.finish_sync()
    r["w_sampler_sync"] = _sha(out[0][1]) == rec["w_mo_sha256"]
    b = rec["bias"].to(dev) if rec["bias"] is not None else None
    y = F.linear(xq.reshape(X.shape), Wq, b)
    r["linear_vs_modelopt"] = _ndiff(y, rec["y_mo"])
    r["linear_bitwise"] = _eq(y, rec["y_mo"])
    if export is not None:
        r["vllm_emulation"] = vllm_emulation(n, X.reshape(-1, H), b, export, rec["y_mo"].reshape(-1, rec["y_mo"].shape[-1]),
                                             Wq, dev)
    del W, Wq, out
    torch.cuda.empty_cache()
    return r


def vllm_emulation(n, x, b, export, y_mo, wq_ours, dev):
    """vLLM 0.19.1's NVFP4 emulation (VLLM_USE_NVFP4_CT_EMULATIONS=1) for a Model Optimizer checkpoint, on the formats
    job's export of this layer: exactly the arguments ModelOptNvFp4LinearMethod passes (input_global_scale_inv =
    1 / input_scale, weight_global_scale = weight_scale_2, block scales as loaded), then the two fixes."""
    import torch
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (dequantize_to_dtype,
                                                                                         run_nvfp4_emulations)
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale
    w = export[n + ".weight"]
    ws = export[n + ".weight_scale"]
    s2 = export[n + ".weight_scale_2"].max().to(torch.float32)
    isc = export[n + ".input_scale"].max().to(torch.float32)
    inv = (1.0 / isc).to(torch.float32)
    sw = swizzle_blockscale(ws)
    out = {}
    for tag, sf, g in (("as_shipped", ws, s2), ("fix_global_only", ws, (1.0 / s2).to(torch.float32)),
                       ("fix_both", sw, (1.0 / s2).to(torch.float32))):
        y = run_nvfp4_emulations(x=x, input_global_scale=inv, weight=w, weight_scale_swizzled=sf, weight_global_scale=g)
        if b is not None:
            y = y + b
        wd = dequantize_to_dtype(w.view(torch.uint8), sf, g, x.dtype, x.device, 16)
        out[tag] = dict(output_vs_modelopt=_ndiff(y, y_mo), weight_vs_modelopt_fake_quant=_ndiff(wd, wq_ours),
                        weight_norm_ratio=float(wd.float().norm() / wq_ours.float().norm()))
    out["weight_scale_2"] = float(s2)
    out["input_scale"] = float(isc)
    return out


def check_compiled(nv, rec, dev):
    """C: the op inside torch.compile and inside CUDA graphs (see the module docstring)."""
    import torch
    X = rec["x"].to(dev)
    H = X.shape[-1]
    h = X.reshape(-1, H)
    amax = float(rec["amax_in"])
    g = torch.Generator(device=dev).manual_seed(7)
    wn = (1.0 + 0.1 * torch.randn(H, generator=g, device=dev)).to(torch.bfloat16)
    op = nv.define_op()

    def f(t):
        v = t.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)
        v = (v * wn.float()).to(torch.bfloat16)
        return op(v, amax)

    def f_unrounded(t):  # what the quantizer would see if a compiler elided the bf16 rounding in front of it
        v = t.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)
        return op(v * wn.float(), amax).to(torch.bfloat16)

    res = {}
    eager = f(h)
    comp = torch.compile(f, fullgraph=True, dynamic=False)
    res["compiled_eq_eager"] = _eq(comp(h), eager)
    res["if_unrounded_input"] = _ndiff(f_unrounded(h), eager)
    h2 = (h.float() * 1.5 + 0.01).to(torch.bfloat16)
    want2 = f(h2)
    sin = h.clone()
    o = op(sin, amax)  # warm
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        o = op(sin, amax)
    sin.copy_(h2)
    gr.replay()
    res["graph_op_eq_eager"] = _eq(o, op(h2, amax))
    sin2 = h.clone()
    comp(sin2)
    gr2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr2):
        o2 = comp(sin2)
    sin2.copy_(h2)
    gr2.replay()
    res["graph_compiled_eq_eager"] = _eq(o2, want2)
    torch.cuda.synchronize()
    return res


def run(a):
    import torch
    import triton

    from verl.utils import specopd_nvfp4 as nv
    from verl.utils import specopd_quant as sq
    dev = "cuda"
    cap = torch.load(a.capture, map_location="cpu", weights_only=False)
    static = sq.load_static(a.static)
    names = [n + s for n in cap["modules"] for s in (".weight", ".weight_scale", ".weight_scale_2", ".input_scale")]
    export = _export_tensors(a.export_dir, set(names), dev) if a.export_dir else None
    mods = {}
    for n, rec in cap["modules"].items():
        mods[n] = check_module(n, rec, static, nv, sq, export, dev)
        r = mods[n]
        emu = (r.get("vllm_emulation") or {})
        print(f"[mo_nvfp4_check] {n}: " + " ".join(f"{k}={int(bool(r[k]))}" for k in A_CHECKS + W_CHECKS)
              + f" | linear rel {r['linear_vs_modelopt']['rel']:.2e} | vLLM emulation rel: "
              + " ".join(f"{t}={emu[t]['output_vs_modelopt']['rel']:.3g}" for t in ("as_shipped", "fix_global_only",
                                                                                      "fix_both") if t in emu),
              flush=True)
    first = cap["modules"][sorted(cap["modules"])[0]]
    comp = check_compiled(nv, first, dev)
    v, bad = verdict(mods, comp)
    out = dict(verdict=v, failures=bad, compiled=comp, modules=mods, meta=dict(
        capture=a.capture, capture_meta=cap.get("meta"), static_sha256=static["_sha256"],
        torch=torch.__version__, triton=triton.__version__, gpu=torch.cuda.get_device_name(),
        vendored_md5=C.md5_file(nv.__file__), engine_input_dtype="torch.bfloat16 (vLLM activations; the op "
        "materializes its bf16 input -- compiled_eq_eager)"))
    if a.json:
        C.write_json(a.json, out)
    print(f"[mo_nvfp4_check] compiled: {comp}")
    print(f"[mo_nvfp4_check] VERDICT {v} ({len(mods)} modules; failures {bad[:6]})")
    return 0 if v == "BIT-EXACT" else 5


def selftest():
    good = {k: True for k in A_CHECKS + W_CHECKS}
    comp = {k: True for k in C_CHECKS}
    assert verdict({"m": dict(good)}, comp) == ("BIT-EXACT", [])
    assert verdict({}, comp)[0] == "NOT BIT-EXACT"
    v, bad = verdict({"m": dict(good, w_sha=False)}, comp)
    assert v == "NOT BIT-EXACT" and bad == ["m:w_sha"]
    v, bad = verdict({"m": dict(good)}, dict(comp, graph_op_eq_eager=None))
    assert v == "NOT BIT-EXACT" and bad == ["compiled:graph_op_eq_eager"]
    print("[mo_nvfp4_check] SELFTEST PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capture", default="")
    ap.add_argument("--static", default="")
    ap.add_argument("--export-dir", default="")
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
