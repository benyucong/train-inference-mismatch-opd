#!/usr/bin/env python3
"""The refit gate of the EMULATED NVFP4 sampler in training: does verl's rollout lifecycle -- the weight stream
through SamplerQuant at every sync, vLLM's post-load processing, the level-2 sleep and both wake-ups -- leave the engine
exactly as the offline measurement built it (configuration mo_nvfp4emu_kv, or mo_nvfp4emu_kv_s0 with layer 0's cache
bf16)? GPU, venv-specopd, one process per engine (VLLM_ENABLE_V1_MULTIPROCESSING=0).

    $PYTHON_BIN bench/mo_emu_refit.py ref-digest   --static S [--kv-skip 0] --out D/digest_ref.json
    $PYTHON_BIN bench/mo_emu_refit.py refit-digest --static S [--kv-skip 0] --out D/digest_refit.json
    $PYTHON_BIN bench/mo_emu_refit.py compare --ref D/digest_ref.json --refit D/digest_refit.json --static S \
                                             [--kv-skip 0] [--offline-engine E] --json D/refit_check.json   # CPU; 0 = PASS
    python3     bench/mo_emu_refit.py selftest

Both engines are a training rollout server's: the bf16 student with SPECOPD_SAMPLER_QUANT=w=nvfp4,a=nvfp4 (verl's worker
extension installs the engine patches: the NVFP4 input quantizer with the static file's activation amax on every decoder
linear layer, the file's k / v scales on every FP8 cache layer, re-applied after post-load processing and every KV
wake-up), kv_cache_dtype fp8 (and, with layer 0's cache bf16, kv_cache_dtype_skip_layers ['0'] with the FA3 kv-skip
patch), sleep mode, FULL_AND_PIECEWISE CUDA graphs. A sync is the worker's update_weights_from_ipc path with the stream replaced
by the student's safetensors shards (one bucket per shard): SamplerQuant.apply, the worker's _update_weights, vLLM's
process_weights_after_loading, SamplerQuant.finish_sync, and after the first sync audit_weights.
  ref-digest:   one sync -> 'after_sync1'; greedy generations.
  refit-digest: sync 1 -> 'after_sync1'; sleep(level=2) + wake_up(['weights']) -> 'after_wake_weights'; sync 2 ->
                'after_sync2'; wake_up(['kv_cache']) (vLLM resets the k / v scales; the patch restores them) ->
                'after_wake_kv'; greedy generations. Also, from the checkpoint, the expected live weight of every fused
                linear layer: concat over its shards of specopd_nvfp4.nvfp4_qdq_weight(shard) (sha256).
A digest: per decoder linear layer the sha256 of the live weight bytes, its shape and dtype, its linear-method class and
static amax; per attention layer the cache dtype, the _k/_v/_q/_prob_scale tensors and the k / v float mirrors.
PASS iff in the refit engine 'after_sync1', 'after_sync2' and 'after_wake_kv' equal the reference engine's
'after_sync1' in every linear layer (weight bytes, method, static amax) and equal the expected live weights; every
FP8 cache layer's k / v (tensor and float) equals the static file's (bf16-served) value after the KV wake-up and in the
reference engine, q / prob stay 1.0; the skipped layers' cache is bf16 in both engines (and no other); the number of
NVFP4 linear layers is 7 x 28 = 196 fused into 112; both engines generate the same greedy tokens on both prompts; the
sync records quantized every decoder weight (196 tensors). Descriptive: what the level-2 sleep did ('after_wake_weights'),
and whether the reference engine's AIME24-problem-0 tokens equal the offline measurement's (engine_<label>.json).
"""
import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mo_common as C  # noqa: E402

SPEC = "w=nvfp4,a=nvfp4"
N_WEIGHTS = 7 * C.N_LAYERS
N_FUSED = 4 * C.N_LAYERS
FUSED = {"qkv_proj": ("q_proj", "k_proj", "v_proj"), "gate_up_proj": ("gate_proj", "up_proj")}


def _f32(x):
    import struct
    return struct.unpack("f", struct.pack("f", float(x)))[0]


def _set_env(static):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"       # vLLMReplica sets it for a sampler-quant rollout server
    for k in ("SPECOPD_SAMPLER_MODELOPT", "SPECOPD_SAMPLER_MODELOPT_SCALES", "SPECOPD_SAMPLER_MODELOPT_MODEL",
              "SPECOPD_SAMPLER_NOISE"):
        os.environ.pop(k, None)
    os.environ.update(SPECOPD_SAMPLER_QUANT=SPEC, SPECOPD_SAMPLER_QUANT_MODEL=C.STUDENT,
                      SPECOPD_SAMPLER_QUANT_STATIC=os.path.abspath(static))


def _engine(static, kv_skip):
    import mo_refit_check as R
    _set_env(static)
    from vllm import LLM
    extra, kvs = {}, None
    if kv_skip:
        extra = dict(kv_cache_dtype_skip_layers=[x for x in kv_skip.split(",") if x])
        kvs = R._install_kvskip_patch()
    return LLM(model=C.STUDENT, **R.engine_kwargs(**extra)), kvs


def _digest_worker(worker):
    """INSIDE the worker: the live state of every decoder linear layer and attention layer."""
    import torch
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.model_executor.layers.linear import LinearBase
    model = worker.model_runner.get_model()
    lin, attn = {}, {}
    for name, m in model.named_modules():
        if isinstance(m, LinearBase) and ".layers." in name:
            w = m.weight.data
            qm = getattr(m, "quant_method", None)
            lin[name] = dict(sha256=hashlib.sha256(w.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
                             shape=list(w.shape), dtype=str(w.dtype), method=type(qm).__name__,
                             amax=getattr(qm, "_specopd_nvfp4_amax", None))
        elif isinstance(m, Attention):
            attn[name] = dict(kv_cache_dtype=str(m.kv_cache_dtype), k=float(m._k_scale), v=float(m._v_scale),
                              q=float(m._q_scale), prob=float(m._prob_scale), kf=m._k_scale_float, vf=m._v_scale_float)
    return dict(linear=lin, attn=attn)


def _shards():
    import glob
    return sorted(glob.glob(os.path.join(C.STUDENT, "*.safetensors")))


def _sync_worker(worker, step):
    """INSIDE the worker: one weight sync as update_weights_from_ipc runs it under SPECOPD_SAMPLER_QUANT (the stream
    replaced by the student's safetensors shards, one bucket per shard)."""
    import torch
    from safetensors import safe_open
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    from verl.utils.specopd_quant import SamplerQuant, assert_engine_patched, audit_weights
    if getattr(worker, "_specopd_quant", None) is None:
        assert_engine_patched(worker.model_runner.vllm_config)
        worker._specopd_quant = SamplerQuant.from_env(worker.model_runner.vllm_config)
    sq = worker._specopd_quant
    dev = torch.device("cuda", torch.cuda.current_device())
    for f in _shards():
        with safe_open(f, framework="pt", device=str(dev)) as fh:
            bucket = [(k, fh.get_tensor(k)) for k in fh.keys()]
        worker._update_weights(sq.apply(bucket), peft_config=None, base_sync_done=True)
        del bucket
    for model, model_config in worker._iter_all_models_with_config():
        process_weights_after_loading(model, model_config, dev)
    rec = sq.finish_sync()
    aud = audit_weights(worker.model_runner.get_model(), sq.spec) if rec.get("sync") == 0 else None
    torch.cuda.synchronize()
    return dict(step=step, audit=json.loads(json.dumps(aud, default=str)) if aud is not None else None,
                **{k: rec[k] for k in ("sync", "spec", "n_tensors", "n_elems", "rel_rms_err")})


def _expected_worker(worker):
    """INSIDE the worker: the expected live weight of every fused decoder linear layer from the checkpoint --
    concat over its shards (vLLM's order) of Model Optimizer's NVFP4 weight fake quantization (sha256)."""
    import torch
    from safetensors import safe_open

    from verl.utils import specopd_nvfp4
    dev = torch.device("cuda", torch.cuda.current_device())
    where = {}
    for f in _shards():
        with safe_open(f, framework="pt") as fh:
            where.update({k: f for k in fh.keys()})

    def get(n):
        with safe_open(where[n], framework="pt", device=str(dev)) as fh:
            return fh.get_tensor(n)

    out = {}
    for i in range(C.N_LAYERS):
        for blk, last in (("self_attn", "qkv_proj"), ("self_attn", "o_proj"), ("mlp", "gate_up_proj"), ("mlp", "down_proj")):
            parts = FUSED.get(last, (last,))
            ws = [specopd_nvfp4.nvfp4_qdq_weight(get(f"model.layers.{i}.{blk}.{p}.weight").to(torch.bfloat16)) for p in parts]
            w = torch.cat(ws, 0).contiguous()
            out[f"model.layers.{i}.{blk}.{last}"] = hashlib.sha256(w.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            del ws, w
    torch.cuda.synchronize()
    return out


def cmd_ref(a):
    import mo_refit_check as R
    llm, kvs = _engine(a.static, a.kv_skip)
    rec = dict(engine="ref", static=os.path.abspath(a.static), kv_skip=a.kv_skip, syncs=[])
    rec["syncs"].append(llm.collective_rpc(_sync_worker, args=(1,))[0])
    rec["after_sync1"] = llm.collective_rpc(_digest_worker)[0]
    rec["generation"] = R._generate(llm)
    rec["kvskip_patch"] = dict(kvs) if kvs is not None else None
    C.write_json(a.out, rec)
    print(f"[mo_emu_refit] ref digest: {len(rec['after_sync1']['linear'])} linear layers -> {a.out}; syncs {rec['syncs']}; "
          f"generation {rec['generation']['text'][:120]!r}", flush=True)
    return 0


def cmd_refit(a):
    import mo_refit_check as R
    llm, kvs = _engine(a.static, a.kv_skip)
    rec = dict(engine="refit", static=os.path.abspath(a.static), kv_skip=a.kv_skip, syncs=[])
    rec["syncs"].append(llm.collective_rpc(_sync_worker, args=(1,))[0])
    rec["after_sync1"] = llm.collective_rpc(_digest_worker)[0]
    llm.sleep(level=2)
    llm.wake_up(tags=["weights"])
    rec["after_wake_weights"] = llm.collective_rpc(_digest_worker)[0]
    rec["syncs"].append(llm.collective_rpc(_sync_worker, args=(2,))[0])
    rec["after_sync2"] = llm.collective_rpc(_digest_worker)[0]
    llm.wake_up(tags=["kv_cache"])
    rec["after_wake_kv"] = llm.collective_rpc(_digest_worker)[0]
    rec["generation"] = R._generate(llm)
    rec["expected"] = llm.collective_rpc(_expected_worker)[0]
    rec["kvskip_patch"] = dict(kvs) if kvs is not None else None
    C.write_json(a.out, rec)
    print(f"[mo_emu_refit] refit digest -> {a.out}; syncs {rec['syncs']}; generation {rec['generation']['text'][:120]!r}",
          flush=True)
    return 0


def _li(name):
    m = re.search(r"layers\.(\d+)\.", name)
    return m.group(1) if m else None


def compare(ref, fit, static, kv_skip="", offline_gen=None):
    """The gate (module docstring). Returns (problems, descriptive record)."""
    bad, desc = [], {}
    skip = {x for x in str(kv_skip).split(",") if x}
    R = ref["after_sync1"]["linear"]
    nv = {n for n, r in R.items() if r["method"].startswith("SpecopdNvfp4")}
    if len(R) != N_FUSED or len(nv) != N_FUSED:
        bad.append(f"reference engine: {len(nv)} NVFP4 linear layers of {len(R)} (want {N_FUSED} of {N_FUSED})")
    exp = fit.get("expected") or {}
    if sorted(exp) != sorted(R):
        bad.append(f"expected weights for {len(exp)} layers vs {len(R)} live")
    for tag in ("after_sync1", "after_sync2", "after_wake_kv"):
        D = fit[tag]["linear"]
        if sorted(D) != sorted(R):
            bad.append(f"{tag}: linear layers differ from the reference engine's ({len(D)} vs {len(R)})")
            continue
        for k in ("sha256", "shape", "dtype", "method", "amax"):
            n = sum(1 for ln in R if R[ln][k] != D[ln][k])
            if n:
                bad.append(f"{tag}: {n}/{len(R)} layers differ from the reference engine in {k}")
        n = sum(1 for ln in R if exp.get(ln) != D[ln]["sha256"])
        if n:
            bad.append(f"{tag}: {n}/{len(R)} layers' live weight != the expected NVFP4 weight from the checkpoint")
    amax = static.get("act_amax") or {}
    for ln, r in R.items():
        prefix, last = ln.rsplit(".", 1)
        want = amax.get(f"{prefix}.{FUSED.get(last, (last,))[0]}")
        if want is None or r["amax"] is None or float(r["amax"]) != float(want):
            bad.append(f"{ln}: static amax {r['amax']} != the file's {want}")
            break
    kv = {str(k): v for k, v in (static.get("kv") or {}).items()}
    for who, att in (("refit after the KV wake-up", fit["after_wake_kv"]["attn"]), ("reference", ref["after_sync1"]["attn"])):
        fp8 = {n: r for n, r in att.items() if r["kv_cache_dtype"].startswith("fp8")}
        bf = {_li(n) for n, r in att.items() if not r["kv_cache_dtype"].startswith("fp8")}
        if bf != skip:
            bad.append(f"{who}: bf16-cache layers {sorted(bf)} != {sorted(skip)}")
        if len(fp8) != C.N_LAYERS - len(skip):
            bad.append(f"{who}: {len(fp8)} FP8 cache layers (want {C.N_LAYERS - len(skip)})")
        for n, r in fp8.items():
            w = kv.get(_li(n))
            if w is None or (r["k"], r["v"], r["kf"], r["vf"]) != (_f32(w["k"]), _f32(w["v"]), _f32(w["k"]), _f32(w["v"])):
                bad.append(f"{who}: {n} k/v {r['k']}/{r['v']} (floats {r['kf']}/{r['vf']}) != file {w}")
                break
            if (r["q"], r["prob"]) != (1.0, 1.0):
                bad.append(f"{who}: {n} q/prob {r['q']}/{r['prob']} (want vLLM's 1.0)")
                break
    for s in ref.get("syncs", []) + fit.get("syncs", []):
        if s.get("spec") != SPEC or s.get("n_tensors") != N_WEIGHTS:
            bad.append(f"sync record {s} (want spec {SPEC}, {N_WEIGHTS} tensors)")
    if len(fit.get("syncs", [])) != 2:
        bad.append(f"refit engine: {len(fit.get('syncs', []))} syncs (want 2)")
    pr, pf = (x["generation"].get("prompts") or {"gate": x["generation"]} for x in (ref, fit))
    for tag in sorted(pr):
        if pr[tag]["token_ids"] != pf.get(tag, {}).get("token_ids"):
            bad.append(f"greedy generation ({tag}) differs between the reference and the refit engine")
    W = fit["after_wake_weights"]
    desc["after_wake_weights"] = dict(
        weight_sha_equal=sum(1 for ln in R if W["linear"].get(ln, {}).get("sha256") == R[ln]["sha256"]),
        amax_equal=sum(1 for ln in R if W["linear"].get(ln, {}).get("amax") == R[ln]["amax"]), n_linear=len(R),
        kv_equal_file=sum(1 for n, r in W["attn"].items() if r["kv_cache_dtype"].startswith("fp8")
                          and r["k"] == _f32((kv.get(_li(n)) or {}).get("k", -1))))
    desc["generation_ref"] = {t: v["text"][:200] for t, v in pr.items()}
    desc["generation_refit"] = {t: v["text"][:200] for t, v in pf.items()}
    if offline_gen is not None:
        desc["ref_aime24_0_equals_offline_measurement"] = (pr.get("aime24_0", {}).get("token_ids") == offline_gen)
    desc["rel_rms_err"] = [s.get("rel_rms_err") for s in fit.get("syncs", [])]
    return bad, desc


def cmd_compare(a):
    ref, fit, st = (json.load(open(p)) for p in (a.ref, a.refit, a.static))
    og = None
    if a.offline_engine and os.path.exists(a.offline_engine):
        og = (json.load(open(a.offline_engine)).get("short_generation") or {}).get("token_ids")
    bad, desc = compare(ref, fit, st, a.kv_skip, og)
    out = dict(verdict="PASS" if not bad else "FAIL", problems=bad, descriptive=desc, kv_skip=a.kv_skip,
               static=os.path.abspath(a.static), static_sha256=hashlib.sha256(open(a.static, "rb").read()).hexdigest(),
               n_linear=len(ref["after_sync1"]["linear"]))
    if a.json:
        C.write_json(a.json, out)
    print(f"[mo_emu_refit] REFIT GATE {out['verdict']}: {len(bad)} problems {bad[:6]}; descriptive {desc}")
    return 0 if not bad else 8


def _fake(skip=""):
    sk = {x for x in skip.split(",") if x}
    lin, exp = {}, {}
    for i in range(C.N_LAYERS):
        for blk, last in (("self_attn", "qkv_proj"), ("self_attn", "o_proj"), ("mlp", "gate_up_proj"), ("mlp", "down_proj")):
            n = f"model.layers.{i}.{blk}.{last}"
            lin[n] = dict(sha256=f"h{n}", shape=[4, 4], dtype="torch.bfloat16",
                          method="SpecopdNvfp4ActUnquantizedLinearMethod", amax=float(1 + i))
            exp[n] = f"h{n}"
    attn = {f"model.layers.{i}.self_attn.attn": dict(kv_cache_dtype="auto" if str(i) in sk else "fp8",
                                                     k=_f32(0.05 + i), v=_f32(0.02), q=1.0, prob=1.0,
                                                     kf=_f32(0.05 + i), vf=_f32(0.02)) for i in range(C.N_LAYERS)}
    static = dict(act_amax={f"model.layers.{i}.{b}.{p}": float(1 + i) for i in range(C.N_LAYERS)
                            for b, p in (("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"),
                                         ("self_attn", "o_proj"), ("mlp", "gate_proj"), ("mlp", "up_proj"),
                                         ("mlp", "down_proj"))},
                  kv={str(i): dict(k=0.05 + i, v=0.02) for i in range(C.N_LAYERS)})
    sync = dict(spec=SPEC, n_tensors=N_WEIGHTS, rel_rms_err=0.1)
    gen = dict(token_ids=[1, 2], text="x", prompts=dict(gate=dict(token_ids=[1, 2], text="x"),
                                                         aime24_0=dict(token_ids=[3], text="y")))
    d = dict(linear=lin, attn=attn)
    ref = dict(after_sync1=d, syncs=[sync], generation=gen)
    garbage = {n: dict(r, sha256="zz") for n, r in lin.items()}
    fit = dict(after_sync1=d, after_sync2=d, after_wake_kv=d, after_wake_weights=dict(linear=garbage, attn=attn),
               syncs=[sync, sync], generation=gen, expected=exp)
    return ref, fit, static


def selftest():
    ref, fit, st = _fake()
    bad, desc = compare(ref, fit, st, "", [3])
    assert not bad, bad
    assert desc["after_wake_weights"]["weight_sha_equal"] == 0 and desc["ref_aime24_0_equals_offline_measurement"]
    f2 = json.loads(json.dumps(fit))
    f2["after_sync2"]["linear"]["model.layers.3.mlp.down_proj"]["sha256"] = "other"
    assert any("after_sync2" in p and "sha256" in p for p in compare(ref, f2, st)[0])
    f3 = json.loads(json.dumps(fit))
    f3["expected"]["model.layers.0.self_attn.qkv_proj"] = "other"
    assert any("expected NVFP4 weight" in p for p in compare(ref, f3, st)[0])
    f4 = json.loads(json.dumps(fit))
    for r in f4["after_wake_kv"]["attn"].values():
        r["k"] = r["v"] = 1.0
    assert any("after the KV wake-up" in p for p in compare(ref, f4, st)[0])
    f5 = json.loads(json.dumps(fit))
    f5["generation"]["prompts"]["gate"]["token_ids"] = [9]
    assert any("greedy generation (gate)" in p for p in compare(ref, f5, st)[0])
    f6 = json.loads(json.dumps(fit))
    f6["syncs"][1] = dict(f6["syncs"][1], n_tensors=195)
    assert any("sync record" in p for p in compare(ref, f6, st)[0])
    st2 = json.loads(json.dumps(st))
    st2["act_amax"]["model.layers.5.self_attn.q_proj"] = 99.0
    assert any("static amax" in p for p in compare(ref, fit, st2)[0])
    # layer 0's cache bf16 in both engines, and only it
    r0, f0, s0 = _fake("0")
    assert not compare(r0, f0, s0, "0")[0], compare(r0, f0, s0, "0")[0]
    assert any("bf16-cache layers" in p for p in compare(r0, f0, s0, "")[0])
    assert any("bf16-cache layers" in p for p in compare(ref, fit, st, "0")[0])
    assert C.selftest() == 0
    print("[mo_emu_refit] SELFTEST PASS")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("ref-digest", "refit-digest", "compare"))
    ap.add_argument("--static", required=True)
    ap.add_argument("--kv-skip", default="", help="the bf16-cache layers (layer 0's cache bf16: 0)")
    ap.add_argument("--ref", default="")
    ap.add_argument("--refit", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--offline-engine", default="", help="compare: the offline measurement's engine_<label>.json")
    a = ap.parse_args(argv)
    return {"ref-digest": cmd_ref, "refit-digest": cmd_refit, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
