#!/usr/bin/env python3
"""One engine configuration of the Model Optimizer formats job, measured with our offline harness (venv-specopd).

The harness is bench/tim_calibrate.py, IMPORTED UNCHANGED, with its decode method: the 16 calibration prompts
(AIME24 0-7, MATH-500 0-7), 2 samples each, T 1, top-p 1, per-request seeds from 777, at most 16,384 tokens, decode
path (y ~ q, log q as the engine computed it), re-scored by a clean bf16 vLLM prefill in a separate process; mismatch =
mean of log q - log p over the sampled tokens, 95% bootstrap over prompts. Nothing in tim_calibrate.py is edited. In
THIS process the wrapper changes only:
  * the engine's model path, during tim_calibrate.build_llm only, to the configuration's Model Optimizer export
    (mo_common.FORMATS[label]['ckpt']); the prompts are still rendered with the student's own tokenizer files (the
    export carries a copy of them);
  * environment variables of the configuration (e.g. VLLM_USE_NVFP4_CT_EMULATIONS=1), set before vLLM is imported;
  * for an NVFP4 export: verl's ModelOpt NVFP4 rollout patch (verl/utils/modelopt, installed by verl's worker
    extension for any ModelOptNvFp4Config engine; it forces Marlin and needs the modelopt package, which venv-specopd
    does not have) is replaced by a recorder, so vLLM's own backend selection applies (recorded in engine_<label>.json);
and adds, after the engine is built and again after the decode: an engine audit (every attention layer's cache dtype,
storage, backend, _k/_v/_q/_prob_scale tensors and float mirrors; every decoder linear layer's quant method, weight
dtype, scales and NVFP4 backend) and one short greedy generation (48 tokens of prompt AIME24:0) -> engine_<label>.json.
For the emulated NVFP4 configurations (mo_common.NVFP4_EMU, tim_calibrate --quant w=nvfp4...) the env value @STATIC is
the --static file, and engine_<label>.json also records the sampler-quant attach record (specopd_quant._PATCHED), the
dtypes the NVFP4 activation op ran on, the weight re-stream's sync record and grid audit (the RESULT of tim_calibrate's
llm.collective_rpc(_reload_worker), recorded by wrapping the engine object's collective_rpc; tim_calibrate.py itself is
unchanged) and a second short generation on the re-streamed (quantized) weights.

    $PYTHON_BIN bench/mo_measure.py q --label mo_fp8_kv --ckpt-root D/ckpt --out-dir D
    $PYTHON_BIN bench/mo_measure.py q --label mo_nvfp4emu_kv --static D/nvfp4_static.json --out-dir D   # emulated NVFP4
    $PYTHON_BIN bench/mo_measure.py score --labels a,b,c --out-dir D
    $PYTHON_BIN bench/mo_measure.py sleepwake --export fp8_kv --ckpt-root D/ckpt --out-dir D   # scales across sleep
    python3     bench/mo_measure.py selftest
Run from the repository root.
"""
import argparse
import json
import os
import re
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mo_common as C  # noqa: E402

ENV_KNOBS = ("VLLM_USE_NVFP4_CT_EMULATIONS", "VLLM_NVFP4_GEMM_BACKEND", "VLLM_USE_FBGEMM", "SPECOPD_SAMPLER_QUANT",
             "SPECOPD_SAMPLER_QUANT_MODEL", "SPECOPD_SAMPLER_QUANT_STATIC")
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def log(*a):
    print(f"[mo_measure {time.strftime('%H:%M:%S')}]", *a, flush=True)


def tim_argv(label, out_dir):
    cfg = C.formats_dict()[label]
    return (["tim_calibrate.py", "--mode", "q", "--arm", label] + C.DECODE_ARGS.split() + cfg["args"].split()
            + ["--out-dir", out_dir])


def import_tim_calibrate():
    import tim_calibrate as tc  # sets VLLM_ENABLE_V1_MULTIPROCESSING=0 before anything imports vllm
    return tc


def _audit_worker(worker):
    """Runs INSIDE the in-process vLLM worker: what the engine really runs (json-able)."""
    import torch
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.model_executor.layers.linear import LinearBase

    def tv(x):
        if isinstance(x, torch.Tensor):
            return [float(v) for v in x.detach().float().reshape(-1).cpu().tolist()]
        return None if x is None else [float(x)]

    def li(name):
        m = _LAYER.search(name)
        return int(m.group(1)) if m else -1

    mr = worker.model_runner
    model = mr.get_model()
    attn, lin = [], []
    for name, m in model.named_modules():
        if isinstance(m, Attention):
            kv = getattr(m, "kv_cache", None)
            t = kv[0] if isinstance(kv, (list, tuple)) and kv else kv
            be = getattr(m, "attn_backend", None)
            attn.append(dict(layer=li(name), name=name, kv_cache_dtype=str(m.kv_cache_dtype),
                             kv_torch=str(getattr(m, "kv_cache_torch_dtype", None)),
                             storage=str(getattr(t, "dtype", None)),
                             backend=be.get_name() if hasattr(be, "get_name") else str(be),
                             quant_method=type(getattr(m, "quant_method", None)).__name__,
                             calculate_kv_scales=bool(getattr(m, "calculate_kv_scales", False)),
                             k=tv(getattr(m, "_k_scale", None)), v=tv(getattr(m, "_v_scale", None)),
                             q=tv(getattr(m, "_q_scale", None)), prob=tv(getattr(m, "_prob_scale", None)),
                             kf=getattr(m, "_k_scale_float", None), vf=getattr(m, "_v_scale_float", None),
                             qf=getattr(m, "_q_scale_float", None)))
        elif isinstance(m, LinearBase) and ".layers." in name:
            qm = getattr(m, "quant_method", None)
            rec = dict(layer=li(name), name=name, method=type(qm).__name__,
                       backend=str(getattr(qm, "backend", None)) if qm is not None else None)
            sch = getattr(m, "scheme", None)   # compressed-tensors scheme (None elsewhere)
            rec["scheme"] = type(sch).__name__ if sch is not None else None
            w = getattr(m, "weight", None)
            if isinstance(w, torch.Tensor):
                rec.update(weight_dtype=str(w.dtype), weight_shape=list(w.shape))
            for attr in ("weight_scale", "input_scale", "weight_scale_2", "weight_global_scale", "input_global_scale",
                         "alpha"):
                x = getattr(m, attr, None)
                if isinstance(x, torch.Tensor):
                    rec[attr] = tv(x) if x.numel() <= 8 else dict(shape=list(x.shape), dtype=str(x.dtype))
            lin.append(rec)
    vc = mr.vllm_config
    cc = vc.cache_config
    comp = vc.compilation_config
    sqm, nvm = sys.modules.get("verl.utils.specopd_quant"), sys.modules.get("verl.utils.specopd_nvfp4")
    sq_rec = None
    if sqm is not None and id(worker.vllm_config) in getattr(sqm, "_PATCHED", {}):
        p = sqm._PATCHED[id(worker.vllm_config)]
        sq_rec = {k: p.get(k) for k in ("n_linear", "lm_head", "n_attn", "nvfp4_widths", "nvfp4_warm", "static_kv",
                                         "nvfp4_missing", "nvfp4_unused", "unexpected", "smooth_missing",
                                         "smooth_unused", "smooth_split", "smooth_fused")}
    smm = sys.modules.get("verl.utils.specopd_smooth")
    return dict(quant_config=type(getattr(vc, "quant_config", None)).__name__, cache_dtype=str(cc.cache_dtype),
                sampler_quant_patched=sq_rec, nvfp4_op_seen=dict(getattr(nvm, "SEEN", {}) or {}) if nvm else None,
                smooth_op_seen=dict(getattr(smm, "SEEN", {}) or {}) if smm else None,
                compilation=dict(mode=str(getattr(comp, "mode", getattr(comp, "level", None))),
                                 cudagraph_mode=str(getattr(comp, "cudagraph_mode", None)).split(".")[-1],
                                 enforce_eager=bool(vc.model_config.enforce_eager)),
                kv_skip_layers=[str(x) for x in (getattr(cc, "kv_cache_dtype_skip_layers", None) or [])],
                calculate_kv_scales=bool(getattr(cc, "calculate_kv_scales", False)),
                model_path=str(vc.model_config.model), attn=attn, linear=lin)


def with_cudagraph_mode(kw, mode):
    """tim_calibrate.build_llm's LLM kwargs with compilation_config.cudagraph_mode replaced (compilation stays on)."""
    kw = dict(kw)
    cc = dict(kw.get("compilation_config") or {})
    cc["cudagraph_mode"] = mode
    kw["compilation_config"] = cc
    return kw


def _set_kv_worker(worker, served):
    """Runs INSIDE the in-process vLLM worker (a configuration's kv_from): every FP8 attention layer's k / v scale
    tensors and float mirrors := the export's calibrated scales as vLLM serves a checkpoint's (bf16; q / prob scales stay
    1.0, as vLLM leaves them), and the same again after any KV wake-up, which resets them to 1.0 (tim_calibrate never
    sleeps the engine). Returns what was written."""
    import torch
    from vllm.model_executor.layers.attention.attention import Attention

    def li(name):
        m = _LAYER.search(name)
        return int(m.group(1)) if m else -1

    layers = {li(n): m for n, m in worker.model_runner.get_model().named_modules()
              if isinstance(m, Attention) and str(m.kv_cache_dtype).startswith("fp8")}
    kv = {int(k): v for k, v in served.items()}

    def apply():
        with torch.no_grad():
            for i, m in layers.items():
                k, v = kv[i]
                m._k_scale.fill_(k)
                m._v_scale.fill_(v)
                m._k_scale_float = float(torch.tensor(k, dtype=torch.float32))
                m._v_scale_float = float(torch.tensor(v, dtype=torch.float32))

    missing = sorted(i for i in layers if i not in kv)
    if missing:
        raise RuntimeError(f"kv_from: no exported scale for FP8 layers {missing}")
    apply()
    for i, m in layers.items():   # vLLM's post-load processing (a weight reload) resets them to 1.0 as well
        pw = getattr(m, "process_weights_after_loading", None)
        if pw is not None:
            def pw_then_restore(*a_, _o=pw, **k_):
                out = _o(*a_, **k_)
                apply()
                return out
            m.process_weights_after_loading = pw_then_restore
    mr = worker.model_runner
    orig = getattr(mr, "init_fp8_kv_scales", None)
    if orig is not None:
        def init_fp8_kv_scales_then_restore(*a, **k):
            out = orig(*a, **k)
            apply()
            return out
        mr.init_fp8_kv_scales = init_fp8_kv_scales_then_restore
    return dict(written=sorted(layers), unused=sorted(set(kv) - set(layers)), wake_hook=orig is not None)


def install_llm_kwargs(extra, rec):
    """tim_calibrate.build_llm does `from vllm import LLM` at call time: a thin constructor that adds the configuration's
    extra engine arguments (e.g. calculate_kv_scales=True: vLLM computes the cache scales itself)."""
    import vllm
    orig = vllm.LLM

    def LLM(*a, **kw):   # noqa: N802
        kw = dict(kw, **extra)
        rec.update(extra)
        return orig(*a, **kw)

    vllm.LLM = LLM


def install_cudagraph_mode(mode, rec):
    """tim_calibrate.build_llm does `from vllm import LLM` at call time: replace that attribute by a thin constructor that
    only changes compilation_config.cudagraph_mode (torch.compile stays on: enforce_eager is not touched)."""
    import vllm
    orig = vllm.LLM

    def LLM(*a, **kw):   # noqa: N802
        kw = with_cudagraph_mode(kw, mode)
        rec.update(cudagraph_mode=mode, enforce_eager=kw.get("enforce_eager"))
        return orig(*a, **kw)

    vllm.LLM = LLM


def install_w4a8_view_fix(rec):
    """vLLM 0.19.1's convert_bf16_scales_to_fp8 (quantization/utils/quant_utils.py) ends with
    chan_scales.view(orig_shape[:-1], -1), which torch rejects (a Size and an int), so its CUTLASS W4A8 kernel loads no
    checkpoint. Installs the same function with view(*orig_shape[:-1], -1), in quant_utils and in the
    kernel module that imported the name; refuses if vLLM's source differs; counts the calls and the scale shapes."""
    import inspect

    import torch
    import vllm.model_executor.kernels.linear.mixed_precision.cutlass as K
    import vllm.model_executor.layers.quantization.utils.quant_utils as Q

    src = inspect.getsource(Q.convert_bf16_scales_to_fp8)
    assert "chan_scales = chan_scales.view(orig_shape[:-1], -1)" in src, "vLLM's convert_bf16_scales_to_fp8 changed"
    assert K.convert_bf16_scales_to_fp8 is Q.convert_bf16_scales_to_fp8
    rec.update(calls=0, shapes=[], fixed_line="chan_scales.view(*orig_shape[:-1], -1)")

    def convert_bf16_scales_to_fp8(quant_fp8, scales):
        assert scales.is_contiguous(), f"scale tensor must be contiguous, got {scales.stride()=}"
        assert scales.is_cuda, "scales must be on gpu"
        orig_shape = scales.shape
        k_groups = orig_shape[-1]
        flat_scales = scales.view(-1, k_groups)
        fp8_scales, chan_scales = quant_fp8(flat_scales)
        fp8_scales = (fp8_scales.float() / 8.0).to(torch.float8_e4m3fn)
        chan_scales *= 8.0
        fp8_scales = fp8_scales.view(orig_shape)
        chan_scales = chan_scales.view(*orig_shape[:-1], -1)
        rec["calls"] += 1
        if len(rec["shapes"]) < 8:
            rec["shapes"].append([list(orig_shape), list(chan_scales.shape)])
        return fp8_scales, chan_scales

    Q.convert_bf16_scales_to_fp8 = convert_bf16_scales_to_fp8
    K.convert_bf16_scales_to_fp8 = convert_bf16_scales_to_fp8


def install_nvfp4_patch_recorder(rec):
    """verl's worker extension does `from verl.utils.modelopt import apply_modelopt_nvfp4_patches` for a
    ModelOptNvFp4Config engine. Pre-seed that module with a recorder (no patch) so vLLM's own NVFP4 backend runs."""
    stub = types.ModuleType("verl.utils.modelopt")

    def apply_modelopt_nvfp4_patches():
        rec["calls"] = rec.get("calls", 0) + 1

    stub.apply_modelopt_nvfp4_patches = apply_modelopt_nvfp4_patches
    stub.__doc__ = "mo_measure stub: verl's ModelOpt NVFP4 rollout patch NOT installed (recorded)"
    sys.modules["verl.utils.modelopt"] = stub
    rec["installed_stub"] = True


def cmd_q(a):
    fd = C.formats_dict()
    if a.label not in fd:
        raise SystemExit(f"unknown label {a.label}")
    cfg = fd[a.label]
    for k in ENV_KNOBS:
        os.environ.pop(k, None)
    env = dict(cfg["env"])
    for k, v in env.items():
        if v == C.STATIC_PLACEHOLDER:
            if not a.static or not os.path.exists(a.static):
                raise SystemExit(f"{a.label}: needs --static <file> for {k} (got {a.static!r})")
            env[k] = os.path.abspath(a.static)
    os.environ.update(env)
    ckpt = os.path.join(a.ckpt_root, cfg["ckpt"]) if cfg["ckpt"] else None
    erec = None
    if ckpt:
        erec = json.load(open(os.path.join(ckpt, "export_record.json")))
    stub = {}
    if erec and str((erec.get("quantization_config") or {}).get("quant_algo", "")).upper() == "NVFP4":
        if not cfg["expect"].get("verl_nvfp4_patch", False):
            install_nvfp4_patch_recorder(stub)
    tc = import_tim_calibrate()
    w4fix = {}
    if cfg.get("vllm_w4a8_view_fix"):
        install_w4a8_view_fix(w4fix)
    graph = {}
    if cfg.get("cudagraph"):
        install_cudagraph_mode(cfg["cudagraph"], graph)
    llm_extra = {}
    if cfg.get("llm_kwargs"):
        install_llm_kwargs(dict(cfg["llm_kwargs"]), llm_extra)
    kv_served = None
    if cfg.get("kv_from"):
        kvrec = json.load(open(os.path.join(a.export_root or a.ckpt_root, cfg["kv_from"], "export_record.json")))
        kv_served = {int(li): (C.bf16(v["k"]), C.bf16(v["v"])) for li, v in kvrec["kv_scales"].items()}
    kvskip_stats = None
    if cfg.get("fa3_kvskip_patch"):
        # in-process engine: the patch on the FlashAttention backend module before the engine is built (bench/g0_kernels)
        import torch
        import vllm.v1.attention.backends.flash_attn as fa

        import g0_kernels
        kvskip_stats = g0_kernels.install_fa3_kvskip_patch(fa, torch)
    out = os.path.join(a.out_dir, f"engine_{a.label}.json")
    state = dict(label=a.label, cfg=cfg, ckpt=ckpt, env={k: os.environ.get(k) for k in ENV_KNOBS}, llm_kwargs=llm_extra,
                 kv_from=cfg.get("kv_from"), kv_from_served={str(k): v for k, v in (kv_served or {}).items()} or None,
                 verl_nvfp4_patch=stub or "not applicable",
                 export_quant_algo=((erec or {}).get("quantization_config") or {}).get("quant_algo"),
                 export_kv_scales=(erec or {}).get("kv_scales"), cudagraph_override=graph,
                 fa3_kvskip_patch=kvskip_stats, w4a8_view_fix=w4fix or "not applicable")
    orig_load, orig_build = tc.load_traces, tc.build_llm
    held = {}
    orig_reload = tc._reload_worker

    def short_generation(llm):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        t0 = held["traces"][0]
        o = llm.generate([TokensPrompt(prompt_token_ids=t0["prompt_ids"])],
                         [SamplingParams(max_tokens=C.SHORT_GEN_TOKENS, temperature=0.0)], use_tqdm=False)[0]
        ids = list(o.outputs[0].token_ids)
        return dict(prompt=f"{t0['bench']}:{t0['i']}", token_ids=ids, text=llm.get_tokenizer().decode(ids))

    def record_reload(llm):
        """tim_calibrate re-streams the weights through SamplerQuant with llm.collective_rpc(_reload_worker) after
        build_llm returns: keep that call's result (the sync record and the grid audit) and generate the short text
        again on the quantized weights. The rpc's RESULT is recorded (not a side effect inside the worker)."""
        orig_rpc = llm.collective_rpc

        def collective_rpc(method, *args, **kwargs):
            res = orig_rpc(method, *args, **kwargs)
            if method is orig_reload:
                rec = res[0] if res else None
                state["quant_sync"] = rec.get("quant") if isinstance(rec, dict) else None
                state["quant_audit"] = rec.get("quant_audit") if isinstance(rec, dict) else None
                state["short_generation_after_sync"] = short_generation(llm)
                C.write_json(out, state)
            return res

        llm.collective_rpc = collective_rpc

    def load_traces(n):
        tr = orig_load(n)
        held["traces"] = tr
        return tr

    def build_llm(mode, load_format, args=None):
        if mode == "q" and ckpt:
            old = tc.STUDENT
            tc.STUDENT = ckpt
            try:
                llm = orig_build(mode, load_format, args)
            finally:
                tc.STUDENT = old
        else:
            llm = orig_build(mode, load_format, args)
        held["llm"] = llm
        if kv_served is not None:
            state["kv_written"] = llm.collective_rpc(_set_kv_worker, args=(kv_served,))[0]
        state["audit_built"] = llm.collective_rpc(_audit_worker)[0]
        state["short_generation"] = short_generation(llm)
        record_reload(llm)
        C.write_json(out, state)
        log(f"{a.label}: engine built; quant_config={state['audit_built']['quant_config']}; "
            f"layer-0 attn={state['audit_built']['attn'][0] if state['audit_built']['attn'] else None}; "
            f"short generation: {state['short_generation']['text'][:160]!r}")
        return llm

    tc.load_traces, tc.build_llm = load_traces, build_llm
    sys.argv = tim_argv(a.label, a.out_dir)
    log(f"{a.label}: tim_calibrate {sys.argv[1:]} ckpt={ckpt} env={cfg['env']}")
    rc = tc.main()
    if "llm" in held:
        state["audit_after_decode"] = held["llm"].collective_rpc(_audit_worker)[0]
        state["fa3_kvskip_patch"] = dict(kvskip_stats) if kvskip_stats is not None else None
        C.write_json(out, state)
    return rc


def cmd_sleepwake(a):
    """What verl's sleep / wake order does to the static and cache scales, on the GPU: a Model Optimizer FP8 +
    calibrated-cache engine with sleep mode, audited after the build, after sleep(level=2) + wake_up(['weights'])
    (verl's order before a weight stream) and after wake_up(['kv_cache']) (verl's order before a rollout). No weights
    are streamed, so the weights themselves are garbage after the level-2 sleep; only the scales are read.
    -> sleepwake.json"""
    for k in ENV_KNOBS:
        os.environ.pop(k, None)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM
    ckpt = os.path.join(a.ckpt_root, a.export)
    llm = LLM(model=ckpt, dtype="bfloat16", gpu_memory_utilization=0.75, max_model_len=18433 + 1024,
              enable_prefix_caching=False, enforce_eager=False, seed=0, max_num_seqs=16, kv_cache_dtype="fp8",
              enable_sleep_mode=True, compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"})
    rec = dict(ckpt=ckpt, export_kv_scales=json.load(open(os.path.join(ckpt, "export_record.json"))).get("kv_scales"))
    rec["built"] = llm.collective_rpc(_audit_worker)[0]
    llm.sleep(level=2)
    llm.wake_up(tags=["weights"])
    rec["after_wake_weights"] = llm.collective_rpc(_audit_worker)[0]
    llm.wake_up(tags=["kv_cache"])
    rec["after_wake_kv_cache"] = llm.collective_rpc(_audit_worker)[0]
    out = os.path.join(a.out_dir, "sleepwake.json")
    C.write_json(out, rec)

    def l0(aud):
        a0 = next(r for r in aud["attn"] if r["layer"] == 0)
        q0 = next((r for r in aud["linear"] if r["layer"] == 0 and r["name"].endswith("qkv_proj")), {})
        return dict(k=a0["k"], v=a0["v"], q=a0["q"], kf=a0["kf"], vf=a0["vf"], qkv_input_scale=q0.get("input_scale"),
                    qkv_weight_scale=q0.get("weight_scale"))
    for tag in ("built", "after_wake_weights", "after_wake_kv_cache"):
        log(f"sleepwake {a.export} {tag}: layer 0 {l0(rec[tag])}")
    return 0


def cmd_score(a):
    for k in ENV_KNOBS:
        os.environ.pop(k, None)
    tc = import_tim_calibrate()
    sys.argv = ["tim_calibrate.py", "--mode", "score", "--arms", a.labels, "--n-per-bench", "8", "--out-dir", a.out_dir]
    return tc.main()


def selftest():
    argv = tim_argv("mo_fp8_kv_fl", "/d")
    assert argv[:5] == ["tim_calibrate.py", "--mode", "q", "--arm", "mo_fp8_kv_fl"]
    assert "--no-tf" in argv and argv[argv.index("--decode-seed") + 1] == "777"
    assert argv[argv.index("--kv-skip-layers") + 1] == "0,27" and argv[-2:] == ["--out-dir", "/d"]
    assert "--kv-cache-dtype" not in tim_argv("mo_z_none", "/d")
    rec = {}
    install_nvfp4_patch_recorder(rec)
    from verl.utils.modelopt import apply_modelopt_nvfp4_patches  # the stub (package verl need not be importable)
    apply_modelopt_nvfp4_patches()
    assert rec == {"installed_stub": True, "calls": 1}
    del sys.modules["verl.utils.modelopt"]
    kw = with_cudagraph_mode(dict(model="m", compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"},
                                  enforce_eager=False), "NONE")
    assert kw["compilation_config"] == {"cudagraph_mode": "NONE"} and kw["enforce_eager"] is False
    argv = tim_argv("mo_nvfp4emu_kv", "/d")
    assert argv[argv.index("--quant") + 1] == "w=nvfp4,a=nvfp4" and "--kv-cache-dtype" in argv
    assert C.selftest() == 0
    print("[mo_measure] SELFTEST PASS")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("q", "score", "sleepwake"))
    ap.add_argument("--label", default="")
    ap.add_argument("--labels", default="")
    ap.add_argument("--ckpt-root", default="")
    ap.add_argument("--export-root", default="", help="q: where a kv_from export lives (default --ckpt-root)")
    ap.add_argument("--export", default="fp8_kv", help="sleepwake: the export to probe")
    ap.add_argument("--static", default="", help="q: the static file of an emulated NVFP4 configuration")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    return dict(q=cmd_q, score=cmd_score, sleepwake=cmd_sleepwake)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
