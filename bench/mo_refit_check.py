#!/usr/bin/env python3
"""The refit gate of the Model Optimizer FP8 sampler: does the per-sync re-quantization inside vLLM reproduce, BYTE
FOR BYTE, the engine vLLM builds from Model Optimizer's own export of the same weights, and do the static and KV scales
survive verl's sleep / wake order? GPU, venv-specopd, one process per engine (VLLM_ENABLE_V1_MULTIPROCESSING=0).

    $PYTHON_BIN bench/mo_refit_check.py export-digest --export-dir X --out D/digest_export.json
    $PYTHON_BIN bench/mo_refit_check.py refit-digest  --scales S --out D/digest_refit.json
    $PYTHON_BIN bench/mo_refit_check.py compare --export D/digest_export.json --refit D/digest_refit.json --scales S \
                                                 --json D/refit_check.json            # CPU; exit 0 = PASS
    python3     bench/mo_refit_check.py selftest

export-digest: vLLM loads the Model Optimizer export X (FP8 weights, scales, calibrated cache) -> digest.
refit-digest: vLLM loads the bf16 student as a Model Optimizer FP8 engine exactly as a training rollout server does
  (quantization=modelopt with the scale file's quantization_config as hf_overrides, kv_cache_dtype=fp8, sleep mode,
  load_format=auto, verl's worker extension); then, inside the worker, the training code path:
    sync 1 (the student's bf16 weights streamed shard by shard through SamplerModelOpt.apply, the rest through the
           worker's own _update_weights, vLLM's post-load processing with the ModelOpt linear layers suppressed,
           SamplerModelOpt.finish_sync) -> digest 'after_sync1';
    sleep(level=2) + wake_up(['weights'])                                                      -> 'after_wake_weights';
    sync 2 (the same weights)                                                                  -> 'after_sync2';
    wake_up(['kv_cache']) (vLLM resets the k / v scales; SamplerModelOpt's hook restores them) -> 'after_wake_kv'.
  Both engines also generate 64 greedy tokens of one fixed prompt.
Recipe fp8fl (the export with the first and last layers unquantized, their cache bf16): both engines keep the scale
file's kv_bf16_layers in bf16 (kv_cache_dtype_skip_layers) and run with the FA3 kv-skip patch
(verl/utils/specopd_kvskip.py) as the training rollout does; the gate also requires those layers' cache to be bf16 in
both engines.
A digest: per quantized linear layer the sha256 of the live FP8 weight bytes, its shape, weight_scale and input_scale;
per attention layer the _k/_v/_q/_prob_scale tensors and the float mirrors.
PASS iff both engines generate the same greedy tokens on both prompts, and
'after_sync1', 'after_sync2' and 'after_wake_kv' equal the export digest in every linear layer (weight bytes,
weight_scale, input_scale), every k/v scale tensor and float equals the scale file's served value (the export's, rounded
to bf16 as vLLM serves it) and the export engine's own after the KV wake-up, q / prob scales
stay 1.0, the [SPECOPD-MO] sync records say layers=all, in_ok=1, kv_ok=1, and the KV wake-up hook saw vLLM's reset
(reset = every layer). Descriptive: what the level-2 sleep did to the static scales ('after_wake_weights'), and whether
the two greedy generations are identical.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mo_common as C  # noqa: E402

VERL_EXT = "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension"
PROMPT = "Compute 17 * 23 and explain each step."
GEN_TOKENS = 64
# the formats job's own short-generation prompt (tim_calibrate.load_traces' first trace, AIME24 problem 0, 48 tokens):
# the export engine here must reproduce engine_mo_fp8_kv.json's tokens (descriptive cross-check of the engine itself)
AIME_TOKENS = C.SHORT_GEN_TOKENS
FORMATS_D = C.FORMATS_DIR


def engine_kwargs(**kw):
    """tim_calibrate.build_llm's mode-q engine (the formats job's), plus the FP8 cache and sleep mode."""
    base = dict(dtype="bfloat16", gpu_memory_utilization=0.75, max_model_len=18433 + 1024, enable_prefix_caching=False,
                enforce_eager=False, seed=0, max_num_seqs=64, max_num_batched_tokens=4096, kv_cache_dtype="fp8",
                enable_sleep_mode=True, worker_extension_cls=VERL_EXT,
                compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"}, load_format="auto")
    base.update(kw)
    return base


def _install_kvskip_patch():
    """The FA3 kv-skip patch (verl/utils/specopd_kvskip.py, as a training rollout worker installs it) in this
    in-process engine's flash-attention backend module."""
    import torch
    import vllm.v1.attention.backends.flash_attn as fa

    from verl.utils.specopd_kvskip import install_fa3_kvskip_patch
    return install_fa3_kvskip_patch(fa, torch)


def _digest_worker(worker):
    """INSIDE the worker: the live state of every ModelOpt linear layer and attention layer."""
    import torch
    from vllm.model_executor.layers.attention.attention import Attention
    mr = worker.model_runner
    model = mr.get_model()
    lin, attn = {}, {}
    for name, m in model.named_modules():
        if type(getattr(m, "quant_method", None)).__name__ == "ModelOptFp8LinearMethod":
            w = m.weight.data
            b = w.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            lin[name] = dict(sha256=hashlib.sha256(b).hexdigest(), shape=list(w.shape), stride=list(w.stride()),
                             dtype=str(w.dtype), weight_scale=float(m.weight_scale), input_scale=float(m.input_scale))
        elif isinstance(m, Attention):
            attn[name] = dict(kv_cache_dtype=str(m.kv_cache_dtype), k=float(m._k_scale), v=float(m._v_scale),
                              q=float(m._q_scale), prob=float(m._prob_scale), kf=m._k_scale_float, vf=m._v_scale_float,
                              qf=getattr(m, "_q_scale_float", None))
    return dict(linear=lin, attn=attn, quant_config=type(getattr(mr.vllm_config, "quant_config", None)).__name__)


def _refit_worker(worker, step):
    """INSIDE the worker: one weight sync exactly as update_weights_from_ipc runs it under SPECOPD_SAMPLER_MODELOPT
    (the stream replaced by the student's safetensors shards, one bucket per shard)."""
    import glob

    import torch
    from safetensors import safe_open
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    from verl.utils.specopd_modelopt import SamplerModelOpt
    if getattr(worker, "_specopd_mo", None) is None:
        worker._specopd_mo = SamplerModelOpt.from_env(worker.model_runner)
    mo = worker._specopd_mo
    dev = torch.device("cuda", torch.cuda.current_device())
    for f in sorted(glob.glob(os.path.join(C.STUDENT, "*.safetensors"))):
        with safe_open(f, framework="pt", device=str(dev)) as fh:
            bucket = [(k, fh.get_tensor(k)) for k in fh.keys()]
        rest = mo.apply(bucket)
        worker._update_weights(rest, peft_config=None, base_sync_done=True)
        del bucket, rest
    with mo.no_linear_reprocess():
        for model, model_config in worker._iter_all_models_with_config():
            process_weights_after_loading(model, model_config, dev)
    rec = mo.finish_sync(step)
    torch.cuda.synchronize()
    return rec


def _aime0_prompt_ids():
    """AIME24 problem 0 exactly as tim_calibrate renders it (its load_traces, imported unchanged)."""
    import mo_measure
    tc = mo_measure.import_tim_calibrate()
    t = tc.load_traces(8)[0]
    assert (t["bench"], t["i"]) == ("AIME24", 0)
    return list(t["prompt_ids"])


def _generate(llm):
    """Greedy generations: the gate prompt (64 tokens) and, for the main pair, AIME24 problem 0 (48 tokens, the formats
    job's; tim_calibrate's traces are the main student's tokens, so another pair gets the gate prompt only)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    tok = llm.get_tokenizer()
    text = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    out = {}
    prompts = [("gate", ids, GEN_TOKENS)] + ([("aime24_0", _aime0_prompt_ids(), AIME_TOKENS)] if C.PAIR == "main" else [])
    for tag, pids, n in prompts:
        o = llm.generate([TokensPrompt(prompt_token_ids=pids)], [SamplingParams(max_tokens=n, temperature=0.0)],
                         use_tqdm=False)[0]
        toks = list(o.outputs[0].token_ids)
        out[tag] = dict(token_ids=toks, text=tok.decode(toks), prompt_len=len(pids))
    # the gate prompt's fields also at the top level, for the reader's descriptive line
    return dict(out["gate"], prompts=out)


def cmd_export_digest(a):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    for k in ("SPECOPD_SAMPLER_MODELOPT", "SPECOPD_SAMPLER_MODELOPT_SCALES", "SPECOPD_SAMPLER_MODELOPT_MODEL",
              "SPECOPD_SAMPLER_QUANT"):
        os.environ.pop(k, None)
    from vllm import LLM
    extra, kvs = {}, None
    if a.kv_skip:   # fp8fl's export engine, or fp8's control (the export with layer 0's cache in bf16, descriptive)
        extra = dict(kv_cache_dtype_skip_layers=[x for x in a.kv_skip.split(",") if x])
        kvs = _install_kvskip_patch()
    llm = LLM(model=a.export_dir, **engine_kwargs(**extra))
    rec = dict(engine="export" + (f"_skip{a.kv_skip}" if a.kv_skip else ""), export_dir=a.export_dir,
               digest=llm.collective_rpc(_digest_worker)[0], generation=_generate(llm),
               kvskip_patch=dict(kvs) if kvs is not None else None)
    C.write_json(a.out, rec)
    print(f"[mo_refit] export digest: {len(rec['digest']['linear'])} linear layers -> {a.out}; "
          f"generation {rec['generation']['text'][:120]!r}", flush=True)
    return 0


def cmd_refit_digest(a):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    sc = json.load(open(a.scales))
    os.environ.update(SPECOPD_SAMPLER_MODELOPT=sc["recipe"], SPECOPD_SAMPLER_MODELOPT_SCALES=os.path.abspath(a.scales),
                      SPECOPD_SAMPLER_MODELOPT_MODEL=C.STUDENT)
    os.environ.pop("SPECOPD_SAMPLER_QUANT", None)
    from vllm import LLM
    extra, kvs = {}, None
    if sc.get("kv_bf16_layers"):
        extra = dict(kv_cache_dtype_skip_layers=[str(i) for i in sc["kv_bf16_layers"]])
        kvs = _install_kvskip_patch()
    llm = LLM(model=C.STUDENT, quantization="modelopt", hf_overrides={"quantization_config": sc["quantization_config"]},
              **engine_kwargs(**extra))
    rec = dict(engine="refit", scales=os.path.abspath(a.scales), syncs=[])
    rec["syncs"].append(llm.collective_rpc(_refit_worker, args=(1,))[0])
    rec["after_sync1"] = llm.collective_rpc(_digest_worker)[0]
    llm.sleep(level=2)
    llm.wake_up(tags=["weights"])
    rec["after_wake_weights"] = llm.collective_rpc(_digest_worker)[0]
    rec["syncs"].append(llm.collective_rpc(_refit_worker, args=(2,))[0])
    rec["after_sync2"] = llm.collective_rpc(_digest_worker)[0]
    llm.wake_up(tags=["kv_cache"])
    rec["after_wake_kv"] = llm.collective_rpc(_digest_worker)[0]
    rec["wakes"] = llm.collective_rpc(lambda w: w._specopd_mo.wakes)[0]
    rec["generation"] = _generate(llm)
    rec["kvskip_patch"] = dict(kvs) if kvs is not None else None
    C.write_json(a.out, rec)
    print(f"[mo_refit] refit digest -> {a.out}; syncs {rec['syncs']}; generation {rec['generation']['text'][:120]!r}",
          flush=True)
    return 0


def f32(x):
    import struct
    return struct.unpack("f", struct.pack("f", float(x)))[0]


def compare(exp, ref, sc, formats_gen=None, control=None):
    """The gate (see the module docstring). Returns (problems, descriptive record)."""
    bad, desc = [], {}
    E = exp["digest"]["linear"]
    if not E:
        bad.append("the export engine has no ModelOpt FP8 linear layers")
    for tag in ("after_sync1", "after_sync2", "after_wake_kv"):
        D = ref[tag]["linear"]
        if sorted(D) != sorted(E):
            bad.append(f"{tag}: linear layers differ from the export's ({len(D)} vs {len(E)})")
            continue
        for k in ("sha256", "shape", "stride", "weight_scale", "input_scale"):
            n = sum(1 for ln in E if E[ln][k] != D[ln][k])
            if n:
                bad.append(f"{tag}: {n}/{len(E)} layers differ from the export in {k}")
    kv = {str(k): v for k, v in sc["kv"].items()}
    A = ref["after_wake_kv"]["attn"]
    fp8 = {n: r for n, r in A.items() if r["kv_cache_dtype"].startswith("fp8")}
    import re
    for n, r in fp8.items():
        li = re.search(r"layers\.(\d+)\.", n).group(1)
        want = kv.get(li)
        if want is None:
            bad.append(f"{n}: no k/v scale in the file")
            continue
        if (r["k"], r["v"], r["kf"], r["vf"]) != (f32(want["k"]), f32(want["v"]), f32(want["k"]), f32(want["v"])):
            bad.append(f"{n}: after the KV wake-up k/v {r['k']}/{r['v']} (floats {r['kf']}/{r['vf']}) != file {want}")
        if (r["q"], r["prob"]) != (1.0, 1.0):
            bad.append(f"{n}: q/prob scale {r['q']}/{r['prob']} (want vLLM's 1.0)")
    if len(fp8) != len(kv):
        bad.append(f"{len(fp8)} FP8 cache layers vs {len(kv)} in the file")
    for li in sc.get("kv_bf16_layers", []):
        for who, att in (("refit", A), ("export", exp["digest"]["attn"])):
            rows = [r for n, r in att.items() if re.search(rf"layers\.{li}\.", n)]
            if not rows or any(r["kv_cache_dtype"] != "auto" for r in rows):
                bad.append(f"{who} engine: layer {li}'s cache is not bf16 ({[r['kv_cache_dtype'] for r in rows]})")
    ex_attn = exp["digest"]["attn"]
    n_kv_exp = sum(1 for n, r in ex_attn.items() if r["kv_cache_dtype"].startswith("fp8")
                   and (r["k"], r["v"]) != (A[n]["k"], A[n]["v"]))
    if n_kv_exp:
        bad.append(f"{n_kv_exp} layers: the export engine's k/v scales differ from the refit engine's after the wake-up")
    for s in ref.get("syncs", []):
        if not (str(s.get("layers", "")).split("/")[0] == str(s.get("layers", "")).split("/")[-1] and
                s.get("in_ok") == 1 and s.get("kv_ok") == 1):
            bad.append(f"sync record {s}")
    if ref.get("wakes") != 1:
        bad.append(f"the KV wake-up hook ran {ref.get('wakes')} times (want 1)")
    W = ref["after_wake_weights"]
    desc["after_wake_weights"] = dict(
        weight_sha_equal=sum(1 for ln in E if W["linear"].get(ln, {}).get("sha256") == E[ln]["sha256"]),
        input_scale_equal=sum(1 for ln in E if W["linear"].get(ln, {}).get("input_scale") == E[ln]["input_scale"]),
        weight_scale_equal=sum(1 for ln in E if W["linear"].get(ln, {}).get("weight_scale") == E[ln]["weight_scale"]),
        kv_equal_file=sum(1 for n, r in W["attn"].items() if r["kv_cache_dtype"].startswith("fp8")
                          and r["k"] == f32(kv[re.search(r"layers\.(\d+)\.", n).group(1)]["k"])),
        n_linear=len(E))
    pe, pr = (x["generation"].get("prompts") or {"gate": x["generation"]} for x in (exp, ref))
    for tag in sorted(pe):
        if pe[tag]["token_ids"] != pr.get(tag, {}).get("token_ids"):
            bad.append(f"greedy generation ({tag}) differs between the export engine and the refit engine "
                       "(identical live state must give identical tokens)")
    desc["generation_identical"] = all(pe[t]["token_ids"] == pr.get(t, {}).get("token_ids") for t in pe)
    desc["generation_export"] = {t: v["text"][:200] for t, v in pe.items()}
    desc["generation_refit"] = {t: v["text"][:200] for t, v in pr.items()}
    if formats_gen is not None:
        desc["export_aime24_0_equals_formats_job"] = (pe.get("aime24_0", {}).get("token_ids") == formats_gen)
    if control is not None:
        desc["control_layer0_cache_bf16"] = {t: v["text"][:200] for t, v in
                                             (control["generation"].get("prompts") or {}).items()}
    return bad, desc


def cmd_compare(a):
    exp, ref, sc = (json.load(open(p)) for p in (a.export, a.refit, a.scales))
    fg = None
    fe = os.path.join(FORMATS_D, "engine_mo_fp8_kv_fl.json" if sc.get("recipe") == "fp8fl" else "engine_mo_fp8_kv.json")
    if C.PAIR == "main" and os.path.exists(fe):
        fg = (json.load(open(fe)).get("short_generation") or {}).get("token_ids")
    ctl = json.load(open(a.control)) if a.control and os.path.exists(a.control) else None
    bad, desc = compare(exp, ref, sc, fg, ctl)
    out = dict(verdict="PASS" if not bad else "FAIL", problems=bad, descriptive=desc,
               n_linear=len(exp["digest"]["linear"]))
    if a.json:
        C.write_json(a.json, out)
    print(f"[mo_refit] REFIT GATE {out['verdict']}: {len(bad)} problems {bad[:6]}; descriptive {desc}")
    return 0 if not bad else 8


def selftest():
    lin = {f"model.layers.{i}.self_attn.qkv_proj": dict(sha256=f"h{i}", shape=[3584, 4608], stride=[1, 3584],
                                                       dtype="torch.float8_e4m3fn", weight_scale=0.01, input_scale=0.1)
           for i in range(2)}
    attn = {f"model.layers.{i}.self_attn.attn": dict(kv_cache_dtype="fp8", k=f32(0.05 * (i + 1)), v=f32(0.02), q=1.0,
                                                     prob=1.0, kf=f32(0.05 * (i + 1)), vf=f32(0.02), qf=1.0)
            for i in range(2)}
    sc = dict(kv={str(i): dict(k=0.05 * (i + 1), v=0.02) for i in range(2)})
    exp = dict(digest=dict(linear=lin, attn=attn), generation=dict(token_ids=[1, 2], text="x"))
    garbage = {n: dict(r, sha256="zz", input_scale=7.0) for n, r in lin.items()}
    ref = dict(after_sync1=dict(linear=lin, attn=attn), after_sync2=dict(linear=lin, attn=attn),
               after_wake_kv=dict(linear=lin, attn=attn), after_wake_weights=dict(linear=garbage, attn=attn),
               syncs=[dict(layers="2/2", in_ok=1, kv_ok=1)] * 2, wakes=1, generation=dict(token_ids=[1, 2], text="x"))
    bad, desc = compare(exp, ref, sc)
    assert not bad, bad
    assert desc["after_wake_weights"]["input_scale_equal"] == 0 and desc["generation_identical"]
    ref2 = json.loads(json.dumps(ref))
    ref2["after_sync2"]["linear"]["model.layers.1.self_attn.qkv_proj"]["sha256"] = "other"
    assert any("after_sync2" in p and "sha256" in p for p in compare(exp, ref2, sc)[0])
    ref4 = json.loads(json.dumps(ref))
    ref4["generation"]["token_ids"] = [1, 3]
    assert any("greedy generation" in p for p in compare(exp, ref4, sc)[0])
    ref3 = json.loads(json.dumps(ref))
    for r in ref3["after_wake_kv"]["attn"].values():
        r["k"] = r["v"] = 1.0
    assert any("after the KV wake-up" in p for p in compare(exp, ref3, sc)[0])
    # fp8fl: layer 1 is in the scale file's kv_bf16_layers -> its cache must be bf16 in both engines
    scfl = dict(recipe="fp8fl", kv={"0": sc["kv"]["0"]}, kv_bf16_layers=[1])
    attn_fl = json.loads(json.dumps(attn))
    attn_fl["model.layers.1.self_attn.attn"]["kv_cache_dtype"] = "auto"
    exfl = dict(exp, digest=dict(linear=lin, attn=attn_fl))
    reffl = json.loads(json.dumps(ref))
    for tag in ("after_sync1", "after_sync2", "after_wake_kv", "after_wake_weights"):
        reffl[tag]["attn"] = attn_fl
    assert not compare(exfl, reffl, scfl)[0], compare(exfl, reffl, scfl)[0]
    assert any("layer 1's cache is not bf16" in p for p in compare(exp, reffl, scfl)[0])
    assert C.selftest() == 0
    print("[mo_refit] SELFTEST PASS")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("export-digest", "refit-digest", "compare"))
    ap.add_argument("--export-dir", default="")
    ap.add_argument("--scales", default="")
    ap.add_argument("--export", default="")
    ap.add_argument("--refit", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--kv-skip", default="", help="export-digest: a control engine with these layers' cache in bf16")
    ap.add_argument("--control", default="", help="compare: the control engine's digest (descriptive)")
    a = ap.parse_args(argv)
    return {"export-digest": cmd_export_digest, "refit-digest": cmd_refit_digest, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
