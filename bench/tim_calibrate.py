"""Sampler-trainer mismatch calibration: per-token KL between the clean student p and a perturbed SAMPLER q, offline,
one GPU.

    python bench/tim_calibrate.py --mode bf16 --targets '' --traces T --out-dir D      # clean ref + noise grid
    python bench/tim_calibrate.py --mode bf16 --sigmas '' --traces T --out-dir D       # targets (+1 secant step)
    python bench/tim_calibrate.py --mode fp8w_fp8kv --traces T --out-dir D             # FP8 weights + FP8 KV cache
    python bench/tim_calibrate.py --mode fp8w | kvfp8 --traces T --out-dir D           # its two halves
    python bench/tim_calibrate.py --mode report --out-dir D

  SAMPLER QUANTIZATION (Score Centering's fake-quant sampler, verl/utils/specopd_quant.py) and vLLM-native KV options,
  one engine configuration per process; every configuration records BOTH estimators:
    python bench/tim_calibrate.py --mode bf16 --sigmas '' --targets '' --traces T --out-dir D   # clean reference
    python bench/tim_calibrate.py --mode q --arm fp8_emu --quant w=e4m3,a=e4m3,kv=e4m3_unit --traces T --out-dir D
    python bench/tim_calibrate.py --mode q --arm kv_fp8 --kv-cache-dtype fp8 [--kv-skip-layers 0,27]
                                  [--attention-backend TRITON_ATTN] --traces T --out-dir D
    python bench/tim_calibrate.py --mode q --arm none --traces T --out-dir D   # clean engine: decode-vs-prefill floor
    python bench/tim_calibrate.py --mode score [--arms a,b] --out-dir D # clean engine scores unscored samples_*
    python bench/tim_calibrate.py --mode attr --traces T --out-dir D    # per-layer / per-head KV attribution (eager)
  (1) prefill, teacher-forced: the patched engine re-scores the 16 base traces -> KL(q||p) IS estimate as above
      (kl_q_<arm>.jsonl; degenerate when the mismatch is large: watch mean r).
  (2) decode: the patched engine SAMPLES y ~ q (T=1, logprobs=0 = log q of each sampled token, the rollout path) from
      the same 16 prompts (n per prompt, fixed per-request seeds shared by every configuration, up to 16384 tokens = the
      training response cap), then a CLEAN bf16 engine teacher-forces the same tokens for log p:  kl_samp_train = mean
      over sampled tokens of (log q - log p) = E_q[log q - log p] = KL(q||p), exactly Score Centering's metric (its
      losses.py), plus the k3 form E_q[p/q - 1 - log(p/q)], a bootstrap CI over PROMPTS and the same per
      response-position bin (0-1k = Score Centering's seq_len regime, 1k-4k, 4k-16k): dec_<arm>.json + per-token
      dec_<arm>.npz. p is vLLM's bf16 prefill, a proxy for the FSDP trainer (the BF16 vLLM engine's mismatch against
      the FSDP trainer, rollout_corr/kl, is the --natural input; --mode q --arm none measures the decode-vs-prefill
      part).

Traces (--traces): 16 of the base student's OWN T=1.0 samples (the gen_T1_p1.jsonl that refresh/t3_acc.py gen writes
for the base student: AIME24 problems 0-7 sample 0 + MATH-500 problems 0-7), prompts built exactly as refresh/t3_acc.py
does. The decode path uses only their prompts: without --traces the same prompts are built from the benchmark files,
and the teacher-forced modes refuse. Every engine TEACHER-FORCES the same token ids (prompt_logprobs=0 -> log-prob of
the actual next token at every position), so the traces y ~ p and, per response token, lp = log p(y), lq = log q(y),
r = q(y)/p(y):
    KL(q||p)  = E_p[r log r]          (importance-weighted; the direction rollout_corr/kl measures: E_q[log q - log p])
    KL(p||q)  = E_p[log p - log q] (k1)  and  E_p[r - 1 - log r] (k3, >= 0 per token)
For a small perturbation the two directions agree to second order; the report gives all three plus mean(r) (~1).

The sampler perturbations are applied with the TRAINING code, not a re-implementation:
  * noise: verl.utils.specopd_tim.SamplerNoise.apply on the HF-named bf16 checkpoint tensors (= the actor stream's
    names and dtype), then vLLM's model.load_weights -- the same calls update_weights_from_ipc makes. A sigma=0 reload
    must reproduce the clean reference (checks the loader), and its d_eps / d_n must equal the training run's
    step-0 [SPECOPD-TIM] line (checks that calibration and training noise are the same vector).
  * fp8: the engine is built as verl's vllm_async_server._apply_quantization builds it (quantization=fp8 + the
    blockwise 128x128 e4m3 hf_overrides, apply_vllm_quant_patches before the engine) and refit through verl's
    prepare/load/process_quanted_weights; fp8 KV = kv_cache_dtype="fp8". load_format=auto: the engine loads the
    checkpoint's weights at init.
The engine runs IN-PROCESS (VLLM_ENABLE_V1_MULTIPROCESSING=0) so collective_rpc can hand the worker a callable.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

STUDENT = os.environ.get("CLUSTER_ROOT", "/path/to/workdir") + "/hf/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60"
TRACES = ""   # --traces: the base student's own T=1 samples (jsonl); "" = the prompts only, built from DATA
DATA = os.environ.get("CLUSTER_ROOT", "/path/to/workdir") + "/workspace/datasets/test_data"
FP8_BLOCK_QUANT_KWARGS = {  # verbatim from vllm_async_server._apply_quantization (no MoE gates in Qwen2)
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
    "ignored_layers": [],
}


def load_traces(n_per_bench):
    import pandas as pd
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STUDENT)
    prompts = {}
    for b in ("AIME24", "MATH-500"):
        df = pd.read_parquet(os.path.join(DATA, b, "test.parquet"))
        prompts[b] = [[dict(m) for m in r["prompt"]] for r in df.to_dict("records")]
    if TRACES:
        rows = (json.loads(line) for line in open(TRACES))
    else:   # the same (bench, problem) prompts without response tokens: all the decode path needs
        print("[cal] no --traces: prompts only (AIME24 / MATH-500 problems 0..n-1), no response tokens", flush=True)
        rows = [dict(bench=b, i=i, s=0, response_ids=[]) for b in prompts
                for i in range(min(n_per_bench, len(prompts[b])))]
    out = []
    for r in rows:
        if r["s"] == 0 and r["i"] < n_per_bench:
            text = tok.apply_chat_template(prompts[r["bench"]][r["i"]], tokenize=False, add_generation_prompt=True)
            pids = tok(text, add_special_tokens=False)["input_ids"]
            out.append(dict(bench=r["bench"], i=r["i"], prompt_ids=pids, response_ids=r["response_ids"]))
    out.sort(key=lambda t: (t["bench"], t["i"]))
    return out


VERL_EXT = "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension"


def build_llm(mode, load_format, a=None):
    from vllm import LLM
    kw = dict(model=STUDENT, dtype="bfloat16", gpu_memory_utilization=0.75, max_model_len=18433 + 1024,
              enable_prefix_caching=False, enforce_eager=False, seed=0, max_num_seqs=16, max_num_batched_tokens=4096,
              load_format=load_format)
    if mode in ("q", "score", "attr"):
        # The rollout's engine as verl builds it: verl's worker extension (whose __new__ installs the sampler-quant
        # patches when SPECOPD_SAMPLER_QUANT is set -- the training code path), FULL_AND_PIECEWISE CUDA graphs.
        # attr: the same engine, clean and EAGER (bench-only per-layer forward overrides need a per-call forward).
        kw.update(worker_extension_cls=VERL_EXT, compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"},
                  max_num_seqs=64, enforce_eager=bool(mode == "attr" or (a is not None and a.enforce_eager)))
        if mode == "q" and a is not None:
            if a.kv_cache_dtype != "auto":
                kw.update(kv_cache_dtype=a.kv_cache_dtype)
            if a.kv_skip_layers:
                kw.update(kv_cache_dtype_skip_layers=[x.strip() for x in a.kv_skip_layers.split(",") if x.strip()])
            if a.attention_backend:
                kw.update(attention_backend=a.attention_backend)
    if mode in ("fp8w", "fp8w_fp8kv"):
        from verl.utils.vllm.vllm_quant_utils import apply_vllm_quant_patches
        apply_vllm_quant_patches()
        os.environ["VERL_VLLM_FP8_QUANT_ENABLED"] = "1"
        kw.update(quantization="fp8", hf_overrides={"quantization_config": dict(FP8_BLOCK_QUANT_KWARGS)})
    if mode in ("kvfp8", "fp8w_fp8kv"):
        kw.update(kv_cache_dtype="fp8")
    return LLM(**kw)


def teacher_force(llm, traces):
    """Per trace: float32 array of log-probs of the RESPONSE tokens under the engine's current weights."""
    import numpy as np
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0)
    reqs = [TokensPrompt(prompt_token_ids=t["prompt_ids"] + t["response_ids"]) for t in traces]
    outs = llm.generate(reqs, [sp] * len(reqs), use_tqdm=False)
    res = []
    for t, o in zip(traces, outs):
        ids = t["prompt_ids"] + t["response_ids"]
        pl = o.prompt_logprobs
        lp = [pl[k][ids[k]].logprob for k in range(len(t["prompt_ids"]), len(ids))]
        res.append(np.asarray(lp, dtype=np.float64))
    return res


def _shards():
    return sorted(glob.glob(os.path.join(STUDENT, "*.safetensors")))


def _reload_worker(worker, sigma, seed, fp8, quant=""):
    """Runs INSIDE the vLLM worker: stream the checkpoint shard by shard through SamplerNoise.apply (and, with
    ``quant``, specopd_quant.SamplerQuant.apply after it) into the model, exactly as update_weights_from_ipc does with
    the actor's buckets (fp8: verl's quantized refit)."""
    import torch
    from safetensors import safe_open

    from verl.utils.specopd_tim import SamplerNoise
    runner = worker.model_runner
    model = runner.model
    dev = torch.device("cuda", torch.cuda.current_device())
    tim = SamplerNoise(sigma, seed) if (sigma > 0 or not quant) else None
    squant = None
    if quant:
        from verl.utils.specopd_quant import QuantSpec, SamplerQuant
        squant = SamplerQuant(QuantSpec.parse(quant))
    if fp8:
        from verl.utils.vllm.vllm_quant_utils import (
            load_quanted_weights, prepare_quanted_weights_for_loading, process_quanted_weights_after_loading)
        state = prepare_quanted_weights_for_loading(model)
    for f in _shards():
        with safe_open(f, framework="pt", device=str(dev)) as fh:
            bucket = [(k, fh.get_tensor(k)) for k in fh.keys()]
        if tim is not None:
            bucket = tim.apply(bucket)
        if squant is not None:
            bucket = squant.apply(bucket)
        if fp8:
            load_quanted_weights(bucket, runner)
        else:
            model.load_weights(bucket)
        del bucket
    rec = tim.finish_sync() if tim is not None else {}
    if squant is not None:
        rec = dict(rec, quant=squant.finish_sync())
    if fp8:
        process_quanted_weights_after_loading(model, state)
    else:
        from vllm.model_executor.model_loader.utils import process_weights_after_loading
        process_weights_after_loading(model, runner.vllm_config.model_config, dev)
    torch.cuda.synchronize()
    if squant is not None:
        from verl.utils.specopd_quant import audit_weights
        rec["quant_audit"] = audit_weights(runner.get_model(), squant.spec)
    return rec


def _engine_audit(worker):
    """Runs INSIDE the vLLM worker: what the sampler-quant patch attached to this engine (None = nothing)."""
    import sys
    m = sys.modules.get("verl.utils.specopd_quant")
    if m is None:
        return None
    rec = m._PATCHED.get(id(worker.vllm_config))
    if rec is None:
        return dict(patched=False, registered=id(worker.vllm_config) in m._ENGINE_SPECS)
    return dict(patched=True, n_linear=rec["n_linear"], lm_head=rec["lm_head"], n_attn=rec["n_attn"],
                attn_skipped=rec.get("attn_skipped", []), drafter=rec["drafter"])


def prompts_of(traces):
    return [dict(bench=t["bench"], i=t["i"], prompt_ids=t["prompt_ids"]) for t in traces]


def decode_sample(llm, traces, n, max_tokens, seed):
    """y ~ q on the DECODE path: per prompt n samples with fixed per-request seeds (identical across arms, so arms are
    paired by their random numbers), logprobs=0 -> log q of every sampled token as the engine computed it."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    reqs, sps, meta = [], [], []
    for t in prompts_of(traces):
        for j in range(n):
            sd = seed + 1000 * (0 if t["bench"] == "AIME24" else 1) + 10 * t["i"] + j
            reqs.append(TokensPrompt(prompt_token_ids=t["prompt_ids"]))
            sps.append(SamplingParams(temperature=1.0, top_p=1.0, max_tokens=max_tokens, logprobs=0, seed=sd))
            meta.append(dict(bench=t["bench"], i=t["i"], j=j, seed=sd, max_tokens=max_tokens,
                             prompt_ids=t["prompt_ids"]))
    outs = llm.generate(reqs, sps, use_tqdm=False)
    res = []
    for m, o in zip(meta, outs):
        c = o.outputs[0]
        ids = list(c.token_ids)
        lq = [float(c.logprobs[k][ids[k]].logprob) for k in range(len(ids))]
        res.append(dict(m, response_ids=ids, lq=lq, finish_reason=c.finish_reason))
    return res


POS_BINS = ((0, 1024), (1024, 4096), (4096, 16384), (0, 16384))  # response-position bins; 0-1k = paper-comparable


def _cluster_ci(sums, counts, groups, n_boot=2000, seed=0):
    """Token-pooled ratio sum/count with a bootstrap over CLUSTERS (prompts): the samples of one prompt are
    correlated, so the prompt -- not the sample -- is resampled (all of a drawn prompt's samples together)."""
    import numpy as np
    keys = sorted(set(groups))
    gs = np.array([sum(s for s, g in zip(sums, groups) if g == k) for k in keys], np.float64)
    gc = np.array([sum(c for c, g in zip(counts, groups) if g == k) for k in keys], np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), (n_boot, len(keys)))
    num, den = gs[idx].sum(1), gc[idx].sum(1)
    ok = den > 0
    if not ok.any():
        return [float("nan"), float("nan")], len(keys)
    bs = num[ok] / den[ok]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], len(keys)


def kl_decode_stats(lq, lp, groups, n_boot=2000, seed=0):
    """y ~ q: d = log q - log p per sampled token. kl_samp_train = mean d (the paper's metric, KL(q||p) k1, token-pooled
    = their sums/count), k3 = mean(exp(-d) - 1 + d) >= 0 per token; bootstrap CI over PROMPTS (``groups``: one key per
    sample; samples of the same prompt are resampled together). Also per response-position bin (POS_BINS): the 0-1k bin
    is the paper-comparable number (their seq_len <= 1024), the full range the training-comparable one (16k cap)."""
    import numpy as np
    d = [np.asarray(a, np.float64) - np.asarray(b, np.float64) for a, b in zip(lq, lp)]
    cat = np.concatenate(d) if d else np.zeros(0)
    out = dict(n_tok=int(cat.size), n_samples=len(d), n_prompts=len(set(groups)),
               kl_samp_train=float(np.mean(cat)) if cat.size else float("nan"),
               kl_k3=float(np.mean(np.exp(-cat) - 1 + cat)) if cat.size else float("nan"),
               mean_abs_d=float(np.mean(np.abs(cat))) if cat.size else float("nan"),
               max_abs_d=float(np.max(np.abs(cat))) if cat.size else float("nan"),
               mean_len=float(np.mean([x.size for x in d])) if d else 0.0)
    out["kl_samp_train_ci95"], _ = _cluster_ci([float(x.sum()) for x in d], [x.size for x in d], groups, n_boot, seed)
    bins = {}
    for lo, hi in POS_BINS:
        seg = [x[lo:hi] for x in d]
        n = sum(s.size for s in seg)
        if n == 0:
            bins[f"{lo}-{hi}"] = dict(n_tok=0)
            continue
        c = np.concatenate(seg)
        ci, _ = _cluster_ci([float(s.sum()) for s in seg], [s.size for s in seg], groups, n_boot, seed)
        bins[f"{lo}-{hi}"] = dict(n_tok=int(n), n_samples=int(sum(s.size > 0 for s in seg)),
                                  kl_samp_train=float(np.mean(c)), ci95=ci, kl_k3=float(np.mean(np.exp(-c) - 1 + c)))
    out["bins"] = bins
    return out


def score_samples(llm, out_dir, arms=None, rescore=False):
    """CLEAN engine: teacher-force samples_<arm>.jsonl in out_dir (prompt_logprobs=0) -> log p of the sampled tokens.
    Per arm (one file each, so concurrent jobs never share a file): dec_<arm>.json (kl_decode_stats) and dec_<arm>.npz
    (per-token lq, lp, per-sample offsets / prompt keys, for any later re-binning). ``arms``: only these labels (None =
    every samples_*). Arms already scored are skipped unless rescore."""
    import numpy as np
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    for f in sorted(glob.glob(os.path.join(out_dir, "samples_*.jsonl"))):
        rows = [json.loads(l) for l in open(f)]
        if not rows:
            continue
        arm = rows[0]["arm"]
        if arms is not None and arm not in arms:
            continue
        js = os.path.join(out_dir, f"dec_{arm}.json")
        if os.path.exists(js) and not rescore:
            print(f"[cal] score: {arm} already scored, skip", flush=True)
            continue
        t = time.time()
        n_all = len(rows)
        rows = [r for r in rows if r["response_ids"]]
        sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0)
        reqs = [TokensPrompt(prompt_token_ids=r["prompt_ids"] + r["response_ids"]) for r in rows]
        outs = llm.generate(reqs, [sp] * len(reqs), use_tqdm=False)
        lp = []
        for r, o in zip(rows, outs):
            ids = r["prompt_ids"] + r["response_ids"]
            pl = o.prompt_logprobs
            lp.append([pl[k][ids[k]].logprob for k in range(len(r["prompt_ids"]), len(ids))])
        lq = [r["lq"] for r in rows]
        groups = [f"{r['bench']}:{r['i']}" for r in rows]
        st = kl_decode_stats(lq, lp, groups)
        st.update(arm=arm, path="decode", cfg=rows[0].get("cfg"), n_empty=n_all - len(rows),
                  frac_truncated=float(sum(r["finish_reason"] == "length" for r in rows) / max(1, len(rows))),
                  max_tokens=rows[0].get("max_tokens"), s=round(time.time() - t, 1))
        print("[cal] " + json.dumps(st), flush=True)
        lens = np.array([len(x) for x in lq], np.int64)
        tmp = os.path.join(out_dir, f"dec_{arm}.tmp.npz")
        np.savez(tmp, lq=np.concatenate([np.asarray(x, np.float32) for x in lq]) if lq else np.zeros(0, np.float32),
                 lp=np.concatenate([np.asarray(x, np.float32) for x in lp]) if lp else np.zeros(0, np.float32),
                 lens=lens, groups=np.array(groups), finish=np.array([r["finish_reason"] or "" for r in rows]))
        os.replace(tmp, os.path.join(out_dir, f"dec_{arm}.npz"))
        with open(js + ".tmp", "w") as fo:
            fo.write(json.dumps(st) + "\n")
        os.replace(js + ".tmp", js)


def _attr_set(worker, cfg):
    """Runs INSIDE an EAGER clean vLLM worker (attribution only; never a training path): override forward() of the
    target model's attention layers so K and/or V are snapped on ``cfg['grid']`` with specopd_quant.kv_qdq (the
    sampler's own quantizer) on the selected layers (None = all) and KV heads (None = all). cfg None = restore clean.
    Instance-level forward overrides take effect per call only because the engine is eager (no Dynamo trace / graph).
    Returns the number of layers snapped."""
    import types

    import torch
    from vllm.model_executor.layers.attention.attention import Attention

    from verl.utils.specopd_quant import _layer_idx, kv_qdq, parse_grid
    model = worker.model_runner.get_model()
    n = 0
    for name, m in model.named_modules():
        if not isinstance(m, Attention):
            continue
        m.__dict__.pop("forward", None)
        if cfg is None:
            continue
        li = _layer_idx(name + ".")
        if cfg.get("layers") is not None and li not in cfg["layers"]:
            continue
        g, part, heads = parse_grid(cfg["grid"], "kv"), cfg.get("part", "kv"), cfg.get("heads")

        def fwd(self, query, key, value, output_shape=None, _g=g, _part=part, _heads=heads, _base=type(m).forward):
            def snap(u, hd):
                u3 = u.reshape(-1, self.num_kv_heads, hd)
                s = kv_qdq(u3, _g)
                if _heads is not None:
                    mask = torch.zeros(self.num_kv_heads, dtype=torch.bool, device=u.device)
                    mask[list(_heads)] = True
                    s = torch.where(mask[None, :, None], s, u3)
                return s.reshape(u.shape)

            if key is not None and "k" in _part:
                key = snap(key, self.head_size)
            if value is not None and "v" in _part:
                value = snap(value, self.head_size_v)
            return _base(self, query, key, value, output_shape)

        m.forward = types.MethodType(fwd, m)
        n += 1
    return n


def run_attr(llm, traces, out_dir, ref_path, a):
    """Per-layer / per-head attribution of the EMULATED KV KL on one eager clean engine, teacher-forced prefill only.
    Every configuration is scored against this engine's OWN clean pass (no compiled-vs-eager floor); attr.jsonl gets
    k1 / k3 KL(p||q) and the IS KL(q||p) with the trace-level CI. Single-layer KLs need not add up to the all-layer KL
    (the errors interact through the residual stream): the sum is reported beside it."""
    import numpy as np
    path = os.path.join(out_dir, "attr.jsonl")
    fout = open(path, "a")
    assert llm.collective_rpc(_attr_set, args=(None,))[0] == 0
    p0 = teacher_force(llm, traces)
    p0b = teacher_force(llm, traces)
    rep = max(float(np.max(np.abs(x - y))) for x, y in zip(p0, p0b))
    refd = None
    if os.path.exists(ref_path):
        ref = np.load(ref_path)
        refd = kl_stats([ref[f"t{k}"] for k in range(len(traces))], p0, n_boot=200)
    fout.write(json.dumps(dict(attr="clean", repeat_max_abs_dlogp=rep, vs_compiled_ref=refd)) + "\n")
    fout.flush()
    print(f"[attr] clean eager engine: repeat max|dlogp|={rep:.3g}; vs compiled ref {refd}", flush=True)
    res = {}

    def run(tag, grid, layers=None, part="kv", heads=None):
        if tag in res:      # the same configuration twice (e.g. --attr-layers = the first and last layers): reuse it
            return res[tag]
        n = llm.collective_rpc(_attr_set, args=(dict(grid=grid, layers=layers, part=part, heads=heads),))[0]
        t = time.time()
        lq = teacher_force(llm, traces)
        st = kl_stats(p0, lq, n_boot=500)
        st.update(attr=tag, grid=grid, layers=layers, part=part, heads=heads, n_layers_snapped=n,
                  s=round(time.time() - t, 1))
        fout.write(json.dumps(st) + "\n")
        fout.flush()
        res[tag] = st
        print(f"[attr] {tag:34s} n={n:2d} k3pq={st['kl_pq_k3']:.4f} k1pq={st['kl_pq_k1']:.4f} "
              f"ISqp={st['kl_qp_is']:.4f} meanr={st['mean_r']:.3f}", flush=True)
        return st

    L = list(range(a.attr_num_layers))
    sel = sorted({int(x) for x in a.attr_layers.split(",") if x.strip()})
    if not sel or not all(0 <= i < a.attr_num_layers for i in sel):
        raise SystemExit(f"--attr-layers {a.attr_layers!r}: want a comma-separated subset of "
                         f"0..{a.attr_num_layers - 1}")
    sel_tag = ",".join(str(i) for i in sel)
    for grid in [g for g in a.attr_grids.split(",") if g]:
        tot = run(f"{grid}:all:kv", grid)
        run(f"{grid}:all:k", grid, part="k")
        vv = run(f"{grid}:all:v", grid, part="v")
        run(f"{grid}:skip0,27:kv", grid, layers=[i for i in L if i not in (0, 27)])
        run(f"{grid}:skip{sel_tag}:kv", grid, layers=[i for i in L if i not in sel])
        run(f"{grid}:only0,27:kv", grid, layers=[0, 27])
        run(f"{grid}:only{sel_tag}:kv", grid, layers=sel)
        per_k = {i: run(f"{grid}:L{i}:k", grid, layers=[i], part="k")["kl_pq_k3"] for i in L}
        if vv["kl_pq_k3"] > 0.05 * max(tot["kl_pq_k3"], 1e-9):
            for i in L:
                run(f"{grid}:L{i}:v", grid, layers=[i], part="v")
        top = sorted(L, key=lambda i: -per_k[i])[: a.attr_top_layers]
        for i in top:
            for h in range(a.attr_num_kv_heads):
                run(f"{grid}:L{i}:k:h{h}", grid, layers=[i], part="k", heads=[h])
        print(f"[attr] {grid}: all-layer K+V k3={tot['kl_pq_k3']:.4f}; sum of single-layer K k3={sum(per_k.values()):.4f};"
              f" top layers (K) {[(i, round(per_k[i], 4)) for i in top]}", flush=True)
    llm.collective_rpc(_attr_set, args=(None,))
    fout.close()


def kl_stats(lp, lq, n_boot=2000, seed=0):
    """Token-pooled estimators + a trace-level bootstrap CI (the trace is the unit) for KL(q||p)."""
    import numpy as np
    d = [b - a for a, b in zip(lp, lq)]  # log r per token
    cat = np.concatenate(d)
    r = np.exp(cat)
    out = dict(n_tok=int(cat.size), kl_qp_is=float(np.mean(r * cat)), kl_pq_k1=float(np.mean(-cat)),
               kl_pq_k3=float(np.mean(r - 1 - cat)), mean_r=float(np.mean(r)), max_abs_logr=float(np.max(np.abs(cat))))
    per = [(float(np.sum(np.exp(x) * x)), x.size) for x in d]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(per), len(per))
        bs.append(sum(per[i][0] for i in idx) / sum(per[i][1] for i in idx))
    out["kl_qp_is_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["bf16", "fp8w_fp8kv", "fp8w", "kvfp8", "q", "score", "attr", "report"])
    ap.add_argument("--arm", default="", help="mode q: label of this sampler configuration")
    ap.add_argument("--quant", default="", help="mode q: SPECOPD_SAMPLER_QUANT spec ('' = none), e.g. w=e4m3,a=e4m3,kv=e4m3_unit")
    ap.add_argument("--kv-cache-dtype", default="auto", help="mode q: vLLM-native kv_cache_dtype (auto | fp8)")
    ap.add_argument("--kv-skip-layers", default="", help="mode q: vLLM kv_cache_dtype_skip_layers, e.g. 0,27")
    ap.add_argument("--attention-backend", default="", help="mode q: vLLM attention_backend, e.g. TRITON_ATTN")
    ap.add_argument("--enforce-eager", action="store_true", help="mode q: no compile / CUDA graphs")
    ap.add_argument("--decode-n", type=int, default=2, help="mode q: samples per prompt on the decode path")
    ap.add_argument("--decode-max-tokens", type=int, default=16384, help="= the training response cap (MAX_RESP)")
    ap.add_argument("--decode-seed", type=int, default=777)
    ap.add_argument("--no-tf", action="store_true", help="mode q: skip the teacher-forced prefill estimate")
    ap.add_argument("--no-decode", action="store_true", help="mode q: skip decode sampling")
    ap.add_argument("--rescore", action="store_true", help="mode score: re-score arms already scored (dec_<arm>.json)")
    ap.add_argument("--arms", default="", help="mode score: comma-separated labels to score ('' = every samples_*)")
    ap.add_argument("--attr-grids", default="e4m3_unit,int4,e2m1,int8", help="mode attr: KV grids to attribute")
    ap.add_argument("--attr-num-layers", type=int, default=28)
    ap.add_argument("--attr-num-kv-heads", type=int, default=4)
    ap.add_argument("--attr-top-layers", type=int, default=4, help="mode attr: per-KV-head K split of the top-k layers")
    ap.add_argument("--attr-layers", default="0,27",
                    help="mode attr: a comma-separated layer set whose KV is left unquantized (skip<set>) and quantized "
                         "alone (only<set>), beside the first and last layers' pair (default: that pair)")
    ap.add_argument("--sigmas", default="0,0.01,0.02,0.04,0.08,0.16,0.32", help="grid measured in THIS process ('' = none)")
    ap.add_argument("--targets", default="0.01,0.05", help="TOTAL per-token sampler-trainer KL to aim sigma at ('' = none)")
    ap.add_argument("--natural", type=float, default=0.0,
                    help="the user's measured rollout_corr/kl of the BF16 vLLM engine against the FSDP trainer (the "
                         "natural mismatch every target already contains; 0 = aim the noise at the whole target)")
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--n-per-bench", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--traces", default="", help="the base student's own T=1 samples (refresh/t3_acc.py gen's "
                    "gen_T1_p1.jsonl); needed by every teacher-forced mode, not by the decode path or score / report")
    ap.add_argument("--load-format", default="auto",
                    help="auto = load the checkpoint at engine init; dummy = verl's default (random init)")
    ap.add_argument("--tol", type=float, default=0.25, help="accept a target's sigma iff |measured/wanted - 1| <= tol")
    a = ap.parse_args()
    global TRACES
    TRACES = a.traces or TRACES
    if not TRACES and (a.mode in ("bf16", "fp8w_fp8kv", "fp8w", "kvfp8", "attr") or (a.mode == "q" and not a.no_tf)):
        ap.error(f"--mode {a.mode} teacher-forces the traces' response tokens: pass --traces <jsonl>")
    os.makedirs(a.out_dir, exist_ok=True)
    if a.mode == "report":
        decs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.out_dir, "dec_*.json")))]
        if decs:
            print(f"{'arm (decode, y~q)':>24} {'kl_samp_train':>13} {'95% CI (prompts)':>22} {'k3':>9} "
                  f"{'0-1k':>9} {'1k-4k':>9} {'4k-16k':>9} {'n_tok':>7} {'trunc':>6}")
            for r in decs:
                b = r.get("bins", {})
                fb = lambda k: f"{b[k]['kl_samp_train']:9.4f}" if b.get(k, {}).get("n_tok") else f"{'-':>9}"
                print(f"{r['arm']:>24} {r['kl_samp_train']:13.5f} {str([round(x, 4) for x in r['kl_samp_train_ci95']]):>22} "
                      f"{r['kl_k3']:9.4f} {fb('0-1024')} {fb('1024-4096')} {fb('4096-16384')} {r['n_tok']:7d} "
                      f"{r['frac_truncated']:6.2f}")
            print()
        rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(a.out_dir, "kl_*.jsonl")))
                if not f.endswith("kl_decode.jsonl") for l in open(f)]
        print(f"{'arm':>22} {'KL(q||p) IS':>12} {'95% CI':>22} {'KL(p||q) k1':>12} {'k3':>10} {'mean r':>8} {'n_tok':>7}")
        for r in rows:
            if "kl_qp_is" not in r:
                print(f"{r['arm']:>22} " + json.dumps({k: v for k, v in r.items() if k != "arm"}))
                continue
            print(f"{r['arm']:>22} {r['kl_qp_is']:12.5f} {str([round(x, 5) for x in r['kl_qp_is_ci95']]):>22} "
                  f"{r['kl_pq_k1']:12.5f} {r['kl_pq_k3']:10.5f} {r['mean_r']:8.4f} {r['n_tok']:7d}")
        return
    if a.mode == "q":
        if not a.arm or not re.fullmatch(r"[A-Za-z0-9_.+-]+", a.arm):
            ap.error("--mode q needs --arm <label> ([A-Za-z0-9_.+-])")
        # BEFORE vLLM starts: the verl worker extension reads these in the engine's worker (in-process here).
        if a.quant:
            os.environ["SPECOPD_SAMPLER_QUANT"] = a.quant
            os.environ["SPECOPD_SAMPLER_QUANT_MODEL"] = STUDENT
            os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
        else:
            os.environ.pop("SPECOPD_SAMPLER_QUANT", None)
    elif a.mode in ("score", "attr"):
        os.environ.pop("SPECOPD_SAMPLER_QUANT", None)
    traces = load_traces(a.n_per_bench)
    print(f"[cal] {len(traces)} traces, {sum(len(t['response_ids']) for t in traces)} response tokens, "
          f"prompt lens {[len(t['prompt_ids']) for t in traces]}", flush=True)
    t0 = time.time()
    llm = build_llm(a.mode, a.load_format, a)
    print(f"[cal] engine mode={a.mode} load_format={a.load_format} up in {time.time() - t0:.0f}s", flush=True)
    ref_path = os.path.join(a.out_dir, "ref_bf16.npz")
    # mode q: one file per arm (concurrent jobs never append to the same file)
    out_path = os.path.join(a.out_dir, f"kl_q_{a.arm}.jsonl" if a.mode == "q" else f"kl_{a.mode}.jsonl")
    if a.mode == "q":
        assert os.path.exists(ref_path) or a.no_tf, "run --mode bf16 --sigmas '' --targets '' first (clean reference)"
    fout = open(out_path, "a")

    def record(arm, lq, extra):
        ref = np.load(ref_path)
        lp = [ref[f"t{k}"] for k in range(len(traces))]
        st = kl_stats(lp, lq)
        st.update(arm=arm, mode=a.mode, **extra)
        print("[cal] " + json.dumps(st), flush=True)
        fout.write(json.dumps(st) + "\n")
        fout.flush()
        return st

    if a.mode in ("score", "attr"):
        fout.close()
        os.remove(out_path) if os.path.getsize(out_path) == 0 else None
        if a.mode == "attr":
            run_attr(llm, traces, a.out_dir, ref_path, a)
            return 0
        score_samples(llm, a.out_dir, [x for x in a.arms.split(",") if x] or None, a.rescore)
        return 0
    if a.mode == "q":
        cfg = dict(quant=a.quant or "none", kv_cache_dtype=a.kv_cache_dtype, kv_skip_layers=a.kv_skip_layers,
                   attention_backend=a.attention_backend or "default", enforce_eager=a.enforce_eager)
        audit = llm.collective_rpc(_engine_audit)[0]
        print(f"[cal] arm={a.arm} cfg={cfg} engine_audit={audit}", flush=True)
        if a.quant and any(k in a.quant for k in ("a=", "kv=")) and not (audit and audit.get("patched")):
            print("[cal] FATAL: activation/KV quant requested but the engine was not patched", flush=True)
            return 4
        rec = {}
        t = time.time()
        if "w=" in a.quant:
            rec = llm.collective_rpc(_reload_worker, args=(0.0, a.seed, False, a.quant))[0]
            print(f"[cal] weights re-streamed through SamplerQuant in {time.time() - t:.0f}s: {rec}", flush=True)
        extra = dict(cfg=cfg, engine_audit=audit, quant_sync=rec.get("quant"), quant_audit=rec.get("quant_audit"))
        if not a.no_tf:
            t = time.time()
            lq = teacher_force(llm, traces)
            st = record(a.arm, lq, dict(extra, path="prefill_tf", s=round(time.time() - t, 1)))
        if not a.no_decode:
            t = time.time()
            samples = decode_sample(llm, traces, a.decode_n, a.decode_max_tokens, a.decode_seed)
            sp_path = os.path.join(a.out_dir, f"samples_{a.arm}.jsonl")
            with open(sp_path + ".tmp", "w") as f:
                for r in samples:
                    f.write(json.dumps(dict(r, arm=a.arm, cfg=cfg)) + "\n")
            os.replace(sp_path + ".tmp", sp_path)
            n_tok = sum(len(r["response_ids"]) for r in samples)
            print(f"[cal] decode: {len(samples)} samples, {n_tok} tokens -> {sp_path} ({time.time() - t:.0f}s)",
                  flush=True)
        fout.close()
        return 0
    if a.mode == "bf16":
        t = time.time()
        lp = teacher_force(llm, traces)  # the engine's own load_format=auto weights = the clean student
        if os.path.exists(ref_path):
            # A later bf16 process (the targets step of a split calibration) KEEPS the first reference and checks
            # that this engine reproduces it: the KL of every later point is against the same p.
            ref = np.load(ref_path)
            dev = max(float(np.max(np.abs(ref[f"t{k}"] - x))) for k, x in enumerate(lp))
            print(f"[cal] clean reference exists; this engine reproduces it to max |dlogp| = {dev:.3g}", flush=True)
            fout.write(json.dumps(dict(arm="ref_repro", mode=a.mode, max_abs_dlogp=dev)) + "\n")
            fout.flush()
        else:
            np.savez(ref_path, **{f"t{k}": x for k, x in enumerate(lp)})
        print(f"[cal] clean reference: {sum(x.size for x in lp)} tokens, mean logp {np.mean(np.concatenate(lp)):.4f} "
              f"({time.time() - t:.0f}s)", flush=True)

        def measure(sigma, arm):
            t = time.time()
            rec = llm.collective_rpc(_reload_worker, args=(sigma, a.seed, False))[0]
            lq = teacher_force(llm, traces)
            return record(arm, lq, dict(sigma=sigma, tim=rec, s=round(time.time() - t, 1)))["kl_qp_is"]

        for sigma in [float(s) for s in a.sigmas.split(",") if s.strip()]:
            measure(sigma, f"noise_sigma{sigma:g}")

        def grid_points():
            """Every noise point measured so far in this out dir (earlier processes included): (sigma, KL)."""
            pts = {}
            for line in open(out_path):
                r = json.loads(line)
                if r.get("arm", "").startswith("noise_sigma") and r.get("sigma", 0) > 0 and r.get("kl_qp_is", 0) > 0:
                    pts[r["sigma"]] = r["kl_qp_is"]
            return sorted(pts.items())

        def loglog(p1, p2, want):
            b = (np.log(p2[1]) - np.log(p1[1])) / (np.log(p2[0]) - np.log(p1[0]))
            return float(f"{np.exp(np.log(p1[0]) + (np.log(want) - np.log(p1[1])) / b):.3g}"), float(b)

        # Targets are TOTAL sampler-trainer KL: the natural bf16-vLLM vs FSDP mismatch (--natural, the measured
        # rollout_corr/kl) is added by the run itself, so the noise alone is aimed at target - natural. KL ~ sigma^2
        # locally: interpolate log KL linearly in log sigma between the BRACKETING grid points (the grid reaches
        # 0.32, so no extrapolation is expected; one is logged if it happens), MEASURE it, refine once by the
        # secant through the two measured points nearest the target, measure again, and keep the measured sigma
        # closest to the target. It is written for the training runs only if its MEASURED KL is within
        # +-TOL of the target; otherwise nothing is written and the job exits non-zero (afterok dependents stay held).
        failed = []
        for tgt in [float(x) for x in a.targets.split(",") if x]:
            want = tgt - a.natural
            pts = grid_points()
            below = [p for p in pts if p[1] <= want]
            above = [p for p in pts if p[1] >= want]
            if below and above:
                lo, hi = below[-1], min(above)
            else:
                lo, hi = (pts[-2], pts[-1]) if not above else (pts[0], pts[1])
                print(f"[cal] WARNING: target {want:.4f} outside the grid -> EXTRAPOLATING from {lo}, {hi}", flush=True)
            sig1, b = loglog(lo, hi, want)
            print(f"[cal] target total {tgt} -> noise KL {want:.4f}: bracket {lo} {hi}, slope {b:.2f} -> sigma {sig1}",
                  flush=True)
            meas = {sig1: measure(sig1, f"noise_sigma{sig1:g}_tgt{tgt:g}")}
            # secant refinement through the two measured points (grid + this one) nearest the target in log KL
            pts = sorted(set(grid_points()) | set(meas.items()))
            near = sorted(pts, key=lambda p: abs(np.log(p[1] / want)))[:2]
            if abs(np.log(meas[sig1] / want)) > np.log(1.05) and near[0][0] != near[1][0]:
                sig2, b2 = loglog(near[0], near[1], want)
                if sig2 not in meas and sig2 > 0:
                    print(f"[cal] refine: secant through {near} slope {b2:.2f} -> sigma {sig2}", flush=True)
                    meas[sig2] = measure(sig2, f"noise_sigma{sig2:g}_tgt{tgt:g}_refine")
            best = min(meas, key=lambda s: abs(np.log(meas[s] / want)))
            ok = abs(meas[best] / want - 1.0) <= a.tol
            print(f"[cal] target {tgt}: chose sigma {best} measured noise KL {meas[best]:.5f} vs wanted {want:.5f} "
                  f"({100 * (meas[best] / want - 1):+.1f}%) -> {'ACCEPT' if ok else 'REJECT'} (tol {100 * a.tol:.0f}%)",
                  flush=True)
            fout.write(json.dumps(dict(arm=f"choice_tgt{tgt:g}", mode=a.mode, target_total=tgt, want_noise_kl=want,
                                       sigma=best, measured=meas[best], tried={str(k): v for k, v in meas.items()},
                                       accepted=bool(ok))) + "\n")
            fout.flush()
            if not ok:
                failed.append(tgt)
                continue
            # The chosen sigma, for a training run submitted with SAMPLER_NOISE_FILE (afterok on this job).
            tmp = os.path.join(a.out_dir, f"sigma_tgt{tgt:g}.txt.tmp")
            with open(tmp, "w") as f:
                f.write(f"{best:g}\n")
            os.replace(tmp, os.path.join(a.out_dir, f"sigma_tgt{tgt:g}.txt"))
        if failed:
            fout.close()
            print(f"[cal] FAILED targets {failed}: no sigma file written for them", flush=True)
            return 3
    else:
        assert os.path.exists(ref_path), "run --mode bf16 first (writes the clean reference)"
        fp8 = a.mode in ("fp8w", "fp8w_fp8kv")
        t = time.time()
        rec = llm.collective_rpc(_reload_worker, args=(0.0, a.seed, fp8))[0]  # verl's refit path, no noise
        lq = teacher_force(llm, traces)
        record(a.mode, lq, dict(sigma=0.0, tim=rec, load_format=a.load_format, s=round(time.time() - t, 1)))
    fout.close()


if __name__ == "__main__":
    sys.exit(main())
