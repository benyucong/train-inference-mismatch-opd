"""REAL-KERNEL rollout benchmark of lossy vLLM 0.19.1 configurations on ONE GPU (GH200).

What it answers: does a lossy configuration whose mismatch is under the admissibility bar (nats/token, given to the
report with --bar or as the tolerance D_tol of a file) buy real rollout speed / memory? For the R1-Distill-Qwen-7B
student, standalone vLLM 0.19.1 (no verl, no Ray), engine built the way verl builds the rollout
(TP=1, max_num_seqs 256, max_num_batched_tokens 8192, chunked prefill, prefix caching, FULL_AND_PIECEWISE CUDA graphs,
processed logprobs, seed 42 = replica 0, gpu_memory_utilization 0.65 = the r1_7b ROLLOUT_GPU_UTIL), requests built the
way verl sends them (token ids, T=1, top_p=1, max_tokens 16384, logprobs=0, detokenize off, n separate requests per
prompt), DAPO-Math-17k prompts through the student's chat template, no per-request seeds (verl sets none).

Configurations (one engine per PROCESS -- a deleted vLLM engine does not reliably return its GPU memory), grammar
`part(_part)*`:
    bf16                exact baseline (bf16 weights, bf16 KV)
    w8                  FP8 weights = verl's ROLLOUT_QUANT=fp8 recipe (= rollout.quantization=fp8 in
                        verl/workers/rollout/vllm_rollout/vllm_async_server.py): blockwise 128x128 e4m3 weights, dynamic
                        activations, quantization="fp8" + hf_overrides.quantization_config (verl's FP8_BLOCK_QUANT_KWARGS,
                        weight_block_size [128, 128], ignored_layers model.layers.<i>.mlp.gate). The engine is built
                        through verl's OWN vLLMHttpServer._apply_quantization (which calls verl's
                        apply_vllm_quant_patches() and sets VERL_VLLM_FP8_QUANT_ENABLED=1) with verl's worker extension
                        (vLLMColocateWorkerExtension, which applies the same patches in the engine worker), then the
                        bf16 checkpoint is pushed through verl's refit path exactly as the first weight sync does
                        (prepare_quanted_weights_for_loading -> the worker's _update_weights per 2048 MB bucket =
                        load_quanted_weights, blockwise scaled_fp8_blockwise -> process_quanted_weights_after_loading).
                        The EAGLE-3 drafter stays bf16, as in training: vLLM takes a separate drafter's quant config
                        from its OWN draft model config (get_draft_quant_config; verl's speculative_config sets none)
                        and verl's refit covers only the target (+ an MTP head). FATAL (exit 5) unless the live quant
                        config shows weight_block_size [128, 128], the refit quantized all 7 x 28 linear weights and
                        the drafter has no FP8 linear. vLLM's ONLINE per-tensor fp8 is NOT measured here.
    kv8all              FP8 KV cache on every layer (kv_cache_dtype="fp8" = e4m3 at unit k/v scale; FA3 also quantizes Q)
    kv8s0 | kv8s0-27    FP8 KV with layers {0} / {0,27} kept bf16 = vLLM 0.19.1 CacheConfig.kv_cache_dtype_skip_layers
                        (attention.py: str(layer_idx) in the list -> that layer's kv_cache_dtype = "auto")
    kv8only5            FP8 KV on layer 5 ONLY (skip every other layer) -- the one-layer-at-a-time scan
    e3 | e3k2           EAGLE-3 speculative decoding, k=3 (default) or k=N; drafter = r1-7b-eagle3-maxpos131072
                        (layer index 28 -> never in a skip list; its KV follows kv_cache_dtype)
    e.g. bf16 w8 kv8s0 kv8s0-27 w8_kv8s0 kv8all bf16_e3 w8_e3 kv8s0_e3 kv8s0-27_e3 w8_kv8s0_e3

Sub-commands:
    python bench/g0_kernels.py gen --cfg kv8s0_e3 --out-dir D [--profile full|smoke|scan] [--dry-run]
        After one untimed warm-up batch, for each (concurrency C, prompt set) in an INTERLEAVED schedule, gated shapes
        (16/64/128) first and the ungated 256 last (every gated C gets a set before any gets a second, so a time budget
        cuts 256 first and never starves one gated shape): submit C requests at once (C/spp prompts x spp samples, the
        sync-OPD rollout step of one replica), drive LLMEngine.step() until all finish and record:
          makespan (= the step's generation time), output tokens, output tok/s, per-request (len, finish time, reason),
          per-concurrency aggregates by_n[n] = (time, tokens, iters) (n = unfinished requests before the step),
          preemptions, peak KV usage fraction, max waiting requests (both from a wrapper on the engine's stat-logger
          record), EAGLE-3 drafts / accepted tokens / per-position acceptance, Prometheus counter deltas (cross-check),
          sampler NLL (mean -log q of the sampled tokens; an entropy proxy), NVML device memory peak, worker torch peak.
        Once per engine: init time, KV capacity (num_gpu_blocks x block_size tokens, max concurrency at max_model_len),
          weights GiB, KV GiB, per-attention-layer kv dtype + backend (target AND drafter), linear quant methods.
        Saves the sampled token ids + vLLM's returned log q for samples r < --save-samples of every prompt (tok_*/).
        w8* only: engine.weight_block_size (live quant config) and engine.fp8_weights (verl's _apply_quantization
          output, the vllm_async_server.py sha256, refit counts and time, drafter quant).
        Output: D/gen_<label>__<group>.json, D/tok_<label>__<group>/c<C>_s<i>.npz. Refuses to overwrite.
    python bench/g0_kernels.py score --ref hf|vllm --out-dir D [--per-c 16]
        REAL-KERNEL MISMATCH. Per gen file, a deterministic subset (per C: round-robin over prompts, sample 0 of every
        prompt first, the SAME prompts for every config) is teacher-forced through
          hf  : HF transformers bf16, flash_attention_2, torch.autocast(bf16), lm_head in bf16 then verl's own
                logprobs_from_logits (flash-attn CE) -- the FSDP trainer's log-prob path without FSDP (trainer stand-in)
          vllm: a clean bf16 vLLM engine, prompt_logprobs=0 (= the offline reference of tim_calibrate / tim_kl_eval)
        k1 = token mean of (log q - log p) over the sampled tokens = E_q[log q - log p] = KL(q||p) = what the training
        run logs as rollout_corr/kl; also k3 (>= 0 per token), a prompt-cluster bootstrap CI, position bins
        (0-1k / 1k-4k / 4k-16k), per-C breakdown, tail rates (|log q - log p| > 1, > 5).
        Output: D/kl_<ref>_<label>__<group>.json + .npz (per-token lq, lp). Refuses to overwrite.
        Validity check of the stand-in (encoded in report, G0_RULES['standin']): bf16_e3's hf k1 at C=16 -- the cell
        comparable to a training run with the bf16 rollout engine (SPEC=eagle3 k=3, 16 seqs/replica, verify path) --
        within [0.5x, 2x] of the rollout_corr/kl that run logs (report --in-run-kl; without it the check is skipped).
    python bench/g0_kernels.py report --out-dir D (--bar B | --dtol-source <tolerance JSON>) [--in-run-kl K]
        Speed / capacity / acceptance / mismatch tables and the verdict, with every decision rule FIXED in
        G0_RULES:
          PRIMARY reading only: each lossy config vs BOTH clean refs (bf16, bf16_e3) of its OWN group (same GPU), over
            the prompt sets both ran, ratio of MEDIAN makespans; a cell clears iff that ratio AND every paired per-set
            ratio are >= thr * max(1, anchor variance); < 3 paired sets, or CV > 5% with < 5, -> INCOMPLETE.
          admissible = the 95% CI upper bound of k1 <= bar under BOTH refs (hf, vllm), pooled AND at each gated C;
            bar = --bar (nats/token) or the D_tol of the tolerance file given with --dtol-source. Exactly one of the
            two is required; there is no default.
          PASS iff an admissible config clears at C=16 (1.15) OR 64 (1.30) OR 128 (1.30); FAIL iff nothing clears and
            nothing missing could change that; else INCOMPLETE. Stand-in check (bf16_e3 hf k1 at C=16 within
            [0.5x, 2x] of --in-run-kl) FAIL or MISSING -> INCOMPLETE; SKIPPED (no --in-run-kl) leaves the verdict.
        Secondary readings (same-spec, vs production bf16+EAGLE-3) are descriptive; a cross-group ratio goes through
        the bf16 anchor over the sets ALL FOUR cells ran. Output: D/g0_report_<ts>.json.
    python bench/g0_kernels.py selftest            # CPU: grammar, kwargs, sets, bookkeeping (fake engine), npz,
                                                     # selection, stats, report, verdict, HF alignment (tiny model)
"""
import argparse
import glob
import json
import math
import os
import random
import re
import sys
import time

CLUSTER = os.environ.get("CLUSTER_ROOT", "/path/to/workdir")
STUDENT = f"{CLUSTER}/hf/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60"
DRAFTER = f"{CLUSTER}/models/r1-7b-eagle3-maxpos131072"
DATA = f"{CLUSTER}/workspace/datasets/dapo-math-17k.parquet"
NUM_LAYERS = 28            # R1-Distill-Qwen-7B (Qwen2): 28 layers, 4 KV heads x 128, no QK-norm

# every w8* config (with and without FP8 KV, with and without EAGLE-3) is verl's ROLLOUT_QUANT=fp8 recipe, the one an
# FP8 training run uses (SPEC=eagle3, load_format=auto, rollout.quantization=fp8). The values below are DECLARED so
# that `gen --dry-run` shows them in its printed plan without importing verl; at run time gen re-derives them by
# calling verl's own vLLMHttpServer._apply_quantization and refuses (exit 5) unless they are equal.
VERL_SERVER_SRC = "verl/workers/rollout/vllm_rollout/vllm_async_server.py"   # relative to the verl repo root
VERL_WORKER_EXT = "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension"
FP8_WEIGHT_BLOCK = [128, 128]
FP8_LINEAR_PER_LAYER = 7       # Qwen2: q, k, v, o, gate, up, down (HF names; vLLM fuses them into 4 modules)
VERL_UPDATE_BUCKET_MB = 2048   # verl rollout.checkpoint_engine.update_weights_bucket_megabytes (the refit bucket)


def verl_fp8_quant_config(num_layers=NUM_LAYERS):
    """verl's FP8_BLOCK_QUANT_KWARGS for a dense model of num_layers (vllm_async_server.py _apply_quantization)."""
    return {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8", "weight_block_size": list(FP8_WEIGHT_BLOCK),
            "ignored_layers": [f"model.layers.{i}.mlp.gate" for i in range(num_layers)]}


def verl_rollout_fp8(hf_config, server_cls=None):
    """Run verl's OWN code for ROLLOUT_QUANT=fp8 (the launcher maps it to actor_rollout_ref.rollout.quantization=fp8
    and nothing else): vLLMHttpServer._apply_quantization and _get_worker_extension_cls on a stand-in `self` whose
    config is verl's RolloutConfig default apart from quantization (qat None, quantization_config_file None).
    Side effects are verl's: apply_vllm_quant_patches() in THIS process and VERL_VLLM_FP8_QUANT_ENABLED=1 in
    os.environ, which the engine core inherits and verl's worker extension turns into the same patches there.
    Returns {quantization, hf_overrides, worker_extension_cls} in engine_kwargs' keys."""
    from types import SimpleNamespace
    if server_cls is None:
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer as server_cls
    me = SimpleNamespace(config=SimpleNamespace(quantization="fp8", qat=None, quantization_config_file=None),
                         model_config=SimpleNamespace(hf_config=hf_config))
    quantization, hf_overrides = server_cls._apply_quantization(me)
    return dict(quantization=quantization, hf_overrides=hf_overrides,
                worker_extension_cls=server_cls._get_worker_extension_cls(me))


def _canon(x):
    return json.loads(json.dumps(x, sort_keys=True, default=str))

# Decision rules of the report's verdict. Its only inputs besides the data are the admissibility bar (report --bar, or
# the D_tol of a tolerance file, report --dtol-source; there is no default) and the rollout_corr/kl that a training run
# with the bf16 rollout engine logs (report --in-run-kl, for the stand-in check). Everything else is fixed here.
GATED_C = (16, 64, 128)
G0_RULES = dict(
    version=("verdict over the primary reading (decision rules below); w8* = verl's blockwise 128x128 FP8-W recipe "
             "with a worker-side FP8 audit (patches active, e4m3 weights, weight_scale_inv (ceil(out/128), "
             "ceil(in/128))); a w8* cell without a passing audit is void"),
    # FAIL iff, over the admissible configs, max speedup < 1.15 at 16 AND < 1.30 at 64 AND < 1.30 at 128, i.e.
    # PASS iff ANY admissible config clears at ANY gated shape (one config per shape; a single 1.3x cell at 64 passes).
    # 256 is measured and reported, never gated.
    gated={16: 1.15, 64: 1.30, 128: 1.30},
    # primary reading: vs the best clean config at that shape = the faster of bf16 and bf16+EAGLE-3. Both must have
    # run in the lossy config's OWN group (same GPU, paired prompt sets); the cell must clear against BOTH (= against
    # the faster one, whichever it is on the paired sets). Cross-group numbers never decide.
    clean_refs=("bf16", "bf16_e3"),
    # 3 timed batches, the MEDIAN makespan, 2 more repeats if CV > 5%, counted over the PAIRED sets (the sets both
    # cells ran); fewer -> the cell is INCOMPLETE.
    min_paired=3, cv_max=0.05, min_paired_if_cv=5,
    # a cell CLEARS iff, against each clean ref, the ratio of median makespans AND every paired per-set ratio
    # are >= thr_eff = thr * max(1, AV_C), AV_C = the largest cross-group max/min of a clean config's median makespan at
    # C (over the sets common to its groups); AV_C unmeasurable -> thr_eff = thr, flagged.
    anchor_flag=1.03,            # AV_C above this: cross-group numbers at C are flagged (descriptive only)
    # admissibility (one decision per config): the prompt-cluster bootstrap 95% CI UPPER bound of k1 <= bar for
    # EVERY ref (hf = trainer stand-in; vllm = the offline clean-bf16-prefill reference) and EVERY scope (pooled,
    # and each gated C separately). Any upper bound > bar -> inadmissible; any missing -> unscored.
    adm_refs=("hf", "vllm"), adm_scopes=("all",) + GATED_C,
    # stand-in validity: the cell comparable to a training run with the bf16 rollout engine (SPEC=eagle3 k=3, 16
    # seqs/replica, the target's verify path) is bf16_e3 at C=16; its hf k1 at C=16 (point) must lie in
    # [band_x[0], band_x[1]] x the rollout_corr/kl that run logs (report --in-run-kl). FAIL or MISSING -> the verdict is INCOMPLETE (the would-be
    # verdict is still reported); no --in-run-kl -> the check is SKIPPED and does not change the verdict.
    standin=dict(cfg="bf16_e3", C=16, ref="hf", band_x=(0.5, 2.0)),
    # the matrix the verdict is over (every lossy config with and without EAGLE-3, minus kv8all_e3: kv8all, FP8 KV on
    # every layer, is the worst-case reference and is not expected to be admissible). A missing config blocks a FAIL
    # (never a PASS).
    expected_lossy=("w8", "kv8all", "kv8s0", "kv8s0-27", "w8_kv8s0", "w8_e3", "kv8s0_e3", "kv8s0-27_e3", "w8_kv8s0_e3"),
)

PROFILES = {
    # full: the speed matrix. 6/5/5/2 sets: every gated shape can reach the rules' 3 + 2 timed batches; the C=16
    # makespan is set by its longest response, so it gets one more. Gated shapes are scheduled first (schedule()), so
    # the time budget can only cut the ungated C=256 sets before it cuts a gated set.
    "full": dict(conc="16,64,128,256", sets="6,5,5,2", spp=8, max_tokens=16384, save_samples=2, time_budget_s=4200),
    # smoke: does the engine build, generate, return logprobs and audit; every request saved so that the scorer sees
    # up to 16 x 2048 tokens (an order of magnitude for k1 only -- minutes, not a result)
    "smoke": dict(conc="16", sets="1", spp=8, max_tokens=2048, save_samples=8, time_budget_s=300),
    # scan: mismatch only (one-layer-at-a-time KV scan): 2 x 16 distinct prompts, 4096 tokens, every request saved
    "scan": dict(conc="16", sets="2", spp=1, max_tokens=4096, save_samples=1, time_budget_s=900),
}
SCORE_PER_C = {"full": 16, "smoke": 16, "scan": 32}
PROM_KEEP = ("vllm:num_preemptions", "vllm:prompt_tokens", "vllm:generation_tokens", "vllm:prompt_tokens_cached",
             "vllm:prompt_tokens_recomputed", "vllm:prefix_cache_queries", "vllm:prefix_cache_hits",
             "vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens", "vllm:spec_decode_num_accepted_tokens",
             "vllm:spec_decode_num_accepted_tokens_per_pos", "vllm:request_success")
POS_BINS = ((0, 1024), (1024, 4096), (4096, 1 << 30))


# ================================================================================================= configurations
def parse_cfg(name, num_layers=NUM_LAYERS):
    """Config grammar -> attributes. Raises ValueError on anything unknown, duplicated or out of range."""
    base = name.split(".", 1)[0]  # an optional ".tag" suffix labels a repeat of the same config
    if not re.fullmatch(r"[A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)*", base):
        raise ValueError(f"bad config name {name!r}")
    wq, kv, skip, spec_k, seen = "bf16", "auto", [], 0, set()
    for p in base.split("_"):
        if p in ("bf16", "w8"):
            key, wq = "w", ("fp8" if p == "w8" else "bf16")
        elif p.startswith("kv8"):
            key, kv, rest = "kv", "fp8", p[3:]
            m_s, m_o = re.fullmatch(r"s(\d+(?:-\d+)*)", rest), re.fullmatch(r"only(\d+(?:-\d+)*)", rest)
            if rest == "all":
                skip = []
            elif m_s:
                skip = sorted({int(x) for x in m_s.group(1).split("-")})
            elif m_o:
                fp8 = {int(x) for x in m_o.group(1).split("-")}
                if any(not 0 <= i < num_layers for i in fp8):
                    raise ValueError(f"{name}: layer out of range 0..{num_layers - 1}")
                skip = [i for i in range(num_layers) if i not in fp8]
            else:
                raise ValueError(f"{name}: kv8 needs all | s<i>[-<j>...] | only<i>[-<j>...] (got {p!r})")
            if any(not 0 <= i < num_layers for i in skip):
                raise ValueError(f"{name}: skip layer out of range 0..{num_layers - 1}")
            if len(skip) >= num_layers:
                raise ValueError(f"{name}: every layer skipped -> no fp8 KV left")
        elif re.fullmatch(r"e3(?:k\d+)?", p):
            key, spec_k = "spec", int(p[3:] or 3) if p.startswith("e3k") else 3
            if not 1 <= spec_k <= 8:
                raise ValueError(f"{name}: speculative depth must be 1..8")
        else:
            raise ValueError(f"{name}: unknown part {p!r}")
        if key in seen:
            raise ValueError(f"{name}: duplicate {key} part")
        seen.add(key)
    fp8_layers = [i for i in range(num_layers) if i not in skip] if kv == "fp8" else []
    return dict(name=name, base=base, wq=wq, kv=kv, kv_skip=skip, kv_fp8_layers=fp8_layers, spec_k=spec_k,
                lossy=(wq != "bf16" or kv != "auto"), exact=(wq == "bf16" and kv == "auto"))


def engine_kwargs(cfg, model, drafter, util=0.65, max_model_len=18433, max_num_seqs=256, max_batched_tokens=8192,
                  seed=42, enforce_eager=False, async_scheduling=None):
    """vLLM LLM(**kwargs) for one config: verl's rollout engine args (vllm_async_server.py) for this student."""
    kw = dict(model=model, tokenizer=model, dtype="bfloat16", tensor_parallel_size=1, seed=seed,
              gpu_memory_utilization=util, max_model_len=max_model_len, max_num_seqs=max_num_seqs,
              max_num_batched_tokens=max_batched_tokens, enable_chunked_prefill=True, enable_prefix_caching=True,
              logprobs_mode="processed_logprobs", enforce_eager=enforce_eager, disable_log_stats=False,
              load_format="auto", compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"})
    if async_scheduling is not None:
        kw["async_scheduling"] = bool(async_scheduling)
    if cfg["wq"] == "fp8":  # verl's blockwise FP8 recipe (verl_rollout_fp8 must reproduce these three keys)
        kw["quantization"] = "fp8"
        kw["hf_overrides"] = {"quantization_config": verl_fp8_quant_config()}
        kw["worker_extension_cls"] = VERL_WORKER_EXT
    if cfg["kv"] == "fp8":
        kw["kv_cache_dtype"] = "fp8"
        if cfg["kv_skip"]:
            kw["kv_cache_dtype_skip_layers"] = [str(i) for i in cfg["kv_skip"]]
    if cfg["spec_k"]:
        kw["speculative_config"] = {"method": "eagle3", "model": drafter, "num_speculative_tokens": cfg["spec_k"]}
    return kw


# ===================================================================================================== prompt sets
def parse_int_list(s):
    return [int(x) for x in str(s).split(",") if str(x).strip()]


def build_sets(n_pool, conc, sets, spp, set_seed):
    """{C: [sorted dataset indices of set i]}; deterministic, identical for every config (paired comparisons)."""
    out = {}
    for c, n_sets in zip(conc, sets):
        if c % spp:
            raise ValueError(f"concurrency {c} is not a multiple of samples-per-prompt {spp}")
        out[c] = [sorted(random.Random(set_seed * 1_000_003 + c * 1_009 + i).sample(range(n_pool), c // spp))
                  for i in range(n_sets)]
    return out


def schedule(conc, sets, gated=GATED_C):
    """(set index, C) order: the GATED shapes first, interleaved (set 0 of every gated C, then set 1, ...), then the
    ungated shapes, interleaved. A time budget therefore cuts the ungated sets first, then the LAST gated sets; it never
    starves one gated shape while another gets extra sets."""
    out = []
    for part in ([(c, n) for c, n in zip(conc, sets) if c in gated], [(c, n) for c, n in zip(conc, sets) if c not in gated]):
        if part:
            out += [(i, c) for i in range(max(n for _, n in part)) for c, n in part if i < n]
    return out


def render_prompt_ids(tok, prompt, max_prompt):
    chat = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else [dict(m) for m in prompt]
    text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    ids = list(tok(text, add_special_tokens=False)["input_ids"])
    return (ids[-max_prompt:], True) if len(ids) > max_prompt else (ids, False)  # verl: data.truncation=left


# ========================================================================================================= trackers
class SchedTracker:
    """Wraps StatLoggerManager.record: per-step scheduler stats the public API only exposes as last-value gauges."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.steps = self.wait_steps = self.run_max = self.wait_max = self.preempted = self.gen_tokens = 0
        self.kv_max = self.kv_sum = 0.0
        self.drafts = self.draft_tokens = self.accepted = 0
        self.acc_pos = []

    def record(self, scheduler_stats=None, iteration_stats=None, *args, **kwargs):
        s = scheduler_stats
        if s is not None:
            self.steps += 1
            u = float(getattr(s, "kv_cache_usage", 0.0) or 0.0)
            self.kv_max, self.kv_sum = max(self.kv_max, u), self.kv_sum + u
            self.run_max = max(self.run_max, int(getattr(s, "num_running_reqs", 0) or 0))
            w = int(getattr(s, "num_waiting_reqs", 0) or 0)
            self.wait_max = max(self.wait_max, w)
            self.wait_steps += int(w > 0)
            sd = getattr(s, "spec_decoding_stats", None)
            if sd is not None:
                self.drafts += int(sd.num_drafts)
                self.draft_tokens += int(sd.num_draft_tokens)
                self.accepted += int(sd.num_accepted_tokens)
                pp, ap = list(sd.num_accepted_tokens_per_pos or []), self.acc_pos
                self.acc_pos = [(ap[i] if i < len(ap) else 0) + (pp[i] if i < len(pp) else 0)
                                for i in range(max(len(ap), len(pp)))]
        it = iteration_stats
        if it is not None:
            self.preempted += int(getattr(it, "num_preempted_reqs", 0) or 0)
            self.gen_tokens += int(getattr(it, "num_generation_tokens", 0) or 0)

    def summary(self):
        d = self.drafts
        return dict(sched_steps=self.steps, kv_usage_max=round(self.kv_max, 4),
                    kv_usage_mean=round(self.kv_sum / self.steps, 4) if self.steps else None,
                    running_max=self.run_max, waiting_max=self.wait_max, waiting_steps=self.wait_steps,
                    preemptions=self.preempted, iter_gen_tokens=self.gen_tokens, spec_drafts=d,
                    spec_draft_tokens=self.draft_tokens, spec_accepted=self.accepted,
                    accept_len=round(1 + self.accepted / d, 4) if d else None,
                    accept_rate=round(self.accepted / self.draft_tokens, 4) if self.draft_tokens else None,
                    accept_per_pos=[round(x / d, 4) for x in self.acc_pos] if d else None)


def install_tracker(eng, tracker):
    lm = getattr(eng, "logger_manager", None)
    if lm is None:
        return False
    orig = lm.record

    def record(*args, **kwargs):
        try:
            tracker.record(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 -- never let bookkeeping kill the measurement
            print(f"[g0] tracker error {e!r}", flush=True)
        return orig(*args, **kwargs)

    lm.record = record
    return True


class Nvml:
    """Device memory (NVML, no CUDA context in this process). The device is the one vLLM allocated on (max used)."""

    def __init__(self, period=0.5):
        self.ok, self.period, self.h, self.info = False, period, None, {}
        try:
            try:
                from vllm.third_party import pynvml
            except Exception:  # noqa: BLE001
                import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.handles = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                try:
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    pynvml.nvmlDeviceGetMemoryInfo(h)
                    self.handles.append((i, h))
                except Exception:  # noqa: BLE001 -- devices outside our cgroup
                    pass
            self.ok = bool(self.handles)
        except Exception as e:  # noqa: BLE001
            self.info = dict(error=repr(e)[:200])
        self.reset()

    def pick(self):
        """After the engine is up: the only accessible device, else CUDA_VISIBLE_DEVICES' first entry (index or UUID),
        else the accessible device with the most used memory (the one vLLM just filled)."""
        if not self.ok:
            return self.info
        used = [(self.nv.nvmlDeviceGetMemoryInfo(h).used, i, h) for i, h in self.handles]
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        how = "max_used"
        u, i, h = max(used, key=lambda x: x[0])
        if len(used) == 1:
            how = "only_device"
        elif cvd.isdigit() and any(j == int(cvd) for _, j, _ in used):
            u, i, h = next(x for x in used if x[1] == int(cvd))
            how = "cuda_visible_devices_index"
        elif cvd.startswith("GPU-"):
            for x in used:
                uu = self.nv.nvmlDeviceGetUUID(x[2])
                if (uu.decode() if isinstance(uu, bytes) else str(uu)) == cvd:
                    u, i, h = x
                    how = "cuda_visible_devices_uuid"
        self.h = h
        name = self.nv.nvmlDeviceGetName(h)
        uuid = self.nv.nvmlDeviceGetUUID(h)
        self.info = dict(nvml_index=i, name=name.decode() if isinstance(name, bytes) else str(name),
                         uuid=uuid.decode() if isinstance(uuid, bytes) else str(uuid),
                         total_gib=round(self.nv.nvmlDeviceGetMemoryInfo(h).total / 2**30, 3),
                         used_after_init_gib=round(u / 2**30, 3), accessible=len(self.handles), picked_by=how,
                         cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
        return self.info

    def reset(self):
        self.peak, self.last = 0, 0.0

    def sample(self, t):
        if self.h is None or t - self.last < self.period:
            return
        self.last = t
        try:
            self.peak = max(self.peak, self.nv.nvmlDeviceGetMemoryInfo(self.h).used)
        except Exception:  # noqa: BLE001
            pass

    def peak_gib(self):
        return round(self.peak / 2**30, 3) if self.peak else None


def prom_snapshot(llm):
    try:
        ms = llm.get_metrics()
    except Exception as e:  # noqa: BLE001
        return {"_error": repr(e)[:200]}
    out = {}
    for m in ms:
        if not m.name.startswith(PROM_KEEP):
            continue
        cls = type(m).__name__
        if cls == "Counter":
            out[m.name] = out.get(m.name, 0) + m.value
        elif cls == "Vector":
            prev = out.get(m.name, [])
            vals = list(m.values)
            n = max(len(prev), len(vals))
            out[m.name] = [(prev[i] if i < len(prev) else 0) + (vals[i] if i < len(vals) else 0) for i in range(n)]
    return out


def prom_delta(a, b):
    if "_error" in a or "_error" in b:
        return {"_error": a.get("_error") or b.get("_error")}
    out = {}
    for k, v in b.items():
        p = a.get(k)
        if isinstance(v, list):
            p = p or []
            out[k] = [x - (p[i] if i < len(p) else 0) for i, x in enumerate(v)]
        else:
            out[k] = v - (p or 0)
    return out


# ================================================================================================ worker-side (rpc)
def _rpc_audit(worker):
    """Runs in the engine's worker (cloudpickled; needs VLLM_ALLOW_INSECURE_SERIALIZATION=1 under multiprocessing)."""
    import torch
    out = {}
    try:
        mr = worker.model_runner
        out["model_memory_usage_gib"] = round(float(getattr(mr, "model_memory_usage", 0) or 0) / 2**30, 3)
        try:
            from vllm.model_executor.layers.attention.attention import Attention
        except Exception:  # noqa: BLE001
            from vllm.attention.layer import Attention
        models = [("target", mr.get_model())]
        dm = getattr(getattr(mr, "drafter", None), "model", None)
        if dm is not None:
            models.append(("drafter", dm))
        layers, lin = [], {}
        for tag, m in models:
            for name, mod in m.named_modules():
                if isinstance(mod, Attention):
                    be = getattr(mod, "attn_backend", None)
                    layers.append(dict(model=tag, name=name, kv_cache_dtype=str(getattr(mod, "kv_cache_dtype", "?")),
                                       kv_torch_dtype=str(getattr(mod, "kv_cache_torch_dtype", "?")),
                                       backend=be.get_name() if hasattr(be, "get_name") else str(be),
                                       impl=type(getattr(mod, "impl", None)).__name__,
                                       k_scale=float(getattr(mod, "_k_scale_float", float("nan"))),
                                       v_scale=float(getattr(mod, "_v_scale_float", float("nan")))))
                qm, w = getattr(mod, "quant_method", None), getattr(mod, "weight", None)
                if qm is not None and isinstance(w, torch.Tensor) and hasattr(mod, "input_size"):
                    key = f"{type(qm).__name__}:{w.dtype}"
                    lin.setdefault(tag, {})[key] = lin.setdefault(tag, {}).get(key, 0) + 1
        out["attn_layers"], out["linear_methods"] = layers, lin
        kcc = getattr(mr, "kv_cache_config", None)
        if kcc is not None:
            out["kv_cache_gib"] = round(sum(int(t.size) for t in kcc.kv_cache_tensors) / 2**30, 3)
            out["kv_num_blocks"] = int(kcc.num_blocks)
            out["kv_groups"] = [dict(n_layers=len(g.layer_names), spec=type(g.kv_cache_spec).__name__)
                                for g in kcc.kv_cache_groups]
        free, total = torch.cuda.mem_get_info()
        out.update(device_used_gib=round((total - free) / 2**30, 3), device_total_gib=round(total / 2**30, 3),
                   torch_max_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
                   torch_reserved_gib=round(torch.cuda.memory_reserved() / 2**30, 3))
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)[:500]
    return out


def _rpc_mem(worker, reset=False):
    import torch
    free, total = torch.cuda.mem_get_info()
    out = dict(torch_max_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
               torch_reserved_gib=round(torch.cuda.memory_reserved() / 2**30, 3),
               device_used_gib=round((total - free) / 2**30, 3))
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return out


FA3_KVSKIP_PATCH = "fa3_kvskip"  # the label in the patch's stats: see install_fa3_kvskip_patch


def install_fa3_kvskip_patch(fa, torch):
    """vLLM 0.19.1 + kv_cache_dtype_skip_layers + FA3: keep vLLM's AOT scheduler metadata except where the bf16 kernel
    would reject it.

    FlashAttentionMetadataBuilder.build's `schedule` passes qkv_dtype from the GLOBAL cache_config.cache_dtype ("fp8")
    also for the attention group of the layers that kv_cache_dtype_skip_layers keeps in bf16 (that group's builder has
    kv_cache_dtype == torch.bfloat16). Where FA3's metadata layout depends on the dtype, the bf16 layer's kernel raises
    "scheduler_metadata must have shape (metadata_size)" and the engine fails. That happens whenever a bf16-cache layer
    of an FP8-cache engine runs outside CUDA graphs (num_splits == 0): an engine without CUDA graphs, or a batch with
    more tokens than the largest captured full CUDA graph (for example an EAGLE-3 verify step of 256 sequences x 4
    tokens = 1024 tokens with a 512-token graph limit). The patch changes nothing else:
      - it acts only inside build() of a builder whose own kv_cache_dtype is bf16/fp16 while the passed qkv_dtype
        differs (= a skipped layer of an FP8-KV engine), and only on the non-CUDA-graph path (num_splits == 0), so a
        captured graph never sees a different tensor and every other builder / engine is untouched;
      - there it ALSO computes the metadata for the layer's own dtype and uses it ONLY if its shape differs from vLLM's
        -- exactly the case in which the stock kernel raises. Wherever stock vLLM runs, it runs bit-for-bit as stock.
    `fa` is vllm.v1.attention.backends.flash_attn (or a stand-in module with the same two names). Returns the live
    stats dict."""
    import threading
    st = getattr(fa, "_g0_fa3_kvskip_stats", None)
    if st is not None:
        return st
    st = dict(patch=FA3_KVSKIP_PATCH, installed=True, checked=0, fallback=0, first_fallback=None)
    tl = threading.local()
    orig_gsm, cls = fa.get_scheduler_metadata, fa.FlashAttentionMetadataBuilder
    orig_build = cls.build
    own16 = (torch.bfloat16, torch.float16)

    def build(self, *args, **kwargs):
        prev, tl.builder = getattr(tl, "builder", None), self
        try:
            return orig_build(self, *args, **kwargs)
        finally:
            tl.builder = prev

    def get_scheduler_metadata(*args, **kwargs):
        md = orig_gsm(*args, **kwargs)
        b = getattr(tl, "builder", None)
        own, q = getattr(b, "kv_cache_dtype", None), kwargs.get("qkv_dtype")
        if b is None or args or own not in own16 or q is None or q == own or kwargs.get("num_splits", 0) != 0:
            return md
        st["checked"] += 1
        md_own = orig_gsm(**dict(kwargs, qkv_dtype=own))
        if tuple(md_own.shape) == tuple(md.shape):
            return md
        st["fallback"] += 1
        if st["first_fallback"] is None:
            st["first_fallback"] = dict(stock_shape=list(md.shape), own_shape=list(md_own.shape), qkv_dtype=str(q),
                                        own_dtype=str(own), batch_size=kwargs.get("batch_size"),
                                        max_seqlen_q=kwargs.get("max_seqlen_q"), causal=kwargs.get("causal"))
        return md_own

    cls.build = build
    fa.get_scheduler_metadata = get_scheduler_metadata
    fa._g0_fa3_kvskip_stats = st
    return st


def _rpc_fa3_kvskip_patch(worker):
    """Worker side (cloudpickled): install install_fa3_kvskip_patch in the engine core process; return its stats."""
    try:
        import torch
        import vllm.v1.attention.backends.flash_attn as fa
        return dict(install_fa3_kvskip_patch(fa, torch))
    except Exception as e:  # noqa: BLE001
        return {"installed": False, "error": repr(e)[:500]}


def _verl_refit_api():
    from verl.utils.vllm import vllm_quant_utils as q
    return q


def _rpc_verl_fp8_refit(worker, model_dir, bucket_mb=VERL_UPDATE_BUCKET_MB, api=None):
    """w8*: push the bf16 checkpoint through verl's FP8 refit path, as verl's FIRST weight sync does
    (vLLMColocateWorkerExtension.update_weights_from_ipc, is_quantized_model branch): step 1
    prepare_quanted_weights_for_loading per model in worker._iter_all_models(); step 2 the worker's own _update_weights
    per bucket (verl's code: split buffers, drop tied aliases, load_quanted_weights = scaled_fp8_blockwise with the live
    weight_block_size, then model.load_weights); step 3 process_quanted_weights_after_loading. Runs in the engine
    worker (needs verl's worker extension, i.e. engine_kwargs' worker_extension_cls). The tensors are the
    checkpoint's, in checkpoint (HF) names = what the FSDP actor streams at step 0. Never raises: errors are returned."""
    out = {}
    try:
        import glob as _glob

        import torch
        from safetensors import safe_open
        q = api or _verl_refit_api()
        t0 = time.time()
        mr = worker.model_runner
        vc = mr.vllm_config
        qc = getattr(vc, "quant_config", None)
        out["is_quantized_model"] = bool(q.is_quantized_model(vc))
        out["quant_config_cls"] = type(qc).__name__
        out["weight_block_size"] = list(getattr(qc, "weight_block_size", None) or [])
        out["activation_scheme"] = getattr(qc, "activation_scheme", None)
        out["is_checkpoint_fp8_serialized"] = getattr(qc, "is_checkpoint_fp8_serialized", None)
        out["ignored_layers_n"] = len(getattr(qc, "ignored_layers", None) or [])
        out["worker_patches_applied"] = len(getattr(q.fp8_state, "vllm_patches", []) or [])
        out["has_update_weights"] = hasattr(worker, "_update_weights")
        if not (out["is_quantized_model"] and out["has_update_weights"]):
            out["error"] = "not an FP8 engine with verl's worker extension"
            return out
        device = getattr(worker, "device", None) or torch.device("cuda", torch.cuda.current_device())
        states = [(m, q.prepare_quanted_weights_for_loading(m)) for m in worker._iter_all_models()]
        out["models_refit"] = len(states)
        files = sorted(_glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not files:
            out["error"] = f"no *.safetensors in {model_dir}"
            return out
        fp8_names, bucket, nbytes, n_t, n_b, limit = set(), [], 0, 0, 0, int(bucket_mb) << 20

        def flush():
            nonlocal bucket, nbytes, n_b
            if bucket:
                worker._update_weights(bucket, peft_config=None, base_sync_done=False)
                fp8_names.update(q.fp8_state.fp8_param_names)  # quant_weights clears it per call: union per bucket
                n_b += 1
            bucket, nbytes = [], 0

        for f in files:
            with safe_open(f, framework="pt", device=str(device)) as sf:
                for k in sf.keys():
                    t = sf.get_tensor(k)
                    if bucket and nbytes + t.numel() * t.element_size() > limit:
                        flush()
                    bucket.append((k, t))
                    nbytes += t.numel() * t.element_size()
                    n_t += 1
        flush()
        for m, s in states:
            q.process_quanted_weights_after_loading(m, s)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        out.update(files=len(files), tensors=n_t, buckets=n_b, fp8_weights_quantized=len(fp8_names),
                   fp8_example=sorted(fp8_names)[:2], refit_s=round(time.time() - t0, 2))
        # the live target after the refit: every LinearBase must hold e4m3 weights with finite positive block scales.
        # worker-side, before the first timed batch: EVERY FP8 target linear must carry a
        # `weight_scale_inv` (the blockwise Fp8LinearMethod's name) of shape (ceil(out/128), ceil(in/128)), with
        # out/in = the module's output_/input_size_per_partition (vLLM's logical sizes; the weight itself may be
        # padded). There is NO fallback to `weight_scale` (the per-tensor / online path's name): a module that has
        # only that is counted as missing, and fp8_refit_problems voids the engine.
        bad, n_fp8, scale_shapes = [], 0, set()
        bn, bk = FP8_WEIGHT_BLOCK
        inv_ok, inv_missing, inv_bad, by_kind = 0, [], [], {}
        for name, mod in mr.model.named_modules():
            w = getattr(mod, "weight", None)
            if not hasattr(mod, "input_size") or getattr(mod, "quant_method", None) is None or w is None:
                continue
            if w.dtype != torch.float8_e4m3fn:
                continue
            n_fp8 += 1
            s = getattr(mod, "weight_scale_inv", None)
            if s is None:
                inv_missing.append(name + (" (has weight_scale only)" if getattr(mod, "weight_scale", None) is not None
                                           else ""))
                continue
            n_out = getattr(mod, "output_size_per_partition", None)
            n_in = getattr(mod, "input_size_per_partition", None)
            want = (-(-int(n_out) // bn), -(-int(n_in) // bk)) if isinstance(n_out, int) and isinstance(n_in, int) else None
            got = tuple(int(x) for x in s.shape)
            scale_shapes.add(got)
            kind = name.rsplit(".", 1)[-1]
            k = by_kind.setdefault(kind, dict(n=0, ok=0, out=n_out, in_=n_in, scale_inv=list(got),
                                              want=list(want) if want else None, dtype=str(s.dtype)))
            k["n"] += 1
            if want is not None and got == want:
                inv_ok += 1
                k["ok"] += 1
            else:
                inv_bad.append(f"{name}: weight_scale_inv {list(got)} != {list(want) if want else '?'} "
                               f"(out={n_out}, in={n_in}, block={FP8_WEIGHT_BLOCK})")
            if not bool(torch.isfinite(s).all()) or not bool((s > 0).all()):
                bad.append(name)
        out.update(target_fp8_linear_modules=n_fp8, target_bad_scales=bad[:5], target_bad_scales_n=len(bad),
                   scale_shape_example=[list(x) for x in sorted(scale_shapes)[:2]],
                   target_scale_inv_block=list(FP8_WEIGHT_BLOCK), target_scale_inv_ok=inv_ok,
                   target_scale_inv_missing_n=len(inv_missing), target_scale_inv_missing=inv_missing[:5],
                   target_scale_inv_bad_shape_n=len(inv_bad), target_scale_inv_bad_shape=inv_bad[:5],
                   target_scale_inv_by_kind=by_kind)
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)[:800]
    return out


def fp8_refit_problems(refit, audit, num_layers=NUM_LAYERS):
    """w8*: the reasons this engine is NOT verl's blockwise recipe (empty = OK). Any reason -> gen exits 5, no data."""
    p = []
    if not isinstance(refit, dict) or refit.get("error") or refit.get("rpc_error"):
        return [f"refit failed: {(refit or {}).get('error') or (refit or {}).get('rpc_error')}"]
    if refit.get("weight_block_size") != FP8_WEIGHT_BLOCK:
        p.append(f"live weight_block_size {refit.get('weight_block_size')} != {FP8_WEIGHT_BLOCK}")
    if refit.get("quant_config_cls") != "Fp8Config" or refit.get("activation_scheme") != "dynamic":
        p.append(f"quant config {refit.get('quant_config_cls')}/{refit.get('activation_scheme')} != Fp8Config/dynamic")
    if refit.get("is_checkpoint_fp8_serialized") is not True:  # False = vLLM's ONLINE per-tensor path (not measured)
        p.append("is_checkpoint_fp8_serialized is not True: this is not the blockwise Fp8LinearMethod path")
    if not refit.get("worker_patches_applied"):
        p.append("verl's apply_vllm_quant_patches() is not active in the engine worker")
    want = FP8_LINEAR_PER_LAYER * num_layers
    if refit.get("fp8_weights_quantized") != want:
        p.append(f"refit quantized {refit.get('fp8_weights_quantized')} linear weights != {want}")
    if refit.get("target_bad_scales_n"):
        p.append(f"{refit['target_bad_scales_n']} target layers with non-finite / non-positive block scales")
    n_fp8 = refit.get("target_fp8_linear_modules")
    if not n_fp8:
        p.append("no FP8 linear module in the target after the refit")
    # every FP8 target linear has weight_scale_inv of shape (ceil(out/128), ceil(in/128)); no fallback
    inv_ok = refit.get("target_scale_inv_ok")
    if inv_ok is None:
        p.append("weight_scale_inv audit: no worker-side weight_scale_inv shape audit in the refit result")
    elif n_fp8 and inv_ok != n_fp8:
        p.append(f"weight_scale_inv audit: {n_fp8 - inv_ok} of {n_fp8} FP8 target linears fail the shape check "
                 f"(ceil(out/128), ceil(in/128)): missing {refit.get('target_scale_inv_missing_n')} "
                 f"{refit.get('target_scale_inv_missing')}, bad shape {refit.get('target_scale_inv_bad_shape_n')} "
                 f"{refit.get('target_scale_inv_bad_shape')}")
    lin = (audit or {}).get("linear_methods") if isinstance(audit, dict) else None
    if not lin:
        p.append("worker audit has no linear_methods (cannot verify target / drafter quantization)")
    else:
        tgt = lin.get("target", {})
        if not tgt or any(k != "Fp8LinearMethod:torch.float8_e4m3fn" for k in tgt):
            p.append(f"target linear methods {sorted(tgt)} != only Fp8LinearMethod:torch.float8_e4m3fn")
        elif n_fp8 and tgt.get("Fp8LinearMethod:torch.float8_e4m3fn") != n_fp8:  # "every" FP8 linear: same module set
            p.append(f"weight_scale_inv audit: the audit sees {tgt.get('Fp8LinearMethod:torch.float8_e4m3fn')} FP8 target linears but "
                     f"the weight_scale_inv check covered {n_fp8}")
        if any(k.startswith("Fp8") or "float8" in k for k in lin.get("drafter", {})):
            p.append(f"drafter has FP8 linear methods {sorted(lin['drafter'])} (verl's drafter is bf16)")
    return p


def fp8_audit_ok(engine):
    """A w8* engine's worker-side audit passed (verl's patches active in the worker, e4m3 target weights,
    every FP8 target linear's weight_scale_inv of shape (ceil(out/128), ceil(in/128)), no refit problem). A w8* cell
    for which this is False is VOID: not scored into the verdict, not admissible, not the budget config; it counts as
    missing."""
    fw = (engine or {}).get("fp8_weights") or {}
    rf = fw.get("refit") if isinstance(fw.get("refit"), dict) else {}
    n = rf.get("target_fp8_linear_modules")
    return (fw.get("problems") == [] and bool(rf.get("worker_patches_applied")) and isinstance(n, int) and n > 0
            and rf.get("target_scale_inv_ok") == n and not rf.get("target_scale_inv_missing_n")
            and not rf.get("target_scale_inv_bad_shape_n"))


def rpc(llm, fn, *args):
    try:
        return llm.collective_rpc(fn, args=args)[0]
    except Exception as e:  # noqa: BLE001
        return {"rpc_error": repr(e)[:300]}


# ======================================================================================================= one set
def run_set(eng, reqs, sp, tracker=None, nvml=None, clock=time.perf_counter):
    """Submit every request at once (one sync-OPD rollout step of one replica) and drive step() to completion.
    reqs: [(rid, prompt_ids, pidx, sample)]. Returns (summary, per-request state)."""
    state = {rid: dict(pidx=p, sample=s, plen=len(ids), prompt=list(ids), ids=[], lq=[], finish_s=None, reason=None)
             for rid, ids, p, s in reqs}
    if tracker is not None:
        tracker.reset()
    if nvml is not None:
        nvml.reset()
    t0 = clock()
    for rid, ids, _, _ in reqs:
        eng.add_request(rid, {"prompt_token_ids": list(ids)}, sp)
    by_n, iters, t_prev, lp_missing = {}, 0, t0, 0
    while eng.has_unfinished_requests():
        n_before = eng.get_num_unfinished_requests()
        outs = eng.step()
        t_now = clock()
        if nvml is not None:
            nvml.sample(t_now)
        if not outs:
            continue  # its time is charged to the next step that returns tokens
        ntok = 0
        for o in outs:
            st, co = state[o.request_id], o.outputs[0]
            tids = list(co.token_ids)
            st["ids"].extend(tids)
            lps = co.logprobs
            if lps is None or len(lps) != len(tids):
                lp_missing += len(tids)
                st["lq"].extend([float("nan")] * len(tids))
            else:
                st["lq"].extend(float(d[t].logprob) for t, d in zip(tids, lps))
            ntok += len(tids)
            if o.finished:
                st["finish_s"], st["reason"] = round(t_now - t0, 4), co.finish_reason
        b = by_n.setdefault(n_before, [0.0, 0, 0])
        b[0] += t_now - t_prev
        b[1] += ntok
        b[2] += 1
        iters += 1
        t_prev = t_now
    makespan = clock() - t0
    lens = sorted(len(v["ids"]) for v in state.values())
    n_tok = sum(lens)
    lq_all = [x for v in state.values() for x in v["lq"] if x == x]
    q = lambda f: lens[min(len(lens) - 1, int(f * len(lens)))]  # noqa: E731
    summ = dict(makespan_s=round(makespan, 3), iters=iters, out_tokens=n_tok,
                out_tok_per_s=round(n_tok / makespan, 2) if makespan > 0 else None,
                prompt_tokens=sum(v["plen"] for v in state.values()), n_requests=len(state),
                len_p50=q(0.5), len_p90=q(0.9), len_max=lens[-1] if lens else 0,
                capped=sum(1 for v in state.values() if v["reason"] == "length"),
                unfinished=sum(1 for v in state.values() if v["reason"] is None),
                sampler_nll=round(-sum(lq_all) / len(lq_all), 5) if lq_all else None, logprob_missing=lp_missing,
                by_n={str(n): [round(v[0], 4), v[1], v[2]] for n, v in sorted(by_n.items())})
    if tracker is not None:
        summ.update(tracker.summary())
    if nvml is not None:
        summ["nvml_peak_gib"] = nvml.peak_gib()
    return summ, state


def save_tokens(path, state, save_samples):
    """Prompt ids, sampled ids and vLLM's log q of every request with sample < save_samples (ragged + offsets)."""
    import numpy as np
    keep = [(rid, v) for rid, v in state.items() if v["sample"] < save_samples]
    keep.sort(key=lambda kv: (kv[1]["pidx"], kv[1]["sample"]))
    roff, resp, lq, poff, prompt = [0], [], [], [0], []
    for _, v in keep:
        resp.extend(v["ids"])
        lq.extend(v["lq"])
        roff.append(len(resp))
        prompt.extend(v["prompt"])
        poff.append(len(prompt))
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, rid=np.array([k for k, _ in keep]), pidx=np.array([v["pidx"] for _, v in keep], np.int64),
                        sample=np.array([v["sample"] for _, v in keep], np.int32),
                        reason=np.array([str(v["reason"]) for _, v in keep]),
                        poff=np.array(poff, np.int64), prompt=np.array(prompt, np.int32),
                        roff=np.array(roff, np.int64), resp=np.array(resp, np.int32), lq=np.array(lq, np.float32))
    os.replace(tmp, path)
    return len(keep)


# ============================================================================================================== gen
def resolve_profile(a):
    prof = PROFILES[a.profile]
    for k, v in prof.items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    return a


def out_paths(d, label, group):
    return os.path.join(d, f"gen_{label}__{group}.json"), os.path.join(d, f"tok_{label}__{group}")


def cmd_gen(a):
    resolve_profile(a)
    cfg = parse_cfg(a.cfg)
    label, group = a.cfg, str(a.group or os.environ.get("SLURM_JOB_ID") or "local")
    conc, sets = parse_int_list(a.conc), parse_int_list(a.sets)
    if len(conc) != len(sets):
        raise SystemExit("--conc and --sets need the same length")
    if max(conc) > a.max_num_seqs:
        raise SystemExit(f"concurrency {max(conc)} > max_num_seqs {a.max_num_seqs}: requests would queue, not run")
    model = a.model or STUDENT
    max_model_len = a.max_prompt + a.max_tokens + 1
    kw = engine_kwargs(cfg, model, a.drafter, util=a.util, max_model_len=max_model_len, max_num_seqs=a.max_num_seqs,
                       max_batched_tokens=a.max_batched_tokens, seed=a.seed, enforce_eager=a.enforce_eager,
                       async_scheduling=a.async_scheduling)
    out_json, tok_dir = out_paths(a.out_dir, label, group)
    sched = schedule(conc, sets)
    plan = dict(label=label, group=group, cfg=cfg, engine_kwargs=kw, schedule=sched, out_json=out_json,
                tok_dir=tok_dir, args=vars(a))
    print("[g0] plan " + json.dumps(plan, default=str), flush=True)
    if a.dry_run:
        return 0
    if os.path.exists(out_json) or os.path.exists(tok_dir):
        print(f"[g0] REFUSING: {out_json} or {tok_dir} exists (never overwrite)", flush=True)
        return 3
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    tok = AutoTokenizer.from_pretrained(a.tokenizer or model)
    pool = pd.read_parquet(a.data, columns=["prompt"])["prompt"].tolist()
    psets = build_sets(len(pool), conc, sets, a.spp, a.set_seed)
    pids, n_trunc = {}, 0
    for c in conc:
        for s in psets[c]:
            for j in s:
                if j not in pids:
                    pids[j], tr = render_prompt_ids(tok, pool[j], a.max_prompt)
                    n_trunc += int(tr)
    print(f"[g0] {len(pids)} distinct prompts rendered ({n_trunc} left-truncated), sets "
          f"{ {c: [len(s) for s in psets[c]] for c in conc} }", flush=True)
    fp8w = None
    if cfg["wq"] == "fp8":
        # verl's OWN ROLLOUT_QUANT=fp8 code builds the quantization args (and, as in verl, applies
        # apply_vllm_quant_patches() and exports VERL_VLLM_FP8_QUANT_ENABLED=1 for the engine core); the declared
        # engine_kwargs must equal what it returns, or nothing runs
        import hashlib

        from transformers import AutoConfig
        import verl.workers.rollout.vllm_rollout.vllm_async_server as vas
        verl_q = verl_rollout_fp8(AutoConfig.from_pretrained(model))
        mine = {k: kw.get(k) for k in ("quantization", "hf_overrides", "worker_extension_cls")}
        src = vas.__file__
        with open(src, "rb") as fh:
            src_sha = hashlib.sha256(fh.read()).hexdigest()
        fp8w = dict(recipe="verl ROLLOUT_QUANT=fp8: blockwise 128x128 e4m3 weights, dynamic activations",
                    verl_apply_quantization=_canon(verl_q), verl_server_src=src,
                    verl_server_src_sha256=src_sha,
                    env_VERL_VLLM_FP8_QUANT_ENABLED=os.environ.get("VERL_VLLM_FP8_QUANT_ENABLED"),
                    drafter_quant=("bf16: vLLM get_draft_quant_config uses the drafter's own model config (verl sets no "
                                   "speculative quantization); verl's refit covers only the target") if cfg["spec_k"] else None)
        if _canon(verl_q) != _canon(mine) or fp8w["env_VERL_VLLM_FP8_QUANT_ENABLED"] != "1":
            print("[g0] FATAL fp8 recipe: verl's _apply_quantization gives " + json.dumps(_canon(verl_q)) +
                  " but the bench declares " + json.dumps(_canon(mine)) +
                  f" (VERL_VLLM_FP8_QUANT_ENABLED={fp8w['env_VERL_VLLM_FP8_QUANT_ENABLED']})", flush=True)
            return 5
    nvml = Nvml()
    t0 = time.time()
    llm = LLM(**kw)
    init_s = time.time() - t0
    if fp8w is not None:  # verl's first weight sync: the bf16 checkpoint through verl's blockwise refit
        fp8w["refit"] = rpc(llm, _rpc_verl_fp8_refit, model, VERL_UPDATE_BUCKET_MB)
    eng = llm.llm_engine
    tracker = SchedTracker()
    tracked = install_tracker(eng, tracker)
    vc = eng.vllm_config
    cc, sc = vc.cache_config, vc.scheduler_config
    nb, bs = int(cc.num_gpu_blocks or 0), int(cc.block_size)
    blocks_per_req = math.ceil(max_model_len / bs)
    engine = dict(init_s=round(init_s, 1), num_gpu_blocks=nb, block_size=bs, kv_tokens=nb * bs,
                  max_concurrency=round(nb / blocks_per_req, 2), cache_dtype=str(cc.cache_dtype),
                  kv_skip_layers=list(getattr(cc, "kv_cache_dtype_skip_layers", []) or []),
                  async_scheduling=getattr(sc, "async_scheduling", None), max_num_seqs=sc.max_num_seqs,
                  max_num_batched_tokens=sc.max_num_batched_tokens, max_model_len=vc.model_config.max_model_len,
                  quantization=str(vc.model_config.quantization), cudagraph_mode=str(vc.compilation_config.cudagraph_mode),
                  spec=str(vc.speculative_config) if vc.speculative_config else None, tracker_installed=tracked,
                  vllm_version=__import__("vllm").__version__, nvml=nvml.pick(), audit=rpc(llm, _rpc_audit))
    if fp8w is not None:
        engine["weight_block_size"] = (fp8w["refit"] or {}).get("weight_block_size")  # live vllm_config.quant_config
        engine["fp8_weights"] = fp8w
        fp8w["problems"] = fp8_refit_problems(fp8w["refit"], engine["audit"])
        print("[g0] fp8 weights " + json.dumps({"weight_block_size": engine["weight_block_size"],
                                                "refit": fp8w["refit"], "problems": fp8w["problems"]}, default=str),
              flush=True)
        rf = fp8w["refit"] if isinstance(fp8w["refit"], dict) else {}
        fp8w["fp8_audit_ok"] = not fp8w["problems"] and fp8_audit_ok(engine)
        print("[g0] fp8 worker audit " + json.dumps(dict(
            ok=fp8w["fp8_audit_ok"], patches=rf.get("worker_patches_applied"),
            fp8_linears=rf.get("target_fp8_linear_modules"), scale_inv_ok=rf.get("target_scale_inv_ok"),
            missing=rf.get("target_scale_inv_missing_n"), bad_shape=rf.get("target_scale_inv_bad_shape_n"),
            by_kind={k: [v.get("scale_inv"), v.get("want"), f"{v.get('ok')}/{v.get('n')}"]
                     for k, v in (rf.get("target_scale_inv_by_kind") or {}).items()}), default=str), flush=True)
        if fp8w["problems"]:
            print(f"[g0] FATAL fp8 refit: {fp8w['problems']}", flush=True)
            return 5
    if cfg["kv"] == "fp8" and cfg["kv_skip"]:  # FA3 metadata for the bf16 skip layers (see install_fa3_kvskip_patch)
        engine["fa3_kvskip_patch"] = rpc(llm, _rpc_fa3_kvskip_patch)
        print("[g0] fa3 kvskip patch " + json.dumps(engine["fa3_kvskip_patch"], default=str), flush=True)
        if engine["fa3_kvskip_patch"].get("installed") is not True:
            print("[g0] FATAL fa3 kvskip patch not installed in the worker", flush=True)
            return 6
    au = engine["audit"]
    if isinstance(au, dict) and au.get("attn_layers"):
        tl = [x for x in au["attn_layers"] if x["model"] == "target"]
        engine["kv_fp8_layers_seen"] = [i for i, x in enumerate(tl) if "fp8" in x["kv_cache_dtype"]]
        engine["kv_audit_ok"] = engine["kv_fp8_layers_seen"] == cfg["kv_fp8_layers"]
        engine["drafter_kv_dtypes"] = sorted({x["kv_cache_dtype"] for x in au["attn_layers"] if x["model"] == "drafter"})
        engine["backends"] = sorted({x["backend"] for x in au["attn_layers"]})
    print("[g0] engine " + json.dumps({k: v for k, v in engine.items() if k != "audit"}, default=str), flush=True)
    if engine.get("kv_audit_ok") is False:
        print(f"[g0] FATAL kv audit: fp8 layers seen {engine['kv_fp8_layers_seen']} != wanted {cfg['kv_fp8_layers']}",
              flush=True)
        return 4
    from vllm.inputs import TokensPrompt
    # warm-up: one untimed batch of the first scheduled shape (its set-0 prompts x spp), 512 tokens;
    # the prefix cache is reset before every timed set, so the reuse of set 0's prompts gives no cache benefit
    i0, c0 = sched[0]
    wu = [TokensPrompt(prompt_token_ids=pids[j]) for j in psets[c0][i0] for _ in range(a.spp)]
    t_wu = time.time()
    llm.generate(wu, SamplingParams(temperature=1.0, max_tokens=512, logprobs=0, detokenize=False), use_tqdm=False)
    engine["warmup"] = dict(requests=len(wu), max_tokens=512, concurrency=c0, s=round(time.time() - t_wu, 1))
    sp = SamplingParams(n=1, temperature=1.0, top_p=1.0, top_k=0, max_tokens=a.max_tokens, logprobs=0,
                        detokenize=False, repetition_penalty=1.0, ignore_eos=a.ignore_eos,
                        output_kind=RequestOutputKind.DELTA)
    doc = dict(label=label, group=group, cfg=cfg, engine_kwargs=kw, engine=engine, args=vars(a), schedule=sched,
               sets={str(c): psets[c] for c in conc}, results=[], skipped=[], done=False,
               host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"))
    os.makedirs(tok_dir)  # only now: an engine that fails to build leaves nothing behind
    _dump(doc, out_json)
    t_gen0 = time.time()
    for i, c in sched:
        if time.time() - t_gen0 > a.time_budget_s:
            doc["skipped"].append(dict(set=i, concurrency=c, why=f"time budget {a.time_budget_s}s"))
            print(f"[g0] skip C={c} set={i}: time budget", flush=True)
            continue
        llm.reset_prefix_cache()
        rpc(llm, _rpc_mem, True)
        p0 = prom_snapshot(llm)
        reqs = [(f"c{c}_s{i}_p{j}_r{s}", pids[j], j, s) for j in psets[c][i] for s in range(a.spp)]
        summ, state = run_set(eng, reqs, sp, tracker if tracked else None, nvml)
        summ.update(concurrency=c, set=i, prom=prom_delta(p0, prom_snapshot(llm)), worker_mem=rpc(llm, _rpc_mem, False),
                    t_start_s=round(time.time() - t_gen0 - summ["makespan_s"], 1))
        if "fa3_kvskip_patch" in engine:  # cumulative counters after this set
            summ["fa3_kvskip"] = rpc(llm, _rpc_fa3_kvskip_patch)
        summ["saved"] = save_tokens(os.path.join(tok_dir, f"c{c}_s{i}.npz"), state, a.save_samples)
        summ["requests"] = {rid: dict(pidx=v["pidx"], sample=v["sample"], len=len(v["ids"]), finish_s=v["finish_s"],
                                      reason=v["reason"]) for rid, v in state.items()}
        doc["results"].append(summ)
        _dump(doc, out_json)
        print("[g0] set " + json.dumps({k: v for k, v in summ.items() if k not in ("by_n", "requests", "prom")}),
              flush=True)
    doc["done"], doc["gen_wall_s"] = True, round(time.time() - t_gen0, 1)
    _dump(doc, out_json)
    print(f"[g0] done {label} group={group} -> {out_json}", flush=True)
    return 0


def _dump(doc, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, default=str)
    os.replace(tmp, path)


# ===================================================================================================== scoring
def load_gen_docs(d):
    docs = []
    for f in sorted(glob.glob(os.path.join(d, "gen_*__*.json"))):
        try:
            doc = json.load(open(f))
        except Exception as e:  # noqa: BLE001
            print(f"[g0] unreadable {f}: {e!r}", flush=True)
            continue
        doc["_path"] = f
        docs.append(doc)
    return docs


def select_for_scoring(doc, tok_dir, per_c):
    """Per concurrency: round-robin over the prompts of the saved sets (set order, then dataset index): sample 0 of
    every prompt, then sample 1, ...; the first per_c. Same prompts (and sample slots) for every config."""
    import numpy as np
    sel = []
    concs = sorted({r["concurrency"] for r in doc["results"]})
    for c in concs:
        items = []
        for r in sorted((r for r in doc["results"] if r["concurrency"] == c), key=lambda r: r["set"]):
            f = os.path.join(tok_dir, f"c{c}_s{r['set']}.npz")
            if not os.path.exists(f):
                continue
            z = np.load(f)
            for k in range(len(z["rid"])):
                a0, a1, p0, p1 = int(z["roff"][k]), int(z["roff"][k + 1]), int(z["poff"][k]), int(z["poff"][k + 1])
                items.append(dict(C=c, set=r["set"], pidx=int(z["pidx"][k]), sample=int(z["sample"][k]),
                                  rid=str(z["rid"][k]), reason=str(z["reason"][k]),
                                  prompt=[int(x) for x in z["prompt"][p0:p1]],
                                  resp=z["resp"][a0:a1].astype(np.int64), lq=z["lq"][a0:a1].astype(np.float64)))
        order = sorted(range(len(items)), key=lambda k: (items[k]["sample"], items[k]["set"], items[k]["pidx"]))
        sel.extend(items[k] for k in order[:per_c] if len(items[k]["resp"]) > 0)
    return sel


def _cluster_ci(num, den, clusters, n_boot=2000, seed=0):
    import numpy as np
    keys = sorted(set(clusters))
    if len(keys) < 2:
        return [None, None]
    idx = {k: i for i, k in enumerate(keys)}
    sn, sd = np.zeros(len(keys)), np.zeros(len(keys))
    for x, y, k in zip(num, den, clusters):
        sn[idx[k]] += x
        sd[idx[k]] += y
    rng = np.random.default_rng(seed)
    b = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    r = sn[b].sum(1) / np.maximum(sd[b].sum(1), 1)
    return [float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))]


def kl_stats(items, n_boot=2000, seed=0):
    """items: dicts with lq, lp (per sampled token), pidx (cluster), C, reason. d = log q - log p."""
    import numpy as np
    ds = [np.asarray(it["lq"], np.float64) - np.asarray(it["lp"], np.float64) for it in items]
    ok = [np.isfinite(d) for d in ds]
    dsum = [float(d[m].sum()) for d, m in zip(ds, ok)]
    dn = [int(m.sum()) for m in ok]
    k3sum = [float((np.exp(np.clip(-d[m], -50, 50)) - 1 + d[m]).sum()) for d, m in zip(ds, ok)]
    N = sum(dn)
    if N == 0:
        return dict(n_tok=0)
    allabs = np.concatenate([np.abs(d[m]) for d, m in zip(ds, ok)])
    out = dict(n_seq=len(items), n_tok=N, n_clusters=len({it["pidx"] for it in items}),
               n_nonfinite=int(sum(len(d) for d in ds) - N),
               k1=sum(dsum) / N, k1_ci95=_cluster_ci(dsum, dn, [it["pidx"] for it in items], n_boot, seed),
               k1_seq_mean=float(np.mean([s / n for s, n in zip(dsum, dn) if n])), k3=sum(k3sum) / N,
               frac_absd_gt1=float((allabs > 1).mean()), frac_absd_gt5=float((allabs > 5).mean()),
               max_absd=float(allabs.max()),
               nll_q=float(-sum(float(np.asarray(it["lq"])[m].sum()) for it, m in zip(items, ok)) / N),
               nll_p=float(-sum(float(np.asarray(it["lp"])[m].sum()) for it, m in zip(items, ok)) / N),
               frac_capped=float(np.mean([it.get("reason") == "length" for it in items])))
    bins = {}
    for lo, hi in POS_BINS:
        s = n = 0
        for d, m in zip(ds, ok):
            w = m[lo:hi]
            s += float(d[lo:hi][w].sum())
            n += int(w.sum())
        bins[f"{lo}-{hi if hi < 1 << 30 else 'end'}"] = dict(k1=s / n if n else None, n_tok=n)
    out["bins"] = bins
    byc = {}
    for it, s, n in zip(items, dsum, dn):
        b = byc.setdefault(str(it["C"]), dict(s=[], n=[], p=[]))
        b["s"].append(s)
        b["n"].append(n)
        b["p"].append(it["pidx"])
    # per concurrency, with its own prompt-cluster CI: admissibility (G0_RULES) is judged at each gated C separately
    out["by_C"] = {c: dict(k1=sum(v["s"]) / sum(v["n"]) if sum(v["n"]) else None,
                           k1_ci95=_cluster_ci(v["s"], v["n"], v["p"], n_boot, seed),
                           n_tok=sum(v["n"]), n_seq=len(v["n"]), n_clusters=len(set(v["p"])))
                   for c, v in sorted(byc.items(), key=lambda kv: int(kv[0]))}
    return out


def hf_logprobs(model, lp_fn, prompt_ids, resp_ids, device="cuda", chunk=4096, autocast=True):
    """log p(resp_t | prompt, resp_<t) under the HF model: the trainer's path (hidden states -> lm_head in the model
    dtype -> verl's logprobs_from_logits) one sequence at a time, lm_head in chunks (bounded logits memory)."""
    import numpy as np
    import torch
    P, R = len(prompt_ids), len(resp_ids)
    ids = torch.tensor([list(prompt_ids) + list(resp_ids)], device=device, dtype=torch.long)
    dev_type = torch.device(device).type
    with torch.inference_mode(), torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=autocast):
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
        hs, labels = h[P - 1:P + R - 1], ids[0, P:]
        out = [lp_fn(model.lm_head(hs[c0:c0 + chunk]), labels[c0:c0 + chunk]).float() for c0 in range(0, R, chunk)]
    return torch.cat(out).cpu().numpy().astype(np.float64)


def _torch_lp(logits, labels):
    import torch
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, labels[:, None])[:, 0]


def load_hf(model_dir, attn, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM
    used = attn
    try:
        m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, attn_implementation=attn)
    except Exception as e:  # noqa: BLE001
        print(f"[g0] attn_implementation={attn} failed ({e!r}); falling back to sdpa", flush=True)
        used = "sdpa"
        m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, attn_implementation="sdpa")
    m = m.to(device).eval()
    lp_fn, impl = _torch_lp, "torch.log_softmax(fp32)"
    try:
        from verl.utils.torch_functional import logprobs_from_logits
        lp_fn = lambda lg, lb: logprobs_from_logits(lg, lb, inplace_backward=False)  # noqa: E731
        impl = "verl.utils.torch_functional.logprobs_from_logits"
    except Exception as e:  # noqa: BLE001
        print(f"[g0] verl logprobs_from_logits unavailable ({e!r}); using torch log_softmax", flush=True)
    return m, lp_fn, dict(attn_implementation=used, lp_impl=impl,
                          flash_ce=os.environ.get("VERL_DISABLE_FLASH_ATTN_CE", "0") != "1")


def cmd_score(a):
    import numpy as np
    model_dir = a.model or STUDENT
    docs = [d for d in load_gen_docs(a.out_dir) if d.get("results")]
    for d in [d for d in docs if parse_cfg(d["label"])["wq"] == "fp8" and not fp8_audit_ok(d.get("engine", {}))]:
        print(f"[g0] VOID {d['label']}__{d['group']}: no passing worker-side FP8 audit -> not scored", flush=True)
        docs.remove(d)
    if a.labels:
        want = set(a.labels.split(","))
        docs = [d for d in docs if d["label"] in want]
    todo = []
    for doc in docs:
        base = os.path.join(a.out_dir, f"kl_{a.ref}_{doc['label']}__{doc['group']}")
        if os.path.exists(base + ".json") or os.path.exists(base + ".npz"):
            print(f"[g0] {base}.json exists -> skipped (never overwrite)", flush=True)
            continue
        if not doc.get("done"):
            print(f"[g0] WARNING {doc['label']}__{doc['group']} not done (partial gen); scoring what exists", flush=True)
        per_c = a.per_c or SCORE_PER_C.get(doc["args"].get("profile", "full"), 16)
        sel = select_for_scoring(doc, out_paths(a.out_dir, doc["label"], doc["group"])[1], per_c)
        if sel:
            todo.append((doc, base, sel, per_c))
    print(f"[g0] score ref={a.ref}: {len(todo)} configs, {sum(len(t[2]) for t in todo)} sequences, "
          f"{sum(len(s['resp']) for t in todo for s in t[2])} tokens", flush=True)
    if not todo:
        return 0
    if a.ref == "hf":
        model, lp_fn, meta = load_hf(model_dir, a.attn, a.device)

        def score_all(sel):
            for s in sel:
                s["lp"] = hf_logprobs(model, lp_fn, s["prompt"], s["resp"], device=a.device, chunk=a.chunk)
    else:
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        maxlen = max(len(s["prompt"]) + len(s["resp"]) for _, _, sel, _ in todo for s in sel)
        llm = LLM(model=model_dir, dtype="bfloat16", gpu_memory_utilization=a.vllm_util, max_model_len=maxlen + 16,
                  enable_prefix_caching=False, seed=0, max_num_seqs=32, max_num_batched_tokens=8192)
        sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0, detokenize=False)
        meta = dict(engine="vllm bf16 prefill (prompt_logprobs=0), as tim_kl_eval.py",
                    vllm_version=__import__("vllm").__version__)

        def score_all(sel):
            for c0 in range(0, len(sel), 32):  # chunks bound the host memory of prompt_logprobs dicts
                part = sel[c0:c0 + 32]
                ids = [list(s["prompt"]) + [int(x) for x in s["resp"]] for s in part]
                outs = llm.generate([TokensPrompt(prompt_token_ids=x) for x in ids], [sp] * len(part), use_tqdm=False)
                for s, x, o in zip(part, ids, outs):
                    P = len(s["prompt"])
                    s["lp"] = np.asarray([o.prompt_logprobs[k][x[k]].logprob for k in range(P, len(x))], np.float64)
    for doc, base, sel, per_c in todo:
        t = time.time()
        score_all(sel)
        for s in sel:
            assert len(s["lp"]) == len(s["resp"]), (s["rid"], len(s["lp"]), len(s["resp"]))
        st = kl_stats(sel)
        st.update(label=doc["label"], group=doc["group"], cfg=doc["cfg"], ref=a.ref, ref_meta=meta, per_c=per_c,
                  rids=[s["rid"] for s in sel], score_s=round(time.time() - t, 1))
        np.savez_compressed(base + ".tmp.npz", **{f"lq|{s['rid']}": s["lq"].astype(np.float32) for s in sel},
                            **{f"lp|{s['rid']}": s["lp"].astype(np.float32) for s in sel})
        os.replace(base + ".tmp.npz", base + ".npz")
        _dump(st, base + ".json")
        print(f"[g0] kl {a.ref} {doc['label']}__{doc['group']}: k1={st['k1']:.5f} {[round(x, 5) if x is not None else None for x in st['k1_ci95']]} "
              f"k3={st['k3']:.5f} n_tok={st['n_tok']} n_seq={st['n_seq']} |d|>1={st['frac_absd_gt1']:.2e} "
              f"by_C={ {c: round(v['k1'], 5) for c, v in st['by_C'].items() if v['k1'] is not None} } ({st['score_s']}s)",
              flush=True)
    return 0


# ====================================================================================================== report
def aggregate(doc):
    """Per concurrency: per-set makespans / tokens / by_n (all keyed by SET index, so every comparison can be restricted
    to the sets both sides ran), plus means of the other metrics."""
    out = {}
    for r in doc.get("results", []):
        a = out.setdefault(r["concurrency"], dict(mk={}, tok={}, by_n={}, pre=[], kvmax=[], waitmax=[], capped=[],
                                                  nreq=[], acc=[], nll=[], nvml=[]))
        s = r["set"]
        a["mk"][s], a["tok"][s] = r["makespan_s"], r["out_tokens"]
        a["by_n"][s] = {int(n): [t, k] for n, (t, k, _) in r.get("by_n", {}).items()}
        a["pre"].append(r.get("preemptions"))
        a["kvmax"].append(r.get("kv_usage_max"))
        a["waitmax"].append(r.get("waiting_max"))
        a["capped"].append(r.get("capped", 0))
        a["nreq"].append(r.get("n_requests", 0))
        a["acc"].append((r.get("spec_drafts") or 0, r.get("spec_accepted") or 0))
        a["nll"].append(r.get("sampler_nll"))
        a["nvml"].append(r.get("nvml_peak_gib"))
    res = {}
    for c, a in out.items():
        mks = [a["mk"][s] for s in sorted(a["mk"])]
        d, acc = sum(x for x, _ in a["acc"]), sum(y for _, y in a["acc"])
        f = lambda xs: [x for x in xs if x is not None]  # noqa: E731
        res[c] = dict(n_sets=len(mks), sets=sorted(a["mk"]), mk=a["mk"], tok=a["tok"], by_n=a["by_n"],
                      makespan_mean=sum(mks) / len(mks), makespan_median=_median(mks), makespan_sd=_sd(mks),
                      makespan_cv=_cv(mks), tok_s=sum(a["tok"].values()) / sum(mks),
                      preempt_mean=_mean(f(a["pre"])), kv_usage_max=max(f(a["kvmax"]), default=None),
                      waiting_max=max(f(a["waitmax"]), default=None),
                      capped_frac=sum(a["capped"]) / max(1, sum(a["nreq"])),
                      accept_len=1 + acc / d if d else None, sampler_nll=_mean(f(a["nll"])),
                      nvml_peak_gib=max(f(a["nvml"]), default=None))
    return res


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return None
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _sd(xs):
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _cv(xs):
    s = _sd(xs)
    return s / (sum(xs) / len(xs)) if s is not None and sum(xs) > 0 else None


def stitched_time(work, rate):
    """Time to produce work[n] tokens at each concurrency n at rate[n] tok/s (nearest measured n if missing)."""
    ns = sorted(rate)
    if not ns:
        return None
    tot = 0.0
    for n, w in work.items():
        r = rate.get(n) or rate[min(ns, key=lambda m: (abs(m - n), m))]
        tot += w / r
    return tot


def _rate(ag, sets):
    """tok/s at each number n of unfinished requests, pooled over the given sets only."""
    t, k = {}, {}
    for s in sets:
        for n, (tt, kk) in ag["by_n"].get(s, {}).items():
            t[n], k[n] = t.get(n, 0.0) + tt, k.get(n, 0) + kk
    return {n: k[n] / t[n] for n in t if t[n] > 0 and k[n] > 0}


def speedup_direct(ag_l, ag_r):
    """Same group (same GPU). EVERY statistic is over the sets BOTH ran (paired by prompt set): speedup = median_R /
    median_L (the rules' estimator), the mean ratio, per-set ratios R_s / L_s and their minimum, each side's CV over those
    sets, and the stitched estimate (the reference's per-concurrency work on those sets priced at each side's rate(n)
    on those sets)."""
    common = sorted(set(ag_l["mk"]) & set(ag_r["mk"]))
    if not common:
        return None
    L, R = [ag_l["mk"][s] for s in common], [ag_r["mk"][s] for s in common]
    work = {}
    for s in common:
        for n, (_, k) in ag_r["by_n"].get(s, {}).items():
            work[n] = work.get(n, 0.0) + k / len(common)
    tl, tr = stitched_time(work, _rate(ag_l, common)), stitched_time(work, _rate(ag_r, common))
    per = [r / x for r, x in zip(R, L)]
    return dict(speedup=_median(R) / _median(L), speedup_mean=sum(R) / sum(L), n_paired=len(common), sets=common,
                per_set=[round(x, 4) for x in per], per_set_min=min(per), cv_l=_cv(L), cv_r=_cv(R),
                stitched=(tr / tl) if tl and tr else None, how="direct (same group, paired sets)")


def speedup_anchored(ag_l, ag_r, ag_al, ag_ar, how=""):
    """Different groups (GPUs): L and its group's anchor A_l, R and its group's anchor A_r. Every statistic is over the
    sets ALL FOUR cells ran: speedup = median_R * (median_Al / median_Ar) / median_L; per-set R_s * Al_s / Ar_s / L_s.
    Never a decision input (G0_RULES)."""
    common = sorted(set(ag_l["mk"]) & set(ag_r["mk"]) & set(ag_al["mk"]) & set(ag_ar["mk"]))
    if not common:
        return None
    med = lambda ag: _median([ag["mk"][s] for s in common])  # noqa: E731
    f = med(ag_al) / med(ag_ar)
    per = [ag_r["mk"][s] * ag_al["mk"][s] / ag_ar["mk"][s] / ag_l["mk"][s] for s in common]
    return dict(speedup=med(ag_r) * f / med(ag_l), anchor_factor=f, n_paired=len(common), sets=common,
                per_set=[round(x, 4) for x in per], per_set_min=min(per),
                cv_l=_cv([ag_l["mk"][s] for s in common]), cv_r=_cv([ag_r["mk"][s] for s in common]),
                stitched=None, how=f"{how} x{f:.4f} over the {len(common)} sets common to all four cells")


def admissibility(kl_by_ref, bar, rules=G0_RULES):
    """One decision per config. kl_by_ref: {ref: kl doc or None}. For every (ref, scope) in the rules: the upper end of
    the prompt-cluster bootstrap 95% CI of k1 (pooled 'all', or that concurrency's own CI). Any upper bound > bar ->
    inadmissible; else any missing -> unscored; else admissible."""
    ups = {}
    for ref in rules["adm_refs"]:
        d = kl_by_ref.get(ref)
        for sc in rules["adm_scopes"]:
            ci = None
            if d:
                ci = d.get("k1_ci95") if sc == "all" else ((d.get("by_C") or {}).get(str(sc)) or {}).get("k1_ci95")
            ups[f"{ref}:{sc}"] = ci[1] if ci and ci[1] is not None else None
    have = [v for v in ups.values() if v is not None]
    status = "inadmissible" if any(v > bar for v in have) else ("unscored" if len(have) < len(ups) else "admissible")
    return dict(status=status, bar=bar, ci95_upper=ups, worst_upper=max(have) if have else None)


def cell_status(vs, c, thr_eff, rules=G0_RULES):
    """One (lossy config, C) cell of the PRIMARY reading. vs: {clean ref base: speedup_direct or None}."""
    got = [s for s in vs.values() if s]
    out = dict(speedup=min((s["speedup"] for s in got), default=None),
               worst_set=min((s["per_set_min"] for s in got), default=None), thr_eff=thr_eff)
    if c not in rules["gated"]:
        return dict(out, status="reported")
    missing = sorted(b for b, s in vs.items() if s is None)
    if missing:
        return dict(out, status="incomplete", why=f"no paired sets vs {missing} in this group")
    for b, s in vs.items():
        cv = max((x for x in (s["cv_l"], s["cv_r"]) if x is not None), default=None)
        if s["n_paired"] < rules["min_paired"]:
            return dict(out, status="incomplete", why=f"{s['n_paired']} paired sets vs {b} < {rules['min_paired']}")
        if cv is not None and cv > rules["cv_max"] and s["n_paired"] < rules["min_paired_if_cv"]:
            return dict(out, status="incomplete",
                        why=f"CV {cv:.3f} > {rules['cv_max']} with {s['n_paired']} paired sets vs {b} "
                            f"< {rules['min_paired_if_cv']}")
    ok = all(s["speedup"] >= thr_eff and s["per_set_min"] >= thr_eff for s in vs.values())
    return dict(out, status="clears" if ok else "fails")


def standin_check(cells, kl_docs, rules=G0_RULES, in_run_kl=None):
    """bf16_e3's hf k1 at C=16 (the shape and path of a training run with the bf16 rollout engine) in the band
    band_x x in_run_kl (the rollout_corr/kl that run logs). FAIL if any such cell is outside it; MISSING if any is unscored (or none ran); else
    PASS. in_run_kl None -> SKIPPED (no band). Every exact config's k1 is listed (descriptive)."""
    s = rules["standin"]
    vals = {}
    for k, v in cells.items():
        if v["cfg"]["base"] == s["cfg"]:
            d = kl_docs.get(s["ref"], {}).get(k) or {}
            vals[f"{k[0]}__{k[1]}"] = ((d.get("by_C") or {}).get(str(s["C"])) or {}).get("k1")
    got = [x for x in vals.values() if x is not None]
    if in_run_kl is None:
        verdict, band = "SKIPPED", None
    else:
        lo, hi = s["band_x"][0] * in_run_kl, s["band_x"][1] * in_run_kl
        band = [lo, hi]
        verdict = ("FAIL" if any(not lo <= x <= hi for x in got) else
                   "MISSING" if not vals or len(got) < len(vals) else "PASS")
    desc = {}
    for k, v in cells.items():
        if v["cfg"]["exact"]:
            row = {}
            for ref in ("hf", "vllm"):
                d = kl_docs.get(ref, {}).get(k) or {}
                row[f"k1_{ref}_all"] = d.get("k1")
                row[f"k1_{ref}_C{s['C']}"] = ((d.get("by_C") or {}).get(str(s["C"])) or {}).get("k1")
            desc[f"{k[0]}__{k[1]}"] = row
    return dict(verdict=verdict, cfg=s["cfg"], C=s["C"], ref=s["ref"], band=band, band_x=list(s["band_x"]),
                in_run_clean=in_run_kl, values=vals, all_exact=desc)


def g0_verdict(primary, adm, standin, present_bases, rules=G0_RULES):
    """The verdict over the PRIMARY reading. PASS iff some admissible config clears at some gated shape.
    FAIL iff nothing clears and nothing missing could change that. Else INCOMPLETE. A stand-in check that FAILs or is
    MISSING turns any verdict into INCOMPLETE (the would-be verdict is kept); a SKIPPED one changes nothing."""
    gated = [str(c) for c in sorted(rules["gated"])]
    st = lambda k, c: primary[k]["by_C"].get(c, {}).get("status", "incomplete")  # noqa: E731
    clears = [[k, c] for k in sorted(primary) if adm[k]["status"] == "admissible" for c in gated if st(k, c) == "clears"]
    blockers = []
    missing = sorted(set(rules["expected_lossy"]) - set(present_bases))
    if missing:
        blockers.append(f"expected lossy configs with no data: {missing}")
    for k in sorted(primary):
        a = adm[k]["status"]
        if a == "inadmissible":
            continue
        open_ = [c for c in gated if st(k, c) != "fails"]
        if a == "unscored" and open_:
            blockers.append(f"{k}: mismatch unscored, cells {open_} are not failing")
        if a == "admissible":
            inc = [c for c in gated if st(k, c) == "incomplete"]
            if inc:
                blockers.append(f"{k}: speed incomplete at C={inc}: "
                                + "; ".join(str(primary[k]['by_C'].get(c, {}).get('why', 'shape not measured')) for c in inc))
    would = "PASS" if clears else ("INCOMPLETE" if blockers else "FAIL")
    verdict = (would if standin["verdict"] in ("PASS", "SKIPPED")
               else f"INCOMPLETE (stand-in check {standin['verdict']})")
    # the budget config: the admissible config with the highest speedup at 64 (ties -> the lower mismatch)
    c64 = [(primary[k]["by_C"]["64"]["speedup"], -(primary[k]["k1_hf"] or 0.0), k) for k in primary
           if adm[k]["status"] == "admissible" and st(k, "64") in ("clears", "fails")]
    return dict(verdict=verdict, verdict_if_standin_ok=would, clearing_cells=clears, blockers=blockers,
                admissible=sorted(k for k in adm if adm[k]["status"] == "admissible"),
                inadmissible=sorted(k for k in adm if adm[k]["status"] == "inadmissible"),
                unscored=sorted(k for k in adm if adm[k]["status"] == "unscored"),
                budget_config=max(c64)[2] if c64 and would == "PASS" else None,
                rule="FAIL iff no admissible config clears at any of C=" + ", ".join(
                    f"{c} (>= {t})" for c, t in sorted(rules["gated"].items())) + " [primary reading]")


def build_report(gen_docs, kl_docs, dtol=None, dtol_meta=None, rules=G0_RULES, anchor="bf16", in_run_kl=None,
                 bar=None):
    """gen_docs: gen JSON docs; kl_docs: {ref: {(label, group): kl doc}}. Pure python (CPU-testable). The only
    decision inputs besides the data are the admissibility bar -- exactly one of `bar` (report --bar) and `dtol` (the
    D_tol of a tolerance file) -- and in_run_kl (the stand-in check's reference; None skips that check); everything
    else is G0_RULES."""
    if (bar is None) == (dtol is None):
        raise ValueError("build_report needs exactly one admissibility bar: bar (--bar) or dtol (--dtol-source)")
    cells, void = {}, []
    for d in gen_docs:
        if d.get("results"):
            cfg = parse_cfg(d["label"])
            if cfg["wq"] == "fp8" and not fp8_audit_ok(d.get("engine", {})):  # void = missing data
                void.append(f"{d['label']}__{d['group']}")
                continue
            cells[(d["label"], d["group"])] = dict(cfg=cfg, agg=aggregate(d), engine=d.get("engine", {}))
    lab = lambda k: f"{k[0]}__{k[1]}"  # noqa: E731
    groups = sorted({g for _, g in cells})
    concs = sorted({c for v in cells.values() for c in v["agg"]})
    exact = sorted(k for k, v in cells.items() if v["cfg"]["exact"])
    lossy = sorted(k for k, v in cells.items() if v["cfg"]["lossy"])

    # ---- anchor variance: a clean config that ran in >= 2 groups, over the sets common to all its copies
    anchors, av = {}, {}
    for c in concs:
        per_base = {}
        for b in rules["clean_refs"]:
            ks = [k for k in exact if cells[k]["cfg"]["base"] == b and c in cells[k]["agg"]]
            if len({k[1] for k in ks}) < 2:
                continue
            common = sorted(set.intersection(*(set(cells[k]["agg"][c]["mk"]) for k in ks)))
            if not common:
                continue
            med = {lab(k): _median([cells[k]["agg"][c]["mk"][s] for s in common]) for k in ks}
            per_base[b] = dict(medians=med, n_common=len(common), max_over_min=max(med.values()) / min(med.values()))
        if per_base:
            av[c] = max(x["max_over_min"] for x in per_base.values())
            anchors[str(c)] = dict(per_config=per_base, AV=av[c], flagged=av[c] > rules["anchor_flag"])
    thr_eff = {c: t * max(1.0, av.get(c, 1.0)) for c, t in rules["gated"].items()}

    def kl(k, ref):
        return kl_docs.get(ref, {}).get(k)

    # ---- PRIMARY reading (decides): vs every clean ref in the SAME group, paired sets, median makespans
    primary = {}
    for lk in lossy:
        refs = {b: next((k for k in exact if k[1] == lk[1] and cells[k]["cfg"]["base"] == b), None)
                for b in rules["clean_refs"]}
        by_c = {}
        for c in sorted(set(concs) | set(rules["gated"])):
            vs = {b: (speedup_direct(cells[lk]["agg"][c], cells[rk]["agg"][c])
                      if rk and c in cells[lk]["agg"] and c in cells[rk]["agg"] else None) for b, rk in refs.items()}
            by_c[str(c)] = dict(cell_status(vs, c, thr_eff.get(c), rules), vs=vs,
                                refs={b: lab(rk) if rk else None for b, rk in refs.items()})
        primary[lab(lk)] = dict(base=cells[lk]["cfg"]["base"], k1_hf=(kl(lk, "hf") or {}).get("k1"),
                                k1_vllm=(kl(lk, "vllm") or {}).get("k1"), by_C=by_c)

    # ---- admissibility and the verdict at the bar, named by its source ("bar" = --bar, "D_tol" = the tolerance file)
    bars = {"D_tol": dtol} if dtol is not None else {"bar": bar}
    standin = standin_check(cells, kl_docs, rules, in_run_kl)
    present = {cells[k]["cfg"]["base"] for k in lossy}
    adm, verdicts = {}, {}
    for name, bar in bars.items():
        adm[name] = {lab(k): admissibility({r: kl(k, r) for r in rules["adm_refs"]}, bar, rules) for k in lossy}
        verdicts[name] = dict(g0_verdict(primary, adm[name], standin, present, rules), bar=bar)
    final = next(iter(bars))

    # ---- SECONDARY readings (descriptive, never decision inputs): same-spec reference, vs production bf16+EAGLE-3
    def ref_for(lk, pred):
        cands = [k for k in exact if pred(cells[k]["cfg"])]
        return ([k for k in cands if k[1] == lk[1]] or cands or [None])[0]

    def speedup_any(lk, rk, c):
        al, ar = cells[lk]["agg"].get(c), cells[rk]["agg"].get(c)
        if al is None or ar is None:
            return None
        if lk[1] == rk[1]:
            return speedup_direct(al, ar)
        a_l = next((k for k in exact if k[1] == lk[1] and cells[k]["cfg"]["base"] == anchor), None)
        a_r = next((k for k in exact if k[1] == rk[1] and cells[k]["cfg"]["base"] == anchor), None)
        if not a_l or not a_r or c not in cells[a_l]["agg"] or c not in cells[a_r]["agg"]:
            return None
        s = speedup_anchored(al, ar, cells[a_l]["agg"][c], cells[a_r]["agg"][c], f"anchor({anchor}) {rk[1]}->{lk[1]}")
        if s:
            s["anchor_flagged"] = av.get(c) is None or av[c] > rules["anchor_flag"]
        return s

    secondary = {}
    for name, pred_of in (("same_spec", lambda lc: (lambda rc: rc["spec_k"] == lc["spec_k"])),
                          ("vs_production_bf16_e3", lambda lc: (lambda rc: rc["spec_k"] == 3))):
        rows = {}
        for lk in lossy:
            rk = ref_for(lk, pred_of(cells[lk]["cfg"]))
            if rk is not None:
                rows[lab(lk)] = dict(ref=lab(rk), by_C={str(c): speedup_any(lk, rk, c) for c in concs})
        secondary[name] = rows

    capacity = {}
    for k, v in cells.items():
        e = v["engine"]
        ref = ref_for(k, lambda rc, s=v["cfg"]["spec_k"]: rc["spec_k"] == s)
        base_tok = cells[ref]["engine"].get("kv_tokens") if ref else None
        capacity[lab(k)] = dict(kv_tokens=e.get("kv_tokens"), max_concurrency=e.get("max_concurrency"),
                                kv_ratio_vs_exact=(e["kv_tokens"] / base_tok) if base_tok and e.get("kv_tokens") else None,
                                weights_gib=(e.get("audit") or {}).get("model_memory_usage_gib"),
                                kv_gib=(e.get("audit") or {}).get("kv_cache_gib"),
                                kv_audit_ok=e.get("kv_audit_ok"), init_s=e.get("init_s"))
    table = {}
    for k, v in cells.items():
        e, rows_c = v["engine"], {}
        kv_gib, w_gib = (e.get("audit") or {}).get("kv_cache_gib"), (e.get("audit") or {}).get("model_memory_usage_gib")
        for c, a in sorted(v["agg"].items()):
            row = {x: a[x] for x in ("n_sets", "sets", "makespan_mean", "makespan_median", "makespan_sd", "makespan_cv",
                                     "tok_s", "preempt_mean", "kv_usage_max", "waiting_max", "capped_frac", "accept_len",
                                     "sampler_nll", "nvml_peak_gib")}
            u = a["kv_usage_max"]
            # KV actually occupied at the peak (a LOWER bound on the need if requests waited or were preempted) and
            # weights + that KV = the memory this config needs at this concurrency (excl. activations / graphs)
            row["kv_peak_tokens"] = int(u * e["kv_tokens"]) if u is not None and e.get("kv_tokens") else None
            row["kv_peak_gib"] = round(u * kv_gib, 3) if u is not None and kv_gib else None
            row["weights_plus_kv_peak_gib"] = round(w_gib + row["kv_peak_gib"], 3) if w_gib and row["kv_peak_gib"] else None
            row["kv_bound"] = bool((a["preempt_mean"] or 0) > 0 or (a["waiting_max"] or 0) > 0)
            rows_c[str(c)] = row
        table[lab(k)] = rows_c
    mism = {ref: {lab(k): {x: d.get(x) for x in ("k1", "k1_ci95", "k3", "n_tok", "n_seq", "frac_absd_gt1",
                                                 "frac_absd_gt5", "nll_q", "by_C", "bins")}
                  for k, d in docs.items()} for ref, docs in kl_docs.items()}
    audit_unverified = sorted(lab(k) for k, v in cells.items() if v["engine"].get("kv_audit_ok") is not True)
    return dict(rules=dict(rules, gated={str(c): t for c, t in rules["gated"].items()}), bar=bars[final],
                bar_source="--bar" if final == "bar" else "--dtol-source", dtol=dtol, dtol_source=dtol_meta,
                groups=groups, concurrencies=concs, table=table, capacity=capacity, mismatch=mism,
                standin_check=standin, anchor_variance=anchors, thr_eff={str(c): t for c, t in thr_eff.items()},
                primary=dict(reading="vs the best clean config (bf16, bf16_e3) in the SAME group, paired sets, median "
                                     "makespans", rows=primary),
                admissibility=adm, verdicts=verdicts, final=final, verdict=verdicts[final]["verdict"],
                audit_unverified=audit_unverified, void_fp8_cells=sorted(void), secondary=secondary)


def load_dtol(path):
    """A tolerance file -> (D_tol, provenance). The file must hold a top-level positive finite "D_tol"."""
    import hashlib
    raw = open(path, "rb").read()
    v = json.loads(raw).get("D_tol")
    if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        raise SystemExit(f"{path}: no positive finite top-level 'D_tol' (got {v!r}); D_tol comes from this file only")
    return float(v), dict(path=os.path.abspath(path), sha256=hashlib.sha256(raw).hexdigest(), D_tol=float(v))


def cmd_report(a):
    if (a.bar is None) == (a.dtol_source is None):
        raise SystemExit("report: give the admissibility bar as exactly one of --bar <nats/token> and "
                         "--dtol-source <tolerance JSON>")
    if a.bar is not None and not (math.isfinite(a.bar) and a.bar > 0):
        raise SystemExit(f"report: --bar must be a positive finite number of nats/token (got {a.bar!r})")
    gens = load_gen_docs(a.out_dir)
    kls = {}
    for f in sorted(glob.glob(os.path.join(a.out_dir, "kl_*_*__*.json"))):
        d = json.load(open(f))
        kls.setdefault(d["ref"], {})[(d["label"], d["group"])] = d
    dtol, meta = load_dtol(a.dtol_source) if a.dtol_source else (None, None)
    if a.in_run_kl is not None and not (math.isfinite(a.in_run_kl) and a.in_run_kl > 0):
        raise SystemExit(f"--in-run-kl must be a positive finite number (got {a.in_run_kl!r})")
    rep = build_report(gens, kls, dtol, meta, in_run_kl=a.in_run_kl, bar=a.bar)
    path = os.path.join(a.out_dir, f"g0_report_{time.strftime('%Y%m%d_%H%M%S')}.json")
    _dump(rep, path)
    print_report(rep)
    print(f"[g0] report -> {path}", flush=True)
    return 0


def print_report(rep):
    f = lambda x, p=3: "-" if x is None else (f"{x:.{p}f}" if isinstance(x, float) else str(x))  # noqa: E731
    print(f"G0_RULES version: {rep['rules'].get('version')}")
    if rep.get("void_fp8_cells"):
        print(f"VOID w8* cells (no passing worker-side FP8 audit; counted as missing data): {rep['void_fp8_cells']}")
    print(f"{'config__group':>30} {'C':>4} {'n':>2} {'median':>8} {'mean':>8} {'cv':>5} {'tok/s':>8} {'preempt':>7} "
          f"{'kvmax':>6} {'wait':>4} {'cap%':>5} {'acc_len':>7} {'nll_q':>6} {'W+KVpk':>7}")
    for k, by_c in rep["table"].items():
        for c, a in by_c.items():
            print(f"{k:>30} {c:>4} {a['n_sets']:>2} {f(a['makespan_median'], 1):>8} {f(a['makespan_mean'], 1):>8} "
                  f"{f(a['makespan_cv'], 3):>5} {f(a['tok_s'], 0):>8} {f(a['preempt_mean'], 1):>7} "
                  f"{f(a['kv_usage_max'], 2):>6} {f(a['waiting_max']):>4} {f(100 * a['capped_frac'], 0):>5} "
                  f"{f(a['accept_len'], 2):>7} {f(a['sampler_nll'], 3):>6} {f(a['weights_plus_kv_peak_gib'], 1):>7}")
    print(f"\n{'config__group':>30} {'kv_tokens':>10} {'max_conc':>8} {'kv_x':>6} {'W_GiB':>6} {'KV_GiB':>7} {'audit':>5}")
    for k, v in rep["capacity"].items():
        print(f"{k:>30} {f(v['kv_tokens']):>10} {f(v['max_concurrency'], 1):>8} {f(v['kv_ratio_vs_exact'], 3):>6} "
              f"{f(v['weights_gib'], 2):>6} {f(v['kv_gib'], 2):>7} {f(v['kv_audit_ok']):>5}")
    for ref, m in rep["mismatch"].items():
        print(f"\nmismatch vs {ref}: {'config__group':>30} {'k1':>9} {'CI95':>22} {'k3':>9} {'n_tok':>8} {'|d|>1':>8} by_C")
        for k, v in m.items():
            ci = v.get("k1_ci95") or [None, None]
            byc = " ".join(f"C{c}={f(b.get('k1'), 5)}" for c, b in (v.get("by_C") or {}).items())
            print(f"{'':>17}{k:>30} {f(v['k1'], 5):>9} {('[' + f(ci[0], 5) + ', ' + f(ci[1], 5) + ']'):>22} "
                  f"{f(v['k3'], 5):>9} {f(v['n_tok']):>8} {f(v['frac_absd_gt1'], 5):>8} {byc}")
    s = rep["standin_check"]
    if s["verdict"] == "SKIPPED":
        print(f"\nSTAND-IN CHECK [SKIPPED]: no in-run reference k1 (report --in-run-kl); {s['cfg']} {s['ref']} k1 at "
              f"C={s['C']}: {json.dumps(s['values'])}")
    else:
        print(f"\nSTAND-IN CHECK [{s['verdict']}]: {s['cfg']} {s['ref']} k1 at C={s['C']} in {s['band']} "
              f"(in-run clean {s['in_run_clean']}): {json.dumps(s['values'])}")
    print(f"  all exact configs (descriptive): {json.dumps(s['all_exact'])}")
    if rep["anchor_variance"]:
        print(f"\nanchor variance (clean configs across groups, common sets): "
              f"{json.dumps({c: dict(AV=round(v['AV'], 4), flagged=v['flagged']) for c, v in rep['anchor_variance'].items()})}")
    print(f"thresholds after anchor deflation: {json.dumps({c: round(t, 4) for c, t in rep['thr_eff'].items()})}")
    print(f"\nPRIMARY speedup [{rep['primary']['reading']}]")
    for k, r in rep["primary"]["rows"].items():
        cs = " ".join(f"C{c}={f(x['speedup'], 3)}[{x['status']}" + (f" min_set {f(x['worst_set'], 3)}" if x.get('worst_set') else "")
                      + "]" for c, x in r["by_C"].items())
        print(f"  {k:>30} k1_hf={f(r['k1_hf'], 4)} k1_vllm={f(r['k1_vllm'], 4)}: {cs}")
    for name, adm in rep["admissibility"].items():
        print(f"\nadmissibility [{name}, bar {rep['verdicts'][name]['bar']}]: "
              f"{json.dumps({k: (v['status'], v['worst_upper']) for k, v in adm.items()})}")
        print(f"VERDICT [{name}]: {json.dumps(rep['verdicts'][name])}")
    print(f"\nFINAL ({rep['final']}): {rep['verdict']}")
    if rep["audit_unverified"]:
        print(f"kv audit NOT verified for: {rep['audit_unverified']} (check the per-layer kv dtypes in engine.audit)")
    for name, rows in rep["secondary"].items():
        print(f"\nsecondary speedup [{name}] (descriptive; never a decision input)")
        for k, r in rows.items():
            cs = " ".join(f"C{c}={f(x['speedup'], 3)}(n={x['n_paired']}" + (", anchored" if x["how"].startswith("anchor") else "")
                          + ")" if x else f"C{c}=-" for c, x in r["by_C"].items())
            print(f"  {k:>30} ref={r['ref']}: {cs}")


# ===================================================================================================== self-test
class _Lp:
    def __init__(self, v):
        self.logprob = v


class _Co:
    def __init__(self, ids, lps, reason):
        self.token_ids, self.logprobs, self.finish_reason = ids, lps, reason


class _Out:
    def __init__(self, rid, co, fin):
        self.request_id, self.outputs, self.finished = rid, [co], fin


class FakeEngine:
    """Deterministic stand-in for LLMEngine (add_request / step / has_unfinished / get_num_unfinished): k tokens per
    step per request (speculation-like), random lengths, some capped at max_tokens, empty steps sprinkled in."""

    def __init__(self, max_tokens, k=2, seed=0, clock=None):
        self.rng, self.max_tokens, self.k, self.reqs, self.clock, self.n_step = random.Random(seed), max_tokens, k, {}, clock, 0

    def add_request(self, rid, prompt, sp):
        n = self.rng.choice([self.max_tokens, self.rng.randint(1, self.max_tokens)])
        self.reqs[rid] = dict(n=n, done=0, plen=len(prompt["prompt_token_ids"]))

    def has_unfinished_requests(self):
        return any(r["done"] < r["n"] for r in self.reqs.values())

    def get_num_unfinished_requests(self):
        return sum(r["done"] < r["n"] for r in self.reqs.values())

    def step(self):
        self.n_step += 1
        if self.clock is not None:
            self.clock.t += 0.01
        if self.n_step % 7 == 0:
            return []
        outs = []
        for rid, r in self.reqs.items():
            if r["done"] >= r["n"]:
                continue
            m = min(self.k, r["n"] - r["done"])
            ids = [(r["done"] + j) * 7 % 1000 + 1 for j in range(m)]
            lps = [{t: _Lp(-0.001 * (r["done"] + j))} for j, t in enumerate(ids)]
            r["done"] += m
            fin = r["done"] >= r["n"]
            outs.append(_Out(rid, _Co(ids, lps, ("length" if r["n"] == self.max_tokens else "stop") if fin else None), fin))
        return outs


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _tiny_hf_check():
    """Alignment of hf_logprobs (positions, chunking) vs a plain full-logits forward, tiny random Qwen2 on CPU."""
    import numpy as np
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=256)
    m = Qwen2ForCausalLM(cfg).eval()
    p, r = [3, 5, 7, 11, 13], [17, 19, 23, 29, 31, 37, 41]
    got = hf_logprobs(m, _torch_lp, p, r, device="cpu", chunk=3, autocast=False)
    with torch.no_grad():
        lg = m(torch.tensor([p + r])).logits[0].float().log_softmax(-1)
    ref = np.array([float(lg[len(p) - 1 + t, tok]) for t, tok in enumerate(r)])
    err = float(np.abs(got - ref).max())
    assert got.shape == (len(r),) and err < 1e-5, err
    # an off-by-one would score the wrong token: make sure the check can fail
    wrong = np.array([float(lg[len(p) + t, tok]) for t, tok in enumerate(r[:-1])])
    assert float(np.abs(got[:-1] - wrong).max()) > 1e-3
    return err


def cmd_selftest(a):
    import tempfile

    import numpy as np
    ok = []
    # 1. grammar
    matrix = ["bf16", "w8", "kv8all", "kv8s0", "kv8s0-27", "w8_kv8s0", "bf16_e3", "w8_e3", "kv8s0_e3", "kv8s0-27_e3",
              "w8_kv8s0_e3", "kv8only5", "e3k2", "bf16.end"]
    P = {n: parse_cfg(n) for n in matrix}
    assert P["bf16"]["exact"] and not P["bf16"]["lossy"] and P["bf16"]["spec_k"] == 0
    assert P["w8"]["wq"] == "fp8" and P["w8"]["kv"] == "auto" and P["w8"]["lossy"]
    assert P["kv8all"]["kv_skip"] == [] and P["kv8all"]["kv_fp8_layers"] == list(range(28))
    assert P["kv8s0"]["kv_skip"] == [0] and len(P["kv8s0"]["kv_fp8_layers"]) == 27
    assert P["kv8s0-27"]["kv_skip"] == [0, 27] and 27 not in P["kv8s0-27"]["kv_fp8_layers"]
    assert P["kv8only5"]["kv_fp8_layers"] == [5] and len(P["kv8only5"]["kv_skip"]) == 27
    assert P["w8_kv8s0_e3"]["wq"] == "fp8" and P["w8_kv8s0_e3"]["spec_k"] == 3 and P["e3k2"]["spec_k"] == 2
    assert P["bf16.end"]["exact"] and P["bf16.end"]["base"] == "bf16"
    for bad in ("bf16_w8", "kv8", "kv8s28", "kv8s0_kv8all", "e3k9", "fp8", "w8__e3", "kv8only", "e3_e3", "x;rm"):
        try:
            parse_cfg(bad)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    ok.append("grammar")
    # 2. engine kwargs
    kw = engine_kwargs(P["w8_kv8s0_e3"], "/m", "/d")
    assert kw["quantization"] == "fp8" and kw["kv_cache_dtype"] == "fp8" and kw["kv_cache_dtype_skip_layers"] == ["0"]
    assert kw["speculative_config"] == {"method": "eagle3", "model": "/d", "num_speculative_tokens": 3}
    # every w8* config is verl's blockwise recipe
    for n in ("w8", "w8_e3", "w8_kv8s0", "w8_kv8s0_e3"):
        kw8 = engine_kwargs(P[n], "/m", "/d")
        qc = kw8["hf_overrides"]["quantization_config"]
        assert kw8["quantization"] == "fp8" and kw8["worker_extension_cls"] == VERL_WORKER_EXT
        assert qc["weight_block_size"] == [128, 128] and qc["quant_method"] == "fp8" and qc["fmt"] == "e4m3"
        assert qc["activation_scheme"] == "dynamic" and len(qc["ignored_layers"]) == NUM_LAYERS
        assert qc["ignored_layers"][0] == "model.layers.0.mlp.gate" and qc["ignored_layers"][-1] == "model.layers.27.mlp.gate"
    kw = engine_kwargs(P["bf16"], "/m", "/d")
    assert not ({"quantization", "kv_cache_dtype", "kv_cache_dtype_skip_layers", "speculative_config", "hf_overrides",
                 "worker_extension_cls"} & set(kw))
    for n in ("kv8all", "kv8s0", "kv8s0-27", "bf16_e3", "kv8s0_e3", "kv8s0-27_e3", "kv8only5"):
        assert not ({"quantization", "hf_overrides", "worker_extension_cls"} & set(engine_kwargs(P[n], "/m", "/d")))
    # the refit verdict: a healthy blockwise engine passes; each single defect is named
    good = dict(weight_block_size=[128, 128], quant_config_cls="Fp8Config", activation_scheme="dynamic",
                is_checkpoint_fp8_serialized=True, worker_patches_applied=3, fp8_weights_quantized=196,
                target_bad_scales_n=0, target_fp8_linear_modules=112, target_scale_inv_ok=112,
                target_scale_inv_missing_n=0, target_scale_inv_bad_shape_n=0)
    aud = {"linear_methods": {"target": {"Fp8LinearMethod:torch.float8_e4m3fn": 112},
                              "drafter": {"UnquantizedLinearMethod:torch.bfloat16": 5}}}
    assert fp8_refit_problems(good, aud) == []
    assert fp8_audit_ok(dict(fp8_weights=dict(refit=good, problems=[])))
    # the weight_scale_inv shape audit must exist, cover every FP8 target linear, and pass for all of them
    for bad_kv in (dict(target_scale_inv_ok=None), dict(target_scale_inv_ok=111, target_scale_inv_bad_shape_n=1),
                   dict(target_scale_inv_ok=111, target_scale_inv_missing_n=1)):
        assert any(x.startswith("weight_scale_inv audit")
                   for x in fp8_refit_problems(dict(good, **bad_kv), aud)), bad_kv
        assert not fp8_audit_ok(dict(fp8_weights=dict(refit=dict(good, **bad_kv), problems=[]))), bad_kv
    assert any(x.startswith("weight_scale_inv audit")
               for x in fp8_refit_problems(dict(good, target_fp8_linear_modules=100, target_scale_inv_ok=100), aud))
    assert not fp8_audit_ok(dict(fp8_weights=dict(refit=good, problems=["x"]))) and not fp8_audit_ok({})
    for bad_kv, bad_aud in ((dict(weight_block_size=[]), aud), (dict(is_checkpoint_fp8_serialized=False), aud),
                            (dict(worker_patches_applied=0), aud), (dict(fp8_weights_quantized=168), aud),
                            (dict(target_bad_scales_n=2), aud), (dict(error="x"), aud), ({}, {}),
                            ({}, {"linear_methods": {"target": {"Fp8OnlineLinearMethod:torch.float8_e4m3fn": 112}}}),
                            ({}, {"linear_methods": {"target": aud["linear_methods"]["target"],
                                                     "drafter": {"Fp8LinearMethod:torch.float8_e4m3fn": 5}}})):
        assert fp8_refit_problems(dict(good, **bad_kv), bad_aud), (bad_kv, bad_aud)
    assert fp8_refit_problems({"rpc_error": "boom"}, aud)
    assert kw["max_model_len"] == 18433 and kw["max_num_seqs"] == 256 and kw["max_num_batched_tokens"] == 8192
    assert kw["gpu_memory_utilization"] == 0.65 and kw["enable_prefix_caching"] and kw["seed"] == 42
    assert kw["logprobs_mode"] == "processed_logprobs" and kw["compilation_config"]["cudagraph_mode"] == "FULL_AND_PIECEWISE"
    assert "kv_cache_dtype_skip_layers" not in engine_kwargs(P["kv8all"], "/m", "/d")
    assert engine_kwargs(P["kv8s0-27"], "/m", "/d")["kv_cache_dtype_skip_layers"] == ["0", "27"]
    ok.append("engine_kwargs")
    # 3. sets / schedule
    conc, sets = parse_int_list(PROFILES["full"]["conc"]), parse_int_list(PROFILES["full"]["sets"])
    assert conc == [16, 64, 128, 256] and sets == [6, 5, 5, 2]
    assert all(n >= G0_RULES["min_paired_if_cv"] for c, n in zip(conc, sets) if c in G0_RULES["gated"])
    s1, s2 = build_sets(17917, conc, sets, 8, 7), build_sets(17917, conc, sets, 8, 7)
    assert s1 == s2 and all(len(s1[c]) == n and all(len(x) == c // 8 for x in s1[c]) for c, n in zip(conc, sets))
    assert all(len(set(x)) == len(x) for c in conc for x in s1[c])
    assert s1[16][0] != s1[16][1] and build_sets(17917, conc, sets, 8, 8) != s1
    sch = schedule(conc, sets)
    # gated shapes first (interleaved), the ungated 256 last: a time budget cuts 256 before any gated set
    assert len(sch) == sum(sets) and sch[:3] == [(0, 16), (0, 64), (0, 128)] and sch[-3:] == [(5, 16), (0, 256), (1, 256)]
    assert all(c != 256 for _, c in sch[:16]) and schedule([16], [1]) == [(0, 16)] and schedule([256], [2]) == [(0, 256), (1, 256)]
    try:
        build_sets(100, [12], [1], 8, 0)
        raise AssertionError("accepted C not multiple of spp")
    except ValueError:
        pass
    ok.append("sets")
    # 4. bookkeeping on a fake engine (+ tracker + save + selection)
    clk = _Clock()
    eng = FakeEngine(max_tokens=50, k=3, seed=1, clock=clk)
    reqs = [(f"c16_s0_p{j}_r{s}", [1, 2, 3, j], j, s) for j in (5, 9) for s in range(8)]
    summ, state = run_set(eng, reqs, None, clock=clk)
    assert summ["n_requests"] == 16 and summ["unfinished"] == 0 and summ["logprob_missing"] == 0
    assert summ["out_tokens"] == sum(r["n"] for r in eng.reqs.values())
    assert abs(sum(v[0] for v in summ["by_n"].values()) - summ["makespan_s"]) < 1e-6
    assert sum(v[1] for v in summ["by_n"].values()) == summ["out_tokens"]
    assert summ["capped"] == sum(r["n"] == 50 for r in eng.reqs.values())
    assert all(len(v["ids"]) == len(v["lq"]) for v in state.values())
    assert max(int(n) for n in summ["by_n"]) == 16
    tr = SchedTracker()

    class _S:
        def __init__(self, u, run, wait, sd=None):
            self.kv_cache_usage, self.num_running_reqs, self.num_waiting_reqs, self.spec_decoding_stats = u, run, wait, sd

    class _SD:
        def __init__(self, d, dt, acc, pp):
            self.num_drafts, self.num_draft_tokens, self.num_accepted_tokens, self.num_accepted_tokens_per_pos = d, dt, acc, pp

    class _I:
        def __init__(self, p, g):
            self.num_preempted_reqs, self.num_generation_tokens = p, g
    tr.record(_S(0.5, 16, 0, _SD(16, 48, 30, [14, 10, 6])), _I(0, 46))
    tr.record(scheduler_stats=_S(0.9, 12, 4, _SD(12, 36, 12, [8, 3, 1])), iteration_stats=_I(2, 24))
    t = tr.summary()
    assert t["kv_usage_max"] == 0.9 and t["waiting_max"] == 4 and t["preemptions"] == 2 and t["spec_drafts"] == 28
    assert abs(t["accept_len"] - (1 + 42 / 28)) < 1e-3 and t["accept_per_pos"] == [round(22 / 28, 4), round(13 / 28, 4), round(7 / 28, 4)]
    assert prom_delta({"a": 1, "v": [1, 2]}, {"a": 4, "v": [2, 5, 1]}) == {"a": 3, "v": [1, 3, 1]}
    with tempfile.TemporaryDirectory() as td:
        tokdir = os.path.join(td, "tok_x__g")
        os.makedirs(tokdir)
        n_saved = save_tokens(os.path.join(tokdir, "c16_s0.npz"), state, 2)
        assert n_saved == 4
        z = np.load(os.path.join(tokdir, "c16_s0.npz"))
        for k in range(n_saved):
            v = state[str(z["rid"][k])]
            assert list(z["resp"][z["roff"][k]:z["roff"][k + 1]]) == v["ids"]
            assert np.allclose(z["lq"][z["roff"][k]:z["roff"][k + 1]], v["lq"])
        doc = dict(results=[dict(concurrency=16, set=0)])
        sel = select_for_scoring(doc, tokdir, 3)
        assert [(s["sample"], s["pidx"]) for s in sel] == [(0, 5), (0, 9), (1, 5)]
    ok.append("bookkeeping")
    # 5. stats
    rng = np.random.default_rng(0)
    items = []
    for j in range(40):
        lq = -rng.exponential(1.0, 300)
        items.append(dict(lq=lq, lp=lq - 0.01 - 0.001 * (j % 3), pidx=j // 2, C=16 if j < 20 else 64, reason="stop"))
    st = kl_stats(items)
    allq = np.concatenate([i["lq"] for i in items])
    allp = np.concatenate([i["lp"] for i in items])
    assert abs(st["k1"] - float((allq - allp).mean())) < 1e-12 and st["n_tok"] == 12000 and st["n_clusters"] == 20
    assert st["k1_ci95"][0] <= st["k1"] <= st["k1_ci95"][1]
    assert abs(st["k3"] - float((np.exp(allp - allq) - 1 - (allp - allq)).mean())) < 1e-12
    assert st["bins"]["0-1024"]["n_tok"] == 12000 and set(st["by_C"]) == {"16", "64"}
    items[0]["lp"] = items[0]["lp"].copy()
    items[0]["lp"][5] = float("nan")
    assert kl_stats(items)["n_nonfinite"] == 1
    ok.append("stats")
    # 6. report: pairing, the primary rule, admissibility, stand-in, verdicts (synthetic docs, exact arithmetic)

    def gdoc(label, group, mks, tok=10000.0):
        """mks: {C: [makespan of set i, or None = set i not run (time budget)]}"""
        res = []
        for c, lst in mks.items():
            for i, m in enumerate(lst):
                if m is not None:
                    res.append(dict(concurrency=c, set=i, makespan_s=m, out_tokens=tok, preemptions=0, kv_usage_max=0.5,
                                    waiting_max=0, capped=1, n_requests=c, spec_drafts=0, spec_accepted=0,
                                    sampler_nll=0.7, nvml_peak_gib=60.0, by_n={str(c): [m, tok, 100]}))
        eng = dict(kv_tokens=10**6 if "kv8" not in label else 2 * 10**6, max_concurrency=50.0,
                   audit=dict(model_memory_usage_gib=10.0, kv_cache_gib=50.0), kv_audit_ok=True)
        if label.startswith("w8"):  # a passing worker-side FP8 audit (else the cell is void)
            eng["fp8_weights"] = dict(problems=[], refit=good)
        return dict(label=label, group=group, results=res, engine=eng)

    def kdoc(k1, up):
        return dict(k1=k1, k1_ci95=[k1 / 2, up], by_C={str(c): dict(k1=k1, k1_ci95=[k1 / 2, up]) for c in GATED_C})

    IN_RUN = 0.001  # a synthetic in-run reference k1 for the stand-in check: band [0.0005, 0.002]
    BAR = 0.11      # a synthetic admissibility bar (report --bar)

    # 6a. a cross-group ratio uses ONLY the sets all four cells ran. Here w8 lost set 5 to the time budget, bf16_e3
    # (other GPU) only ran sets 0-2 and that GPU's anchor lost set 1 -> sets {0, 2}, medians.
    R1 = [gdoc("bf16", "1", {16: [100, 102, 104, 106, 108, 110]}), gdoc("w8", "1", {16: [80, 81, 82, 83, 84, None]}),
          gdoc("bf16", "2", {16: [110, None, 114, 116, 118, 120]}), gdoc("bf16_e3", "2", {16: [90, 91, 92]})]
    rep1 = build_report(R1, {}, bar=BAR)
    v = rep1["secondary"]["vs_production_bf16_e3"]["w8__1"]["by_C"]["16"]
    assert v["sets"] == [0, 2] and v["n_paired"] == 2 and abs(v["speedup"] - 91 * (102 / 112) / 81) < 1e-12
    assert v["per_set"] == [round(r * a / b / x, 4) for r, a, b, x in ((90, 100, 110, 80), (92, 104, 114, 82))]
    s = rep1["secondary"]["same_spec"]["w8__1"]["by_C"]["16"]  # same group: paired over w8's 5 sets, not bf16's 6
    assert s["how"].startswith("direct") and s["sets"] == [0, 1, 2, 3, 4] and abs(s["speedup"] - 104 / 82) < 1e-12
    p = rep1["primary"]["rows"]["w8__1"]["by_C"]["16"]  # primary never goes cross-group: bf16_e3 is not in group 1
    assert p["status"] == "incomplete" and "bf16_e3" in p["why"]
    assert abs(rep1["anchor_variance"]["16"]["AV"] - 116 / 106) < 1e-12  # bf16 g1 vs g2 over sets {0,2,3,4,5}: medians
    # 6b. the primary rule. Both clean refs in both groups; AV_16 = 82/80 -> thr_eff(16) = 1.15 * 1.025
    G = [gdoc("bf16", "1", {16: [100] * 6, 64: [200] * 5, 128: [300] * 5}),
         gdoc("bf16_e3", "1", {16: [80] * 5 + [160], 64: [180] * 5, 128: [290] * 5}),
         gdoc("kv8s0", "1", {16: [64] * 5 + [None], 64: [150] * 5, 128: [220, 220]}),  # clears at 16 (paired 0-4)
         gdoc("w8", "1", {16: [60, 60, 60, 60, 75, 60], 64: [170] * 5, 128: [280] * 5}),  # one set below -> fails
         gdoc("kv8all", "1", {16: [40] * 6, 64: [100] * 5, 128: [150] * 5}),  # fastest, inadmissible
         gdoc("kv8s0-27", "1", {16: [75] * 6, 64: [100, 120, 140], 128: [280] * 5}),  # C64: CV>5% with 3 sets
         gdoc("bf16", "2", {16: [102] * 6, 64: [200] * 5, 128: [300] * 5}),
         gdoc("bf16_e3", "2", {16: [82] * 6, 64: [180] * 5, 128: [290] * 5}),
         gdoc("kv8s0_e3", "2", {16: [70] * 6, 64: [170] * 5, 128: [280] * 5})]  # 82/70 = 1.171: >= 1.15, < thr_eff
    hf = {("kv8s0", "1"): kdoc(0.004, 0.006), ("w8", "1"): kdoc(0.003, 0.005), ("kv8all", "1"): kdoc(0.5, 0.6),
          ("kv8s0-27", "1"): kdoc(0.003, 0.005), ("kv8s0_e3", "2"): kdoc(0.003, 0.004),
          ("bf16_e3", "1"): kdoc(0.0009, 0.0012), ("bf16_e3", "2"): kdoc(0.0010, 0.0013)}
    vl = {k: d for k, d in hf.items() if k != ("kv8s0_e3", "2")}  # kv8s0_e3 has no vllm-ref score -> unscored
    rep = build_report(G, {"hf": hf, "vllm": vl}, in_run_kl=IN_RUN, bar=BAR)
    assert abs(rep["anchor_variance"]["16"]["AV"] - 82 / 80) < 1e-12 and rep["anchor_variance"]["16"]["flagged"] is False
    assert abs(rep["thr_eff"]["16"] - 1.15 * 82 / 80) < 1e-12 and rep["thr_eff"]["64"] == 1.3
    P_ = rep["primary"]["rows"]
    c = P_["kv8s0__1"]["by_C"]["16"]
    assert c["status"] == "clears" and abs(c["speedup"] - 80 / 64) < 1e-12 and c["vs"]["bf16_e3"]["sets"] == [0, 1, 2, 3, 4]
    assert abs(c["vs"]["bf16_e3"]["speedup_mean"] - 1.25) < 1e-12 and abs(c["vs"]["bf16_e3"]["stitched"] - 1.25) < 1e-12
    assert P_["kv8s0__1"]["by_C"]["64"]["status"] == "fails" and abs(P_["kv8s0__1"]["by_C"]["64"]["speedup"] - 1.2) < 1e-12
    assert P_["kv8s0__1"]["by_C"]["128"]["status"] == "incomplete" and "2 paired" in P_["kv8s0__1"]["by_C"]["128"]["why"]
    w = P_["w8__1"]["by_C"]["16"]  # median ratio 80/60 = 1.33 but the paired set 4 is 80/75 = 1.067 -> fails
    assert w["status"] == "fails" and abs(w["speedup"] - 80 / 60) < 1e-12 and abs(w["worst_set"] - 80 / 75) < 1e-12
    assert P_["kv8s0-27__1"]["by_C"]["64"]["status"] == "incomplete" and "CV" in P_["kv8s0-27__1"]["by_C"]["64"]["why"]
    e3 = P_["kv8s0_e3__2"]["by_C"]["16"]
    assert e3["status"] == "fails" and 1.15 < e3["speedup"] < rep["thr_eff"]["16"]  # the anchor deflation bites
    assert P_["kv8all__1"]["by_C"]["16"]["status"] == "clears" and cell_status({"bf16": None}, 256, None)["status"] == "reported"
    a = rep["admissibility"]["bar"]
    assert (a["kv8s0__1"]["status"], a["kv8all__1"]["status"], a["kv8s0_e3__2"]["status"]) == ("admissible", "inadmissible", "unscored")
    assert rep["standin_check"]["verdict"] == "PASS" and rep["final"] == "bar" and rep["bar"] == BAR
    vp = rep["verdicts"]["bar"]
    assert vp["verdict"] == "PASS" and vp["clearing_cells"] == [["kv8s0__1", "16"]] and vp["budget_config"] == "kv8s0__1"
    assert rep["verdict"] == "PASS"
    # 6c. D_tol from a tolerance file: kv8s0 (CI upper 0.006) drops out -> nothing admissible clears; still INCOMPLETE because
    # expected configs are missing and kv8s0-27 is incomplete at 64; kv8s0_e3 (unscored) fails everywhere -> no blocker
    rep = build_report(G, {"hf": hf, "vllm": vl}, dtol=0.005, dtol_meta=dict(path="x"), in_run_kl=IN_RUN)
    vd = rep["verdicts"]["D_tol"]
    assert rep["final"] == "D_tol" and set(rep["verdicts"]) == {"D_tol"} and rep["bar"] == 0.005
    assert vd["verdict"] == "INCOMPLETE" and rep["bar_source"] == "--dtol-source"
    assert any("no data" in b for b in vd["blockers"]) and any(b.startswith("kv8s0-27__1: speed incomplete") for b in vd["blockers"])
    assert not any(b.startswith("kv8s0_e3__2") for b in vd["blockers"]) and vd["budget_config"] is None
    # 6d. FAIL needs every expected config present and every open cell closed
    Gf = [g for g in G if g["label"] != "kv8s0-27"] + [gdoc("kv8s0-27", "1", {16: [75] * 6, 64: [170] * 5, 128: [280] * 5})]
    rules = dict(G0_RULES, expected_lossy=("w8", "kv8all", "kv8s0", "kv8s0-27", "kv8s0_e3"))
    rep = build_report(Gf, {"hf": hf, "vllm": vl}, dtol=0.005, rules=rules, in_run_kl=IN_RUN)
    assert rep["verdict"] == "FAIL" and rep["verdicts"]["D_tol"]["blockers"] == []
    # ... an UNSCORED config with a cell that is not failing (here incomplete at 128) could still pass -> INCOMPLETE
    Gu = [g for g in Gf if g["label"] != "kv8s0_e3"] + [gdoc("kv8s0_e3", "2", {16: [70] * 6, 64: [170] * 5, 128: [200, 200]})]
    vd = build_report(Gu, {"hf": hf, "vllm": vl}, dtol=0.005, rules=rules, in_run_kl=IN_RUN)["verdicts"]["D_tol"]
    assert vd["verdict"] == "INCOMPLETE" and vd["blockers"] == ["kv8s0_e3__2: mismatch unscored, cells ['128'] are not failing"]
    # 6e. stand-in: bf16_e3 at C=16 (not bf16, not pooled); FAIL / MISSING -> INCOMPLETE, would-be verdict kept
    hf_bad = {**hf, ("bf16_e3", "2"): kdoc(0.003, 0.004)}
    rep = build_report(G, {"hf": hf_bad, "vllm": vl}, in_run_kl=IN_RUN, bar=BAR)
    assert rep["standin_check"]["verdict"] == "FAIL" and rep["verdict"] == "INCOMPLETE (stand-in check FAIL)"
    assert rep["verdicts"]["bar"]["verdict_if_standin_ok"] == "PASS"
    rs = build_report(G, {"hf": hf_bad, "vllm": vl}, bar=BAR)  # no in-run reference: the check is skipped, not failed
    assert rs["standin_check"]["verdict"] == "SKIPPED" and rs["standin_check"]["band"] is None and rs["verdict"] == "PASS"
    hf_pool = {**hf, ("bf16_e3", "2"): dict(kdoc(0.0010, 0.0013), k1=0.05)}  # pooled value is NOT what is checked
    rp = build_report(G, {"hf": hf_pool, "vllm": vl}, in_run_kl=IN_RUN, bar=BAR)
    assert rp["standin_check"]["verdict"] == "PASS"
    hf_miss = {k: d for k, d in hf.items() if k != ("bf16_e3", "2")}
    rm = build_report(G, {"hf": hf_miss, "vllm": vl}, in_run_kl=IN_RUN, bar=BAR)
    assert rm["verdict"] == "INCOMPLETE (stand-in check MISSING)"
    for kw in (dict(), dict(bar=BAR, dtol=0.005)):  # the bar comes from exactly one source; there is no default
        try:
            build_report(G, {"hf": hf, "vllm": vl}, **kw)
            raise AssertionError(f"build_report accepted bar sources {sorted(kw)}")
        except ValueError:
            pass
    # 6f. admissibility units: CI upper (not the point), every ref, every gated C
    d0 = kdoc(0.05, 0.09)
    assert admissibility({"hf": d0, "vllm": d0}, 0.11)["status"] == "admissible"
    assert admissibility({"hf": d0, "vllm": None}, 0.11)["status"] == "unscored"
    assert admissibility({"hf": kdoc(0.10, 0.12), "vllm": d0}, 0.11)["status"] == "inadmissible"  # point 0.10 <= bar
    d1 = kdoc(0.05, 0.09)
    d1["by_C"]["64"]["k1_ci95"] = [0.08, 0.2]
    assert admissibility({"hf": d0, "vllm": d1}, 0.11)["status"] == "inadmissible"  # one C, one ref suffices
    assert admissibility({"hf": kdoc(0.5, 0.6), "vllm": None}, 0.11)["status"] == "inadmissible"
    # 6g. the report subcommand end to end: D_tol only from the tolerance file (sha256 recorded); a bad file is refused;
    # --bar in place of the file; no bar, both sources or a non-positive bar is refused with a one-line message
    import contextlib
    import io
    with tempfile.TemporaryDirectory() as td:
        for g in G:
            _dump(g, os.path.join(td, f"gen_{g['label']}__{g['group']}.json"))
        for ref, docs in (("hf", hf), ("vllm", vl)):
            for (lb, gr), d in docs.items():
                _dump(dict(d, ref=ref, label=lb, group=gr), os.path.join(td, f"kl_{ref}_{lb}__{gr}.json"))
        _dump({"D_tol": 0.005}, os.path.join(td, "dtol.json"))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            assert main(["report", "--out-dir", td, "--dtol-source", os.path.join(td, "dtol.json"),
                         "--in-run-kl", str(IN_RUN)]) == 0
        r = json.load(open(glob.glob(os.path.join(td, "g0_report_*.json"))[0]))
        assert r["final"] == "D_tol" and r["verdict"] == "INCOMPLETE" and len(r["dtol_source"]["sha256"]) == 64
        assert "FINAL (D_tol): INCOMPLETE" in out.getvalue() and "STAND-IN CHECK [PASS]" in out.getvalue()
        _dump({"D_tol": None}, os.path.join(td, "bad.json"))
        try:
            main(["report", "--out-dir", td, "--dtol-source", os.path.join(td, "bad.json")])
            raise AssertionError("accepted a D_tol-less file")
        except SystemExit as e:
            assert "D_tol" in str(e)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            assert main(["report", "--out-dir", td, "--bar", str(BAR)]) == 0
        assert "FINAL (bar): PASS" in out.getvalue() and "STAND-IN CHECK [SKIPPED]" in out.getvalue()
        for argv in ([], ["--bar", str(BAR), "--dtol-source", os.path.join(td, "dtol.json")], ["--bar", "0"]):
            try:
                main(["report", "--out-dir", td] + argv)
                raise AssertionError(f"report accepted {argv}")
            except SystemExit as e:
                assert isinstance(e.code, str) and "\n" not in e.code and "--bar" in e.code, e.code
    # 6h. a w8* cell without a passing worker audit is VOID -> counted as missing (an expected config with no data
    # blocks a FAIL), never in primary / admissibility / the budget config; the report prints the rules version
    Gv = [g for g in Gf if g["label"] != "w8"] + [dict(gdoc("w8", "1", {16: [60] * 6, 64: [100] * 5, 128: [150] * 5}))]
    Gv[-1]["engine"] = dict(Gv[-1]["engine"], fp8_weights=dict(problems=[], refit=dict(good, target_scale_inv_ok=0)))
    rv = build_report(Gv, {"hf": hf, "vllm": vl}, dtol=0.005, rules=rules, in_run_kl=IN_RUN)
    assert rv["void_fp8_cells"] == ["w8__1"] and "w8__1" not in rv["primary"]["rows"]
    assert "w8__1" not in rv["admissibility"]["D_tol"] and rv["verdicts"]["D_tol"]["budget_config"] is None
    assert rv["verdict"] == "INCOMPLETE" and any("no data" in b and "w8" in b for b in rv["verdicts"]["D_tol"]["blockers"])
    assert "worker-side FP8 audit" in G0_RULES["version"] and "void" in G0_RULES["version"]
    with contextlib.redirect_stdout(io.StringIO()) as out:
        print_report(rv)
    assert f"G0_RULES version: {G0_RULES['version']}" in out.getvalue() and "VOID w8* cells" in out.getvalue()
    assert rep1["capacity"]["w8__1"]["kv_ratio_vs_exact"] == 1.0 and abs(stitched_time({1: 100, 4: 400}, {1: 10.0, 3: 40.0}) - 20) < 1e-9
    assert _median([3, 1, 2]) == 2 and _median([4, 1, 2, 3]) == 2.5 and _cv([5.0]) is None
    with contextlib.redirect_stdout(io.StringIO()):
        print_report(rep)
        print_report(rep1)
    ok.append("report")
    # 7. HF alignment (tiny random model, CPU)
    try:
        err = _tiny_hf_check()
        ok.append(f"hf_alignment(max_err={err:.1e})")
    except ImportError as e:
        ok.append(f"hf_alignment SKIPPED ({e})")
    print("[g0] selftest PASS: " + ", ".join(ok), flush=True)
    return 0


# ========================================================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--cfg", required=True, help="config name (grammar in the module docstring)")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--group", default=None, help="comparison group (default SLURM_JOB_ID): same GPU = same group")
    g.add_argument("--profile", default="full", choices=sorted(PROFILES))
    g.add_argument("--conc", default=None, help="comma list of concurrencies (default: profile)")
    g.add_argument("--sets", default=None, help="comma list: prompt sets per concurrency (default: profile)")
    g.add_argument("--spp", type=int, default=None, help="samples per prompt (training n=8)")
    g.add_argument("--max-tokens", type=int, default=None, help="response cap (training MAX_RESP 16384)")
    g.add_argument("--save-samples", type=int, default=None, help="save ids+logq of samples r < this per prompt")
    g.add_argument("--time-budget-s", type=float, default=None, help="no new set starts after this much generation")
    g.add_argument("--max-prompt", type=int, default=2048, help="training MAX_PROMPT (left truncation)")
    g.add_argument("--util", type=float, default=0.65, help="gpu_memory_utilization (r1_7b ROLLOUT_GPU_UTIL)")
    g.add_argument("--max-num-seqs", type=int, default=256, help="verl rollout.max_num_seqs")
    g.add_argument("--max-batched-tokens", type=int, default=8192, help="verl rollout.max_num_batched_tokens")
    g.add_argument("--seed", type=int, default=42, help="engine seed (verl: rollout.seed + replica rank)")
    g.add_argument("--set-seed", type=int, default=20261002)
    g.add_argument("--enforce-eager", action="store_true")
    g.add_argument("--async-scheduling", type=int, default=None, choices=[0, 1], help="default: vLLM's choice")
    g.add_argument("--ignore-eos", action="store_true", help="fixed-length workload (diagnostic, not the training one)")
    g.add_argument("--model", default=None)
    g.add_argument("--tokenizer", default=None, help="prompt rendering only (default: --model)")
    g.add_argument("--drafter", default=DRAFTER)
    g.add_argument("--data", default=DATA)
    g.add_argument("--dry-run", action="store_true", help="print the resolved plan (engine kwargs, schedule) and exit")
    s = sub.add_parser("score")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--ref", required=True, choices=["hf", "vllm"])
    s.add_argument("--per-c", type=int, default=0, help="sequences per concurrency (0 = profile default)")
    s.add_argument("--labels", default="", help="comma list of config labels (default: every gen file in out-dir)")
    s.add_argument("--attn", default="flash_attention_2", help="HF attn_implementation (verl actor default)")
    s.add_argument("--chunk", type=int, default=4096, help="lm_head rows per chunk")
    s.add_argument("--vllm-util", type=float, default=0.6)
    s.add_argument("--model", default=None)
    s.add_argument("--device", default="cuda", help="hf scorer device (cpu: tests only)")
    r = sub.add_parser("report")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--bar", type=float, default=None,
                   help="the admissibility bar in nats/token (the 95%% CI upper bound of k1 must not exceed it); "
                        "this or --dtol-source is required. Every other decision rule is G0_RULES")
    r.add_argument("--dtol-source", default=None,
                   help="tolerance JSON whose top-level positive 'D_tol' (nats/token) is the admissibility bar; "
                        "this or --bar is required")
    r.add_argument("--in-run-kl", type=float, default=None,
                   help="rollout_corr/kl logged by a training run with the bf16 rollout engine (SPEC=eagle3 k=3, 16 "
                        "seqs/replica): the "
                        "stand-in check needs bf16_e3's hf k1 at C=16 within G0_RULES['standin']['band_x'] x this; "
                        "without it that check is skipped")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    return {"gen": cmd_gen, "score": cmd_score, "report": cmd_report, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
