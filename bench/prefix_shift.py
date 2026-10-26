#!/usr/bin/env python3
"""Per-token mismatch against the shift in the visited partial responses (prefix shift), measured at a fixed student
checkpoint. One GPU, no training.

Notation. At a prefix h: p = the trainer's student (bf16 HF forward), p_T = the teacher, q = the sampler,
A = log p_T - log p. For tokens drawn from pi in {p, q}, the expected logit-space update is
    G_pi(h) = pi * A - (pi . A) p                           (gap_probe.logit_fields "P" for pi = p; "P" + "E" for q)
and the token error is E = G_q - G_p (gap_probe's e_opd). gap_probe writes these with the ascent sign; the
loss-gradient sign flips every field together, so every norm, inner product, cosine and share is the same.
In parameter space g_pi(h) = J_h^T G_pi(h) (probe0_lib.pullback_trace). With d_s the token-weighted distribution of
prefixes visited by sampler s, the four cells are C(s, pi) = E_{h ~ d_s}[g_pi(h)], and
    Delta = C(q, q) - C(std, p) = T_q + S_p,   T_q = C(q, q) - C(q, p)  (token mismatch on the sampler's prefixes),
                                               S_p = C(q, p) - C(std, p)  (prefix shift; it needs no q at all),
the other order Delta = T_p + S_q, and the placebo S_0 = C(std', p) - C(std, p) from a second standard-engine set.

Sets (the 64 held-out prompts of gap_probe's prefix set x 4 responses; the smoke: 4 x 2):
    P       the standard bf16 engine's responses (theta0: the saved held-out samples j 0-3; theta200: generated)
    P2      a second, independent standard set (j 4-7): the placebo, true shift 0
    Qn015   weight noise sigma 0.15      Qn0325  weight noise sigma 0.325      (two noise levels)
    Qf8s0   vLLM FP8 KV cache, first layer kept in bf16      Qf8all  FP8 KV cache on every layer      (two FP8 caches)
Every sampler's response (k, j) uses the vLLM seed 100,000 + 16 k + j, the same as P's (paired by random numbers).

Subcommands (one process each; bench/prefix_shift.sbatch runs them from the repository root):
    plan      prompts, response slots, seeds; at theta0 also P and P2 from the saved held-out samples (CPU). Inputs:
              --gap-dir (gap_probe's main output dir), --ho-dir (theta0: ho_eval's samples of the base student),
              --ckpt-path (theta200: the trained checkpoint's HF dir), --expected-step0 (optional); plan.json records
              the gap dir and the expected step-0 record for the later phases
    gen       one vLLM engine: std (theta200: P, P2; then the noise reloads Qn015, Qn0325) | f8all | f8s0
    fp8score  full-vocabulary vLLM log q at 16 cut rows per response: f8all and f8s0 on P's cuts and their own Q's,
              the bf16-cache engine (floor) on P's cuts; prefix caching on, then the check against caching off (its
              threshold is the bf16 engine floor in the gap dir's fp8_base_auto.json; without that file the check is
              skipped and the sets are scored again without caching)
    hf        HF bf16 student (gradient checkpointing) and teacher on the GPU; the noisy samplers' final hidden states
              cached on the host; per response one forward with gradients and one body backward per field into
              CountSketch accumulators (grad_sketch.py); per-row logit measures; the Gram matrix of the sketches
    repro     the exact gradient probe's (tools/probe0) reproduction at sigma 0.15 on its 64 part-(a) traces (sdpa,
              float64), through the sketch; compared with that probe's probe0.json when --reference is given
    gapcheck  CPU, saved data: the per-row teacher gap on gap_probe's prefixes against gap_probe.logit_measures, bit for
              bit, and its mean against --reference when given
    selfcheck CPU: closed forms against autograd
Loaded by path (none of them is edited; this file imports them): gap_probe.py, probe0_lib.py, tim_calibrate.py, the
training noise code verl/verl/utils/specopd_tim.py, and g0_kernels.py (for its FA3 skipped-layer patch).
The engine that keeps the first layer's cache in bf16 (f8s0, in gen and in fp8score) gets that patch right after the
engine is built, as the training rollout does with SPECOPD_FA3_KVSKIP_PATCH=1: stock vLLM 0.19.1 fails with
"scheduler_metadata must have shape (metadata_size)" on batches over 512 tokens there. The patch is bit-for-bit stock
wherever stock runs; its counters are written into the set and score records.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GP = _load("ps_gap_probe", os.path.join(HERE, "gap_probe.py"))
_MODS = {}


def PL():
    if "PL" not in _MODS:
        _MODS["PL"] = GP.load_probe_lib()
    return _MODS["PL"]


def TIM():
    if "TIM" not in _MODS:
        _MODS["TIM"] = GP.load_tim()
    return _MODS["TIM"]


def TC():
    if "TC" not in _MODS:
        _MODS["TC"] = _load("ps_tim_calibrate", os.path.join(HERE, "tim_calibrate.py"))
    return _MODS["TC"]


def GS():
    if "GS" not in _MODS:
        _MODS["GS"] = _load("ps_grad_sketch", os.path.join(HERE, "grad_sketch.py"))
    return _MODS["GS"]


def G0():
    """bench/g0_kernels.py (stdlib at import), for its FA3 skipped-layer patch only."""
    if "G0" not in _MODS:
        _MODS["G0"] = _load("ps_g0_kernels", os.path.join(HERE, "g0_kernels.py"))
    return _MODS["G0"]


# ------------------------------------------------------------------------------------------------------ constants
CK_LABEL = {"theta0": "base", "theta200": "s200"}            # -> gap_probe.CKPT_STEPS
SIGMAS = (("n015", 0.15), ("n0325", 0.325))                  # the two noise levels
SIGMA = dict(SIGMAS)
SAMPLERS = ("n015", "n0325", "f8all", "f8s0")
SAMPLER_SET = {"n015": "Qn015", "n0325": "Qn0325", "f8all": "Qf8all", "f8s0": "Qf8s0"}
FAMILIES = {"noise": dict(smaller="n015", larger="n0325"),   # per family: the smaller and the larger perturbation
            "fp8": dict(smaller="f8s0", larger="f8all")}
SETS = ("P", "P2", "Qn015", "Qn0325", "Qf8all", "Qf8s0")
SCORE_ENGINES = {"f8all": ("fp8", ()), "f8s0": ("fp8", ("0",)), "floor": ("auto", ())}
SCORE_TARGETS = {"f8all": ("P", "Qf8all"), "f8s0": ("P", "Qf8s0"), "floor": ("P",)}
GEN_SETS = {"std": ("P", "P2", "Qn015", "Qn0325"), "f8all": ("Qf8all",), "f8s0": ("Qf8s0",)}
SEED_BASE, SEED_STRIDE = 100000, 16                          # ho_eval.py: seed of (k, j) = 100,000 + 16 k + j
CUT_SEED, K_CUTS = 20261016, 16                              # cut rows: default_rng([20261016, k, j]), 16 per response
MODES = {
    "main": dict(P=64, n=4, max_tokens=16384, n_check=64, gen_min_s=1800, score_min_s=240, prompt_reserve_s=420),
    "smoke": dict(P=4, n=2, max_tokens=1024, n_check=64, gen_min_s=120, score_min_s=60, prompt_reserve_s=120),
}
FORMAT = "prefix_shift_plan/v1"
FLOOR_JSON = "fp8_base_auto.json"   # in the gap dir (gap_probe fp8 --ckpt base --kv auto): the bf16 engine floor
REPRO_SIGMA = 0.15
REPRO_DIGITS = dict(kl_qp=3, Abar_q=4, gap=4)                                          # within half a unit of the decimal
REPRO_REL = 0.005                                                                      # r_bias and cos within 0.5%
EXACT_PAIRS = ("P:Gp", "P:E_n015", "Qn015:Gp", "Qn015:E_n015")                         # the smoke's exact check
EXACT_BOUND = 5.0                                                                      # |dG| <= 5 |x| |y| / sqrt(k)
EXACT_R_REL = 0.01                                                                     # r_T and r_S within 1%
MEASURE_KEYS = ("kl_qp", "kl_pq", "gap", "kl_q_tau", "Abar_q", "Abar_p", "dA", "e_L1", "e_L2", "gp_L2", "gq_L2",
                "drift_L2", "esc_L2", "eis_L2", "d_L1", "d_L2", "H_p", "q_sum", "is_cap_share", "meanabsA_q",
                "maxabsA_q1e-08", "sum_absd_absA")
SUB = 128                                                                              # rows per float64 sub-chunk
HIDDEN = 3584
HOST_BUDGET_GB = 215.0                                                                 # of the job's 240 GB


def log(*a):
    print(f"[ps {time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s]", *a, flush=True)


def ho_seed(k, j):
    return SEED_BASE + SEED_STRIDE * int(k) + int(j)


def sampling_kwargs(seed, max_tokens):
    """ho_eval.sampling_kwargs: T 1, top_p 1, top_k -1, no repetition penalty, logprobs=0 (the sampled token's log q)."""
    return dict(temperature=1.0, top_p=1.0, top_k=-1, repetition_penalty=1.0, max_tokens=int(max_tokens),
                seed=int(seed), logprobs=0)


def draw_cuts_kj(T, K, k, j, seed=CUT_SEED):
    """gap_probe.draw_cuts with the response index in the seed: min(K, T) distinct rows of [0, T), uniform, in draw
    order (row t = the prefix prompt + response[:t], whose next token is response[t])."""
    import numpy as np
    if T <= 0:
        return []
    return [int(x) for x in np.random.default_rng([int(seed), int(k), int(j)]).choice(T, size=min(K, T), replace=False)]


def set_js(name, n):
    return list(range(4, 4 + n)) if name == "P2" else list(range(n))


def half_of(ridx, n):
    return 0 if ridx < n // 2 else 1


# ------------------------------------------------------------------------------------------------- files: plan, sets
def write_jsonl(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def load_plan(D):
    rec = GP.read_json(os.path.join(D, "plan_record.json"))
    path = os.path.join(D, "plan.json")
    if _sha256(path) != rec["sha256"]:
        raise SystemExit(f"[ps] REFUSE: {path} sha256 differs from its record")
    plan = GP.read_json(path)
    if plan.get("format") != FORMAT:
        raise SystemExit(f"[ps] REFUSE: plan format {plan.get('format')!r}")
    return plan


def set_paths(D, name):
    return os.path.join(D, "sets", f"{name}.jsonl"), os.path.join(D, "sets", f"{name}.json")


def write_set(D, name, rows, extra):
    import numpy as np
    pj, pr = set_paths(D, name)
    os.makedirs(os.path.dirname(pj), exist_ok=True)
    write_jsonl(pj, rows)
    lens = np.array([r["T"] for r in rows], dtype=np.int64)
    rec = dict(set=name, sha256=_sha256(pj), n=len(rows), n_tokens=int(lens.sum()), mean_len=float(lens.mean()),
               median_len=float(np.median(lens)), max_len=int(lens.max()),
               frac_length=float(np.mean([r["finish_reason"] == "length" for r in rows])),
               created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
    GP.write_json(pr, rec)
    return rec


def set_done(D, name):
    pj, pr = set_paths(D, name)
    if not (os.path.exists(pj) and os.path.exists(pr)):
        return False
    return GP.read_json(pr).get("sha256") == _sha256(pj)


def load_set(D, name, plan=None):
    pj, pr = set_paths(D, name)
    rec = GP.read_json(pr)
    if _sha256(pj) != rec["sha256"]:
        raise SystemExit(f"[ps] REFUSE: {pj} sha256 differs from its record")
    rows = [json.loads(line) for line in open(pj)]
    if plan is not None:
        js = plan["sets"][name]
        want = [(pr_["i"], j) for pr_ in plan["prompts"] for j in js]
        if [(r["i"], r["j"]) for r in rows] != want:
            raise SystemExit(f"[ps] REFUSE: {name}: response slots differ from plan.json")
    return rows, rec


def cut_rows(rows):
    """Canonical cut-row order of a set: responses in set order, each response's cuts in draw order.
    Returns [(response index, t, draw index)]."""
    return [(ri, t, d) for ri, r in enumerate(rows) for d, t in enumerate(r["cuts"])]


def check_rows(rows, n_check, n):
    """The prefix-caching check rows: P's cuts ordered by (draw index, response slot, prompt), the first n_check."""
    cand = sorted((d, ri % n, ri // n, ri, t) for ri, t, d in cut_rows(rows))
    return [(ri, t) for _, _, _, ri, t in cand[:n_check]]


def load_scales(gap_dir, sha256=None):
    """gap_probe measure's scales_base.json in gap_dir: the base student's frozen per-tensor noise scales. With sha256
    (plan.json's record of the file), the file must be unchanged since plan.json was written."""
    path = os.path.join(gap_dir, "scales_base.json")
    if sha256 is not None and _sha256(path) != sha256:
        raise SystemExit(f"[ps] REFUSE: {path} changed since plan.json was written")
    obj = GP.read_json(path)
    if obj.get("seed") != GP.NOISE_SEED or obj.get("source") != GP.STUDENT:
        raise SystemExit(f"[ps] REFUSE: {path} is not the base student's seed-{GP.NOISE_SEED} scales")
    return {k: float(v) for k, v in obj["scales"].items()}


def plan_scales(plan):
    return load_scales(plan["gap_dir"], plan["scales_sha256"])


def noise_record_check(rec, label, E):
    """E: the expected step-0 record (--expected-step0: the training log's step-0 [SPECOPD-TIM] line), or None: the
    check is skipped. theta0: the record equals E (2e-6 relative); theta200: the frozen base scales give the base
    nnorm1, and the target set is the same."""
    if E is None:
        return dict(ok=True, skipped="no --expected-step0")
    ok_counts = rec.get("n_tensors") == E["n_tensors"] and rec.get("n_elems") == E["n_elems"]
    keys = ("d_eps", "d_n", "nnorm1") if label == "base" else ("nnorm1",)
    rel = {k: abs(rec[k] - E[k]) / abs(E[k]) for k in keys}
    out = dict(ok=bool(ok_counts and all(v <= 2e-6 for v in rel.values())), rel=rel, counts_equal=bool(ok_counts),
               compared=list(keys))
    if label != "base":
        out["d_n_minus_base"] = rec["d_n"] - E["d_n"]
    return out


# ---------------------------------------------------------------------------------------------------------- plan
def ho_rows(ho_dir, sel, js):
    """The saved held-out samples of the base student (ho_eval's samples.jsonl and meta.json in ho_dir; bf16, theta0)
    for prompts `sel` and slots `js`."""
    samples, meta_p = os.path.join(ho_dir, "samples.jsonl"), os.path.join(ho_dir, "meta.json")
    meta = GP.read_json(meta_p)
    ck = meta["config_key"]
    want = dict(seed_base=SEED_BASE, seed_stride=SEED_STRIDE, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=16384,
                logprobs=0, spec="none", dtype="bfloat16", tokids_sha256=GP.HO_TOKIDS_SHA256,
                prompt_sha256=GP.HO_DAPO_SHA256, ckpt=GP.STUDENT)
    bad = {k: ck.get(k) for k, v in want.items() if ck.get(k) != v}
    if bad:
        raise SystemExit(f"[ps] REFUSE: {meta_p}: config differs: {bad}")
    need = {(int(k), int(j)) for k in sel for j in js}
    got = {}
    with open(samples) as f:
        for line in f:
            r = json.loads(line)
            key = (int(r["k"]), int(r["j"]))
            if key in need:
                got[key] = r
    missing = sorted(need - set(got))
    if missing:
        raise SystemExit(f"[ps] REFUSE: {samples} lacks {missing[:4]}")
    for (k, j), r in got.items():
        if r["seed"] != ho_seed(k, j) or r["n_tok"] != len(r["response_ids"]):
            raise SystemExit(f"[ps] REFUSE: {samples} row {(k, j)}: seed {r['seed']} / length mismatch")
    return got, meta


def need_ho_dir(a):
    if not a.ho_dir:
        raise SystemExit("[ps] REFUSE: plan --ckpt theta0 needs --ho-dir (ho_eval's held-out samples of the base student)")
    return a.ho_dir


def cmd_plan(a):
    cfg = MODES[a.mode]
    D = a.out_dir
    os.makedirs(os.path.join(D, "sets"), exist_ok=True)
    if os.path.exists(os.path.join(D, "plan_record.json")):
        plan = load_plan(D)
        log(f"plan exists ({plan['mode']}, {plan['ckpt']})")
        if plan["mode"] != a.mode or plan["ckpt"] != a.ckpt:
            raise SystemExit("[ps] REFUSE: plan.json in D is for another mode / checkpoint")
    else:
        from transformers import AutoTokenizer
        label = CK_LABEL[a.ckpt]
        step = GP.CKPT_STEPS[label]
        if not a.gap_dir:
            raise SystemExit("[ps] REFUSE: plan needs --gap-dir (gap_probe's main output dir)")
        if label == "base":
            path = GP.STUDENT
        elif a.ckpt_path:
            path = a.ckpt_path
        else:
            raise SystemExit(f"[ps] REFUSE: plan --ckpt {a.ckpt} needs --ckpt-path (the step-{step} checkpoint's HF dir)")
        tok = AutoTokenizer.from_pretrained(GP.STUDENT)
        ids_all = GP.render_prompt_ids(tok, GP.load_ho_chats())
        if len(ids_all) != GP.N_HO or GP.tokids_sha256(ids_all) != GP.HO_TOKIDS_SHA256:
            raise SystemExit("[ps] REFUSE: HO-DAPO token ids differ from the expected token-id hash")
        sel64 = GP.select_prompts(GP.N_HO, 64)
        gap_prompts, gap_rec = GP.load_prefixes(a.gap_dir)
        gap_k = [int(p["k"]) for p in gap_prompts]
        if sel64 != gap_k:
            raise SystemExit("[ps] REFUSE: the 64 selected prompts differ from gap_probe's k list")
        sel = sel64[:cfg["P"]]
        load_scales(a.gap_dir)
        expected = GP.load_expected_step0(a.expected_step0, "[ps]")
        plan = dict(format=FORMAT, mode=a.mode, ckpt=a.ckpt, label=label, step=step, path=path,
                    student=GP.STUDENT, teacher=GP.TEACHER, n_resp=cfg["n"], max_tokens=cfg["max_tokens"],
                    K=K_CUTS, cut_seed=CUT_SEED, seed_base=SEED_BASE, seed_stride=SEED_STRIDE,
                    sets={s: set_js(s, cfg["n"]) for s in SETS}, sigmas=dict(SIGMAS), noise_seed=GP.NOISE_SEED,
                    gap_dir=os.path.abspath(a.gap_dir), expected_step0=expected,
                    scales_sha256=_sha256(os.path.join(a.gap_dir, "scales_base.json")),
                    fp8=dict(f8all=dict(kv_cache_dtype="fp8", skip=[]), f8s0=dict(kv_cache_dtype="fp8", skip=["0"])),
                    sampling=sampling_kwargs(0, cfg["max_tokens"]), gap_prefixes_sha256=gap_rec["sha256"],
                    ho_dapo_sha256=GP.HO_DAPO_SHA256, ho_tokids_sha256=GP.HO_TOKIDS_SHA256,
                    prompts=[dict(i=i, k=k, prompt_ids=ids_all[k]) for i, k in enumerate(sel)])
        GP.write_json(os.path.join(D, "plan.json"), plan, canonical=True)
        GP.write_json(os.path.join(D, "plan_record.json"), dict(sha256=_sha256(os.path.join(D, "plan.json")),
                                                                created=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
        log(f"plan written: {a.mode} {a.ckpt} ({label}, {path}); {len(sel)} prompts, {cfg['n']} responses per set")
    if plan["ckpt"] == "theta0":
        todo = [s for s in ("P", "P2") if not set_done(D, s)]
        if todo:
            sel = [p["k"] for p in plan["prompts"]]
            got, meta = ho_rows(need_ho_dir(a), sel, sorted({j for s in todo for j in plan["sets"][s]}))
            cap = plan["max_tokens"]
            for s in todo:
                rows = []
                for pr in plan["prompts"]:
                    for j in plan["sets"][s]:
                        r = got[(pr["k"], j)]
                        ids = [int(x) for x in r["response_ids"]]
                        cut = len(ids) > cap
                        ids = ids[:cap]
                        rows.append(dict(i=pr["i"], k=pr["k"], j=j, seed=ho_seed(pr["k"], j), T=len(ids),
                                         response_ids=ids, lq=None, nll_sum=None if cut else float(r["nll_sum"]),
                                         finish_reason="length" if cut else r["finish_reason"],
                                         cuts=draw_cuts_kj(len(ids), plan["K"], pr["k"], j)))
                rec = write_set(D, s, rows, dict(source=f"ho_eval held-out samples ({a.ho_dir})", vllm=meta.get("vllm"),
                                                 truncated_to=cap if cap < 16384 else None))
                log(f"{s}: {rec['n']} responses from the saved held-out samples, {rec['n_tokens']} tokens")
        mp = os.path.join(D, "sets", "MEMPROBE.jsonl")
        if plan["mode"] == "smoke" and not os.path.exists(mp):
            # the smoke's peak-memory probe: the longest prompt + response of the MAIN P set, at full length
            sel64 = GP.select_prompts(GP.N_HO, 64)
            got, _ = ho_rows(need_ho_dir(a), sel64, set_js("P", MODES["main"]["n"]))
            from transformers import AutoTokenizer
            ids_all = GP.render_prompt_ids(AutoTokenizer.from_pretrained(GP.STUDENT), GP.load_ho_chats())
            (k, j), r = max(got.items(), key=lambda kv: len(ids_all[kv[0][0]]) + len(kv[1]["response_ids"]))
            write_jsonl(mp, [dict(k=k, j=j, prompt_ids=ids_all[k], response_ids=[int(x) for x in r["response_ids"]],
                                  T=len(r["response_ids"]), main_P_tokens=sum(len(x["response_ids"]) for x in got.values()))])
            log(f"MEMPROBE: k {k} j {j}, {len(ids_all[k])} + {len(r['response_ids'])} tokens")
    return 0


# ------------------------------------------------------------------------------------------------- vLLM engines
def gen_engine_kwargs(path, kv="auto", skip=()):
    """tim_calibrate.build_llm(mode='q'): the rollout engine as verl builds it (verl's worker extension, CUDA graphs
    FULL_AND_PIECEWISE), with the checkpoint as a parameter and the base student's tokenizer (as gap_probe.cmd_fp8)."""
    kw = dict(model=path, tokenizer=GP.STUDENT, dtype="bfloat16", gpu_memory_utilization=0.75,
              max_model_len=18433 + 1024, enable_prefix_caching=False, enforce_eager=False, seed=0, max_num_seqs=64,
              max_num_batched_tokens=4096, load_format="auto", worker_extension_cls=TC().VERL_EXT,
              compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"})
    if kv != "auto":
        kw["kv_cache_dtype"] = kv
    if skip:
        kw["kv_cache_dtype_skip_layers"] = [str(x) for x in skip]
    return kw


def score_engine_kwargs(path, kv, skip, cache, maxlen):
    """gap_probe.cmd_fp8's scoring engine (full-vocabulary log-probs), with prefix caching as a parameter."""
    kw = dict(model=path, tokenizer=GP.STUDENT, dtype="bfloat16", kv_cache_dtype=kv, gpu_memory_utilization=0.5,
              max_model_len=int(maxlen) + 8, enable_prefix_caching=bool(cache), seed=0, max_logprobs=-1,
              max_num_seqs=32, max_num_batched_tokens=8192, compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"})
    if skip:
        kw["kv_cache_dtype_skip_layers"] = [str(x) for x in skip]
    return kw


SCORE_SAMPLING = dict(max_tokens=1, temperature=1.0, top_p=1.0, top_k=-1, logprobs=-1, detokenize=False,
                      flat_logprobs=True, seed=0)


def _layer_audit_worker(worker):
    """Inside the in-process vLLM worker: per attention layer its name, cache dtype, cache tensor dtype, backend and
    k / v scales (gap_probe._audit_worker, kept per layer so that the first layer can be checked)."""
    out = {}
    try:
        from vllm.model_executor.layers.attention.attention import Attention
        mr = worker.model_runner
        cc = mr.vllm_config.cache_config
        out.update(cache_dtype=str(cc.cache_dtype), skip_layers=[str(x) for x in (getattr(cc, "kv_cache_dtype_skip_layers",
                                                                                          None) or [])])
        layers = []
        for name, m in mr.get_model().named_modules():
            if isinstance(m, Attention):
                kv = getattr(m, "kv_cache", None)
                t = kv[0] if isinstance(kv, (list, tuple)) and kv else kv
                be = getattr(m, "attn_backend", None)
                layers.append(dict(name=name, kv_cache_dtype=str(m.kv_cache_dtype), tensor_dtype=str(getattr(t, "dtype", None)),
                                   backend=be.get_name() if hasattr(be, "get_name") else str(be),
                                   k_scale=getattr(m, "_k_scale_float", None), v_scale=getattr(m, "_v_scale_float", None)))
        out["layers"] = layers
    except Exception as e:  # noqa: BLE001 (the check below refuses an audit with an error)
        out["error"] = repr(e)
    return out


def audit_ok(audit, kind, n_layers=28):
    """kind std / floor: every layer bf16 ('auto'); f8all: every layer FP8; f8s0: layer 0 bf16, the others FP8. Always:
    FLASH_ATTN and k / v scales 1.0 (uncalibrated, as the training rollout). Returns (ok, reason)."""
    if not audit or "error" in audit:
        return False, f"audit failed: {audit}"
    L = audit.get("layers") or []
    if len(L) != n_layers:
        return False, f"{len(L)} attention layers, expected {n_layers}"
    for x in L:
        m = re.search(r"layers\.(\d+)\.", x["name"])
        idx = int(m.group(1)) if m else -1
        if x["backend"] != "FLASH_ATTN":
            return False, f"{x['name']}: backend {x['backend']}"
        if x["k_scale"] != 1.0 or x["v_scale"] != 1.0:
            return False, f"{x['name']}: scales {x['k_scale']} / {x['v_scale']}"
        fp8 = x["kv_cache_dtype"].startswith("fp8")
        want_fp8 = kind == "f8all" or (kind == "f8s0" and idx != 0)
        if fp8 != want_fp8:
            return False, f"{x['name']} (layer {idx}): cache {x['kv_cache_dtype']} / {x['tensor_dtype']}, {kind} wants " \
                          f"{'FP8' if want_fp8 else 'bf16'}"
        if not want_fp8 and "bfloat16" not in x["tensor_dtype"]:
            return False, f"{x['name']}: cache tensor {x['tensor_dtype']}, expected bfloat16"
    return True, "ok"


def kvskip_patch(llm, skip, where):
    """g0_kernels' FA3 skipped-layer patch in the in-process engine worker, for an engine with
    kv_cache_dtype_skip_layers (f8s0). Stock vLLM 0.19.1 passes the global FP8 qkv_dtype to the bf16 layer's FA3
    scheduler metadata and fails with "scheduler_metadata must have shape (metadata_size)" on the path outside CUDA
    graphs (batches over 512 tokens). The patch is bit-for-bit stock wherever stock runs. Installing is
    idempotent, so a second call returns the cumulative counters (installed / checked / fallback / first_fallback).
    No skip list: nothing is called and None is returned. Refuses unless the worker reports installed True."""
    if not skip:
        return None
    st = llm.collective_rpc(G0()._rpc_fa3_kvskip_patch)
    st = st[0] if isinstance(st, list) else st
    if not isinstance(st, dict) or st.get("installed") is not True:
        raise SystemExit(f"[ps] REFUSE: {where}: the FA3 skipped-layer patch is not installed: {st}")
    return {k: st.get(k) for k in ("patch", "installed", "checked", "fallback", "first_fallback")}


def stream_noisy(path, sigma, seed, scales, tim, dev, sink):
    """The training noise on a trained checkpoint, streamed into an engine tensor by tensor: every tensor of the
    checkpoint's safetensors (fp32 master weights) is read on `dev`, passed through SamplerNoise(sigma, seed).apply with
    the per-tensor scales pre-filled (frozen at the base student's first sync, as training freezes them) and handed to
    sink([(name, tensor)]) (vLLM's load_weights, which casts to bf16). Returns the finish_sync record."""
    from safetensors import safe_open
    sn = tim.SamplerNoise(sigma, seed)
    sn.scales.update(scales)
    fmap = GP.safetensors_map(path)
    by_file = {}
    for n, f in fmap.items():
        by_file.setdefault(f, []).append(n)
    for f in sorted(by_file):
        with safe_open(f, framework="pt", device=str(dev)) as fh:
            for n in fh.keys():
                t = fh.get_tensor(n)
                if tim.is_target(n, t) and n not in scales:
                    raise SystemExit(f"[ps] REFUSE: no frozen scale for noise target {n}")
                sink(sn.apply([(n, t)]))
                del t
    rec = sn.finish_sync()
    return {k: v for k, v in rec.items() if k != "ms"}


def _reload_scaled_worker(worker, path, sigma, seed, scales):
    """Inside the vLLM worker (theta200): stream_noisy into model.load_weights, then process_weights_after_loading, the
    calls update_weights_from_ipc makes (tim_calibrate._reload_worker's order)."""
    import torch
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    runner = worker.model_runner
    model = runner.model
    dev = torch.device("cuda", torch.cuda.current_device())
    rec = stream_noisy(path, sigma, seed, scales, TIM(), dev, model.load_weights)
    process_weights_after_loading(model, runner.vllm_config.model_config, dev)
    torch.cuda.synchronize()
    return rec


def reload_noise(llm, plan, sigma):
    t = time.time()
    if plan["label"] == "base":     # theta0: tim_calibrate._reload_worker as is (base shards, scales of this first sync)
        rec = llm.collective_rpc(TC()._reload_worker, args=(float(sigma), GP.NOISE_SEED, False))[0]
        rec = {k: v for k, v in rec.items() if k != "ms"}
    else:
        rec = llm.collective_rpc(_reload_scaled_worker, args=(plan["path"], float(sigma), GP.NOISE_SEED,
                                                              plan_scales(plan)))[0]
    chk = noise_record_check(rec, plan["label"], plan.get("expected_step0"))
    log(f"noise sigma {sigma}: {rec} check {chk} ({time.time() - t:.0f}s)")
    if not chk["ok"]:
        raise SystemExit(f"[ps] REFUSE: the vLLM noise record at sigma {sigma} differs from the expected step-0 "
                         f"record: {chk}")
    return dict(sigma=sigma, record=rec, check=chk, s=round(time.time() - t, 1))


def generate_set(llm, plan, name):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    reqs, sps, meta = [], [], []
    for pr in plan["prompts"]:
        for j in plan["sets"][name]:
            sd = ho_seed(pr["k"], j)
            reqs.append(TokensPrompt(prompt_token_ids=pr["prompt_ids"]))
            sps.append(SamplingParams(**sampling_kwargs(sd, plan["max_tokens"])))
            meta.append((pr, j, sd))
    t = time.time()
    outs = llm.generate(reqs, sps, use_tqdm=False)
    dt = time.time() - t
    rows = []
    for (pr, j, sd), o in zip(meta, outs):
        c = o.outputs[0]
        ids = [int(x) for x in c.token_ids]
        lps = c.logprobs or []
        if len(lps) != len(ids):
            raise SystemExit(f"[ps] {name}: logprobs for {len(lps)} of {len(ids)} tokens")
        lq = [float(lps[n][ids[n]].logprob) for n in range(len(ids))]
        rows.append(dict(i=pr["i"], k=pr["k"], j=j, seed=sd, T=len(ids), response_ids=ids, lq=lq, nll_sum=None,
                         finish_reason=c.finish_reason, cuts=draw_cuts_kj(len(ids), plan["K"], pr["k"], j)))
    return rows, dt


def decode_prefill_reference(llm, plan, rows):
    """Within one generation engine (the sigma 0.15 noisy standard engine, set Qn015): the engine teacher-forces its own
    sampled responses (tim_calibrate.teacher_force, prompt_logprobs=0) and |decode log q(y) - prefill log q(y)| is
    averaged over the cut rows (and over all rows). The FP8 check's reference at theta0 only, as a documented stand-in:
    theta0's P comes from the saved held-out samples, which keep nll_sum but no per-token decode log q. At theta200 the
    reference is P's decode log q against the bf16-cache floor scoring engine at P's cuts (fp8score floor), the same
    two engines' structure as the FP8 side; this value is then recorded beside it only."""
    import numpy as np
    keep = [r for r in rows if r["T"] > 0]
    tr = [dict(prompt_ids=plan["prompts"][r["i"]]["prompt_ids"], response_ids=r["response_ids"]) for r in keep]
    t = time.time()
    tf = TC().teacher_force(llm, tr)
    cut, allr = [], []
    for r, lp in zip(keep, tf):
        d = np.asarray(r["lq"], np.float64) - np.asarray(lp, np.float64)
        allr.append(d)
        cut.extend(float(d[t_]) for t_ in r["cuts"])
    cut = np.asarray(cut)
    alla = np.concatenate(allr)
    return dict(n_cut=int(cut.size), mean_abs_cut=float(np.abs(cut).mean()), mean_cut=float(cut.mean()),
                n_all=int(alla.size), mean_abs_all=float(np.abs(alla).mean()), mean_all=float(alla.mean()),
                s=round(time.time() - t, 1))


def cmd_gen(a):
    cfg = MODES[a.mode]
    D = a.out_dir
    plan = load_plan(D)
    eng = a.engine
    names = [s for s in GEN_SETS[eng] if not (plan["ckpt"] == "theta0" and s in ("P", "P2"))]
    todo = [s for s in names if not set_done(D, s)]
    if not todo:
        log(f"gen {eng}: every set exists")
        return 0
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import vllm
    from vllm import LLM
    kv, skip = ("auto", ()) if eng == "std" else SCORE_ENGINES[eng]
    kw = gen_engine_kwargs(plan["path"], kv, skip)
    t0 = time.time()
    llm = LLM(**kw)
    t_engine = time.time() - t0
    fa3 = kvskip_patch(llm, skip, f"gen {eng}")
    audit = llm.collective_rpc(_layer_audit_worker)
    audit = audit[0] if isinstance(audit, list) else audit
    ok, why = audit_ok(audit, eng)
    log(f"gen {eng}: engine up in {t_engine:.0f}s; audit {why}" + (f"; FA3 skipped-layer patch {fa3}" if fa3 else ""))
    if not ok:
        raise SystemExit(f"[ps] REFUSE: gen {eng}: engine audit: {why}")
    rc = 0
    for s in todo:
        left = (a.deadline - time.time()) if a.deadline else float("inf")
        if left < cfg["gen_min_s"]:
            log(f"gen {s}: SKIPPED, {left:.0f}s left < {cfg['gen_min_s']}s")
            rc = 4
            break
        noise = reload_noise(llm, plan, SIGMA[s[1:]]) if s in ("Qn015", "Qn0325") else None
        rows, dt = generate_set(llm, plan, s)
        extra = dict(source="generated", engine=eng, engine_kwargs=kw, engine_s=round(t_engine, 1), audit=audit,
                     vllm=vllm.__version__, gen_s=round(dt, 1), noise=noise)
        if fa3 is not None:     # cumulative counters after this set
            extra["fa3_kvskip"] = dict(after_install=fa3, after_set=kvskip_patch(llm, skip, f"gen {eng}"))
            log(f"gen {s}: FA3 skipped-layer patch {extra['fa3_kvskip']['after_set']}")
        if s == "Qn015":
            extra["decode_prefill_ref"] = decode_prefill_reference(llm, plan, rows)
        rec = write_set(D, s, rows, extra)
        log(f"gen {s}: {rec['n']} responses, {rec['n_tokens']} tokens, mean {rec['mean_len']:.0f}, length-capped "
            f"{rec['frac_length']:.3f} ({dt:.0f}s)")
    return rc


# ----------------------------------------------------------------------------------------------- FP8 scoring
def score_prefixes(llm, prefixes, V):
    """Full-vocabulary float64 log q at each prefix (gap_probe.cmd_fp8's request form, renormalised in float64)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    sp = SamplingParams(**SCORE_SAMPLING)
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prefixes], [sp] * len(prefixes), use_tqdm=False)
    res = []
    for o in outs:
        ids, vals = GP.flat_position0(o.outputs[0].logprobs)
        res.append(GP.dense_logprobs(ids, vals, V))       # raises without the full vocabulary (no top-k fill here)
    return res


def score_set_rows(llm, plan, rows, items, out, V, deadline, group=16, prog=None):
    """items: [(row index into `out`, response index, t)]. Prefix-caching friendly order: groups of `group` responses,
    and within a group one round per cut rank (shortest cut first), so each round reuses the previous round's
    prefixes. out[row] = float32 log q. Returns {row: (log q of the realised next token, tail mass, n returned)}."""
    by = {}
    for row, ri, t in items:
        by.setdefault(ri, []).append((t, row))
    for ri in by:
        by[ri].sort()
    done = {} if prog is None else prog
    order = sorted(by)
    for g0 in range(0, len(order), group):
        grp = [ri for ri in order[g0:g0 + group] if any(row not in done for _, row in by[ri])]
        if not grp:
            continue
        if deadline and time.time() > deadline:
            return done, False
        for d in range(max(len(by[ri]) for ri in grp)):
            batch = [(ri,) + by[ri][d] for ri in grp if d < len(by[ri]) and by[ri][d][1] not in done]
            if not batch:
                continue
            pre = [plan["prompts"][rows[ri]["i"]]["prompt_ids"] + rows[ri]["response_ids"][:t] for ri, t, _ in batch]
            for (ri, t, row), (lq, info) in zip(batch, score_prefixes(llm, pre, V)):
                out[row] = lq.float().numpy()
                done[row] = (float(lq[rows[ri]["response_ids"][t]]), float(info["tail_mass"]), int(info["n_returned"]))
        if hasattr(out, "flush"):
            out.flush()
    return done, True


def score_paths(D, eng, s):
    return os.path.join(D, f"score_{eng}_{s}.npy"), os.path.join(D, f"score_{eng}_{s}.json")


def engine_floor_kl(plan):
    """The bf16 engine floor: the bf16-cache vLLM engine's mean KL(q || p_HF) at gap_probe's prefixes (mean_kl_qp of
    FLOOR_JSON in the plan's gap dir), or None when that file is absent. The prefix-caching check passes when caching
    changes q by no more than this (mean KL(q_off || q_on) over the check rows)."""
    path = os.path.join(plan["gap_dir"], FLOOR_JSON)
    return float(GP.read_json(path)["mean_kl_qp"]) if os.path.exists(path) else None


def cmd_fp8score(a):
    import numpy as np
    cfg = MODES[a.mode]
    D = a.out_dir
    plan = load_plan(D)
    eng = a.engine
    final = os.path.join(D, f"score_{eng}.json")
    st = GP.read_json(final) if os.path.exists(final) else {}
    if st.get("status") == "complete":
        log(f"fp8score {eng}: complete, skipped")
        return 0
    if a.cache == "on" and st.get("status") == "scored":
        log(f"fp8score {eng}: scored with caching on; the caching-off check is next")
        return 0
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import vllm
    from transformers import AutoConfig
    from vllm import LLM
    V = AutoConfig.from_pretrained(GP.STUDENT).vocab_size
    sets = {s: load_set(D, s, plan)[0] for s in SCORE_TARGETS[eng]}
    kv, skip = SCORE_ENGINES[eng]
    maxlen = max(len(plan["prompts"][r["i"]]["prompt_ids"]) + t for s in sets for r in sets[s] for t in r["cuts"])
    kw = score_engine_kwargs(plan["path"], kv, skip, a.cache == "on", maxlen)
    t0 = time.time()
    llm = LLM(**kw)
    t_engine = time.time() - t0
    fa3 = kvskip_patch(llm, skip, f"fp8score {eng}")
    audit = llm.collective_rpc(_layer_audit_worker)
    audit = audit[0] if isinstance(audit, list) else audit
    ok, why = audit_ok(audit, eng)
    log(f"fp8score {eng} (caching {a.cache}): engine up in {t_engine:.0f}s; audit {why}"
        + (f"; FA3 skipped-layer patch {fa3}" if fa3 else ""))
    if not ok:
        raise SystemExit(f"[ps] REFUSE: fp8score {eng}: engine audit: {why}")
    deadline = a.deadline - cfg["score_min_s"] if a.deadline else 0

    def fa3_now():
        return None if fa3 is None else dict(after_install=fa3, now=kvskip_patch(llm, skip, f"fp8score {eng}"))

    def score_all(caching):
        info = {}
        for s, rows in sets.items():
            npy, js = score_paths(D, eng, s)
            items = [(n, ri, t) for n, (ri, t, _) in enumerate(cut_rows(rows))]
            prev = GP.read_json(js) if os.path.exists(js) else {}
            resume = prev.get("caching") == caching and prev.get("status") == "partial" and os.path.exists(npy)
            mm = np.lib.format.open_memmap(npy, mode="r+" if resume else "w+", dtype=np.float32, shape=(len(items), V))
            prog = {int(k): tuple(v) for k, v in prev.get("rows_done", {}).items()} if resume else {}
            t = time.time()
            done, finished = score_set_rows(llm, plan, rows, items, mm, V, deadline, prog=prog)
            del mm
            rec = dict(set=s, engine=eng, caching=caching, N=len(items), status="complete" if finished else "partial",
                       rows=[[ri, t_] for _, ri, t_ in items], score_s=round(time.time() - t, 1),
                       rows_done={str(k): list(v) for k, v in done.items()}, fa3_kvskip=fa3_now())
            if finished:
                lq_y = np.array([done[n][0] for n in range(len(items))])
                tails = np.array([done[n][1] for n in range(len(items))])
                rec.update(lq_y=lq_y.tolist(), tail_mass_max=float(tails.max()),
                           returned_min=int(min(done[n][2] for n in range(len(items)))))
                # decode against prefill at the cuts: an FP8 engine on its own sampler's set (the FP8 check), and the
                # bf16-cache floor engine on P when P has per-token decode log q (theta200: the standard engine's own
                # generation), which is the FP8 check's reference with the same two engines' structure
                own = s != "P" or eng == "floor"
                if own and all(rows[ri]["lq"] is not None for _, ri, _ in items):
                    dec = np.array([rows[ri]["lq"][t_] for _, ri, t_ in items], np.float64)
                    dd = lq_y - dec
                    rec["decode_prefill"] = dict(n=int(dd.size), mean_abs=float(np.abs(dd).mean()), mean=float(dd.mean()),
                                                 decode="generation engine (set's own decode log q)",
                                                 prefill=f"scoring engine {eng} (caching {caching})")
            GP.write_json(js, rec)
            info[s] = {k: v for k, v in rec.items() if k not in ("rows", "rows_done", "lq_y")}
            log(f"fp8score {eng} {s}: {rec['status']} ({rec['score_s']:.0f}s) {info[s].get('decode_prefill', '')}")
            if not finished:
                return info, False
        return info, True

    base = dict(engine=eng, vllm=vllm.__version__, audit=audit, maxlen=maxlen, engine_s_last=round(t_engine, 1))
    if a.cache == "on":
        info, fin = score_all("on")
        GP.write_json(final, dict(base, engine_kwargs_on=kw, status="scored" if fin else "partial_on", sets=info,
                                  fa3_kvskip_on=fa3_now()))
        return 0 if fin else 4
    # caching off: the check on n_check of P's cuts against the caching-on pass, or a full pass without caching
    rowsP = sets["P"]
    allP = cut_rows(rowsP)
    pos = {(ri, t): n for n, (ri, t, _) in enumerate(allP)}
    crow = check_rows(rowsP, cfg["n_check"], plan["n_resp"])
    have_on = st.get("status") == "scored"
    floor = engine_floor_kl(plan)
    check = None
    if have_on and floor is None:
        check = f"skipped: no {FLOOR_JSON} in the gap dir (the engine floor)"
        log(f"fp8score {eng}: the prefix-caching check is {check}; rescoring without caching")
    elif have_on:
        t = time.time()
        pre = [plan["prompts"][rowsP[ri]["i"]]["prompt_ids"] + rowsP[ri]["response_ids"][:t_] for ri, t_ in crow]
        res = score_prefixes(llm, pre, V)
        on = np.load(score_paths(D, eng, "P")[0], mmap_mode="r")
        kls, dys = [], []
        for (ri, t_), (lq, _) in zip(crow, res):
            lq_on = np.asarray(on[pos[(ri, t_)]], np.float64)
            lq_on = lq_on - np.logaddexp.reduce(lq_on)
            lo = lq.numpy()
            kls.append(float(np.sum(np.exp(lo) * (lo - lq_on))))
            y = rowsP[ri]["response_ids"][t_]
            dys.append(float(abs(lo[y] - lq_on[y])))
        check = dict(n=len(crow), mean_kl_off_on=float(np.mean(kls)), max_kl_off_on=float(np.max(kls)),
                     mean_abs_dlogq_y=float(np.mean(dys)), threshold=floor,
                     threshold_source=f"{FLOOR_JSON} mean KL(q||p) (the bf16 engine floor)",
                     s=round(time.time() - t, 1))
        check["ok"] = bool(check["mean_kl_off_on"] <= floor)
        log(f"fp8score {eng}: prefix-caching check {check}")
    if isinstance(check, dict) and check["ok"]:
        GP.write_json(final, dict(st, status="complete", caching="on", check=check, engine_kwargs_off=kw,
                                  fa3_kvskip_off=fa3_now()))
        return 0
    info, fin = score_all("off")
    GP.write_json(final, dict(base, engine_kwargs_off=kw, status="complete" if fin else "partial_off", caching="off",
                              check=check if check is not None else "not run: no caching-on pass", sets=info,
                              fa3_kvskip_off=fa3_now()))
    return 0 if fin else 4


# ------------------------------------------------------------------------------------------------- the HF pass
def set_fields(name, all_cuts=True):
    """(field, rows, sampler) per set: 'all' = every response row, 'cut' = the 16 cut rows with weight T / |K|."""
    c = [("Gpc", "cut", None)] if all_cuts else []
    return {"P": [("Gp", "all", None), ("E_n015", "all", "n015"), ("E_n0325", "all", "n0325"),
                  ("E_f8all", "cut", "f8all"), ("E_f8s0", "cut", "f8s0")] + c,
            "P2": [("Gp", "all", None)],
            "Qn015": [("Gp", "all", None), ("E_n015", "all", "n015")],
            "Qn0325": [("Gp", "all", None), ("E_n0325", "all", "n0325")],
            "Qf8all": [("Gp", "all", None), ("E_f8all", "cut", "f8all")] + c,
            "Qf8s0": [("Gp", "all", None), ("E_f8s0", "cut", "f8s0")] + c}[name]


def set_qs(name):
    """Samplers whose q is measured on a set's rows: (all-row noise samplers, cut-row vLLM engines)."""
    return {"P": (("n015", "n0325"), ("f8all", "f8s0", "floor")), "P2": ((), ()), "Qn015": (("n015",), ()),
            "Qn0325": (("n0325",), ()), "Qf8all": ((), ("f8all",)), "Qf8s0": ((), ("f8s0",))}[name]


class Layout:
    """Sketch slots, prompt-major: per prompt, per (set, field) pair, one slot per unit. A unit is a response half
    (A = the first n/2 response slots, B = the rest) or, for the G_p pairs when gp_units == 'response', one response
    (the per-prefix sensitivity needs each response's own sum)."""

    def __init__(self, n_prompts, n_resp, all_cuts=True, gp_units="response"):
        self.K, self.n, self.all_cuts, self.gp_units = int(n_prompts), int(n_resp), bool(all_cuts), gp_units
        self.pairs, self.index, off = [], {}, 0
        for s in SETS:
            for f, kind, qn in set_fields(s, all_cuts):
                units = "response" if (f == "Gp" and gp_units == "response") else "half"
                nu = self.n if units == "response" else 2
                self.index[(s, f)] = len(self.pairs)
                self.pairs.append(dict(name=f"{s}:{f}", set=s, field=f, rows=kind, q=qn, units=units, n_units=nu,
                                       offset=off))
                off += nu
        self.per_prompt = off
        self.n_slots = self.K * off

    def slot(self, i, s, f, ridx):
        p = self.pairs[self.index[(s, f)]]
        u = ridx if p["units"] == "response" else half_of(ridx, self.n)
        return int(i) * self.per_prompt + p["offset"] + u

    def meta(self):
        import numpy as np
        sp, sk, su = [], [], []
        for i in range(self.K):
            for pi, p in enumerate(self.pairs):
                for u in range(p["n_units"]):
                    sp.append(pi)
                    sk.append(i)
                    su.append(u)
        return np.array(sp), np.array(sk), np.array(su)

    def to_json(self):
        return dict(K=self.K, n=self.n, all_cuts=self.all_cuts, gp_units=self.gp_units, per_prompt=self.per_prompt,
                    n_slots=self.n_slots, pairs=self.pairs)


def gp_field(lp, lt):
    """G_p = p (A - p.A): the same operations as gap_probe.logit_fields' 'P', so the two are bitwise equal."""
    p = lp.exp()
    A = lt - lp
    return p * (A - (p * A).sum(-1, keepdim=True))


def base_row_measures(lp, lt, gp=None):
    """Per-row quantities that need no sampler, float64. gap = KL(p || p_T) by gap_probe.logit_measures' formula."""
    p = lp.exp()
    out = dict(gap=(p * (lp - lt)).sum(-1), kl_Tp=(lt.exp() * (lt - lp)).sum(-1), H_p=-(p * lp).sum(-1),
               Abar_p=(p * (lt - lp)).sum(-1))
    out["gp_L2"] = (gp if gp is not None else gp_field(lp, lt)).norm(dim=-1)
    return out


def q_row_measures(lp, lq, lt, gp, E):
    """gap_probe.logit_measures (the listed keys) plus the per-row cosine of G_q = G_p + E with G_p."""
    import torch
    m = GP.logit_measures(lp, lq, lt)
    out = {k: m[k] for k in MEASURE_KEYS}
    gq = gp + E
    out["cos_gq_gp"] = (gp * gq).sum(-1) / torch.clamp(gp.norm(dim=-1) * gq.norm(dim=-1), min=1e-300)
    return out


class RowStore:
    def __init__(self):
        self.cols = {}

    def add(self, **kw):
        import numpy as np
        for k, v in kw.items():
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            self.cols.setdefault(k, []).append(np.asarray(v))

    def arrays(self):
        import numpy as np
        return {k: np.concatenate(v) if v else np.zeros(0) for k, v in self.cols.items()}

    def n(self):
        return sum(len(x) for x in self.cols.get("t", []))


class Response:
    """Everything the field function needs for one response."""

    def __init__(self, ri, i, T, y, hT, WT, WS, specs, q_all, q_cut, cuts, lq_dec, dev):
        self.ri, self.i, self.T, self.y, self.hT, self.WT, self.WS = ri, i, T, y, hT, WT, WS
        self.specs, self.q_all, self.q_cut, self.dev = specs, q_all, q_cut, dev
        self.cuts = list(cuts)                                  # [(t, draw index)]
        self.w = float(T) / len(self.cuts) if self.cuts else 0.0
        self.lq_dec = lq_dec


def make_field_fn(resp, base_store, q_stores, fd):
    """The field function for probe0_lib.pullback_trace: per chunk of response rows [c0, c1) it returns every field
    of resp.specs (all-row fields at every row; cut fields only at the cut rows, times T / |K|) and records the
    per-row measures."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    names_all = [f for f, kind, _ in resp.specs if kind == "all"]
    names_cut = [f for f, kind, _ in resp.specs if kind == "cut"]
    cut_t = {t: d for t, d in resp.cuts}

    def ff(z, c0, c1):
        R_ = c1 - c0
        out = {f: torch.zeros_like(z) for f in names_all}
        here = sorted(t for t in cut_t if c0 <= t < c1)
        if here:
            out.update({f: torch.zeros_like(z) for f in names_cut})
        for s0 in range(0, R_, SUB):
            s1 = min(R_, s0 + SUB)
            r0, r1 = c0 + s0, c0 + s1
            lp = torch.log_softmax(z[s0:s1], -1)
            lt = torch.log_softmax(F.linear(resp.hT[r0:r1], resp.WT).to(fd), -1)
            y = resp.y[r0:r1]
            gp = gp_field(lp, lt)
            tt = torch.arange(r0, r1)
            dcol = np.array([cut_t.get(int(t), -1) for t in tt], dtype=np.int64)
            wcol = np.where(dcol >= 0, resp.w, 0.0)
            bm = base_row_measures(lp, lt, gp)
            base_store.add(resp=torch.full((s1 - s0,), resp.ri), i=torch.full((s1 - s0,), resp.i), t=tt,
                           T=torch.full((s1 - s0,), resp.T), cut=dcol, w_cut=wcol,
                           lp_y=lp.gather(1, y[:, None])[:, 0], lt_y=lt.gather(1, y[:, None])[:, 0],
                           lq_dec=resp.lq_dec[r0:r1], **bm)
            if "Gp" in out:
                out["Gp"][s0:s1] = gp
            for qn, hq in resp.q_all.items():
                lq = torch.log_softmax(F.linear(hq[r0:r1].to(resp.dev), resp.WS).to(fd), -1)
                E = GP.logit_fields(lp, lq, lt)["E"]
                if f"E_{qn}" in out:
                    out[f"E_{qn}"][s0:s1] = E
                m = q_row_measures(lp, lq, lt, gp, E)
                q_stores[qn].add(resp=torch.full((s1 - s0,), resp.ri), i=torch.full((s1 - s0,), resp.i), t=tt,
                                 T=torch.full((s1 - s0,), resp.T), cut=dcol, w_cut=wcol,
                                 lq_y=lq.gather(1, y[:, None])[:, 0], **m)
                del lq, E, m
            hs = [t for t in here if r0 <= t < r1]
            if hs:
                loc = torch.tensor([t - r0 for t in hs], device=lp.device)
                dst = [t - c0 for t in hs]
                lpc, ltc, gpc = lp[loc], lt[loc], gp[loc]
                tt_c = np.array(hs, dtype=np.int64)
                dc = np.array([cut_t[t] for t in hs], dtype=np.int64)
                for qn, rows_q in resp.q_cut.items():
                    lqc = torch.stack([rows_q[t] for t in hs])
                    E = GP.logit_fields(lpc, lqc, ltc)["E"]
                    if f"E_{qn}" in out:
                        out[f"E_{qn}"][dst] = resp.w * E
                    m = q_row_measures(lpc, lqc, ltc, gpc, E)
                    q_stores[qn].add(resp=torch.full((len(hs),), resp.ri), i=torch.full((len(hs),), resp.i), t=tt_c,
                                     T=np.full(len(hs), resp.T), cut=dc, w_cut=np.full(len(hs), resp.w),
                                     lq_y=lqc.gather(1, y[loc][:, None])[:, 0], **m)
                if "Gpc" in out:
                    out["Gpc"][dst] = resp.w * gpc
            del lp, lt, gp
        return out, {}
    return ff


def run_response(student, teacher, PLm, ids, n_prompt, resp, accs, base_store, q_stores, chunk, fd):
    """One response: the teacher's final hidden states, then probe0_lib.pullback_trace (one forward with gradients,
    one body backward per field) into the accumulators `accs` (field name -> FieldAcc-like)."""
    resp.hT = PLm.forward_hidden(teacher, ids, grad=False)[n_prompt - 1:n_prompt - 1 + resp.T]
    ff = make_field_fn(resp, base_store, q_stores, fd)
    n = PLm.pullback_trace(student, ids, n_prompt, ff, accs, chunk, fd)
    resp.hT = None
    return n


def host_need_gb(sets, layout, k):
    rows = {s: sum(r["T"] for r in rows_) for s, rows_ in sets.items()}
    cache = (2 * rows["P"] + rows["Qn015"] + rows["Qn0325"]) * HIDDEN * 2
    return (cache + layout.n_slots * k * 4) / 1e9, cache / 1e9


def build_noisy_caches(student, src, scales, tim, plan, sets, dev, deadline=0):
    """For each noise level: the training noise on the student (gap_probe.noise_apply on the fp32 source, frozen base
    scales), the final hidden states at every response row of P and of that level's Q set kept on the host (bf16),
    then the student restored and checked bit for bit."""
    import torch
    PLm = PL()
    caches, recs = {}, {}
    for qn, sigma in SIGMAS:
        t = time.time()
        rec, _ = GP.noise_apply(student, sigma, src, scales, tim, GP.NOISE_SEED, dev)
        chk = noise_record_check(rec, plan["label"], plan.get("expected_step0"))
        log(f"HF noise sigma {sigma}: {rec} {chk}")
        if not chk["ok"]:
            raise SystemExit(f"[ps] REFUSE: the HF noise record at sigma {sigma} differs from the expected step-0 record: "
                             f"{chk}")
        with torch.no_grad():
            for s in ("P", f"Q{qn}"):
                for ri, r in enumerate(sets[s]):
                    if r["T"] == 0:
                        continue
                    pids = plan["prompts"][r["i"]]["prompt_ids"]
                    ids = torch.tensor(pids + r["response_ids"], dtype=torch.long, device=dev)
                    h = PLm.forward_hidden(student, ids, grad=False)
                    caches[(s, qn, ri)] = h[len(pids) - 1:len(pids) - 1 + r["T"]].to("cpu")
                    del h
        GP.restore_clean(student, src, dev)
        params = dict(student.named_parameters())
        with torch.no_grad():
            bad = [n for n in src if not torch.equal(params[n].detach(), src[n].to(dev).to(params[n].dtype))]
        if bad:
            raise SystemExit(f"[ps] REFUSE: the student is not restored bit for bit after sigma {sigma}: {bad[:3]}")
        recs[qn] = dict(sigma=sigma, record=rec, check=chk, restored_bitwise=True, s=round(time.time() - t, 1))
    return caches, recs


def load_scores(D, plan, sets):
    """name -> {set: (memmap [N, V], {(response, t): row})} for every finished scoring pass."""
    import numpy as np
    out, info = {}, {}
    for eng, targets in SCORE_TARGETS.items():
        st = GP.read_json(os.path.join(D, f"score_{eng}.json"))
        if st.get("status") != "complete":
            raise SystemExit(f"[ps] REFUSE: score_{eng}.json status {st.get('status')}")
        info[eng] = dict(caching=st.get("caching"), check=st.get("check"))
        out[eng] = {}
        for s in targets:
            npy, js = score_paths(D, eng, s)
            rec = GP.read_json(js)
            items = cut_rows(sets[s])
            if rec.get("status") != "complete" or rec["rows"] != [[ri, t] for ri, t, _ in items]:
                raise SystemExit(f"[ps] REFUSE: {js}: incomplete or its rows differ from the set's cuts")
            out[eng][s] = (np.load(npy, mmap_mode="r"), {(ri, t): n for n, (ri, t, _) in enumerate(items)})
    return out, info


def save_rows(D, name, base_store, q_stores, resp_table):
    import numpy as np
    arr = base_store.arrays()
    arr.update({f"resp_{k}": np.asarray(v) for k, v in resp_table.items()})
    np.savez_compressed(os.path.join(D, f"rows_{name}.npz.tmp.npz"), **arr)
    os.replace(os.path.join(D, f"rows_{name}.npz.tmp.npz"), os.path.join(D, f"rows_{name}.npz"))
    for qn, st in q_stores.items():
        np.savez_compressed(os.path.join(D, f"rowsq_{name}_{qn}.npz.tmp.npz"), **st.arrays())
        os.replace(os.path.join(D, f"rowsq_{name}_{qn}.npz.tmp.npz"), os.path.join(D, f"rowsq_{name}_{qn}.npz"))


def cmd_hf(a):
    import numpy as np
    import torch
    cfg = MODES[a.mode]
    D = a.out_dir
    plan = load_plan(D)
    out_json = os.path.join(D, "hf.json")
    if os.path.exists(os.path.join(D, "gram.npz")) and os.path.exists(out_json):
        log("hf: gram.npz exists, skipped")
        return 0
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    PLm, tim, gs = PL(), TIM(), GS()
    fd = torch.float64
    sets, srec = {}, {}
    for s in SETS:
        sets[s], srec[s] = load_set(D, s, plan)
    scores, sinfo = load_scores(D, plan, sets)
    K, n = len(plan["prompts"]), plan["n_resp"]
    sk = gs.Sketcher(log2k=a.sketch_log2)
    gp_units = a.gp_units
    lay = Layout(K, n, all_cuts=a.all_cuts, gp_units="response" if gp_units != "half" else "half")
    need, cache_gb = host_need_gb(sets, lay, sk.k)
    fallback = None
    if gp_units == "auto" and need > HOST_BUDGET_GB:
        # the host-memory fallback: G_p sketches per response half (main theta0 plan: 2,944 slots, about 99 GB of
        # sketches, -> 2,176 slots, about 73 GB). The per-prefix weighting (each response counts once) needs G_p per
        # response, so it is NOT computed after this fallback; the reader says so in reading.txt.
        fallback = dict(planned_gb_response_units=round(need, 1), slots_response_units=lay.n_slots)
        lay = Layout(K, n, all_cuts=a.all_cuts, gp_units="half")
        need, cache_gb = host_need_gb(sets, lay, sk.k)
        fallback.update(planned_gb_half_units=round(need, 1), slots_half_units=lay.n_slots)
        log(f"hf: HOST-MEMORY FALLBACK: G_p per response half ({fallback}); the per-prefix weighting is dropped")
    log(f"hf: host memory planned {need:.1f} GB (noisy caches {cache_gb:.1f} GB, {lay.n_slots} sketches, "
        f"G_p units {lay.gp_units}); budget {HOST_BUDGET_GB} GB")
    if need > HOST_BUDGET_GB:
        raise SystemExit(f"[ps] REFUSE: planned host memory {need:.1f} GB > {HOST_BUDGET_GB} GB")
    summ = dict(plan_ckpt=plan["ckpt"], label=plan["label"], path=plan["path"], mode=plan["mode"], attn=a.attn,
                chunk=a.chunk, sub=SUB, sketch=sk.info(), layout=lay.to_json(), host_need_gb=need,
                host_memory_fallback=fallback, checks={},
                timing={}, scores=sinfo, sets={s: {k: v for k, v in srec[s].items() if k not in ("engine_kwargs", "audit")}
                                                for s in SETS})
    import transformers
    summ["versions"] = dict(torch=torch.__version__, transformers=transformers.__version__)
    if dev.type == "cuda":
        summ["gpu"] = torch.cuda.get_device_name(0)
    from transformers import AutoConfig, AutoTokenizer
    tok_s, tok_t = AutoTokenizer.from_pretrained(GP.STUDENT), AutoTokenizer.from_pretrained(GP.TEACHER)
    V = AutoConfig.from_pretrained(GP.STUDENT).vocab_size
    if AutoConfig.from_pretrained(GP.TEACHER).vocab_size != V or tok_s.get_vocab() != tok_t.get_vocab():
        raise SystemExit("[ps] REFUSE: student and teacher vocabularies differ")
    t = time.time()
    student = GP.load_model(plan["path"], dev, a.attn)
    teacher = GP.load_model(GP.TEACHER, dev, a.attn)
    src, src_info = GP.load_noise_source(student, plan["path"], tim)
    scales = plan_scales(plan)
    summ["checks"]["noise_source"] = src_info
    tfp = [float(p.detach().double().sum()) for p in teacher.parameters()]
    summ["timing"]["load_s"] = round(time.time() - t, 1)
    log(f"hf: models and the fp32 noise source loaded ({time.time() - t:.0f}s) {src_info}")
    t = time.time()
    caches, nrec = build_noisy_caches(student, src, scales, tim, plan, sets, dev)
    summ["checks"]["noise"] = nrec
    summ["timing"]["caches_s"] = round(time.time() - t, 1)
    del src
    gc.collect()
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()
    body, lm = PLm.param_list(student)
    body_names = [nm for nm, _ in body]
    WS, WT = lm, teacher.get_output_embeddings().weight
    X = torch.zeros(lay.n_slots, sk.k, dtype=torch.float32)
    accs = {}
    exact = {}
    if a.exact_check:
        exact = {p: PLm.FieldAcc(student) for p in EXACT_PAIRS}
    stores = {s: (RowStore(), {qn: RowStore() for qn in sum(set_qs(s), ())}) for s in SETS}
    resp_table = {s: dict(ri=[], i=[], j=[], T=[], nll_sum=[], finish=[]) for s in SETS}
    per_prompt_s, timing_rows = [], []
    deadline = a.deadline or float("inf")
    n_done = 0
    t_loop = time.time()
    for i in range(K):
        avg = (sum(per_prompt_s) / len(per_prompt_s)) if per_prompt_s else 0.0
        reserve = cfg["prompt_reserve_s"] + lay.n_slots * sk.k * 4 / 2e9
        if time.time() + 1.25 * avg + reserve > deadline:
            log(f"hf: deadline: stopping after {n_done} of {K} prompts")
            break
        tp = time.time()
        for s in SETS:
            rows = sets[s]
            specs = set_fields(s, lay.all_cuts)
            q_all_names, q_cut_names = set_qs(s)
            for ridx in range(n):
                ri = i * n + ridx
                r = rows[ri]
                tab = resp_table[s]
                for k_, v_ in (("ri", ri), ("i", i), ("j", r["j"]), ("T", r["T"]),
                               ("nll_sum", np.nan if r["nll_sum"] is None else r["nll_sum"]), ("finish", r["finish_reason"])):
                    tab[k_].append(v_)
                if r["T"] == 0:
                    continue
                tr = time.time()
                pids = plan["prompts"][i]["prompt_ids"]
                ids = torch.tensor(pids + r["response_ids"], dtype=torch.long, device=dev)
                y = torch.tensor(r["response_ids"], dtype=torch.long, device=dev)
                q_cut = {}
                for qn in q_cut_names:
                    mm, pos = scores[qn][s]
                    ix = [pos[(ri, t_)] for t_ in r["cuts"]]
                    lqc = torch.from_numpy(np.array(mm[ix])).to(dev).double()
                    lqc = lqc - torch.logsumexp(lqc, -1, keepdim=True)
                    q_cut[qn] = {t_: lqc[m_] for m_, t_ in enumerate(r["cuts"])}
                lq_dec = torch.tensor(r["lq"] if r["lq"] is not None else [math.nan] * r["T"], dtype=torch.float64)
                resp = Response(ri, i, r["T"], y, None, WT, WS, specs, {qn: caches[(s, qn, ri)] for qn in q_all_names},
                                q_cut, [(t_, d_) for d_, t_ in enumerate(r["cuts"])], lq_dec, dev)
                fa = {}
                for f, _, _ in specs:
                    if f not in accs:
                        accs[f] = gs.SketchAcc(sk, body_names, WS.shape, dev)
                    fa[f] = gs.TeeAcc(accs[f], exact[f"{s}:{f}"]) if f"{s}:{f}" in exact else accs[f]
                run_response(student, teacher, PLm, ids, len(pids), resp, fa, stores[s][0], stores[s][1], a.chunk, fd)
                for f, _, _ in specs:
                    X[lay.slot(i, s, f, ridx)].add_(accs[f].take().cpu())
                timing_rows.append((s, i, ridx, r["T"], len(specs), round(time.time() - tr, 2)))
                del ids, y, q_cut, resp
        per_prompt_s.append(time.time() - tp)
        n_done = i + 1
        log(f"hf: prompt {n_done}/{K} done ({per_prompt_s[-1]:.0f}s; loop {time.time() - t_loop:.0f}s; peak GPU "
            f"{torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == 'cuda' else 0:.1f} GiB)")
    summ["timing"]["loop_s"] = round(time.time() - t_loop, 1)
    summ["timing"]["per_prompt_s"] = [round(x, 1) for x in per_prompt_s]
    summ["timing"]["responses"] = timing_rows
    if dev.type == "cuda":
        summ["max_mem_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    if a.mem_probe:
        summ["mem_probe"] = mem_probe(student, teacher, PLm, plan, D, accs, sk, dev, a.chunk, fd)
    tfp2 = [float(p.detach().double().sum()) for p in teacher.parameters()]
    summ["checks"]["teacher_unchanged"] = bool(tfp == tfp2)
    if not summ["checks"]["teacher_unchanged"]:
        raise SystemExit("[ps] REFUSE: the teacher's parameters changed during the pass")
    if exact:
        summ["exact_check"] = exact_check(exact, X, lay, sets, sk, PLm, n_done, dev)
        GP.write_json(os.path.join(D, "exact_check.json"), summ["exact_check"])
        del exact
    for s in SETS:
        save_rows(D, s, stores[s][0], stores[s][1], resp_table[s])
    del student, teacher, accs, caches, body, lm, WS, WT, scores
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    t = time.time()
    nr = n_done * lay.per_prompt
    G = gs.gram(X[:nr], dev).numpy() if nr else np.zeros((0, 0))
    summ["timing"]["gram_s"] = round(time.time() - t, 1)
    sp, sk_, su = lay.meta()
    Tarr = {f"T_{s}": np.array([[sets[s][i * n + rr]["T"] for rr in range(n)] for i in range(K)], np.int64) for s in SETS}
    np.savez(os.path.join(D, "gram.npz.tmp.npz"), G=G, slot_pair=sp[:nr], slot_prompt=sk_[:nr], slot_unit=su[:nr],
             n_done=np.array(n_done), layout=np.array(json.dumps(lay.to_json())), **Tarr)
    os.replace(os.path.join(D, "gram.npz.tmp.npz"), os.path.join(D, "gram.npz"))
    summ["n_prompts_done"] = n_done
    # A deadline-stopped pass is final: the sketches live only in memory, so a resumed job skips hf once gram.npz
    # exists. Redoing it needs gram.npz and hf.json removed by hand. The reader reads fewer than 32 prompts as INCOMPLETE.
    summ["status"] = "complete" if n_done == K else "partial"
    summ["elapsed_s"] = round(time.time() - T0, 1)
    GP.write_json(out_json, summ)
    log(f"hf: {summ['status']} ({n_done}/{K} prompts); Gram {G.shape} in {summ['timing']['gram_s']}s")
    return 0 if n_done == K else 4


def pair_vectors(X, lay, n_done, names):
    """name -> the sum of a pair's slot sketches over the finished prompts."""
    out = {}
    for nm in names:
        s, f = nm.split(":")
        p = lay.pairs[lay.index[(s, f)]]
        idx = [i * lay.per_prompt + p["offset"] + u for i in range(n_done) for u in range(p["n_units"])]
        out[nm] = X[idx].double().sum(0)
    return out


def exact_check(exact, X, lay, sets, sk, PLm, n_done, dev):
    """The smoke's exact-against-sketch check: the exact fp32 vectors (FieldAcc) of four pairs,
    their exact Gram (probe0_lib.readout), the Gram of the summed slot sketches, and the sketch of each exact vector."""
    import torch
    vecs = {p: exact[p].tensors() for p in EXACT_PAIRS}
    ex = PLm.readout(vecs, dev)["gram"]
    sv = pair_vectors(X, lay, n_done, EXACT_PAIRS)
    sg = {p: {q: float(torch.dot(sv[p], sv[q])) for q in EXACT_PAIRS} for p in EXACT_PAIRS}
    direct = {}
    for p in EXACT_PAIRS:
        s_ex = sk.sketch(vecs[p], device=dev).double().cpu()
        direct[p] = float((s_ex - sv[p]).norm() / max(float(sv[p].norm()), 1e-300))
    entries, ok = [], True
    for p in EXACT_PAIRS:
        for q in EXACT_PAIRS:
            bound = EXACT_BOUND * math.sqrt(max(ex[p][p], 0.0) * max(ex[q][q], 0.0)) / math.sqrt(sk.k)
            d = abs(sg[p][q] - ex[p][q])
            ok = ok and d <= bound
            entries.append(dict(x=p, y=q, exact=ex[p][q], sketch=sg[p][q], abs_diff=d, bound=bound, ok=bool(d <= bound)))
    DP = sum(r["T"] for r in sets["P"][:n_done * lay.n])
    DQ = sum(r["T"] for r in sets["Qn015"][:n_done * lay.n])

    def rr(G):
        gi = G["P:Gp"]["P:Gp"] / DP ** 2
        tq = G["Qn015:E_n015"]["Qn015:E_n015"] / DQ ** 2
        sp = G["Qn015:Gp"]["Qn015:Gp"] / DQ ** 2 - 2 * G["Qn015:Gp"]["P:Gp"] / (DQ * DP) + gi
        return math.sqrt(max(tq, 0) / gi), math.sqrt(max(sp, 0) / gi)
    rT_ex, rS_ex = rr(ex)
    rT_sk, rS_sk = rr(sg)
    out = dict(pairs=list(EXACT_PAIRS), k=sk.k, bound_factor=EXACT_BOUND, entries=entries, entries_ok=bool(ok),
               r_T=dict(exact=rT_ex, sketch=rT_sk, rel=abs(rT_sk / rT_ex - 1)),
               r_S=dict(exact=rS_ex, sketch=rS_sk, rel=abs(rS_sk / rS_ex - 1)),
               sketch_of_exact_vs_accumulated_rel=direct, n_prompts=n_done)
    out["r_ok"] = bool(out["r_T"]["rel"] <= EXACT_R_REL and out["r_S"]["rel"] <= EXACT_R_REL)
    out["ok"] = bool(out["entries_ok"] and out["r_ok"])
    log(f"exact check: entries {'ok' if ok else 'FAIL'}; r_T {rT_ex:.5f} / {rT_sk:.5f}, r_S {rS_ex:.5f} / {rS_sk:.5f}")
    return out


def mem_probe(student, teacher, PLm, plan, D, accs, sk, dev, chunk, fd):
    """The smoke's peak-memory probe: P's six fields on the longest saved held-out response of the main P set, at full
    length (the clean student's hidden states stand in for the noisy ones; the memory use is the same)."""
    import torch
    p = os.path.join(D, "sets", "MEMPROBE.jsonl")
    if not os.path.exists(p):
        return dict(ran=False, reason="no MEMPROBE set (theta0 smoke plans it)")
    r = json.loads(open(p).readline())
    T = r["T"]
    ids = torch.tensor(r["prompt_ids"] + r["response_ids"], dtype=torch.long, device=dev)
    n0 = len(r["prompt_ids"])
    with torch.no_grad():
        h = PLm.forward_hidden(student, ids, grad=False)[n0 - 1:n0 - 1 + T].to("cpu")
    cuts = draw_cuts_kj(T, K_CUTS, r["k"], r["j"])
    WS = student.get_output_embeddings().weight
    with torch.no_grad():
        q_cut = {}
        for qn in ("f8all", "f8s0", "floor"):
            hh = h[cuts].to(dev)
            lq = torch.log_softmax(torch.nn.functional.linear(hh, WS).to(fd), -1)
            q_cut[qn] = {t: lq[m] for m, t in enumerate(cuts)}
    resp = Response(0, 0, T, torch.tensor(r["response_ids"], device=dev), None, teacher.get_output_embeddings().weight,
                    WS, set_fields("P"), {"n015": h, "n0325": h}, q_cut, [(t, d) for d, t in enumerate(cuts)],
                    torch.full((T,), math.nan, dtype=torch.float64), dev)
    body, lm = PLm.param_list(student)
    fa = {}
    for f, _, _ in set_fields("P"):
        if f not in accs:
            accs[f] = GS().SketchAcc(sk, [nm for nm, _ in body], lm.shape, dev)
        fa[f] = accs[f]
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    run_response(student, teacher, PLm, ids, n0, resp, fa, RowStore(), {q: RowStore() for q in ("n015", "n0325", "f8all",
                                                                                                  "f8s0", "floor")},
                 chunk, fd)
    for f in fa:
        accs[f].take()
    out = dict(ran=True, k=r["k"], j=r["j"], T=T, seq_len=int(ids.numel()), n_fields=len(fa), s=round(time.time() - t, 1),
               peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
               total_gib=round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2))
    log(f"mem probe: {out}")
    return out


# ---------------------------------------------------------------------------- exact gradient probe reproduction
def probe0_reference(path):
    """The numbers of the exact gradient probe (tools/probe0/probe0.py's probe0.json) at its noise level with sigma
    REPRO_SIGMA, or None without a path (the comparison is skipped, and one line says so)."""
    if not path:
        log("repro: no --reference: the comparison with the exact gradient probe's probe0.json is skipped")
        return None
    d = GP.read_json(path)
    lvls = [k for k, v in d["noise"].items() if v.get("sigma") == REPRO_SIGMA]
    if len(lvls) != 1:
        raise SystemExit(f"[ps] REFUSE: {path}: not one noise level at sigma {REPRO_SIGMA}: {sorted(d['noise'])}")
    lvl = lvls[0]
    lv, rw = d["part_a"]["levels"][lvl], d["part_a"]["rows"]
    r, c = lv["r_bias"], lv["cos_bias_KL"]
    return dict(kl_qp=rw[lvl]["token_mean_kl_qp"], Abar_q=-rw[lvl]["token_mean_Eqa"], gap=rw["none"]["token_mean_kl"],
                r_bias=r, cos=(1 + r * c) / math.sqrt(1 + 2 * r * c + r * r), N_rows=d["part_a"]["N_rows"],
                sigma=d["noise"][lvl]["sigma"], level=lvl, path=path,
                note="a = log p - log r, so E_q[A] = -Eqa")


def repro_check(got, ref):
    out = {}
    for k, dig in REPRO_DIGITS.items():
        out[k] = dict(value=got[k], reference=ref[k], ok=bool(abs(got[k] - ref[k]) <= 0.5 * 10 ** (-dig)))
    for k in ("r_bias", "cos"):
        out[k] = dict(value=got[k], reference=ref[k], rel=abs(got[k] / ref[k] - 1),
                      ok=bool(abs(got[k] / ref[k] - 1) <= REPRO_REL))
    return out


def cmd_repro(a):
    import numpy as np
    import torch
    D = a.out_dir
    os.makedirs(D, exist_ok=True)
    out_json = os.path.join(D, "repro.json")
    if os.path.exists(out_json) and GP.read_json(out_json).get("status") == "complete":
        log("repro: exists, skipped")
        return 0
    ref = probe0_reference(a.reference)
    expected = GP.load_expected_step0(a.expected_step0, "[ps]")
    traces = [json.loads(line) for line in open(a.traces)]
    if a.max_traces:
        traces = traces[:a.max_traces]
    if not all(t["prompt_ids"][-3:] == [151645, 151648, 198] for t in traces):
        raise SystemExit("[ps] REFUSE: the traces' prompts do not end with the generation prompt")
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    PLm, tim, gs = PL(), TIM(), GS()
    fd = torch.float64
    t = time.time()
    student = GP.load_model(GP.STUDENT, dev, a.attn)
    teacher = GP.load_model(GP.TEACHER, dev, a.attn)
    src, src_info = GP.load_noise_source(student, GP.STUDENT, tim)
    rec, _ = GP.noise_apply(student, REPRO_SIGMA, src, load_scales(a.gap_dir), tim, GP.NOISE_SEED, dev)
    chk = noise_record_check(rec, "base", expected)
    hq = {}
    with torch.no_grad():
        for n_, tr in enumerate(traces):
            ids = torch.tensor(tr["prompt_ids"] + tr["response_ids"], dtype=torch.long, device=dev)
            n0 = len(tr["prompt_ids"])
            hq[n_] = PLm.forward_hidden(student, ids, grad=False)[n0 - 1:n0 - 1 + len(tr["response_ids"])].to("cpu")
    GP.restore_clean(student, src, dev)
    params = dict(student.named_parameters())
    restored = all(torch.equal(params[n].detach(), src[n].to(dev).to(params[n].dtype)) for n in src)
    del src
    gc.collect()
    if not (chk["ok"] and restored):
        raise SystemExit(f"[ps] REFUSE: repro noise record {chk} / restored {restored}")
    log(f"repro: models, noise and caches ready ({time.time() - t:.0f}s); {len(traces)} traces")
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()
    body, lm = PLm.param_list(student)
    sk = gs.Sketcher()
    specs = [("Gp", "all", None), ("E_n015", "all", "n015")]
    accs = {f: gs.SketchAcc(sk, [nm for nm, _ in body], lm.shape, dev) for f, _, _ in specs}
    tot = {f: torch.zeros(sk.k, dtype=torch.float32, device=dev) for f, _, _ in specs}
    base, qs = RowStore(), {"n015": RowStore()}
    t = time.time()
    deadline = a.deadline or float("inf")
    for n_, tr in enumerate(traces):
        avg = (time.time() - t) / n_ if n_ else 0.0
        if time.time() + 1.5 * avg + 120 > deadline:
            # the deadline guard: stop before the job's limit; a partial reproduction fails its gate (no resume)
            out = dict(status="partial", ok=False, n_traces_done=n_, n_traces=len(traces), s=round(time.time() - t, 1),
                       reason="deadline", max_traces=a.max_traces or None)
            GP.write_json(out_json, out)
            log(f"repro: deadline: stopped after {n_} of {len(traces)} traces: FAIL")
            return 4
        T = len(tr["response_ids"])
        ids = torch.tensor(tr["prompt_ids"] + tr["response_ids"], dtype=torch.long, device=dev)
        resp = Response(n_, n_, T, torch.tensor(tr["response_ids"], device=dev), None,
                        teacher.get_output_embeddings().weight, lm, specs, {"n015": hq[n_]}, {}, [],
                        torch.full((T,), math.nan, dtype=torch.float64), dev)
        run_response(student, teacher, PLm, ids, len(tr["prompt_ids"]), resp, accs, base, qs, 512, fd)
        for f in accs:
            tot[f] += accs[f].take()
        if n_ % 8 == 0 or n_ == len(traces) - 1:
            log(f"repro: trace {n_ + 1}/{len(traces)} ({time.time() - t:.0f}s)")
    g = {a_: {b_: float(torch.dot(tot[a_].double(), tot[b_].double())) for b_ in tot} for a_ in tot}
    P_, E_ = "Gp", "E_n015"
    r_bias = math.sqrt(g[E_][E_] / g[P_][P_])
    cos = (g[P_][P_] + g[P_][E_]) / math.sqrt(g[P_][P_] * (g[P_][P_] + 2 * g[P_][E_] + g[E_][E_]))
    bq = qs["n015"].arrays()
    ba = base.arrays()
    got = dict(kl_qp=float(np.mean(bq["kl_qp"])), Abar_q=float(np.mean(bq["Abar_q"])), gap=float(np.mean(ba["gap"])),
               r_bias=r_bias, cos=cos)
    if ref is None:
        cmp_ = "skipped: no --reference"
        ok = bool(chk["ok"] and restored)
    else:
        cmp_ = repro_check(got, ref)
        ok = bool(all(v["ok"] for v in cmp_.values()) and chk["ok"] and restored and len(ba["gap"]) == ref["N_rows"])
    out = dict(status="complete", ok=ok, got=got, reference=ref, compare=cmp_, noise=dict(record=rec, check=chk),
               restored_bitwise=restored, n_traces=len(traces), N_rows=int(len(ba["gap"])), attn=a.attn, sketch=sk.info(),
               gram=g, s=round(time.time() - t, 1), max_traces=a.max_traces or None)
    GP.write_json(out_json, out)
    shown = cmp_ if ref is None else {k: (round(v["value"], 6), v["ok"]) for k, v in cmp_.items()}
    log(f"repro: {'PASS' if ok else 'FAIL'} {json.dumps(shown)}; got {json.dumps(got)}")
    return 0


# ---------------------------------------------------------------------------------------------------- CPU checks
def cmd_gapcheck(a):
    """CPU, saved data: base_row_measures' gap on gap_probe measure's p_base.pt / p_s200.pt / tau.pt in --gap-dir equals
    gap_probe.logit_measures' per row bit for bit; with --reference (JSON: checkpoint label -> mean gap), its mean over
    the prefixes equals the reference to 1e-12 relative."""
    import numpy as np
    import torch
    d = a.gap_dir
    _, prec = GP.load_prefixes(d)
    want_all = GP.read_json(a.reference) if a.reference else None
    if want_all is None:
        log("gapcheck: no --reference: the mean gap is not compared")

    def logits(name):
        blob = torch.load(os.path.join(d, name), map_location="cpu")
        if blob["prefix_sha256"] != prec["sha256"]:
            raise SystemExit(f"[ps] REFUSE: {d}/{name} is for another prefix set")
        return blob["logits"]
    zt = logits("tau.pt")
    out, ok = {}, True
    for ck in ("base", "s200"):
        zp = logits(f"p_{ck}.pt")
        mine, ref = [], []
        for c0 in range(0, zp.shape[0], a.chunk):
            lp = torch.log_softmax(zp[c0:c0 + a.chunk].double(), -1)
            lt = torch.log_softmax(zt[c0:c0 + a.chunk].double(), -1)
            mine.append(base_row_measures(lp, lt)["gap"].numpy())
            ref.append(GP.logit_measures(lp, lp, lt)["gap"].numpy())
        mine, ref = np.concatenate(mine), np.concatenate(ref)
        mean = float(np.mean(mine))
        rec = dict(n=int(mine.size), mean_gap=mean, bitwise_equal_to_gap_probe=bool(np.array_equal(mine, ref)))
        if want_all is not None:
            want = float(want_all[ck])
            rec.update(reference=want, rel=abs(mean / want - 1))
        rec["ok"] = bool(rec["bitwise_equal_to_gap_probe"] and (want_all is None or rec["rel"] <= 1e-12))
        ok = ok and rec["ok"]
        out[ck] = rec
    res = dict(gapcheck="teacher gap on gap_probe's prefixes", ok=ok, reference=a.reference or None, checkpoints=out)
    print(json.dumps(res))
    if a.json:
        GP.write_json(a.json, res)
    return 0 if ok else 1


def cmd_selfcheck(a):
    import torch
    g = torch.Generator().manual_seed(0)
    worst = 0.0
    for V in (3, 9, 33):
        z = torch.randn(5, V, generator=g, dtype=torch.float64) * 3
        lp = torch.log_softmax(z, -1)
        lq = torch.log_softmax(z + torch.randn(5, V, generator=g, dtype=torch.float64), -1)
        lt = torch.log_softmax(torch.randn(5, V, generator=g, dtype=torch.float64) * 3, -1)
        ref = GP.autograd_reference(z, lq, lt)
        f = GP.logit_fields(lp, lq, lt)
        for x, y in ((gp_field(lp, lt), ref["g_p"]), (f["E"], ref["g_q"] - ref["g_p"]), (f["P"] + f["E"], ref["g_q"])):
            worst = max(worst, float((x - y).abs().max()))
    print(json.dumps(dict(selfcheck="G_p, E, G_q vs autograd", max_abs_err=worst, ok=worst < 1e-10)))
    return 0 if worst < 1e-10 else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "gen", "fp8score", "hf"):
        s = sub.add_parser(name)
        s.add_argument("--out-dir", required=True)
        s.add_argument("--mode", choices=sorted(MODES), required=True)
        s.add_argument("--deadline", type=float, default=0.0, help="epoch seconds; 0 = none")
    pl = sub.choices["plan"]
    pl.add_argument("--ckpt", choices=sorted(CK_LABEL), required=True)
    pl.add_argument("--gap-dir", default="", help="gap_probe's main output dir (prefixes.json, scales_base.json, and "
                                                 f"{FLOOR_JSON} for the prefix-caching check)")
    pl.add_argument("--ho-dir", default="", help="theta0: ho_eval's output dir for the base student (samples.jsonl, "
                                                "meta.json)")
    pl.add_argument("--ckpt-path", default="", help="theta200: the trained checkpoint's HF dir")
    pl.add_argument("--expected-step0", default="", help="JSON with the training log's step-0 [SPECOPD-TIM] fields; "
                                                         "absent = the noise records are not checked")
    sub.choices["gen"].add_argument("--engine", choices=sorted(GEN_SETS), required=True)
    sub.choices["fp8score"].add_argument("--engine", choices=sorted(SCORE_ENGINES), required=True)
    sub.choices["fp8score"].add_argument("--cache", choices=("on", "off"), required=True)
    h = sub.choices["hf"]
    h.add_argument("--attn", default="sdpa")
    h.add_argument("--device", default="cuda:0")
    h.add_argument("--chunk", type=int, default=512, help="response rows per lm_head / field chunk (probe0: 512)")
    h.add_argument("--all-cuts", type=int, default=1, help="1 = also the cut-row G_p fields (the all-cuts version)")
    h.add_argument("--gp-units", choices=("auto", "response", "half"), default="auto")
    h.add_argument("--exact-check", type=int, default=0, help="1 = exact FieldAcc beside the sketch (smoke)")
    h.add_argument("--mem-probe", type=int, default=0, help="1 = the full-length peak-memory probe (smoke)")
    h.add_argument("--sketch-log2", type=int, default=23, help="sketch dimension 2^n (default 23; smaller only for quick checks)")
    r = sub.add_parser("repro")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--traces", required=True, help="the exact gradient probe's (tools/probe0) part-(a) traces, the "
                                                  "file given to its --a-traces (jsonl: prompt_ids, response_ids)")
    r.add_argument("--gap-dir", required=True, help="gap_probe's main output dir (scales_base.json)")
    r.add_argument("--reference", default="", help="the exact gradient probe's probe0.json on the same traces; "
                                                    "absent = not compared")
    r.add_argument("--expected-step0", default="", help="JSON with the training log's step-0 [SPECOPD-TIM] fields; "
                                                        "absent = the noise record is not checked")
    r.add_argument("--attn", default="sdpa")
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--max-traces", type=int, default=0)
    r.add_argument("--deadline", type=float, default=0.0, help="epoch seconds; 0 = none (per-trace stop)")
    k = sub.add_parser("gapcheck")
    k.add_argument("--gap-dir", required=True, help="gap_probe's main output dir (prefixes.json, p_*.pt, tau.pt)")
    k.add_argument("--reference", default="", help="JSON: checkpoint label -> mean gap; absent = not compared")
    k.add_argument("--chunk", type=int, default=16)
    k.add_argument("--json", default="")
    sub.add_parser("selfcheck")
    a = ap.parse_args(argv)
    return dict(plan=cmd_plan, gen=cmd_gen, fp8score=cmd_fp8score, hf=cmd_hf, repro=cmd_repro, gapcheck=cmd_gapcheck,
                selfcheck=cmd_selfcheck)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
