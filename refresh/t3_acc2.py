"""Accuracy composite: AIME24 / AIME25 / AMC23 at 16 samples per problem (avg@16) for one checkpoint, no speculation.

    python t3_acc2.py freeze-check [--freeze <json>]                                    (stdlib only)
    python t3_acc2.py gen       --model <ckpt hf dir> --tokenizer <base student dir> --out-dir <D>/avg16 [--freeze <json>]
    python t3_acc2.py grade     --out-dir <D>/avg16 --tokenizer <base student dir>     (verifier chain: venv-specopd)
    python t3_acc2.py summarize --out-dir <D>/avg16 --reuse-dir <D> [--freeze <json>]  (stdlib only)
    python t3_acc2.py decontam  [--freeze <json> [--check]] [--train <parquet>] [--out <json>] (pandas)

Companion of refresh/t3_acc.py, the held-out KL pass (AIME24 x8 + MATH-500 x1, whose traces bench/tim_kl_eval.py
teacher-forces for the held-out KL). bench/tim_eval2.sbatch runs that pass first, then this.
Everything that defines a sample is t3_acc.py's:
  - prompt: the benchmark parquet's own chat list, formatted with the BASE student's chat template
    (add_generation_prompt=True; R1-Distill's template opens <think>), tokenized with add_special_tokens=False, so every
    checkpoint sees identical prompt ids (the same construction as t3_acc.py's cmd_gen);
  - sampling: T=1.0, top_p=1.0, max 16384 new tokens, one request per (problem, sample), each with its own seed;
  - engine: one vLLM 0.19.1 engine, bf16, gpu_memory_utilization 0.90, max_model_len 16384+1024, max_num_seqs 256,
    prefix caching, engine seed 2000, NO speculative decoding;
  - grading: decode (skip special tokens), custom_reward.ttrl_math.reward_func (last \\boxed{}), the same row fields.
Seeds extend t3_acc.py's  seed = 2000 + 1000*bench_no + 8*problem + sample  (bench_no AIME24 0, MATH-500 1) to
    seed = 2000 + OFFSET[bench] + 8*problem + (sample % 8) + 1_000_000 * (sample // 8)
    OFFSET: AIME24 0, MATH-500 1000 (both exactly t3_acc.py's), AIME25 10000, AMC23 20000.
AIME24 samples 0-7 thus carry t3_acc.py's seeds, and every other (bench, problem, sample) gets a seed that no request of
either harness uses (t3_acc.py's MATH-500 seeds span 3000-6999: a plain bench_no 2/3 for AIME25/AMC23 would collide).
Default plan "AIME24:8-15,AIME25:0-15,AMC23:0-15": AIME24 samples 0-7 already exist in the t3_acc.py pass of the SAME
checkpoint (<D>/graded_T1_p1.jsonl), so `summarize --reuse-dir <D>` merges them -- after checking their seeds, sampling
parameters, model and tokenizer -- instead of regenerating them. Plan "AIME24:0-15,..." regenerates all 16 standalone.
AMC23 in test_data is the 83-problem AMC12 2022+2023 set (AI-MO aimo-validation-amc, float answers; the DeepScaleR
"AMC" convention), NOT the 40-problem AMC 2023 set.

FREEZE RECORD (optional, --freeze <json>, schema t3_acc2_freeze/1). It fixes what a result is judged by: the benchmark
parquets' sha256 and problem counts, the decontamination rule, its thresholds and the exact excluded-problem list, k,
and the endpoint composite. The record for AIME24 / AIME25 / AMC23 ships as refresh/t3_acc2_freeze.json, and
bench/tim_eval2.sbatch passes it by default. Its data_dir and decontam.train_file are relative to $CLUSTER_ROOT
(root_path); data_dir names where --data looks by default, and the data are checked by sha256, not by path. With a
record, gen refuses data that does not match its pins, and summarize applies its exclusion list and endpoint;
bench/tim_eval2.sbatch runs `freeze-check` before ANY stage. Without a record, those checks are skipped (each command
prints one line saying so), no problem is excluded, and the endpoint is DEFAULT_ENDPOINTS: avg@16, the
benchmark-weighted composite of AIME24, AIME25 and AMC23 over the full benches.
LOGS: grade and summarize never print an accuracy (only counts, completeness and file paths); the numbers are only in
summary16_<tag>.json / merged16_<tag>.jsonl.
DECONTAMINATION (decontam/1): AMC23 overlaps the training pool (DAPO-Math-17k, the launcher's TRAIN_FILES; DAPO has a
block of rewritten AMC 2022/2023 problems, rows ~14700-17300). For every eval problem against every training row
(both with their prompt wrappers stripped, then lowercased and cut into [a-z0-9]+ words):
    W  = difflib.SequenceMatcher(autojunk=False) ratio of the two word sequences (exact whenever W >= 0.40)
    S3 = fraction of the eval problem's distinct word 3-grams that occur in the training row
  AUTO-EXCLUDE if max W >= 0.75 or max S3 >= 0.60; FLAGGED if not auto and (max W >= 0.50 or max S3 >= 0.30), each
  flagged problem carries an adjudication recorded in the freeze record (same question on the same data, up to the
  answer's format -> exclude; otherwise keep); everything else is kept. `decontam --freeze <json> --check` recomputes
  this from the parquets and fails unless it reproduces the record's list exactly (--check without --freeze is
  refused: there is no list to compare with); without a record, decontam scores and tiers every bench for audit only
  (DEFAULT_THRESHOLDS, --train required). Summaries report every bench in full AND, where the list
  is non-empty, a <bench>_clean block; the endpoint composite is computed on the subset the endpoint names.
Outputs (out-dir): gen16_<tag>.jsonl (+ _meta.json), graded16_<tag>.jsonl (no text), merged16_<tag>.jsonl (graded rows of
every sample of every bench, `src` = t3_acc | t3_acc2, `excluded` = in the record's decontamination list),
summary16_<tag>.json. None of these names matches t3_acc.py's `gen_T*.jsonl` glob, so the two harnesses can never grade
each other's files.
"""
import argparse
import difflib
import glob
import hashlib
import json
import os
import re
import sys
import time

ROOT = os.environ.get("CLUSTER_ROOT", "/path/to/workdir")
DATA = ROOT + "/workspace/datasets/test_data"
SEED0 = 2000                                   # t3_acc.py's --seed default (engine seed and request-seed base)
OFFSET = {"AIME24": 0, "MATH-500": 1000, "AIME25": 10000, "AMC23": 20000}
WINDOW = {"AIME24": 1000, "MATH-500": 9000, "AIME25": 10000, "AMC23": 10000}  # seed room per bench (no overlap)
BLOCK = 1_000_000                              # samples 8-15, 16-23, ... of a problem live in their own seed block
STRIDE = 8                                     # t3_acc.py's per-problem stride
DEFAULT_PLAN = "AIME24:8-15,AIME25:0-15,AMC23:0-15"
REUSE = ("AIME24",)                            # benches whose samples 0-7 the t3_acc.py pass already holds
T3_ACC_SAMPLES = {"AIME24": 8, "MATH-500": 1}  # t3_acc.py's BENCH: what its gen_T1_p1.jsonl holds

# ---- decontamination (decontam/1): the wrappers are exact (every row of the four parquets carries them; checked)
DECONTAM_METHOD = "decontam/1"
EVAL_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."
DAPO_PREFIX = ("Solve the following math problem step by step. The last line of your response should be of the form "
               "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n")
DAPO_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'
W_FLOOR = 0.40                                 # W is computed exactly at or above this; below it is reported as absent
TOPN = 3                                       # candidate rows kept per measure (the adjudication reads these)
WEIGHTINGS = ("benchmark", "problem")
SUBSETS = ("clean", "full")
DEFAULT_THRESHOLDS = dict(auto_word=0.75, auto_sh3=0.60, flag_word=0.50, flag_sh3=0.30)
# the endpoint without a freeze record: no exclusion list exists, so the composite is over the full benches
DEFAULT_ENDPOINTS = dict(k=16, composite=dict(benches=["AIME24", "AIME25", "AMC23"], weighting="benchmark", subset="full"))


def _rj(path):
    with open(path) as f:
        return json.load(f)


def _rjl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def _wj(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bench_parquet(bench, data=DATA):
    return os.path.join(data, bench, "test.parquet")


def root_path(path):
    """A path from the freeze record: relative paths are relative to $CLUSTER_ROOT, absolute ones are kept."""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


# ---------------------------------------------------------------- freeze record (optional)
def load_freeze(path):
    """(freeze dict, sha256 of the file's bytes), or (None, None) when no record is given."""
    if not path:
        return None, None
    return _rj(path), sha256_file(path)


def validate_freeze(fz):
    """Schema + internal consistency; returns a list of error strings (empty = valid). Needs no data."""
    errs = []
    if fz.get("schema") != "t3_acc2_freeze/1":
        errs.append(f"schema {fz.get('schema')!r} != 't3_acc2_freeze/1'")
    benches = fz.get("benches") or {}
    if not benches:
        errs.append("no benches")
    for b, v in benches.items():
        if b not in OFFSET:
            errs.append(f"bench {b} has no seed offset")
        if not (isinstance(v.get("n_problems"), int) and v["n_problems"] > 0):
            errs.append(f"{b}: n_problems {v.get('n_problems')!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(v.get("sha256", ""))):
            errs.append(f"{b}: sha256 {v.get('sha256')!r}")
    dc = fz.get("decontam") or {}
    if dc.get("method") != DECONTAM_METHOD:
        errs.append(f"decontam method {dc.get('method')!r} != {DECONTAM_METHOD!r} (the code implements only this one)")
    th = dc.get("thresholds") or {}
    for k in ("auto_word", "auto_sh3", "flag_word", "flag_sh3"):
        if not isinstance(th.get(k), (int, float)):
            errs.append(f"threshold {k} missing")
    if not errs and not (th["flag_word"] <= th["auto_word"] and th["flag_sh3"] <= th["auto_sh3"]
                         and th["flag_word"] >= W_FLOOR):
        errs.append(f"thresholds inconsistent {th} (need flag <= auto, flag_word >= W floor {W_FLOOR})")
    if not re.fullmatch(r"[0-9a-f]{64}", str(dc.get("train_sha256", ""))):
        errs.append(f"train_sha256 {dc.get('train_sha256')!r}")
    excl, adj = dc.get("exclude") or {}, dc.get("adjudicated") or {}
    for b in benches:
        n = benches[b].get("n_problems", 0)
        e = excl.get(b)
        if not isinstance(e, list) or e != sorted(set(e)) or not all(isinstance(i, int) and 0 <= i < n for i in e):
            errs.append(f"{b}: exclude must be a sorted list of distinct problem indices in [0, {n}), got {e!r}")
            continue
        for i, d in (adj.get(b) or {}).items():
            if not (i.isdigit() and int(i) < n):
                errs.append(f"{b}: adjudicated key {i!r}")
                continue
            if d.get("decision") not in ("exclude", "keep") or not str(d.get("note", "")).strip():
                errs.append(f"{b} {i}: adjudication needs decision exclude|keep and a note, got {d!r}")
            elif (d["decision"] == "exclude") != (int(i) in e):
                errs.append(f"{b} {i}: adjudicated {d['decision']} but {'not ' if int(i) not in e else ''}in exclude")
    for b in set(excl) | set(adj):
        if b not in benches:
            errs.append(f"decontam names bench {b} that the freeze does not pin")
    ep = fz.get("endpoints") or {}
    if not (isinstance(ep.get("k"), int) and ep["k"] > 0):
        errs.append(f"endpoints.k {ep.get('k')!r}")
    comp = ep.get("composite") or {}
    if not comp.get("benches") or any(b not in benches for b in comp.get("benches", [])):
        errs.append(f"endpoints.composite.benches {comp.get('benches')!r} not within the pinned benches")
    if comp.get("weighting") not in WEIGHTINGS or comp.get("subset") not in SUBSETS:
        errs.append(f"endpoints.composite weighting/subset {comp.get('weighting')!r}/{comp.get('subset')!r}")
    return errs


def freeze_excluded(fz):
    return {b: set(v) for b, v in fz["decontam"]["exclude"].items()}


def check_data_pins(fz, benches, data=DATA):
    """sha256 of each named bench parquet vs the freeze's pin -> ({bench: sha}, [errors])."""
    shas, errs = {}, []
    for b in benches:
        pin = fz["benches"].get(b)
        if pin is None:
            errs.append(f"bench {b} is not pinned by the freeze")
            continue
        p = bench_parquet(b, data)
        if not os.path.exists(p):
            errs.append(f"{p} missing")
            continue
        shas[b] = sha256_file(p)
        if shas[b] != pin["sha256"]:
            errs.append(f"{b}: {p} sha256 {shas[b]} != frozen {pin['sha256']}")
    return shas, errs


def cmd_freeze_check(a):
    fz, sha = load_freeze(a.freeze)
    if fz is None:
        print("[acc2] no freeze record given (--freeze): the benchmark sha256 pins and the decontamination list are not "
              f"checked; endpoint {DEFAULT_ENDPOINTS}", flush=True)
        return 0
    errs = validate_freeze(fz)
    if not errs:
        _, errs = check_data_pins(fz, list(fz["benches"]), a.data)
    excl = {b: len(v) for b, v in (fz.get("decontam") or {}).get("exclude", {}).items()}
    comp = (fz.get("endpoints") or {}).get("composite")
    print(f"[acc2] freeze {a.freeze} sha256={sha} excluded={excl} endpoint={comp}", flush=True)
    for e in errs:
        print(f"[acc2] freeze ERROR: {e}", flush=True)
    return 2 if errs else 0


# ---------------------------------------------------------------- seeds, plan, prompts
def seed_of(bench, i, s, seed0=SEED0):
    assert 0 <= i and STRIDE * (i + 1) <= WINDOW[bench], (bench, i)  # a bench's seeds never reach the next bench's
    return seed0 + OFFSET[bench] + STRIDE * i + (s % STRIDE) + BLOCK * (s // STRIDE)


def parse_plan(plan):
    out = []
    for part in plan.split(","):
        bench, rng = part.split(":")
        lo, hi = (int(x) for x in rng.split("-"))
        assert bench in OFFSET and 0 <= lo <= hi, part
        out.append((bench, lo, hi))
    assert len({b for b, _, _ in out}) == len(out), f"bench listed twice in {plan}"
    return out


def load_bench(name, data=DATA):
    """t3_acc.py's load_bench: the parquet's own chat list and ground truth, row order = problem index."""
    import pandas as pd
    df = pd.read_parquet(bench_parquet(name, data))
    out = []
    for i, r in enumerate(df.to_dict("records")):
        rm = r["reward_model"]
        if isinstance(rm, str):
            import ast
            rm = ast.literal_eval(rm)
        out.append(dict(i=i, id=str(r.get("id", i)), prompt=[dict(m) for m in r["prompt"]], gt=str(rm["ground_truth"])))
    return out


def prompt_ids(tok, chat):
    """t3_acc.py's prompt construction, verbatim."""
    text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


def build_items(tok, plan, seed0=SEED0, data=DATA):
    items, n_problems = [], {}
    for bench, lo, hi in parse_plan(plan):
        probs = load_bench(bench, data)
        n_problems[bench] = len(probs)
        for p in probs:
            ids = prompt_ids(tok, p["prompt"])
            for s in range(lo, hi + 1):
                items.append(dict(bench=bench, i=p["i"], id=p["id"], s=s, gt=p["gt"], prompt_ids=ids,
                                  seed=seed_of(bench, p["i"], s, seed0)))
    seeds = [it["seed"] for it in items]
    assert len(set(seeds)) == len(seeds), "duplicate request seeds in the plan"
    return items, n_problems


def tag_of(temp, top_p):
    return f"T{temp:g}_p{top_p:g}"


def cmd_gen(a):
    fz, fz_sha = load_freeze(a.freeze)
    benches = [b for b, _, _ in parse_plan(a.plan)]
    if fz is None:
        print("[acc2] no freeze record given (--freeze): the benchmark sha256 and problem-count checks are skipped",
              flush=True)
        errs, perr = [], []
        shas = {b: sha256_file(bench_parquet(b, a.data)) for b in benches}
    else:
        errs = validate_freeze(fz)
        shas, perr = check_data_pins(fz, benches, a.data) if not errs else ({}, [])
    if errs or perr:  # wrong data or an unusable freeze: fail BEFORE building an engine
        for e in errs + perr:
            print(f"[acc2] gen ERROR: {e}", flush=True)
        return 2
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    items, n_problems = build_items(tok, a.plan, a.seed, a.data)
    if fz is not None:
        bad = {b: n for b, n in n_problems.items() if n != fz["benches"][b]["n_problems"]}
        assert not bad, f"problem counts {bad} != frozen {({b: fz['benches'][b]['n_problems'] for b in bad})}"
    print(f"[acc2] plan {a.plan}: {len(items)} requests, problems {n_problems}; "
          f"first prompt: {tok.decode(items[0]['prompt_ids'])[-160:]!r}", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    todo = [tp for tp in a.temps.split(",")
            if not os.path.exists(os.path.join(a.out_dir, f"gen16_{tag_of(*(float(x) for x in tp.split(':')))}.jsonl"))]
    if not todo:  # unlike t3_acc.py, do not build an engine just to skip (gen again after a grade failure is free)
        print(f"[acc2] every gen16 file for {a.temps} exists in {a.out_dir}, nothing to do", flush=True)
        return 0
    t0 = time.time()
    llm = LLM(model=a.model, tokenizer=a.tokenizer, dtype="bfloat16", gpu_memory_utilization=a.gpu_util,
              max_model_len=a.max_tokens + 1024, max_num_seqs=a.max_num_seqs, seed=a.seed, enable_prefix_caching=True)
    init_s = time.time() - t0
    for tp in a.temps.split(","):
        temp, top_p = (float(x) for x in tp.split(":"))
        tag = tag_of(temp, top_p)
        path = os.path.join(a.out_dir, f"gen16_{tag}.jsonl")
        if os.path.exists(path):
            print(f"[acc2] {path} exists, skipping", flush=True)
            continue
        sps = [SamplingParams(temperature=temp, top_p=top_p, max_tokens=a.max_tokens, seed=it["seed"]) for it in items]
        t = time.time()
        # one call for every bench (longest first, as the plan lists them): one tail instead of one per bench
        outs = llm.generate([TokensPrompt(prompt_token_ids=it["prompt_ids"]) for it in items], sps, use_tqdm=False)
        wall = time.time() - t
        ntok, by_bench = 0, {}
        with open(path + ".tmp", "w") as f:
            for it, o in zip(items, outs):
                r = o.outputs[0]
                ntok += len(r.token_ids)
                bb = by_bench.setdefault(it["bench"], [0, 0])
                bb[0] += 1
                bb[1] += len(r.token_ids)
                f.write(json.dumps(dict(bench=it["bench"], i=it["i"], id=it["id"], s=it["s"], gt=it["gt"], seed=it["seed"],
                                        resp_len=len(r.token_ids), finish_reason=r.finish_reason,
                                        response_ids=list(r.token_ids))) + "\n")
        os.replace(path + ".tmp", path)
        meta = dict(model=a.model, tokenizer=a.tokenizer, temp=temp, top_p=top_p, max_tokens=a.max_tokens, n=len(items),
                    out_tokens=ntok, wall_s=round(wall, 1), init_s=round(init_s, 1), tok_per_s=round(ntok / max(wall, 1e-9), 1),
                    max_num_seqs=a.max_num_seqs, gpu_util=a.gpu_util, seed=a.seed, speculative=None, plan=a.plan,
                    n_problems=n_problems, by_bench={b: dict(n=v[0], out_tokens=v[1]) for b, v in by_bench.items()},
                    seed_scheme="seed0 + OFFSET[bench] + 8*i + (s%8) + 1e6*(s//8)", offsets=OFFSET,
                    data=a.data, data_sha256=shas, freeze_sha256_at_gen=fz_sha)
        _wj(os.path.join(a.out_dir, f"gen16_{tag}_meta.json"), meta)
        print(f"[acc2] gen {tag} " + json.dumps(meta), flush=True)  # lengths and timing only, no grading
    return 0


def grade_row(tok, reward_func, r):
    """t3_acc.py's per-row grading, verbatim (pops response_ids; a verifier crash is a recorded wrong answer)."""
    text = tok.decode(r.pop("response_ids"), skip_special_tokens=True)
    try:
        res = reward_func(r["bench"], text, r["gt"])
        r["acc"] = float(res["acc"]) if isinstance(res, dict) else float(res > 0)
        r["pred"] = (res.get("pred") or "")[:200] if isinstance(res, dict) else ""
        r["has_boxed"] = "\\boxed" in text
    except Exception as e:  # noqa: BLE001
        r["acc"], r["pred"], r["grade_error"] = 0.0, "", repr(e)[:200]
    r["closed_think"] = "</think>" in text
    return r


def cmd_grade(a):
    from transformers import AutoTokenizer
    from custom_reward.ttrl_math import reward_func
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    for path in sorted(glob.glob(os.path.join(a.out_dir, "gen16_T*.jsonl"))):
        tag = os.path.basename(path)[len("gen16_"):-len(".jsonl")]
        out = os.path.join(a.out_dir, f"graded16_{tag}.jsonl")
        if os.path.exists(out):
            print(f"[acc2] {out} exists, skipping", flush=True)
            continue
        t = time.time()
        rows = [grade_row(tok, reward_func, r) for r in _rjl(path)]
        with open(out + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(out + ".tmp", out)
        quick = {}
        for r in rows:
            q = quick.setdefault(r["bench"], dict(n=0, grade_errors=0))
            q["n"] += 1
            q["grade_errors"] += "grade_error" in r
        # counts only: accuracies live in the graded file and, merged, in summary16 / merged16
        print(f"[acc2] graded {tag} in {time.time() - t:.0f}s -> {out}: " + json.dumps(quick), flush=True)
    return 0


# ---------------------------------------------------------------- summarize
def _same_path(x, y):
    return x == y or os.path.realpath(x) == os.path.realpath(y)


def reuse_rows(reuse_dir, tag, meta16):
    """AIME24 samples 0-7 from the t3_acc.py pass of the same checkpoint, after checking it IS the same sampling."""
    gm = _rj(os.path.join(reuse_dir, f"gen_{tag}_meta.json"))
    for k in ("temp", "top_p", "max_tokens", "seed"):  # what defines the sampling distribution and the seeds
        assert gm[k] == meta16[k], f"reuse: {k} {gm[k]} (t3_acc pass) != {meta16[k]} (this pass)"
    assert gm.get("speculative") is None, f"reuse: t3_acc pass used speculation {gm.get('speculative')}"
    assert _same_path(gm["model"], meta16["model"]), f"reuse: model {gm['model']} != {meta16['model']}"
    assert _same_path(gm["tokenizer"], meta16["tokenizer"]), f"reuse: tokenizer {gm['tokenizer']} != {meta16['tokenizer']}"
    rows = []
    for r in _rjl(os.path.join(reuse_dir, f"graded_{tag}.jsonl")):
        if r["bench"] not in REUSE:
            continue
        assert r["s"] < T3_ACC_SAMPLES[r["bench"]], r
        want = seed_of(r["bench"], r["i"], r["s"], meta16["seed"])
        assert r["seed"] == want, f"reuse: seed {r['seed']} != {want} for {r['bench']} problem {r['i']} sample {r['s']}"
        r["src"] = "t3_acc"
        rows.append(r)
    return rows


def bench_summary(rs, n_problems_expected, k_target):
    per = {}
    for r in rs:
        per.setdefault(r["i"], []).append(r)
    samples = {i: sorted(x["s"] for x in v) for i, v in per.items()}
    full = list(range(k_target))
    complete = len(per) == n_problems_expected and all(v == full for v in samples.values())
    pp = [sum(x["acc"] for x in per[i]) / len(per[i]) for i in sorted(per)]
    n = len(rs)
    mean_pp = sum(pp) / len(pp)
    var_pp = sum((x - mean_pp) ** 2 for x in pp) / max(1, len(pp) - 1)
    out = dict(n=n, n_problems=len(per), n_problems_expected=n_problems_expected,
               k=(len(rs) // len(per)) if per else 0, complete=complete,
               acc=sum(r["acc"] for r in rs) / n,                   # avg@k: the mean over every sample
               acc_problem_mean=mean_pp,                            # mean over problems of per-problem avg@k (= acc if complete)
               pass_at_k=sum(any(x["acc"] > 0 for x in per[i]) for i in per) / len(per),
               se_problem=(var_pp / len(pp)) ** 0.5,                # descriptive: SE over problems (sampling of problems)
               trunc=sum(r["finish_reason"] == "length" for r in rs) / n,
               resp_len=sum(r["resp_len"] for r in rs) / n,
               closed_think=sum(bool(r.get("closed_think")) for r in rs) / n,
               has_boxed=sum(bool(r.get("has_boxed")) for r in rs) / n,
               grade_errors=sum("grade_error" in r for r in rs),
               per_problem_acc={str(i): round(v, 6) for i, v in zip(sorted(per), pp)})
    if not complete:
        bad = {i: v for i, v in samples.items() if v != full}
        out["incomplete_detail"] = dict(problems_seen=len(per), wrong_sample_sets=len(bad))
    return out


def composite(by_bench_rows, benches, excl, n_expected, k, weighting, subset):
    """Composite accuracy over `benches`: per-problem avg@k, then the mean over problems of each bench (benchmark
    weighting: mean of the bench means) or over all problems pooled (problem weighting). subset clean drops the record's
    decontamination list."""
    out, pooled, means, complete = dict(weighting=weighting, subset=subset, by_bench={}), [], [], True
    for b in benches:
        rs = by_bench_rows.get(b, [])
        drop = excl.get(b, set()) if subset == "clean" else set()
        rs = [r for r in rs if r["i"] not in drop]
        if not rs:
            out["by_bench"][b] = dict(n_problems=0, n_excluded=len(drop), acc=None, complete=False)
            complete = False
            continue
        s = bench_summary(rs, n_expected[b] - len(drop), k)
        pp = list(s["per_problem_acc"].values())
        pooled += pp
        means.append(s["acc_problem_mean"])
        complete &= s["complete"]
        out["by_bench"][b] = dict(n_problems=s["n_problems"], n_excluded=len(drop), acc=s["acc_problem_mean"],
                                  complete=s["complete"])
    ok = len(means) == len(benches)
    out["acc"] = (sum(means) / len(means) if weighting == "benchmark" else sum(pooled) / len(pooled)) if ok else None
    out["n_problems"] = len(pooled)
    out["complete"] = complete and ok
    return out


def cmd_summarize(a):
    fz, fz_sha = load_freeze(a.freeze)
    if fz is None:
        print("[acc2] no freeze record given (--freeze): no data pins and no decontamination exclusion list; endpoint "
              f"{DEFAULT_ENDPOINTS}", flush=True)
        endpoints, excl = DEFAULT_ENDPOINTS, {}
    else:
        errs = validate_freeze(fz)
        for e in errs:
            print(f"[acc2] summarize ERROR (freeze): {e}", flush=True)
        if errs:
            return 2
        endpoints, excl = fz["endpoints"], freeze_excluded(fz)
    k = endpoints["k"]
    comp_def = endpoints["composite"]
    tags = sorted(os.path.basename(p)[len("graded16_"):-len(".jsonl")]
                  for p in glob.glob(os.path.join(a.out_dir, "graded16_T*.jsonl")))
    assert tags, f"no graded16_T*.jsonl in {a.out_dir}"
    rc = 0
    for tag in tags:
        meta16 = _rj(os.path.join(a.out_dir, f"gen16_{tag}_meta.json"))
        if fz is not None:
            pins = {b: v["sha256"] for b, v in fz["benches"].items()}
            got = meta16.get("data_sha256") or {}
            bad = {b: got.get(b) for b in meta16["n_problems"] if got.get(b) != pins.get(b)}
            assert not bad, f"generated from data {bad} that the freeze does not pin ({ {b: pins.get(b) for b in bad} })"
            bad = {b: n for b, n in meta16["n_problems"].items() if n != fz["benches"][b]["n_problems"]}
            assert not bad, f"problem counts {bad} != frozen"
        rows = _rjl(os.path.join(a.out_dir, f"graded16_{tag}.jsonl"))
        for r in rows:
            assert r["seed"] == seed_of(r["bench"], r["i"], r["s"], meta16["seed"]), r
            r["src"] = "t3_acc2"
        have = {(r["bench"], r["i"], r["s"]) for r in rows}
        assert len(have) == len(rows), "duplicate (bench, problem, sample) rows in graded16"
        reused = 0
        if a.reuse_dir:
            old = reuse_rows(a.reuse_dir, tag, meta16)
            clash = [x for x in ((r["bench"], r["i"], r["s"]) for r in old) if x in have]
            assert not clash, f"reuse: {len(clash)} samples are in BOTH passes, e.g. {clash[:3]}"
            rows += old
            reused = len(old)
        for r in rows:
            r["excluded"] = r["i"] in excl.get(r["bench"], set())
        n_expected = dict(meta16["n_problems"])
        by_bench = {}
        for r in rows:
            by_bench.setdefault(r["bench"], []).append(r)
        order = sorted(by_bench, key=lambda x: list(OFFSET).index(x))
        summ = dict(tag=tag, model=meta16["model"], plan=meta16["plan"], reuse_dir=a.reuse_dir, reused_rows=reused,
                    k_target=k, data_sha256=meta16["data_sha256"],
                    freeze=dict(path=a.freeze, sha256=fz_sha) if fz is not None else None)
        for b in order:  # every bench in FULL, and a _clean block wherever the exclusion list is non-empty
            summ[b] = bench_summary(by_bench[b], n_expected.get(b, -1), k)
            if excl.get(b):
                clean = [r for r in by_bench[b] if not r["excluded"]]
                summ[f"{b}_clean"] = dict(bench_summary(clean, n_expected.get(b, -1) - len(excl[b]), k) if clean else
                                          dict(n=0, n_problems=0, complete=False), excluded_problems=sorted(excl[b]))
        comps = {f"composite_{w}_{s}": composite(by_bench, comp_def["benches"], excl, n_expected, k, w, s)
                 for w in WEIGHTINGS for s in SUBSETS}
        ep_name = f"composite_{comp_def['weighting']}_{comp_def['subset']}"
        summ["endpoint"] = dict(name=ep_name, benches=comp_def["benches"], **comps.pop(ep_name))
        summ["sensitivity"] = comps   # the other three composites, reported next to the endpoint
        diag = {}
        for b in REUSE:
            if b in by_bench and reused:
                for src, lab in (("t3_acc", "samples_0_7"), ("t3_acc2", "samples_8_15")):
                    h = [r for r in by_bench[b] if r["src"] == src]
                    diag[f"{b}_{lab}_acc"] = sum(r["acc"] for r in h) / len(h) if h else None
        if diag:
            diag["note"] = ("merge consistency check only, NOT an endpoint: AIME24 samples 0-7 come from the t3_acc.py "
                            "pass; the endpoint pools all 16")
        summ["diagnostics"] = diag
        rows.sort(key=lambda r: (list(OFFSET).index(r["bench"]), r["i"], r["s"]))
        mp = os.path.join(a.out_dir, f"merged16_{tag}.jsonl")
        with open(mp + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(mp + ".tmp", mp)
        sp = os.path.join(a.out_dir, f"summary16_{tag}.json")
        _wj(sp, summ)
        # completeness and counts only: no accuracy, pass rate or truncation reaches the log
        short = {b: dict(n_problems=summ[b]["n_problems"], k=summ[b]["k"], complete=summ[b]["complete"]) for b in order}
        print(f"[acc2] summary {tag}: reused {reused} rows, benches {json.dumps(short)}, endpoint {ep_name} complete="
              f"{summ['endpoint']['complete']} over {summ['endpoint']['n_problems']} problems, excluded "
              f"{ {b: len(v) for b, v in excl.items() if v} }, freeze sha256 {fz_sha} -> {sp}", flush=True)
        if not (all(summ[b]["complete"] for b in order) and summ["endpoint"]["complete"]):
            print(f"[acc2] WARNING: a bench is not complete at k={k} (see {sp})", flush=True)
            rc = 3
    return rc


# ---------------------------------------------------------------- decontamination (decontam/1)
def strip_wrap(text, prefix, suffix):
    assert text.startswith(prefix) and text.endswith(suffix), f"unexpected prompt wrapper: {text[:60]!r}...{text[-60:]!r}"
    return text[len(prefix):len(text) - len(suffix)]


def norm_words(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def trigrams(words):
    return {tuple(words[k:k + 3]) for k in range(len(words) - 2)}


def _top(scored, n=TOPN):
    return [[s, j] for s, j in sorted(scored, key=lambda x: (-x[0], x[1]))[:n]]  # unrounded: tiers compare these


def overlap_scores(eval_words, train_words, train_tri, floor=W_FLOOR):
    """Top-n training rows by W (word-sequence ratio, exact for every row with W >= floor: real_quick_ratio and
    quick_ratio are upper bounds of ratio, so pruning on them never drops a row at or above the floor) and by S3."""
    sm = difflib.SequenceMatcher(None, autojunk=False)
    sm.set_seq2(eval_words)
    w = []
    for j, dw in enumerate(train_words):
        sm.set_seq1(dw)
        if sm.real_quick_ratio() < floor or sm.quick_ratio() < floor:
            continue
        r = sm.ratio()
        if r >= floor:
            w.append((r, j))
    e3 = trigrams(eval_words)
    s3 = [(len(e3 & t) / len(e3), j) for j, t in enumerate(train_tri)] if e3 else []
    return dict(word=_top(w), sh3=_top([x for x in s3 if x[0] > 0]), n_words=len(eval_words))


def tier_of(sc, th):
    W = sc["word"][0][0] if sc["word"] else 0.0
    S = sc["sh3"][0][0] if sc["sh3"] else 0.0
    if W >= th["auto_word"] or S >= th["auto_sh3"]:
        return "auto"
    if W >= th["flag_word"] or S >= th["flag_sh3"]:
        return "flag"
    return "clean"


def decide(scored, fz):
    """scored: {bench: [score dicts with i]} -> ({bench: sorted excluded indices}, [errors]). Benches the freeze does not
    pin are scored and tiered for audit only (no decision, no error)."""
    th, adj = fz["decontam"]["thresholds"], fz["decontam"].get("adjudicated") or {}
    out, errs = {}, []
    for b, rows in scored.items():
        if b not in fz["benches"]:
            for r in rows:
                r["tier"], r["decision"] = tier_of(r, th), None
            continue
        ab, ex = adj.get(b) or {}, []
        for r in rows:
            t, d = tier_of(r, th), ab.get(str(r["i"]))
            r["tier"] = t
            if t == "flag":
                if d is None:
                    errs.append(f"{b} {r['i']}: flagged but not adjudicated (W {r['word'][:1]}, S3 {r['sh3'][:1]})")
                    r["decision"] = None
                    continue
                r["decision"], r["note"] = d["decision"], d["note"]
            else:
                if d is not None:
                    errs.append(f"{b} {r['i']}: adjudicated but tier is {t} (stale adjudication)")
                r["decision"] = "exclude" if t == "auto" else "keep"
            if r["decision"] == "exclude":
                ex.append(r["i"])
        out[b] = sorted(ex)
    return out, errs


def cmd_decontam(a):
    if a.check and not a.freeze:
        print("[acc2] decontam ERROR: --check compares with a freeze record's exclusion list and needs --freeze <json>",
              flush=True)
        return 2
    import pandas as pd
    fz, fz_sha = load_freeze(a.freeze)
    if fz is None:
        if not a.train:
            print("[acc2] decontam ERROR: without a freeze record (--freeze) it needs --train <training parquet>",
                  flush=True)
            return 2
        print("[acc2] no freeze record given (--freeze): DEFAULT_THRESHOLDS, every bench scored and tiered for audit "
              "only, no exclusion list to reproduce", flush=True)
        fz, errs = dict(benches={}, decontam=dict(thresholds=DEFAULT_THRESHOLDS, exclude={})), []
    else:
        errs = validate_freeze(fz)
    train = a.train or root_path(fz["decontam"]["train_file"])
    tsha = sha256_file(train)
    if fz_sha is not None and tsha != fz["decontam"]["train_sha256"]:
        errs.append(f"train file {train} sha256 {tsha} != frozen {fz['decontam']['train_sha256']}")
    benches = a.benches.split(",") if a.benches else (list(fz["benches"]) or DEFAULT_ENDPOINTS["composite"]["benches"])
    _, perr = check_data_pins(fz, [b for b in benches if b in fz["benches"]], a.data)
    errs += perr
    t0 = time.time()
    tdf = pd.read_parquet(train)
    if fz_sha is not None:
        assert len(tdf) == fz["decontam"]["train_rows"], f"train rows {len(tdf)} != frozen {fz['decontam']['train_rows']}"
    t_words = [norm_words(strip_wrap(p[0]["content"], DAPO_PREFIX, DAPO_SUFFIX)) for p in tdf["prompt"]]
    t_tri = [trigrams(w) for w in t_words]
    t_gt = [str((rm if isinstance(rm, dict) else {}).get("ground_truth")) for rm in tdf["reward_model"]]
    scored = {}
    for b in benches:
        rows = []
        for p in load_bench(b, a.data):
            assert len(p["prompt"]) == 1 and p["prompt"][0]["role"] == "user", (b, p["i"])
            sc = overlap_scores(norm_words(strip_wrap(p["prompt"][0]["content"], "", EVAL_SUFFIX)), t_words, t_tri)
            for m in ("word", "sh3"):
                for x in sc[m]:
                    x.append(t_gt[x[1]])  # [score, train row, train ground truth]
            rows.append(dict(bench=b, i=p["i"], id=p["id"], gt=p["gt"], **sc))
        scored[b] = rows
    excl, derr = decide(scored, fz)
    errs += derr
    if a.check and not excl:  # --benches named only benches the record does not pin: nothing to compare
        errs.append(f"--check scored no bench the freeze record pins (benches {benches}, pinned {list(fz['benches'])})")
    for b, rows in scored.items():
        n = {t: sum(r["tier"] == t for r in rows) for t in ("auto", "flag", "clean")}
        print(f"[acc2] decontam {b}: {len(rows)} problems, tiers {n}"
              + (f", excluded {len(excl[b])} (frozen {len(fz['decontam']['exclude'].get(b, []))})" if b in excl else
                 " (audit only: not pinned by the freeze)"), flush=True)
    mism = {b: dict(computed_not_frozen=sorted(set(v) - set(fz["decontam"]["exclude"].get(b, []))),
                    frozen_not_computed=sorted(set(fz["decontam"]["exclude"].get(b, [])) - set(v)))
            for b, v in excl.items() if v != fz["decontam"]["exclude"].get(b, [])}
    if mism:
        errs.append(f"computed exclusion list differs from the frozen one: {mism}")
    if a.out:
        _wj(a.out, dict(method=DECONTAM_METHOD, freeze=a.freeze, freeze_sha256=fz_sha, train=train, train_sha256=tsha,
                        thresholds=fz["decontam"]["thresholds"], w_floor=W_FLOOR, excluded=excl, errors=errs,
                        wall_s=round(time.time() - t0, 1), problems=scored))
        print(f"[acc2] decontam scores -> {a.out}", flush=True)
    for e in errs:
        print(f"[acc2] decontam ERROR: {e}", flush=True)
    verdict = "FAILED" if errs else ("OK: reproduces the record's list" if excl else "audit only (no pinned bench scored)")
    print(f"[acc2] decontam {verdict} ({time.time() - t0:.0f}s)", flush=True)
    return 5 if (errs and a.check) else 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--model", required=True)
    g.add_argument("--tokenizer", required=True)
    g.add_argument("--out-dir", required=True)
    g.add_argument("--plan", default=DEFAULT_PLAN)
    g.add_argument("--temps", default="1.0:1.0")
    g.add_argument("--max-tokens", type=int, default=16384)
    g.add_argument("--max-num-seqs", type=int, default=256)
    g.add_argument("--gpu-util", type=float, default=0.90)
    g.add_argument("--seed", type=int, default=SEED0)
    g.add_argument("--data", default=DATA)
    g.add_argument("--freeze", default="", help="optional freeze record (JSON, schema t3_acc2_freeze/1)")
    r = sub.add_parser("grade")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--tokenizer", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--reuse-dir", default="", help="the t3_acc.py pass dir of the SAME checkpoint (AIME24 samples 0-7)")
    s.add_argument("--freeze", default="", help="optional freeze record (JSON, schema t3_acc2_freeze/1)")
    f = sub.add_parser("freeze-check")
    f.add_argument("--freeze", default="", help="optional freeze record (JSON, schema t3_acc2_freeze/1)")
    f.add_argument("--data", default=DATA)
    d = sub.add_parser("decontam")
    d.add_argument("--freeze", default="", help="optional freeze record (JSON, schema t3_acc2_freeze/1)")
    d.add_argument("--data", default=DATA)
    d.add_argument("--train", default="", help="default: the freeze's decontam.train_file, relative to $CLUSTER_ROOT "
                                               "(required without --freeze)")
    d.add_argument("--benches", default="", help="default: the freeze's benches, else AIME24,AIME25,AMC23 "
                                                 "(benches the freeze does not pin: audit only, e.g. MATH-500)")
    d.add_argument("--out", default="")
    d.add_argument("--check", action="store_true",
                   help="exit 5 unless the record's list is reproduced exactly (needs --freeze; exit 2 without it)")
    a = ap.parse_args()
    return {"gen": cmd_gen, "grade": cmd_grade, "summarize": cmd_summarize, "freeze-check": cmd_freeze_check,
            "decontam": cmd_decontam}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
