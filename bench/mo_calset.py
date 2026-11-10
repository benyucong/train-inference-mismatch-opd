#!/usr/bin/env python3
"""The calibration set of every Model Optimizer and llm-compressor export and of the calibrated-cache configurations:
512 responses sampled by our student (T 1, top-p 1, <= 16,384 tokens) on 256 DAPO training problems drawn with seed
20261008, disjoint from every held-out and evaluation set. One set for every tool and every format.

The rule:
  1. pool = the DAPO parquet (dapo-math-17k, sha256 asserted by ho_select.load_dapo), rows 0..17,916;
  2. order = numpy.random.default_rng(20261008).permutation(17917); walk it from the start;
  3. drop a row that is one of the 1,000 held-out problems (the in-distribution held-out set bench/ho_select.py builds;
     its pool indices are listed in mo_calset_excl.json);
  4. drop a row whose eval tier against any of AIME24, AIME25, AMC23, MATH-500 is auto or flag (ho_select.py's scores:
     W = difflib ratio with the 0.40 floor, S_fwd / S_rev trigram overlaps, Tier(W, max(S_fwd, S_rev)); thresholds
     auto W >= 0.75 or S >= 0.60, flag W >= 0.50 or S >= 0.30) -- stricter than the held-out selection, which drops a
     flag only with an equal answer;
  5. keep the first 256 survivors, in walk order (k = 0..255);
  6. prompt = the row's DAPO prompt as stored, rendered by the student's chat template exactly as tim_calibrate /
     verl render it (apply_chat_template(add_generation_prompt=True), then tokenized without special tokens);
  7. two responses per problem, per-request seed 20261008 + 10k + j (j = 0, 1), T 1, top-p 1, no top-k, <= 16,384
     tokens, a clean bf16 vLLM 0.19.1 engine (the student, no quantization).

    $PYTHON_BIN bench/mo_calset.py select   --out-dir D      # CPU: the 256 problems -> D/problems.json
    $PYTHON_BIN bench/mo_calset.py generate --out-dir D      # GPU: the 512 responses -> D/calset.jsonl + calset_record.json
    python3     bench/mo_calset.py selftest                  # stdlib only
Run from the repository root (bench/ and refresh/ importable).
"""
import argparse
import difflib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mo_common as C  # noqa: E402


def log(*a):
    print(f"[mo_calset {time.strftime('%H:%M:%S')}]", *a, flush=True)


def eval_scores(ho, pool, evals, matchers, i, floor):
    """ho_select.score_candidate's eval half, verbatim logic: per bench [max W, max S_fwd, max S_rev, best match]."""
    q, qt = pool.words[i], pool.tri[i]
    ev = {}
    for e, m in enumerate(matchers):
        w = ho.ratio(m, q, floor)
        s = len(evals.tri[e] & qt) / len(evals.tri[e]) if evals.tri[e] else 0.0
        sr = len(qt & evals.tri[e]) / len(qt) if qt else 0.0
        b = evals.ids[e][0]
        cur = ev.get(b, [0.0, 0.0, 0.0, None])
        if max(w, s) > max(cur[0], cur[1]):
            cur[3] = evals.ids[e][1]
        ev[b] = [max(cur[0], w), max(cur[1], s), max(cur[2], sr), cur[3]]
    return ev


def walk(order, excl, tier_of, n_keep):
    """Steps 3-5. tier_of(i) -> {bench: tier}. Returns (kept, record of drops, number of rows walked)."""
    kept, drops, walked = [], {"heldout": [], "eval_auto": [], "eval_flag": []}, 0
    for i in order:
        if len(kept) == n_keep:
            break
        walked += 1
        if i in excl:
            drops["heldout"].append(int(i))
            continue
        tiers = tier_of(i)
        worst = "auto" if "auto" in tiers.values() else ("flag" if "flag" in tiers.values() else "clean")
        if worst in C.CAL_DROP_TIERS:
            drops[f"eval_{worst}"].append(int(i))
            continue
        kept.append(int(i))
    return kept, drops, walked


def cmd_select(a):
    import numpy as np
    import ho_select as ho  # imports refresh/t3_acc2.py from this tree
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "problems.json")
    if os.path.exists(out):
        raise SystemExit(f"REFUSE: {out} exists")
    df, contents, gts, sha = ho.load_dapo(C.DAPO)
    n = len(df)
    pool = ho.Pool(contents, gts, list(range(n)))          # order unused here (identity)
    evals = ho.load_evals(C.EVAL_DATA)
    assert tuple(sorted({b for b, _ in evals.ids})) == tuple(sorted(C.CAL_EVAL_BENCHES))
    matchers = []
    for w in evals.words:
        sm = difflib.SequenceMatcher(None, autojunk=False)
        sm.set_seq2(w)
        matchers.append(sm)
    excl = C.load_excl(HERE)
    assert max(excl) < n
    order = [int(x) for x in np.random.default_rng(C.CAL_SEED).permutation(n)]
    scores = {}

    def tier_of(i):
        ev = eval_scores(ho, pool, evals, matchers, i, ho.FLOOR)
        scores[i] = ev
        return {b: ho.eval_tier(v) for b, v in ev.items()}

    t = time.time()
    kept, drops, walked = walk(order, excl, tier_of, C.CAL_N_PROBLEMS)
    if len(kept) != C.CAL_N_PROBLEMS:
        raise SystemExit(f"FAIL: only {len(kept)} problems survive")
    rec = dict(rule="mo_calset.py docstring", dapo=C.DAPO, dapo_sha256=sha, n_pool=n, seed=C.CAL_SEED,
               eval_benches=list(C.CAL_EVAL_BENCHES), drop_tiers=list(C.CAL_DROP_TIERS), n_walked=walked,
               n_dropped={k: len(v) for k, v in drops.items()}, dropped=drops, pool_indices=kept,
               ground_truths=[gts[i] for i in kept], zh=sum(ho.is_zh(pool.texts[i]) for i in kept),
               eval_scores={str(i): scores[i] for i in kept + drops["eval_auto"] + drops["eval_flag"]},
               seconds=round(time.time() - t, 1))
    C.write_json(out, rec)
    log(f"selected {len(kept)} of {walked} walked; dropped {rec['n_dropped']}; zh {rec['zh']} -> {out}")
    return 0


def render(tok, prompt_msgs):
    text = tok.apply_chat_template([dict(m) for m in prompt_msgs], tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


def cmd_generate(a):
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    probs = json.load(open(os.path.join(a.out_dir, "problems.json")))
    out = os.path.join(a.out_dir, "calset.jsonl")
    if os.path.exists(out):
        raise SystemExit(f"REFUSE: {out} exists")
    df = pd.read_parquet(C.DAPO)
    tok = AutoTokenizer.from_pretrained(C.STUDENT)
    reqs, sps, meta = [], [], []
    for k, i in enumerate(probs["pool_indices"]):
        pids = render(tok, df["prompt"].iloc[i])
        for j in range(C.CAL_N_PER_PROBLEM):
            sd = C.request_seed(k, j)
            sp = SamplingParams(temperature=C.CAL_TEMPERATURE, top_p=C.CAL_TOP_P, max_tokens=C.CAL_MAX_TOKENS, seed=sd)
            assert sp.temperature == 1.0 and sp.top_p == 1.0 and sp.top_k in (0, -1) and sp.min_p == 0.0
            reqs.append(TokensPrompt(prompt_token_ids=pids))
            sps.append(sp)
            meta.append(dict(k=k, j=j, pool_index=int(i), seed=sd, prompt_ids=pids))
    t = time.time()
    llm = LLM(model=C.STUDENT, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=18433 + 1024,
              enable_prefix_caching=False, enforce_eager=False, seed=0, max_num_seqs=256, max_num_batched_tokens=8192)
    log(f"engine up in {time.time() - t:.0f}s; generating {len(reqs)} responses")
    t = time.time()
    outs = llm.generate(reqs, sps, use_tqdm=False)
    rows = []
    for m, o in zip(meta, outs):
        c = o.outputs[0]
        rows.append(dict(m, response_ids=list(c.token_ids), finish_reason=c.finish_reason))
    with open(out + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(out + ".tmp", out)
    lens = [len(r["response_ids"]) for r in rows]
    rec = dict(student=C.STUDENT, n_responses=len(rows), n_problems=len(probs["pool_indices"]),
               problems_sha256=C.sha256_file(os.path.join(a.out_dir, "problems.json")),
               calset_sha256=C.sha256_file(out), mean_len=sum(lens) / len(lens), max_len=max(lens),
               n_tokens_response=sum(lens), n_tokens_total=sum(lens) + sum(len(r["prompt_ids"]) for r in rows),
               frac_truncated=sum(r["finish_reason"] == "length" for r in rows) / len(rows),
               n_empty=sum(1 for x in lens if x == 0), sampling=dict(temperature=1.0, top_p=1.0, top_k=0,
                                                                      max_tokens=C.CAL_MAX_TOKENS),
               seconds=round(time.time() - t, 1))
    C.write_json(os.path.join(a.out_dir, "calset_record.json"), rec)
    log(f"calset: {json.dumps({k: v for k, v in rec.items() if k != 'student'})}")
    return 0


def load_calset(d):
    """Rows of calset.jsonl, checked against calset_record.json's sha256 (every consumer calls this)."""
    rec = json.load(open(os.path.join(d, "calset_record.json")))
    p = os.path.join(d, "calset.jsonl")
    if C.sha256_file(p) != rec["calset_sha256"]:
        raise SystemExit(f"REFUSE: {p} sha256 differs from calset_record.json")
    rows = [json.loads(line) for line in open(p)]
    assert len(rows) == C.CAL_N_PROBLEMS * C.CAL_N_PER_PROBLEM == rec["n_responses"]
    return rows, rec


def selftest():
    order = list(range(20))
    excl = {1, 4}
    tiers = {2: {"AIME24": "flag"}, 3: {"MATH-500": "auto"}, 5: {"AMC23": "clean"}}
    kept, drops, walked = walk(order, excl, lambda i: tiers.get(i, {"AIME24": "clean"}), 5)
    assert kept == [0, 5, 6, 7, 8] and walked == 9, (kept, walked)
    assert drops == {"heldout": [1, 4], "eval_auto": [3], "eval_flag": [2]}
    assert C.request_seed(3, 1) == 20261008 + 31
    assert C.selftest() == 0
    print("[mo_calset] SELFTEST PASS")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("select", "generate"))
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    return cmd_select(a) if a.cmd == "select" else cmd_generate(a)


if __name__ == "__main__":
    sys.exit(main())
