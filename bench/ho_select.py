#!/usr/bin/env python3
"""The in-distribution held-out set HO (DAPO-Math-17k problems the training order never reaches), and its two prompt
sets: HO-DAPO (the rows as stored, the DAPO prompt asking for an "Answer:" line) and HO-JR (the same rows with the
prompt rewritten to the evaluation format: the problem text P, the DAPO wrapper stripped, + EVAL_SUFFIX).

`t3_acc2` is IMPORTED, not copied (norm_words, trigrams, strip_wrap, load_bench, DAPO_PREFIX, DAPO_SUFFIX,
EVAL_SUFFIX). The rule, in every detail:
  - order: torch.Generator().manual_seed(1); the first randperm(17,917) is discarded; the second is the order;
    candidates = the last NC=2,000 positions, walked from 17,916 down;
  - W = difflib.SequenceMatcher(None, autojunk=False).ratio(), 0 when real_quick_ratio() or quick_ratio() < 0.40 (and,
    against training rows and eval problems, when the ratio itself is < 0.40);
  - against training rows (position < 8,000): the candidate is seq2; S = |tri(cand) & tri(row)| / |tri(cand)|; W_pre /
    S_pre = maxima over rows (FIRST maximum in position order); best row = the W-argmax if W_pre >= S_pre, else the
    S-argmax;
  - against AIME24 / AIME25 / AMC23 / MATH-500: the eval problem is seq2; S_fwd over the eval problem's trigrams (decontam/1's
    direction), S_rev over the candidate's; per bench the maxima of W, S_fwd, S_rev and the best match = the problem with
    the largest max(W, S_fwd);
  - Tier(W, S): auto if W >= 0.75 or S >= 0.60; flag if W >= 0.50 or S >= 0.30; else clean. Training tier =
    Tier(W_pre, S_pre); a bench's eval tier = Tier(W, max(S_fwd, S_rev));
  - exclusion precedence (the first that applies is recorded): exact (normalised word sequence = a row at position
    < 8,000) > pre_auto > pre_flag_samegt (raw ground_truth string equality with the best row) > per bench in the order
    AIME24, AIME25, AMC23, MATH-500: ev_auto, else ev_flag_samegt (norm_ans equality with the best match's answer);
  - within-set duplicates: a survivor is dropped if, against any candidate ALREADY KEPT: W (candidate as seq2) >= 0.75,
    or S over the candidate's trigrams >= 0.60, or (W >= 0.50 or S >= 0.30) with equal RAW ground_truth strings;
    walking down from 17,916 keeps the higher-position copy;
  - HO = the first 1,000 survivors in decreasing order position (and the 300 / 500 prefixes).

Subcommands (CPU only; nothing is downloaded):
  order     torch: the order; optionally compared with a reference order (--ref-dir) and with a CSV of (pos,
            pool_index) rows (--per-prompt-csv), such as the per_prompt.csv that bench/rl_data.py summarize writes.
  neardup   the per-candidate scores of the last NC positions (multiprocessing); writes neardup_tail<NC>.json.
  select    the selection from a neardup file -> heldout_Rstar.json (+ clause counts, Chinese share, edge counts).
  check     tests 0-3 in the out dir. Against a reference build (--ref-dir: order_seed1_skip1.json,
            neardup_tail2000.json, heldout_Rstar.json and their SHA256SUMS): the per-candidate scores EXACTLY, then
            the 300/500/1000 sets' pool indices, positions and first positions; against reference clause counts
            (--ref-clauses): the clause counts; always: the selection's determinism. A test whose reference is not
            given is skipped, and one line says so.
  write     HO-DAPO and HO-JR parquets (row k = HO problem k, decreasing order position) + ho_record.json (the inputs
            of bench/ho_eval.py freeze: sha256s, pool indices, positions, drop counts by clause, Chinese share, edge
            counts, builder sha256, equality with the reference heldout_Rstar.json when one was checked). Refuses
            unless `check` passed in the same out dir.

    $PYTHON_BIN bench/ho_select.py order
    $PYTHON_BIN bench/ho_select.py neardup --out-dir $R/ho_build --procs 32
    $PYTHON_BIN bench/ho_select.py check   --out-dir $R/ho_build [--ref-dir <reference build dir>] [--ref-clauses <json>]
    $PYTHON_BIN bench/ho_select.py write   --out-dir $R/ho_build --ho-dir $R/ho
"""
import argparse
import collections
import difflib
import hashlib
import importlib.util
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("CLUSTER_ROOT", "/path/to/workdir")
TRAIN_DATA = f"{R}/workspace/datasets/dapo-math-17k.parquet"
TRAIN_DATA_SHA256 = "039f3afd689c846985bd2bf58e55a2210a8b08a1a9e1f60dbc07107a5f341925"
EVAL_DATA = f"{R}/workspace/datasets/test_data"
N_POOL = 17917
NC = 2000                                  # candidates: the last NC order positions
PMAX = 8000                                # training rows: positions < PMAX
CUTS = (800, 1600, 4000, 8000)             # reported cutoffs (only PMAX is read by the rule)
FLOOR = 0.40
BENCHES = ("AIME24", "AIME25", "AMC23", "MATH-500")
TH = dict(auto_word=0.75, auto_sh3=0.60, flag_word=0.50, flag_sh3=0.30)
SIZES = (300, 500, 1000)
REF_FILES = ("order_seed1_skip1.json", "neardup_tail2000.json", "heldout_Rstar.json")
FORMAT = "ho_select/v1"


def log(*a):
    print(f"[ho_select {time.strftime('%H:%M:%S')}]", *a, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _rj(p):
    with open(p) as f:
        return json.load(f)


def _wj(p, obj):
    with open(p + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(p + ".tmp", p)


def load_acc2():
    """t3_acc2 IMPORTED from the tree this file is in (refresh/ next to bench/)."""
    p = os.path.join(os.path.dirname(HERE), "refresh", "t3_acc2.py")
    spec = importlib.util.spec_from_file_location("t3_acc2", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, p


A, ACC2_PATH = load_acc2()


# ------------------------------------------------------------------------------------------------ the JR prompt
def jr_content_local(dapo_content):
    """P = strip_wrap(c, DAPO_PREFIX, DAPO_SUFFIX) verbatim (asserts the wrapper), JR = P + EVAL_SUFFIX."""
    return A.strip_wrap(dapo_content, A.DAPO_PREFIX, A.DAPO_SUFFIX) + A.EVAL_SUFFIX


def jr_rule():
    """The JR content function `write` uses, and its name for ho_record.json."""
    return jr_content_local, "jr_content_local"


# ------------------------------------------------------------------------------------------------ texts
def problem_text(dapo_content):
    return A.strip_wrap(dapo_content, A.DAPO_PREFIX, A.DAPO_SUFFIX)


def is_zh(text):
    """More than 5% CJK unified ideographs."""
    return sum('一' <= ch <= '鿿' for ch in text) / max(1, len(text)) > 0.05


def edge_counts(texts):
    """Three edge cases, counted on P (recorded in ho_record.json): outer whitespace (P != P.strip()), the string
    "boxed" inside P, an "Answer:" inside P."""
    return dict(outer_whitespace=sum(t != t.strip() for t in texts), boxed=sum("boxed" in t for t in texts),
                answer_colon=sum("Answer:" in t for t in texts))


class Pool:
    """The pool as the scoring and the selection see it: texts P, normalised words, trigrams, raw ground truths,
    exact-duplicate key groups, the order and its inverse. Built from plain lists (no pandas)."""

    def __init__(self, contents, ground_truths, order):
        assert len(contents) == len(ground_truths) == len(order)
        self.n = len(contents)
        self.texts = [problem_text(c) for c in contents]
        self.words = [A.norm_words(t) for t in self.texts]
        self.tri = [A.trigrams(w) for w in self.words]
        self.key = [" ".join(w) for w in self.words]
        self.gt = [str(g) for g in ground_truths]
        self.order = list(order)
        assert sorted(self.order) == list(range(self.n)), "order is not a permutation of the pool"
        self.pos_of = {i: p for p, i in enumerate(self.order)}
        self.grp = collections.defaultdict(list)
        for i, k in enumerate(self.key):
            self.grp[k].append(i)


class Evals:
    """Eval problems (words, trigrams, ids and answers), in bench order AIME24, AIME25, AMC23, MATH-500 and problem
    order within a bench."""

    def __init__(self, benches):
        """benches: {bench: [dict(i, content, gt)]} with content = the parquet's user content (EVAL_SUFFIX-wrapped)."""
        self.words, self.tri, self.ids, self.gt = [], [], [], {}
        for b in BENCHES:
            for p in benches[b]:
                w = A.norm_words(A.strip_wrap(p["content"], "", A.EVAL_SUFFIX))
                self.words.append(w)
                self.tri.append(A.trigrams(w))
                self.ids.append((b, p["i"]))
                self.gt[(b, p["i"])] = p["gt"]


# ------------------------------------------------------------------------------------------------ near-duplicate scores
def ratio(sm, other, floor=FLOOR):
    """W against `other` (0 below the floor): seq2 is already set on sm."""
    sm.set_seq1(other)
    if sm.real_quick_ratio() < floor or sm.quick_ratio() < floor:
        return 0.0
    r = sm.ratio()
    return r if r >= floor else 0.0


_W = {}                                    # worker globals (fork)


def _init_worker(pool, evals, pmax, cuts, floor):
    _W.update(pool=pool, evals=evals, pmax=pmax, cuts=cuts, floor=floor, evm=None)


def _evm():
    if _W["evm"] is None:
        ms = []
        for w in _W["evals"].words:
            sm = difflib.SequenceMatcher(None, autojunk=False)
            sm.set_seq2(w)
            ms.append(sm)
        _W["evm"] = ms
    return _W["evm"]


def score_candidate(p):
    """The scores of the candidate at position p: training rows at positions < pmax, every cut in `cuts`, the eval
    benches."""
    P, E, floor = _W["pool"], _W["evals"], _W["floor"]
    pre_idx = P.order[:_W["pmax"]]
    i = P.order[p]
    q, qt = P.words[i], P.tri[i]
    sm = difflib.SequenceMatcher(None, autojunk=False)
    sm.set_seq2(q)
    best = {c: [0.0, 0.0, None, None] for c in _W["cuts"]}
    for k, j in enumerate(pre_idx):
        w = ratio(sm, P.words[j], floor)
        s = len(qt & P.tri[j]) / len(qt) if qt else 0.0
        for c in _W["cuts"]:
            if k < c:
                bc = best[c]
                if w > bc[0]:
                    bc[0], bc[2] = w, k
                if s > bc[1]:
                    bc[1], bc[3] = s, k
    ev = {}
    for e, m in enumerate(_evm()):
        w = ratio(m, q, floor)
        s = len(E.tri[e] & qt) / len(E.tri[e]) if E.tri[e] else 0.0
        sr = len(qt & E.tri[e]) / len(qt) if qt else 0.0
        b = E.ids[e][0]
        cur = ev.get(b, [0.0, 0.0, 0.0, None])
        if max(w, s) > max(cur[0], cur[1]):
            cur[3] = E.ids[e][1]
        ev[b] = [max(cur[0], w), max(cur[1], s), max(cur[2], sr), cur[3]]
    return dict(pos=p, pool_index=i, n_words=len(q), pre={str(c): v for c, v in best.items()}, ev=ev)


def neardup(pool, evals, nc=NC, pmax=PMAX, cuts=CUTS, floor=FLOOR, procs=1):
    """Candidates = positions n-nc .. n-1 (ascending; the output keeps that order)."""
    assert pmax in cuts, "the rule reads the pmax cut"
    cand = list(range(pool.n - nc, pool.n))
    assert min(cand) >= pmax, "candidates overlap the training prefix"
    if procs <= 1:
        _init_worker(pool, evals, pmax, cuts, floor)
        return [score_candidate(p) for p in cand]
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(procs, initializer=_init_worker, initargs=(pool, evals, pmax, cuts, floor)) as pl:
        return pl.map(score_candidate, cand, chunksize=4)


# ------------------------------------------------------------------------------------------------ exclusion + selection
def tier(w, s, th=TH):
    return "auto" if (w >= th["auto_word"] or s >= th["auto_sh3"]) else (
        "flag" if (w >= th["flag_word"] or s >= th["flag_sh3"]) else "clean")


def eval_tier(v):
    """A bench's eval tier: Tier(W, max(S_fwd, S_rev))."""
    return tier(v[0], max(v[1], v[2]))


def best_row(pool, r, c):
    """The best training row: the W-argmax if W_pre >= S_pre (and it exists), else the S-argmax."""
    w, s, aw, as_ = r["pre"][c]
    k = aw if (w >= s and aw is not None) else (as_ if as_ is not None else aw)
    return pool.order[k] if k is not None else None


def norm_ans(x):
    """Eval answers: strip, spaces removed, numeric values float-normalised."""
    x = str(x).strip().replace(" ", "")
    try:
        return str(float(x)).rstrip("0").rstrip(".")
    except Exception:
        return x


def eval_answer_equal(eval_gt, cand_gt):
    return norm_ans(eval_gt) == norm_ans(cand_gt)


def annotate(pool, records, pmax=PMAX):
    """The per-record fields that the rule reads; returns the records sorted by decreasing position."""
    R = sorted(records, key=lambda r: -r["pos"])
    c = str(pmax)
    for r in R:
        i = r["pool_index"]
        r["exact_dup_pre"] = any(pool.pos_of[j] < pmax for j in pool.grp[pool.key[i]] if j != i)
        r["tier_pre"] = tier(r["pre"][c][0], r["pre"][c][1])
    return R


def excl(pool, evals, r, pmax=PMAX):
    """The exclusion clauses, in their precedence. Returns the clause or None (survives)."""
    i = r["pool_index"]
    if r["exact_dup_pre"]:
        return "exact"
    t = r["tier_pre"]
    if t == "auto":
        return "pre_auto"
    if t == "flag" and pool.gt[best_row(pool, r, str(pmax))] == pool.gt[i]:
        return "pre_flag_samegt"
    for b in BENCHES:                      # r["ev"] holds the benches in insertion order = BENCHES
        v = r["ev"][b]
        et = eval_tier(v)
        if et == "auto":
            return "ev_auto"
        if et == "flag" and v[3] is not None and eval_answer_equal(evals.gt[(b, v[3])], pool.gt[i]):
            return "ev_flag_samegt"
    return None


def within_dup(pool, i, kept, floor=FLOOR):
    """Within-set duplicates: candidate i against the candidates ALREADY KEPT."""
    q = pool.words[i]
    qt = A.trigrams(q)
    sm = difflib.SequenceMatcher(None, autojunk=False)
    sm.set_seq2(q)
    for j in kept:
        w = pool.words[j]
        sm.set_seq1(w)
        W = sm.ratio() if (sm.real_quick_ratio() >= floor and sm.quick_ratio() >= floor) else 0.0
        S = len(qt & A.trigrams(w)) / len(qt) if qt else 0.0
        if W >= 0.75 or S >= 0.60 or ((W >= 0.5 or S >= 0.3) and pool.gt[i] == pool.gt[j]):
            return True
    return False


def select(pool, evals, records, sizes=SIZES, pmax=PMAX):
    """The selection for each N: walk down from the last position, keep survivors of the exclusion that are not
    within-set duplicates of an already-kept candidate. Returns ({N: dict(pool_indices, positions, first_position)},
    details)."""
    R = annotate(pool, records, pmax)
    reasons = {r["pos"]: excl(pool, evals, r, pmax) for r in R}
    out, details = {}, {}
    for N in sizes:
        kept, used, wd, dropped_within = [], 0, 0, []
        for r in R:
            used += 1
            if reasons[r["pos"]] is None:
                if within_dup(pool, r["pool_index"], kept):
                    wd += 1
                    dropped_within.append(r["pos"])
                    continue
                kept.append(r["pool_index"])
                if len(kept) == N:
                    break
        out[N] = dict(pool_indices=kept, positions=[pool.pos_of[i] for i in kept], first_position=pool.n - used)
        details[N] = dict(n_kept=len(kept), used=used, within_set_dropped=wd, within_set_dropped_positions=dropped_within,
                          zh=sum(is_zh(pool.texts[i]) for i in kept))
    clauses = collections.Counter(reasons.values())
    details["clauses_over_candidates"] = {str(k): v for k, v in clauses.items()}
    return out, details


def ho_drop_counts(pool, evals, records, n_target=1000, pmax=PMAX):
    """Drop counts by clause among the positions walked to build the n_target set (recorded in ho_record.json)."""
    R = annotate(pool, records, pmax)
    kept, cnt = [], collections.Counter()
    for r in R:
        c = excl(pool, evals, r, pmax)
        if c is None and within_dup(pool, r["pool_index"], kept):
            c = "within_set"
        cnt[c or "kept"] += 1
        if c is None:
            kept.append(r["pool_index"])
            if len(kept) == n_target:
                break
    return dict(cnt)


# ------------------------------------------------------------------------------------------------ pool audit (descriptive)
def pool_audit(pool, prefixes=(800, 1600, 4000, 8000)):
    dups = [v for v in pool.grp.values() if len(v) > 1]
    raw = collections.defaultdict(list)
    for i, t in enumerate(pool.texts):
        raw[t].append(i)
    rd = [v for v in raw.values() if len(v) > 1]
    tail_hits = {}
    for P in prefixes:
        if P > pool.n:
            continue
        prek = {pool.key[i] for i in pool.order[:P]}
        tail_hits[str(P)] = sum(1 for p in range(P, pool.n) if pool.key[pool.order[p]] in prek)
    return dict(n=pool.n, distinct_normalized=len(pool.grp), dup_groups=len(dups), dup_rows=sum(len(v) for v in dups),
                raw_dup_groups=len(rd), raw_dup_rows=sum(len(v) for v in rd),
                zh_frac=round(sum(is_zh(t) for t in pool.texts) / pool.n, 4), prefix_tail_exact_hits=tail_hits)


# ------------------------------------------------------------------------------------------------ comparison with the reference
def compare_neardup(mine, ref):
    """Test 0: the per-candidate records equal the reference EXACTLY (positions, pool indices, word counts, every
    cut's W/S/argmax, every bench's W/S_fwd/S_rev/best match). Returns (ok, n_diff, first diffs)."""
    diffs = []
    if len(mine) != len(ref):
        return False, abs(len(mine) - len(ref)), [f"length {len(mine)} != {len(ref)}"]
    for a, b in zip(mine, ref):
        if a != b:
            diffs.append(dict(pos=b.get("pos"), mine=a, ref=b))
    return not diffs, len(diffs), diffs[:3]


def compare_selection(mine, ref):
    """Test 1: the 300/500/1000 sets' pool indices and positions, and the first positions."""
    out = {}
    for N in SIZES:
        a, b = mine.get(N) or mine.get(str(N)), ref.get(str(N)) or ref.get(N)
        out[str(N)] = dict(pool_indices=a["pool_indices"] == b["pool_indices"], positions=a["positions"] == b["positions"],
                           first_position=a["first_position"] == b["first_position"],
                           first_position_value=a["first_position"], ref_first_position=b["first_position"])
    ok = all(v["pool_indices"] and v["positions"] and v["first_position"] for v in out.values())
    return ok, out


# ------------------------------------------------------------------------------------------------ I/O (cluster: pandas, torch)
def training_order(n_pool, seed=1, skip=1):
    import torch
    g = torch.Generator()
    g.manual_seed(seed)
    for _ in range(skip):
        torch.randperm(n_pool, generator=g)
    return torch.randperm(n_pool, generator=g).tolist()


def load_dapo(path, allow_other=False):
    import pandas as pd
    sha = sha256_file(path)
    if sha != TRAIN_DATA_SHA256 and not allow_other:
        raise SystemExit(f"REFUSE: {path} sha256 {sha} is not the DAPO parquet ({TRAIN_DATA_SHA256})")
    df = pd.read_parquet(path)
    contents = []
    for p in df["prompt"]:
        assert len(p) == 1 and p[0]["role"] == "user", "DAPO row is not one user message"
        contents.append(p[0]["content"])
    gts = [str(rm["ground_truth"]) for rm in df["reward_model"]]
    return df, contents, gts, sha


def load_evals(data=EVAL_DATA):
    benches = {}
    for b in BENCHES:
        benches[b] = [dict(i=p["i"], content=p["prompt"][0]["content"], gt=p["gt"]) for p in A.load_bench(b, data)]
    return Evals(benches)


def build_pool(a):
    df, contents, gts, sha = load_dapo(a.data, a.allow_other_data)
    order = training_order(len(df)) if not a.order_json else _rj(a.order_json)
    return df, Pool(contents, gts, order), sha


def cmd_order(a):
    df, contents, gts, sha = load_dapo(a.data, a.allow_other_data)
    order = training_order(len(df))
    out = dict(n_pool=len(df), first16=order[:16], data_sha256=sha)
    if a.ref_dir:
        ref = _rj(os.path.join(a.ref_dir, "order_seed1_skip1.json"))
        out["equals_reference"] = order == ref
    else:
        log("order: no --ref-dir: the comparison with a reference order is skipped")
    if a.per_prompt_csv:
        import csv
        with open(a.per_prompt_csv) as f:
            pp = list(csv.DictReader(f))
        out["per_prompt_rows"] = len(pp)
        out["per_prompt_pool_index_mismatches"] = sum(order[int(r["pos"])] != int(r["pool_index"]) for r in pp)
    else:
        log("order: no --per-prompt-csv: the comparison with a training run's recorded order is skipped")
    print(json.dumps(out, indent=1))
    ok = out.get("equals_reference", True) and out.get("per_prompt_pool_index_mismatches", 0) == 0
    if not ok:
        raise SystemExit("order check FAILED")
    return out


def cmd_neardup(a):
    os.makedirs(a.out_dir, exist_ok=True)
    df, pool, sha = build_pool(a)
    evals = load_evals(a.eval_data)
    t = time.time()
    recs = neardup(pool, evals, procs=a.procs)
    p = os.path.join(a.out_dir, f"neardup_tail{NC}.json")
    with open(p + ".tmp", "w") as f:
        json.dump(recs, f)                 # no indent: byte-comparable with a reference build
    os.replace(p + ".tmp", p)
    log(f"neardup: {len(recs)} candidates in {time.time() - t:.0f}s -> {p} sha256 {sha256_file(p)}")
    return recs


def cmd_select(a, pool=None, evals=None):
    if pool is None:
        _, pool, _ = build_pool(a)
        evals = load_evals(a.eval_data)
    recs = _rj(a.neardup or os.path.join(a.out_dir, f"neardup_tail{NC}.json"))
    sel, det = select(pool, evals, recs)
    p = os.path.join(a.out_dir, "heldout_Rstar.json")
    with open(p + ".tmp", "w") as f:
        json.dump({N: v for N, v in sel.items()}, f)
    os.replace(p + ".tmp", p)
    det["drop_counts_1000"] = ho_drop_counts(pool, evals, recs)
    _wj(os.path.join(a.out_dir, "select_details.json"), det)
    log(f"select: {[(N, det[N]['n_kept'], sel[N]['first_position']) for N in SIZES]} clauses "
        f"{det['clauses_over_candidates']}")
    return sel, det


def cmd_check(a):
    """Tests 0-3 (run BEFORE `write`; `write` refuses without its PASS). Tests 0 and 1 and the order comparison need
    a reference build (--ref-dir), test 3 reference clause counts (--ref-clauses); each is skipped when its reference
    is not given. Test 2 (determinism) always runs."""
    _, pool, data_sha = build_pool(a)
    evals = load_evals(a.eval_data)
    mine = _rj(os.path.join(a.out_dir, f"neardup_tail{NC}.json"))
    sel, det = select(pool, evals, mine)
    cl = det["clauses_over_candidates"]
    # test 2 (determinism): the selection run twice gives the same json bytes
    sel2, _ = select(pool, evals, _rj(os.path.join(a.out_dir, f"neardup_tail{NC}.json")))
    t2_ok = json.dumps(sel2, sort_keys=True) == json.dumps(sel, sort_keys=True)
    res = dict(format=FORMAT, data_sha256=data_sha, test2=dict(ok=t2_ok),
               builder=os.path.abspath(__file__), builder_sha256=sha256_file(os.path.abspath(__file__)),
               t3_acc2=ACC2_PATH, t3_acc2_sha256=sha256_file(ACC2_PATH))
    oks = [t2_ok]
    ref = a.ref_dir
    if ref:
        sums = {}
        with open(os.path.join(ref, "SHA256SUMS")) as f:
            for line in f:
                h, n = line.split()
                sums[n] = h
        ref_ok = {n: sha256_file(os.path.join(ref, n)) == sums[n] for n in REF_FILES}
        t0_ok, t0_n, t0_first = compare_neardup(mine, _rj(os.path.join(ref, "neardup_tail2000.json")))
        order_ok = pool.order == _rj(os.path.join(ref, "order_seed1_skip1.json"))
        t1_ok, t1 = compare_selection(sel, _rj(os.path.join(ref, "heldout_Rstar.json")))
        res.update(reference_dir=os.path.abspath(ref), reference_sha256_ok=ref_ok, order_equals_reference=order_ok,
                   test0=dict(ok=t0_ok, n_diff=t0_n, first=t0_first), test1=dict(ok=t1_ok, sets=t1))
        oks += [all(ref_ok.values()), order_ok, t0_ok, t1_ok]
    else:
        log("check: no --ref-dir: tests 0 and 1 and the order comparison against a reference build are skipped")
        res.update(reference_dir=None, test0=dict(ok=None, skipped=True), test1=dict(ok=None, skipped=True))
    if a.ref_clauses:
        expected = _rj(a.ref_clauses)
        t3_ok = all(cl.get(k, 0) == v for k, v in expected.items())
        res["test3"] = dict(ok=t3_ok, clauses=cl, expected=expected)
        oks.append(t3_ok)
    else:
        log("check: no --ref-clauses: test 3 (clause counts against reference counts) is skipped")
        res["test3"] = dict(ok=None, skipped=True, clauses=cl)
    res["PASS"] = bool(all(oks))
    _wj(os.path.join(a.out_dir, "check.json"), res)
    print(json.dumps({k: v for k, v in res.items() if k != "test0"}
                     | {"test0_ok": res["test0"]["ok"], "test0_n_diff": res["test0"].get("n_diff")},
                     indent=1, default=str))
    if not res["PASS"]:
        raise SystemExit("check FAILED: `write` refuses until it passes")
    return res


def write_prompt_sets(df, pool_indices, jr_fn, check_local=True):
    """HO-DAPO = the rows as stored; HO-JR = the same rows with only prompt[0].content replaced by the JR content.
    Row k = HO problem k (the selection's order: decreasing order position)."""
    ho_dapo = df.iloc[list(pool_indices)].reset_index(drop=True)
    ho_jr = ho_dapo.copy()
    new_prompts = []
    for k, p in enumerate(ho_dapo["prompt"]):
        msgs = [dict(m) for m in p]
        assert len(msgs) == 1 and msgs[0]["role"] == "user"
        c = msgs[0]["content"]
        jr = jr_fn(c)
        if check_local and jr != jr_content_local(c):
            raise SystemExit(f"REFUSE: the JR rule and the local definition differ at HO row {k}")
        msgs[0]["content"] = jr
        new_prompts.append(msgs)
    ho_jr["prompt"] = new_prompts
    return ho_dapo, ho_jr


def cmd_write(a):
    chk_p = os.path.join(a.out_dir, "check.json")
    if not os.path.exists(chk_p) or not _rj(chk_p).get("PASS"):
        raise SystemExit("REFUSE: `check` has not PASSED in this out dir")
    chk = _rj(chk_p)
    if chk["builder_sha256"] != sha256_file(os.path.abspath(__file__)):
        raise SystemExit("REFUSE: the builder changed since `check`")
    df, pool, data_sha = build_pool(a)
    evals = load_evals(a.eval_data)
    recs = _rj(os.path.join(a.out_dir, f"neardup_tail{NC}.json"))
    sel, det = select(pool, evals, recs)
    ho = sel[1000]
    jr_fn, jr_src = jr_rule()
    ho_dapo, ho_jr = write_prompt_sets(df, ho["pool_indices"], jr_fn)
    os.makedirs(a.ho_dir, exist_ok=True)
    paths = dict(dapo=os.path.join(a.ho_dir, "ho_dapo.parquet"), jr=os.path.join(a.ho_dir, "ho_jr.parquet"))
    for k, d in (("dapo", ho_dapo), ("jr", ho_jr)):
        if os.path.exists(paths[k]):
            raise SystemExit(f"REFUSE: {paths[k]} exists")
        d.to_parquet(paths[k] + ".tmp.parquet", index=False)
        os.replace(paths[k] + ".tmp.parquet", paths[k])
    # test 5 on the written files
    import pandas as pd
    rd, rj = pd.read_parquet(paths["dapo"]), pd.read_parquet(paths["jr"])
    t5 = (list(rd.columns) == list(rj.columns) and len(rd) == len(rj) == 1000
          and all(rd[c].astype(str).tolist() == rj[c].astype(str).tolist() for c in rd.columns if c != "prompt")
          and all(rj["prompt"][k][0]["content"] == problem_text(rd["prompt"][k][0]["content"]) + A.EVAL_SUFFIX
                  for k in range(len(rd))))
    texts = [pool.texts[i] for i in ho["pool_indices"]]
    import pyarrow
    rec = dict(format=FORMAT, what="inputs of bench/ho_eval.py freeze: HO-DAPO and HO-JR",
               paths=paths, sha256={k: sha256_file(p) for k, p in paths.items()},
               pool_indices=ho["pool_indices"], positions=ho["positions"], first_position=ho["first_position"],
               n=len(ho["pool_indices"]), drop_counts_by_clause=ho_drop_counts(pool, evals, recs),
               chinese_share=sum(is_zh(t) for t in texts) / len(texts), edge_counts=edge_counts(texts),
               equals_reference_heldout_Rstar=chk["test1"]["ok"], check_json_sha256=sha256_file(chk_p),
               builder_sha256=sha256_file(os.path.abspath(__file__)), t3_acc2_sha256=sha256_file(ACC2_PATH),
               jr_rule=jr_src, test5_jr_equals_dapo_except_content=bool(t5), data_sha256=data_sha,
               pandas=pd.__version__, pyarrow=pyarrow.__version__,
               note="token-id sha256 per prompt set: ho_eval.py `freeze` (needs the student's tokenizer)")
    _wj(os.path.join(a.ho_dir, "ho_record.json"), rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("pool_indices", "positions")}, indent=1))
    if not t5:
        raise SystemExit("test 5 FAILED")
    return rec


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--data", default=TRAIN_DATA)
        p.add_argument("--allow-other-data", action="store_true")
        p.add_argument("--eval-data", default=EVAL_DATA)
        p.add_argument("--order-json", default="", help="use a stored order instead of torch (tests only)")
        p.add_argument("--out-dir", default="")

    p = sub.add_parser("order")
    common(p)
    p.add_argument("--ref-dir", default="", help="a reference build dir holding order_seed1_skip1.json")
    p.add_argument("--per-prompt-csv", default="",
                   help="optional: a CSV with integer columns pos (training-order position) and pool_index (row of the "
                        "DAPO parquet); other columns are ignored, and every row must satisfy order[pos] == pool_index. "
                        "bench/rl_data.py summarize writes such a CSV (per_prompt.csv)")
    p = sub.add_parser("neardup")
    common(p)
    p.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p = sub.add_parser("select")
    common(p)
    p.add_argument("--neardup", default="")
    p = sub.add_parser("check")
    common(p)
    p.add_argument("--ref-dir", default="", help="a reference build dir (order, neardup scores, sets, SHA256SUMS)")
    p.add_argument("--ref-clauses", default="", help="a JSON file {clause: count} of reference clause counts (test 3)")
    p = sub.add_parser("write")
    common(p)
    p.add_argument("--ho-dir", required=True)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    return dict(order=cmd_order, neardup=cmd_neardup, select=cmd_select, check=cmd_check, write=cmd_write)[a.cmd](a)


if __name__ == "__main__":
    main()
