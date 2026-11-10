#!/usr/bin/env python3
"""Accuracy on the held-out set HO, sampled for ONE checkpoint on ONE prompt set and graded by the either-format
reward (custom_reward.specopd_reward_g2) in a worker thread.

What a sample is (identical for every checkpoint of both objectives):
  - prompt: row k of the prompt set (HO-DAPO: the DAPO training prompt; HO-JR: the same problem in the evaluation
    prompt format; both written by bench/ho_select.py, sha256 in the prompt-set record that `freeze` writes), through
    the CHECKPOINT DIR's own tokenizer and chat template,
    `apply_chat_template(chat, tokenize=False, add_generation_prompt=True)` then `tok(text, add_special_tokens=False)`;
    the token ids of the whole prompt set must hash to the record's value (so every checkpoint on a set sees identical
    ids); <= 2,048 tokens;
  - sampling: vLLM bf16, T 1.0, top_p 1, top_k -1, repetition_penalty 1, max_tokens 16,384, NO speculative decoding,
    `logprobs=0` (only the sampled token's logprob: its per-sample sum and count are kept), one request per sample;
  - seed of sample j of HO problem k (k = 0..999 in HO order): 100,000 + 16k + j, the same for both prompt sets;
  - decode with skip_special_tokens=True (as verl's reward manager does).
Grades (a one-worker ThreadPoolExecutor, the mode verl's naive reward manager uses):
  - g2       the primary grade: custom_reward.specopd_reward_g2.reward_func in a WORKER THREAD (math_verify fallback
             off);
  - g2m      descriptive: the same reward in the MAIN thread;
  - boxonly  descriptive: custom_reward.specopd_reward.reward_func (the \\boxed-only reward) in a worker thread;
  - ext      descriptive: the g2 grade, except that when g2 finds no \\boxed and no Answer: line
             (answer_source "none"), the EXTENDED answer-line extractor is tried (label `Answer` or `答案`, optional
             markdown `*`; separator `:` or `：`; the value on the same line, or on the next non-empty line when the
             label line's value is empty after stripping `*` and spaces; the last such line after </think>, else
             anywhere) and its value graded by the same ttrl_math grader (`grade(pred, gt, False)`), in a worker
             thread.
Per sample (samples.jsonl stores response token ids, never text): the four grades, the answer source (boxed /
answer_line / none; ext adds ext_line), finished (finish_reason "stop", i.e. EOS before 16,384) or truncated and the
length, </think> present and whether the credited answer starts after the last </think>, strict credit per grade
(correct, finished, </think> present, credited answer after it), the sampled-token NLL sum and token count.
Per checkpoint (summary.json, per_problem.csv): acc, acc_strict, H_HO (mean over problems of the per-problem mean of the
per-sample mean NLL), trunc, answer-source shares, fin, g (gradeable share among finished), a (correct share among
finished gradeable), the only-extended share, each grade mode's counts.

Guards: the prompt set's sha256 must equal the record's; RL checkpoints are sampled on HO-JR and OPD checkpoints on
HO-DAPO, the base on both. Each objective is evaluated on the prompt format it trains on. OPD trains on the
DAPO-Math-17k prompts as stored. The RL runs train on the JR format: the problem text with the DAPO wrapper removed,
followed by t3_acc2.EVAL_SUFFIX (the instruction the evaluation benchmarks use). bench/rl_data.py writes their
training data: its `jr` subcommand rewrites dapo-math-17k.parquet into dapo-math-17k-jr.parquet (row-aligned, only the
user message's content replaced) and its `subset` subcommand writes the RL training subset. HO-JR and
dapo-math-17k-jr.parquet apply one rule, bench/ho_select.py's jr_content_local (P = t3_acc2.strip_wrap(content,
DAPO_PREFIX, DAPO_SUFFIX), JR = P + EVAL_SUFFIX), so an RL checkpoint sees held-out prompts built exactly like its
training prompts. The log never prints an accuracy.

    $PYTHON_BIN bench/ho_eval.py resolve --job <training job id> --step 100 --runs-root <dir>   # exactly one match
    $PYTHON_BIN bench/ho_eval.py freeze  --ho-record <ho_record.json> --tokenizer <base> --out <ho_freeze.json> \
        --resolve-jobs <training job id> ... --runs-root <dir>
    $PYTHON_BIN bench/ho_eval.py gen --freeze-ho <ho_freeze.json> --prompt dapo --objective opd --arm <arm label> \
        --step 100 --ckpt <hf dir> --n 8 --j0 0 [--block]
    $PYTHON_BIN bench/ho_eval.py merge --freeze-ho <...> --prompt jr --arm base --step 0            # base j-blocks
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("CLUSTER_ROOT", "/path/to/workdir")
HO_ROOT = f"{R}/ho"                         # bench/ho_select.py write --ho-dir
STUDENT = f"{R}/hf/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60"
FORMAT = "ho_eval/v1"
N_HO = 1000
SEED_BASE, SEED_STRIDE, N_MAX = 100000, 16, 16
TEMPERATURE, TOP_P, TOP_K, MAX_RESP, MAX_PROMPT = 1.0, 1.0, -1, 16384, 2048
PROMPT_SETS = ("dapo", "jr")
OBJECTIVES = ("rl", "opd", "base")
OWN_PROMPT = {"rl": "jr", "opd": "dapo"}
BASE_BLOCKS = ((0, 4), (4, 4), (8, 4), (12, 4))
GRADES = ("g2", "g2m", "boxonly", "ext")
CHUNK = 100                                 # problems per generate call

T0 = time.time()


def log(*a):
    print(f"[ho_eval {time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s]", *a, flush=True)


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
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(p + ".tmp", p)


# ------------------------------------------------------------------------------------------------ seeds + guards
def ho_seed(k, j):
    return SEED_BASE + SEED_STRIDE * k + j


def check_blocks(n, j0):
    if n not in (4, 8) or j0 < 0 or j0 + n > N_MAX or j0 % n:
        raise SystemExit(f"REFUSE: n={n} j0={j0}: n must be 4 or 8 and j0 a multiple of n with j0 + n <= 16")


def prompt_guard(objective, prompt):
    """RL -> HO-JR, OPD -> HO-DAPO (each objective's own training prompt format); the base on both."""
    if objective not in OBJECTIVES or prompt not in PROMPT_SETS:
        raise SystemExit(f"REFUSE: objective {objective!r} / prompt {prompt!r}")
    if objective == "base":
        return "base"
    if OWN_PROMPT[objective] != prompt:
        raise SystemExit(f"REFUSE: a {objective.upper()} checkpoint is sampled on HO-{OWN_PROMPT[objective].upper()} "
                         f"(its own training prompt), not HO-{prompt.upper()}")
    return "own"


def freeze_entry(freeze, prompt, path):
    """The prompt-set record's entry for the prompt set; refuses unless the file on disk hashes to it."""
    pin = freeze.get("prompt_sets", {}).get(prompt)
    if not pin:
        raise SystemExit(f"REFUSE: the prompt-set record has no {prompt} prompt set")
    sha = sha256_file(path)
    if sha != pin["sha256"]:
        raise SystemExit(f"REFUSE: {path} sha256 {sha} != the record's {prompt} sha256 {pin['sha256']}")
    return pin


# ------------------------------------------------------------------------------------------------ data + prompts
def _plain(x):
    if hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
        x = x.tolist()
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def load_prompt_set(path):
    """Row k = HO problem k: dict(k, chat, data_source, ground_truth, extra_info). Parquet, or a .jsonl of the same fields
    (tests without the cluster data)."""
    rows = []
    if path.endswith(".jsonl"):
        with open(path) as f:
            for k, line in enumerate(f):
                r = json.loads(line)
                rows.append(dict(k=k, chat=r["prompt"], data_source=r["data_source"],
                                 ground_truth=str(r["reward_model"]["ground_truth"]), extra_info=r.get("extra_info") or {}))
        return rows
    import pandas as pd
    df = pd.read_parquet(path)
    for k in range(len(df)):
        row = df.iloc[k]
        rm = _plain(row["reward_model"])
        ei = _plain(row["extra_info"]) if "extra_info" in row.index else {}
        rows.append(dict(k=k, chat=[dict(m) for m in _plain(row["prompt"])], data_source=row["data_source"],
                         ground_truth=str(rm["ground_truth"]), extra_info=ei if isinstance(ei, dict) else {}))
    return rows


def render_prompt_ids(tok, rows, max_prompt=MAX_PROMPT):
    out = []
    for r in rows:
        text = tok.apply_chat_template(r["chat"], tokenize=False, add_generation_prompt=True)
        ids = list(tok(text, add_special_tokens=False)["input_ids"])
        if len(ids) > max_prompt:
            raise SystemExit(f"REFUSE: HO row {r['k']} renders to {len(ids)} > {max_prompt} prompt tokens")
        out.append(ids)
    return out


def tokids_sha256(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def resolve_checkpoint(job, step, root):
    """tim_eval2.sbatch's lookup: <root>/*/checkpoints/specopd/*_<JOBID>/global_step_<S>/actor/huggingface, exactly one
    match holding config.json. root holds the repository checkouts the training jobs ran from (the launcher writes
    checkpoints/specopd/sopd_<arm>_<job id> under the tree it runs from). Job ids are unique, so 0 or > 1 matches
    refuse."""
    import glob
    pat = os.path.join(root, "*", "checkpoints", "specopd", f"*_{job}", f"global_step_{step}", "actor", "huggingface")
    hits = sorted(p for p in glob.glob(pat) if os.path.isfile(os.path.join(p, "config.json")))
    if len(hits) != 1:
        raise SystemExit(f"REFUSE: need exactly 1 checkpoint for job {job} step {step}, found {len(hits)}: {hits}")
    return hits[0]


# ------------------------------------------------------------------------------------------------ answers + grades
G2_ANS_RE = re.compile(r"(?im)^\s*\**\s*answer\s*\**\s*:\s*(.+?)\s*$")          # = specopd_reward_g2.ANS_RE (asserted)
EXT_RE = re.compile(r"^\s*\**\s*(?:answer|答案)\s*\**\s*[:：](.*)$", re.IGNORECASE)


def _clean(x):
    """specopd_reward_g2.answer_line's value cleanup."""
    x = x.strip().strip("*").strip()
    x = re.sub(r"^\$+|\$+$", "", x).strip().rstrip(".").strip()
    return x or None


def _ext_find(segment):
    lines = segment.split("\n")
    found = None
    for i, ln in enumerate(lines):
        m = EXT_RE.match(ln)
        if not m:
            continue
        v = m.group(1).strip().strip("*").strip()
        if not v:
            v = next((nx.strip() for nx in lines[i + 1:] if nx.strip()), "")
        if _clean(v) if v else None:
            found = v
    return found


def ext_answer(text):
    """The extended answer-line extractor: (value, after_think) or (None, False)."""
    tail = text.split("</think>")[-1]
    v = _ext_find(tail)
    after = v is not None and "</think>" in text
    if v is None:
        v = _ext_find(text)
    if v is None:
        return None, False
    return _clean(v), after


def credited_after_think(text, source):
    """Whether the credited answer starts after the last </think> (False without </think>)."""
    if "</think>" not in text:
        return False
    tail = text.split("</think>")[-1]
    if source == "boxed":
        return text.rfind("\\boxed") > text.rfind("</think>")
    if source == "answer_line":
        return bool(G2_ANS_RE.findall(tail))
    return False


class Graders:
    """The rewards (this tree's custom_reward/). Tests inject stubs with the same three callables."""

    def __init__(self, g2=None, boxonly=None, value_grade=None, code_root=None):
        if g2 is None:
            root = code_root or next((r for r in (os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE)))
                                      if os.path.isfile(os.path.join(r, "custom_reward", "specopd_reward_g2.py"))), None)
            if root is None:
                raise SystemExit("REFUSE: custom_reward/ not found next to this tree (repository root)")
            sys.path.insert(0, root)
            from custom_reward.specopd_reward_g2 import reward_func as g2, ANS_RE
            from custom_reward.specopd_reward import reward_func as boxonly
            from custom_reward.ttrl_math import grade
            if ANS_RE.pattern != G2_ANS_RE.pattern:
                raise SystemExit("REFUSE: specopd_reward_g2.ANS_RE differs from the harness copy")
            value_grade = (lambda pred, gt: bool(grade(pred, str(gt), False)))
            cr = os.path.join(root, "custom_reward")
            self.hashes = dict(specopd_reward_g2=sha256_file(os.path.join(cr, "specopd_reward_g2.py")),
                               specopd_reward=sha256_file(os.path.join(cr, "specopd_reward.py")),
                               ttrl_math_init=sha256_file(os.path.join(cr, "ttrl_math", "__init__.py")),
                               math_utils=sha256_file(os.path.join(cr, "ttrl_math", "math_utils.py")))
        else:
            self.hashes = dict(stub=True)
        self.g2, self.boxonly, self.value_grade = g2, boxonly, value_grade


def _score(res):
    return float(res["score"]) if isinstance(res, dict) else float(res)


def grade_one(r, text, gt, ds, ei, graders, main_thread):
    """All grades of one sample. Called in the worker thread (main_thread=False: g2, boxonly, ext) and in the main
    thread (main_thread=True: g2m)."""
    if main_thread:
        try:
            r["g2m"] = _score(graders.g2(ds, text, gt, extra_info=dict(ei)))
        except Exception as e:
            r["g2m"], r["g2m_error"] = 0.0, repr(e)[:200]
        return
    try:
        res = graders.g2(ds, text, gt, extra_info=dict(ei))
        r["g2"], r["src"] = _score(res), (res.get("answer_source", "none") if isinstance(res, dict) else "none")
    except Exception as e:
        r["g2"], r["src"], r["g2_error"] = 0.0, "none", repr(e)[:200]
    try:
        r["boxonly"] = _score(graders.boxonly(ds, text, gt, extra_info=dict(ei)))
    except Exception as e:
        r["boxonly"], r["boxonly_error"] = 0.0, repr(e)[:200]
    r["think"] = "</think>" in text
    r["after"] = credited_after_think(text, r["src"])
    r["ext"], r["ext_src"], r["ext_after"] = r["g2"], r["src"], r["after"]
    r["ext_only"] = False
    if r["src"] == "none":                    # g2 found no \boxed and no Answer: line: try the extended extractor
        v, after = ext_answer(text)
        if v is not None:
            r["ext_only"] = True
            r["ext_src"], r["ext_after"] = "ext_line", after
            try:
                r["ext"] = 1.0 if graders.value_grade(v, gt) else 0.0
            except Exception as e:
                r["ext"], r["ext_error"] = 0.0, repr(e)[:200]
    r["boxonly_src"] = "boxed" if r["src"] == "boxed" else "none"


def finalize_row(r):
    r["fin"] = r["finish_reason"] == "stop"
    r["trunc"] = r["finish_reason"] == "length"
    for m in GRADES:
        src = r["ext_src"] if m == "ext" else (r["boxonly_src"] if m == "boxonly" else r["src"])
        after = r["ext_after"] if m == "ext" else r["after"]
        r[f"{m}_strict"] = bool(r[m] > 0.5 and r["fin"] and r["think"] and after and src != "none")


def grade_rows(rows, prompts, tok, graders, executor):
    """Every reward call of the primary and descriptive worker-thread grades runs inside the one-worker executor
    (`executor.submit(work).result()`); g2m in the main thread."""
    texts = [tok.decode(r["response_ids"], skip_special_tokens=True) for r in rows]

    def work():
        assert threading.current_thread() is not threading.main_thread()
        for r, t in zip(rows, texts):
            p = prompts[r["k"]]
            grade_one(r, t, p["ground_truth"], p["data_source"], p["extra_info"], graders, main_thread=False)
    executor.submit(work).result()
    assert threading.current_thread() is threading.main_thread()
    for r, t in zip(rows, texts):
        p = prompts[r["k"]]
        grade_one(r, t, p["ground_truth"], p["data_source"], p["extra_info"], graders, main_thread=True)
        finalize_row(r)


# ------------------------------------------------------------------------------------------------ aggregation
PP_FIELDS = (["k", "n", "n_fin", "n_trunc", "n_think", "nll_mean_sum", "n_src_boxed", "n_src_answer_line", "n_src_none",
              "n_fin_ext_only", "n_correct_g2_boxed", "n_correct_g2_answer_line"]
             + [f"{x}_{m}" for m in GRADES for x in ("n_correct", "n_strict", "n_fin_gradeable",
                                                       "n_correct_fin_gradeable")])


def per_problem(rows, n_expected):
    """rows: graded samples. One record per problem k with exactly n_expected samples (else refuse)."""
    by = {}
    for r in rows:
        by.setdefault(int(r["k"]), []).append(r)
    out = []
    for k in sorted(by):
        v = by[k]
        if len(v) != n_expected or len({x["j"] for x in v}) != n_expected:
            raise ValueError(f"problem {k}: {len(v)} samples, want {n_expected} distinct j")
        d = dict(k=k, n=len(v), n_fin=sum(x["fin"] for x in v), n_trunc=sum(x["trunc"] for x in v),
                 n_think=sum(x["think"] for x in v),
                 nll_mean_sum=sum(x["nll_sum"] / x["n_tok"] if x["n_tok"] else 0.0 for x in v),
                 n_src_boxed=sum(x["src"] == "boxed" for x in v), n_src_answer_line=sum(x["src"] == "answer_line" for x in v),
                 n_src_none=sum(x["src"] == "none" for x in v), n_fin_ext_only=sum(x["fin"] and x["ext_only"] for x in v),
                 n_correct_g2_boxed=sum(x["src"] == "boxed" and x["g2"] > 0.5 for x in v),
                 n_correct_g2_answer_line=sum(x["src"] == "answer_line" and x["g2"] > 0.5 for x in v))
        for m in GRADES:
            def src(x):
                return x["ext_src"] if m == "ext" else (x["boxonly_src"] if m == "boxonly" else x["src"])
            d[f"n_correct_{m}"] = sum(x[m] > 0.5 for x in v)
            d[f"n_strict_{m}"] = sum(x[f"{m}_strict"] for x in v)
            d[f"n_fin_gradeable_{m}"] = sum(x["fin"] and src(x) != "none" for x in v)
            d[f"n_correct_fin_gradeable_{m}"] = sum(x["fin"] and src(x) != "none" and x[m] > 0.5 for x in v)
        out.append(d)
    return out


def summarize_pp(pp):
    """Checkpoint-level quantities; every share pools samples over problems."""
    def s(f):
        return sum(d[f] for d in pp)
    N = s("n")
    out = dict(n_problems=len(pp), n_samples=N,
               H_HO=sum(d["nll_mean_sum"] / d["n"] for d in pp) / len(pp),
               trunc=s("n_trunc") / N, fin=s("n_fin") / N, think_share=s("n_think") / N,
               answer_source=dict(boxed=s("n_src_boxed") / N, answer_line=s("n_src_answer_line") / N,
                                  none=s("n_src_none") / N),
               only_extended_share_of_finished=(s("n_fin_ext_only") / s("n_fin")) if s("n_fin") else None)
    for m in GRADES:
        key = "" if m == "g2" else f"_{m}"
        out[f"acc{key}"] = sum(d[f"n_correct_{m}"] / d["n"] for d in pp) / len(pp)
        out[f"acc_strict{key}"] = sum(d[f"n_strict_{m}"] / d["n"] for d in pp) / len(pp)
        out[f"g{key}"] = (s(f"n_fin_gradeable_{m}") / s("n_fin")) if s("n_fin") else None
        out[f"a{key}"] = (s(f"n_correct_fin_gradeable_{m}") / s(f"n_fin_gradeable_{m}")) if s(f"n_fin_gradeable_{m}") else None
    return out


# ------------------------------------------------------------------------------------------------ outputs
def out_dir_for(a):
    d = os.path.join(a.ho_root, f"{a.arm}_s{a.step}_{a.prompt}")
    if a.block:
        d = os.path.join(d, "blocks", f"j{a.j0:02d}-{a.j0 + a.n - 1:02d}")
    return d


def chunk_path(d, lo, hi):
    return os.path.join(d, "chunks", f"c{lo:04d}_{hi:04d}.jsonl")


def read_samples(d):
    rows = []
    cdir = os.path.join(d, "chunks")
    for f in sorted(os.listdir(cdir)) if os.path.isdir(cdir) else []:
        if re.fullmatch(r"c\d{4}_\d{4}\.jsonl", f):
            with open(os.path.join(cdir, f)) as fh:
                rows.extend(json.loads(line) for line in fh)
    return rows


def write_outputs(d, rows, n, meta):
    """samples.jsonl (token ids + fields, no text), per_problem.csv, summary.json. Returns the summary."""
    rows = sorted(rows, key=lambda r: (r["k"], r["j"]))
    with open(os.path.join(d, "samples.jsonl.tmp"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    os.replace(os.path.join(d, "samples.jsonl.tmp"), os.path.join(d, "samples.jsonl"))
    pp = per_problem(rows, n)
    with open(os.path.join(d, "per_problem.csv.tmp"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PP_FIELDS)
        w.writeheader()
        w.writerows(pp)
    os.replace(os.path.join(d, "per_problem.csv.tmp"), os.path.join(d, "per_problem.csv"))
    summ = dict(meta, **summarize_pp(pp), status="COMPLETE" if len(pp) == meta["n_problems_expected"] else "PARTIAL")
    _wj(os.path.join(d, "summary.json"), summ)
    return summ


def config_key(a, pin, ckpt):
    return dict(format=FORMAT, ckpt=os.path.realpath(ckpt), prompt=a.prompt, prompt_sha256=pin["sha256"],
                tokids_sha256=pin["tokids_sha256"], objective=a.objective, arm=a.arm, step=a.step, n=a.n, j0=a.j0,
                temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, max_tokens=MAX_RESP,
                seed_base=SEED_BASE, seed_stride=SEED_STRIDE, logprobs=0, spec="none", dtype="bfloat16")


def sampling_kwargs(k, j):
    return dict(temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, repetition_penalty=1.0, max_tokens=MAX_RESP,
                seed=ho_seed(k, j), logprobs=0)


def vllm_engine_factory(model, gpu_util, max_num_seqs, meta):
    """The engine (bf16, gpu util, max_model_len = 2048 + 16384 + 1, seed 0, prefix caching, no SD), with logprobs=0:
    the sampled token's logprob per position."""
    def factory():
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        meta["vllm"] = vllm.__version__
        llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=gpu_util, max_model_len=MAX_PROMPT + MAX_RESP + 1,
                  max_num_seqs=max_num_seqs, seed=0, enable_prefix_caching=True, enforce_eager=False)

        def run(reqs):
            outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p, _ in reqs],
                                [SamplingParams(**kw) for _, kw in reqs], use_tqdm=False)
            res = []
            for o in outs:
                c = o.outputs[0]
                ids = list(c.token_ids)
                lps = c.logprobs or []
                if len(lps) != len(ids):
                    raise RuntimeError(f"logprobs for {len(lps)} of {len(ids)} tokens")
                nll = -sum(lp[t].logprob for lp, t in zip(lps, ids))
                res.append(dict(token_ids=ids, finish_reason=c.finish_reason, nll_sum=nll))
            return res
        return run
    return factory


def cmd_gen(a, engine_factory=None, tok=None, graders=None):
    """engine_factory / tok / graders: injection points for tests without a GPU (the real run builds vLLM, the
    checkpoint dir's tokenizer and this tree's rewards)."""
    check_blocks(a.n, a.j0)
    mode = prompt_guard(a.objective, a.prompt)
    if a.block and a.objective != "base":
        raise SystemExit("REFUSE: --block is the base's four-job split only")
    if a.objective == "base" and a.n != 4:
        raise SystemExit("REFUSE: the base runs as four n=4 j-blocks (0-3, 4-7, 8-11, 12-15), merged by `merge`")
    if a.objective == "base" and not a.block:
        raise SystemExit("REFUSE: a base j-block needs --block (merged by `merge`)")
    freeze = _rj(a.freeze_ho)
    path = a.data or freeze["prompt_sets"][a.prompt]["path"]
    pin = freeze_entry(freeze, a.prompt, path)
    ckpt = STUDENT if a.ckpt == "base" else a.ckpt
    if a.objective == "base" and a.ckpt != "base" and os.path.realpath(a.ckpt) != os.path.realpath(STUDENT):
        raise SystemExit("REFUSE: objective base needs CKPT=base")
    d = out_dir_for(a)
    os.makedirs(os.path.join(d, "chunks"), exist_ok=True)
    key = config_key(a, pin, ckpt)
    meta_p = os.path.join(d, "meta.json")
    if os.path.exists(meta_p) and _rj(meta_p).get("config_key") != key:
        raise SystemExit(f"REFUSE: {meta_p} exists with a different config; not mixing samples")
    rows_all = load_prompt_set(path)
    if len(rows_all) != a.n_problems:
        raise SystemExit(f"REFUSE: {path} has {len(rows_all)} rows, want {a.n_problems}")
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(ckpt)        # the checkpoint dir's own tokenizer + template
    ids = render_prompt_ids(tok, rows_all)
    th = tokids_sha256(ids)
    if th != pin["tokids_sha256"]:
        raise SystemExit(f"REFUSE: prompt token ids of {a.prompt} through {ckpt}'s tokenizer hash to {th}, "
                         f"not the record's {pin['tokids_sha256']}")
    if graders is None:
        graders = Graders()
    meta = dict(task="held-out evaluation", config_key=key, mode=mode,
                ckpt=os.path.realpath(ckpt), prompt_set=dict(path=path, sha256=pin["sha256"], tokids_sha256=th),
                grader=dict(primary="specopd_reward_g2.reward_func, worker thread", hashes=graders.hashes,
                            descriptive=["g2m main thread", "boxonly worker thread", "ext worker thread"]),
                seeds=dict(base=SEED_BASE, stride=SEED_STRIDE, j=list(range(a.j0, a.j0 + a.n))),
                n_problems_expected=a.n_problems, start=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    _wj(meta_p, meta)
    by_k = {r["k"]: r for r in rows_all}
    ranges = [(lo, min(lo + a.chunk, a.n_problems)) for lo in range(0, a.n_problems, a.chunk)]
    todo = [(lo, hi) for lo, hi in ranges if not os.path.exists(chunk_path(d, lo, hi))]
    log(f"{len(todo)} of {len(ranges)} chunks to generate; n={a.n} j0={a.j0} prompt={a.prompt} mode={mode}")
    if a.dry_run:
        log(f"DRY_RUN ckpt={meta['ckpt']} out={d}")
        return dict(meta=meta, out_dir=d, todo=todo)
    if todo:
        run = (engine_factory or vllm_engine_factory(ckpt, a.gpu_util, a.max_num_seqs, meta))()
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="verl_like_executor")
        for lo, hi in todo:
            reqs, keys = [], []
            for k in range(lo, hi):
                for j in range(a.j0, a.j0 + a.n):
                    kw = sampling_kwargs(k, j)
                    reqs.append((ids[k], kw))
                    keys.append((k, j, kw["seed"]))
            t = time.time()
            outs = run(reqs)
            if len(outs) != len(reqs):
                raise RuntimeError(f"engine returned {len(outs)} outputs for {len(reqs)} requests")
            rows = [dict(k=k, j=j, seed=sd, response_ids=o["token_ids"], resp_len=len(o["token_ids"]),
                         n_tok=len(o["token_ids"]), finish_reason=o["finish_reason"], nll_sum=float(o["nll_sum"]))
                    for (k, j, sd), o in zip(keys, outs)]
            grade_rows(rows, by_k, tok, graders, ex)
            dst = chunk_path(d, lo, hi)
            with open(dst + ".tmp", "w") as f:
                for r in rows:
                    f.write(json.dumps(r, sort_keys=True) + "\n")
            os.replace(dst + ".tmp", dst)
            log(f"chunk {lo}-{hi}: {len(rows)} samples, {sum(r['resp_len'] for r in rows)} tokens, "
                f"{time.time() - t:.0f}s")
        ex.shutdown()
    meta["end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _wj(meta_p, meta)
    summ = write_outputs(d, read_samples(d), a.n, meta)
    report(d, summ)
    return summ


def report(d, summ):
    """Never an accuracy in the log: status and counts only."""
    log(f"DONE status={summ['status']} problems={summ['n_problems']} samples={summ['n_samples']} -> {d}")


def cmd_merge(a):
    """The base's four j-blocks (0-3, 4-7, 8-11, 12-15) -> the n=16 output, only if they partition 0..15 and share
    everything else (checkpoint, prompt set, token ids, sampling)."""
    d = os.path.join(a.ho_root, f"{a.arm}_s{a.step}_{a.prompt}")
    bdir = os.path.join(d, "blocks")
    blocks = sorted(os.listdir(bdir)) if os.path.isdir(bdir) else []
    metas, rows = [], []
    for b in blocks:
        m = _rj(os.path.join(bdir, b, "meta.json"))
        s = _rj(os.path.join(bdir, b, "summary.json"))
        if s.get("status") != "COMPLETE":
            raise SystemExit(f"REFUSE: block {b} is not COMPLETE")
        metas.append(m)
        rows.extend(read_samples(os.path.join(bdir, b)))
    js = sorted(j for m in metas for j in m["seeds"]["j"])
    if js != list(range(N_MAX)) or len(metas) != len(BASE_BLOCKS):
        raise SystemExit(f"REFUSE: the blocks' j sets {[m['seeds']['j'] for m in metas]} do not partition 0..15")
    strip = ("n", "j0")
    keys = [{k: v for k, v in m["config_key"].items() if k not in strip} for m in metas]
    if any(k != keys[0] for k in keys):
        raise SystemExit("REFUSE: the blocks differ in more than their j range")
    meta = dict(metas[0], merged_from=blocks, seeds=dict(base=SEED_BASE, stride=SEED_STRIDE, j=js),
                config_key=dict(keys[0], n=N_MAX, j0=0, merged=True))
    for k in ("start", "end"):
        meta.pop(k, None)
    summ = write_outputs(d, rows, N_MAX, meta)
    _wj(os.path.join(d, "meta.json"), meta)
    report(d, summ)
    return summ


def cmd_freeze(a):
    """The prompt-set record: the prompt sets' paths and sha256s (from bench/ho_select.py write's ho_record.json), the
    sha256 of each set's token ids through the base student's chat template, and the resolved step-100 checkpoints of
    the training jobs named by --resolve-jobs."""
    if a.resolve_jobs and not a.runs_root:
        raise SystemExit("REFUSE: --resolve-jobs needs --runs-root")
    rec = _rj(a.ho_record)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    out = dict(schema="ho_freeze/1", ho_record=os.path.abspath(a.ho_record), ho_record_sha256=sha256_file(a.ho_record),
               tokenizer=os.path.realpath(a.tokenizer), prompt_sets={}, pool_indices=rec["pool_indices"],
               positions=rec["positions"])
    for p in PROMPT_SETS:
        path = rec["paths"][p]
        if sha256_file(path) != rec["sha256"][p]:
            raise SystemExit(f"REFUSE: {path} changed since ho_record.json")
        ids = render_prompt_ids(tok, load_prompt_set(path))
        out["prompt_sets"][p] = dict(path=path, sha256=rec["sha256"][p], tokids_sha256=tokids_sha256(ids),
                                     max_prompt_tokens=max(len(x) for x in ids), n=len(ids))
    out["resolved_checkpoints"] = {j: resolve_checkpoint(j, 100, a.runs_root) for j in a.resolve_jobs}
    out["resolved_checkpoints"]["base"] = os.path.realpath(STUDENT)
    if os.path.exists(a.out):
        raise SystemExit(f"REFUSE: {a.out} exists")
    _wj(a.out, out)
    print(json.dumps({p: {k: v for k, v in x.items()} for p, x in out["prompt_sets"].items()}, indent=1))
    return out


def cmd_resolve(a):
    if a.job == "base":
        print(STUDENT)
        return STUDENT
    if not a.runs_root:
        raise SystemExit("REFUSE: resolving a training job's checkpoint needs --runs-root")
    p = resolve_checkpoint(a.job, a.step, a.runs_root)
    print(p)
    return p


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--freeze-ho", required=True)
        p.add_argument("--prompt", required=True, choices=PROMPT_SETS)
        p.add_argument("--arm", required=True)
        p.add_argument("--step", type=int, required=True)
        p.add_argument("--ho-root", default=HO_ROOT)

    p = sub.add_parser("gen")
    common(p)
    p.add_argument("--objective", required=True, choices=OBJECTIVES)
    p.add_argument("--ckpt", required=True, help="an hf dir, or 'base'")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--j0", type=int, default=0)
    p.add_argument("--block", action="store_true", help="a base j-block (merged by `merge`)")
    p.add_argument("--data", default="", help="override the prompt set path (tests); its sha256 is still checked")
    p.add_argument("--n-problems", type=int, default=N_HO)
    p.add_argument("--chunk", type=int, default=CHUNK)
    p.add_argument("--gpu-util", type=float, default=0.90)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("merge")
    common(p)
    p = sub.add_parser("freeze")
    p.add_argument("--ho-record", required=True)
    p.add_argument("--tokenizer", default=STUDENT)
    p.add_argument("--out", required=True)
    p.add_argument("--resolve-jobs", nargs="*", required=True,
                   help="the training job ids whose step-100 checkpoints the record lists (may be empty)")
    p.add_argument("--runs-root", default="",
                   help="the directory holding the repository checkouts the training jobs ran from")
    p = sub.add_parser("resolve")
    p.add_argument("--job", required=True, help="a training job id, or 'base'")
    p.add_argument("--step", type=int, default=100)
    p.add_argument("--runs-root", default="",
                   help="the directory holding the repository checkouts the training jobs ran from")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "gen":
        return cmd_gen(a)
    if a.cmd == "merge":
        return cmd_merge(a)
    if a.cmd == "freeze":
        return cmd_freeze(a)
    if a.cmd == "resolve":
        return cmd_resolve(a)


if __name__ == "__main__":
    main()
