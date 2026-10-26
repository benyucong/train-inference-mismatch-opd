#!/usr/bin/env python3
"""Training data of the RL comparison (GRPO and REINFORCE): the JR prompt file, the training order, the base student's
pass over the first order positions, and the subset of problems the base student answers sometimes but not always.

The RL runs train the 7B student on DAPO-Math-17k problems whose DAPO wrapper is replaced by JustRL's instruction, on
the problems, in OPD's training order, on which the untrained student answers between 1 and 7 of 8 samples correctly
(every GRPO group then carries a learning signal). From dapo-math-17k.parquet to the subset parquet:

  jr         CPU. dapo-math-17k.parquet -> dapo-math-17k-jr.parquet (+ <out>.json: row count and sha256s). For DAPO row
             i with user content c_i = DAPO_PREFIX + P_i + DAPO_SUFFIX,
                 P_i  = strip_wrap(c_i, DAPO_PREFIX, DAPO_SUFFIX)      verbatim: not stripped, normalised or edited
                 JR_i = P_i + EVAL_SUFFIX                              " Please reason step by step, and put your
                                                                       final answer within \\boxed{}." (one space)
             and the prompt stays one user message. Only that string changes: every other column, every other key of
             the message and the row order are the DAPO file's, so row i of both files is the same problem and one
             training order applies to both. DAPO_PREFIX, DAPO_SUFFIX, EVAL_SUFFIX and strip_wrap are imported from
             refresh/t3_acc2.py (the evaluation's own prompt format), not copied. The new file is written under a
             temporary name and renamed only if every check holds: the source is the DAPO parquet (sha256), every row
             carries the exact DAPO wrapper (checked explicitly too, since strip_wrap's own check is an `assert` that
             `python -O` would drop), every JR content is P + EVAL_SUFFIX, every other column and message key is
             equal row by row, the arrow schema is equal, no content holds an escaped "\\boxed{{}}", and every JR
             prompt fits MAX_PROMPT tokens both through the text path `gen` uses and through verl's continuous-token
             builder (the training path; model_family deepseek, hf_model_type qwen2). Otherwise the file is kept as
             <out>.FAILED.parquet, its record names the failed check, and the exit code is 3.
  order      CPU (torch). The training order of the launcher's data sampler at its defaults (data.shuffle=True,
             data.seed=1): verl's RandomSampler with torch.Generator().manual_seed(1) over the whole DAPO parquet.
             verl draws one permutation before the first training step, so the first randperm is discarded and the
             second is the order; step t of a run with BATCH b reads order positions [b(t-1), bt). With torchdata
             installed, the order is also drawn through the sampler class itself and must be equal. Written as a
             JSON list of pool indices (position -> DAPO row), the format bench/ho_select.py reads.
  gen        GPU. The base student's GROUP=8 samples per prompt for order positions [--pos-lo, --pos-hi) of --data,
             with the training run's sampling settings: vLLM bf16, temperature 1, top_p 1, top_k -1, repetition
             penalty 1, at most MAX_RESP=16,384 new tokens (prompts are at most MAX_PROMPT=2,048 tokens, checked, so
             the training rollout's limit min(MAX_RESP, MAX_PROMPT + MAX_RESP - prompt length) is MAX_RESP), one
             request per sample with seed SEED_BASE + GROUP * pos + s. The prompt ids are apply_chat_template(
             add_generation_prompt=True) of the row's chat (no enable_thinking: the launcher passes none for this
             student), tokenized without extra special tokens. The order is computed on --order-data (the DAPO
             parquet) and mapped to rows of --data, which must be row-aligned with it: each row used is checked to be
             the JR form of its DAPO row with every other column equal (or --data is the DAPO parquet itself).
             Grading is the training verifier: solution = tokenizer.decode(response, skip_special_tokens=True),
             score = custom_reward.specopd_reward.reward_func(data_source, solution, ground_truth, extra_info) (a dict
             result contributes its "score"), correct = score > CORRECT_THRESHOLD of
             verl/verl/trainer/ppo/specopd_grpo_metrics.py. The verifier runs in a one-worker thread, as in verl's
             reward loop: its SIGALRM-based sympy timeout only arms in the main thread, so main-thread grading could
             score a response differently from training. A verifier exception is recorded on the row and scored 0.
             Not reproduced: the training rollout's own random stream (the counts estimate the base student's pass
             rate; they do not replay a run), and speculative decoding, which leaves the sampled distribution
             unchanged.
             Output, in --out-dir: chunks/cLLLLL_HHHHH.jsonl (--chunk prompts each, one row per sample: pos, s,
             pool_index, seed, prompt_len, resp_len, finish_reason, response_ids, score, correct[, reward_error]),
             each written atomically, and blocks/bLLLLL_HHHHH.json, the configuration and chunk list of the block.
             Running a block again with the same configuration generates only the missing chunks. Several jobs may
             fill disjoint blocks of one out dir; every block must have the same configuration apart from its
             positions.
             vLLM's batch composition can change bf16 numerics, so files equal to the paper's need the same blocks
             (1,000 positions, bench/rl_data.sbatch) and the same --chunk.
  summarize  CPU. per_prompt.csv and summary.json in --out-dir from the chunk files: one CSV row per position of
             [0, --n-positions) with all GROUP samples (pos, pool_index, n, n_correct, pass_rate, mean_score,
             n_finished, n_truncated, mean_len, n_reward_errors). Checks: the blocks share one configuration and do
             not overlap, every chunk lies in a block, no (pos, s) appears twice, every sample carries its seed and a
             `correct` that equals score > CORRECT_THRESHOLD. status = COMPLETE when every position is complete.
             `pos` and `pool_index` are the columns `bench/ho_select.py order --per-prompt-csv` checks against its
             order.
  subset     CPU. Keep the positions with 1 <= n_correct <= 7 of 8, in increasing order position; K = the number kept.
             The subset is the first BATCH * S kept rows, S = min(--max-steps, floor(K / BATCH)), a whole number of
             RL steps at BATCH=16 prompts per step (the runs read it with SHUFFLE=False, row = training position).
             Refuses (exit 4) when S < --min-steps, and refuses a pass that is not COMPLETE over [0, N), whose
             per_prompt.csv is not the one summarize wrote, or that sampled another file than --data. The rows are
             JR.iloc[pool_indices] with every column; texts repeated within the subset are counted in the record,
             never removed. Writes <out> and <out>.json (sha256, K, steps = S, positions, pool indices).
Positions stay below TRAIN_POS_MAX = 8,000: bench/ho_select.py removes from its held-out problems the near-duplicates
of training positions below 8,000 only, so a training problem from a later position could repeat a held-out one.

    $PYTHON_BIN bench/rl_data.py jr
    $PYTHON_BIN bench/rl_data.py order
    for k in 0 1 2 3 4 5 6 7; do POS_LO=$((1000*k)) POS_HI=$((1000*(k+1))) sbatch bench/rl_data.sbatch; done
    $PYTHON_BIN bench/rl_data.py summarize
    $PYTHON_BIN bench/rl_data.py subset
"""
import argparse
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import os
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                         # the repository root: bench/, refresh/, custom_reward/, verl/
R = os.environ.get("CLUSTER_ROOT", "/path/to/workdir")
DATASETS = f"{R}/workspace/datasets"
DAPO = f"{DATASETS}/dapo-math-17k.parquet"
DAPO_SHA256 = "039f3afd689c846985bd2bf58e55a2210a8b08a1a9e1f60dbc07107a5f341925"
DAPO_CONTENT_SHA256 = "157ddfe34217616feea45b59c90ec271fc346a2e0073cc4c26a3e0b8f1ad132b"  # parquet_content_sha256
JR_DATA = f"{DATASETS}/dapo-math-17k-jr.parquet"
SUBSET = f"{DATASETS}/dapo-math-17k-jr.subset.parquet"
WORK = f"{R}/rl_data"
ORDER_JSON = f"{WORK}/order_seed1_skip1.json"
PASS_DIR = f"{WORK}/base_pass"
# sha256 of the two files the paper's RL runs used. Another pyarrow / pandas version can write the same rows to a file
# with another sha256, and the subset also depends on vLLM's numerics, so a difference is reported, never refused.
JR_SHA256_REF = "ebe730ed28286795ccc38b82e0067393f99816ad8647bb353ce38843b0a6536d"
SUBSET_SHA256_REF = "4992dfcb6bb34f6f787ca0098ccb601d20924d938cc094343dc9a40b3e62c051"

if HERE not in sys.path:
    sys.path.insert(0, HERE)
import mo_common as MC  # noqa: E402  (stdlib only at import)

STUDENT = MC.PAIRS["main"]["student"]               # DeepSeek-R1-Distill-Qwen-7B, the RL comparison's student

GROUP = 8                     # samples per prompt = the launcher's ROLLOUT_N
DATA_SEED = 1                 # the launcher's SEED (data.seed)
SAMPLER_SKIP = 1              # permutations drawn before the first training step
MAX_PROMPT = 2048             # the launcher's MAX_PROMPT (data.max_prompt_length; filter_overlong_prompts=False)
MAX_RESP = 16384              # the launcher's MAX_RESP
TEMPERATURE, TOP_P, TOP_K, REPETITION_PENALTY = 1.0, 1.0, -1, 1.0
SEED_BASE = 300000            # sample seed = SEED_BASE + GROUP * pos + s
CHUNK = 160                   # prompts per generate call
N_POSITIONS = 8000            # order positions sampled and selected from
TRAIN_POS_MAX = 8000          # bench/ho_select.py PMAX: training positions its held-out set is cleaned against
KEEP = (1, GROUP - 1)         # 1 <= #correct <= 7 of 8
RL_BATCH = 16                 # the RL runs' BATCH
MAX_STEPS = 200               # the RL runs' STEPS
MIN_STEPS = 100
DOUBLE_BRACE = "\\boxed{{}}"
JR_SCHEMA = "rl_data_jr/1"
BLOCK_SCHEMA = "rl_data_block/1"
SUMMARY_SCHEMA = "rl_data_summary/1"
SUBSET_SCHEMA = "rl_data_subset/1"
CHUNK_RE = re.compile(r"c(\d{5})_(\d{5})\.jsonl")
BLOCK_RE = re.compile(r"b(\d{5})_(\d{5})\.json")
PER_PROMPT_FIELDS = ("pos", "pool_index", "n", "n_correct", "pass_rate", "mean_score", "n_finished", "n_truncated",
                     "mean_len", "n_reward_errors")

T0 = time.time()


def log(*a):
    print(f"[rl_data {time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s]", *a, flush=True)


def parquet_content_sha256(path):
    """sha256 over the rows of a parquet file (each row as sorted-key JSON, one per line): equal for files with the same
    rows in the same order, whatever library version wrote them, unlike the file's own sha256."""
    import pyarrow.parquet as pq
    h = hashlib.sha256()
    for row in pq.read_table(path).to_pylist():
        h.update(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def is_dapo(path, sha):
    """The DAPO parquet: its file sha256, or the same rows written by another library version."""
    if sha == DAPO_SHA256:
        return True
    if parquet_content_sha256(path) == DAPO_CONTENT_SHA256:
        log(f"{path}: the DAPO parquet's rows, written by another library version (file sha256 {sha})")
        return True
    return False


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, separators=(",", ":")).encode()).hexdigest()


def _rj(path):
    with open(path) as f:
        return json.load(f)


def _wj(path, obj, indent=1):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


# ------------------------------------------------------------------------------------------------ t3_acc2 (imported)
def load_t3_acc2():
    """refresh/t3_acc2.py from the tree this file is in, loaded by path (as bench/ho_select.py does)."""
    p = os.path.join(ROOT, "refresh", "t3_acc2.py")
    spec = importlib.util.spec_from_file_location("t3_acc2", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, p


A, T3_PATH = load_t3_acc2()
DAPO_PREFIX, DAPO_SUFFIX, EVAL_SUFFIX, strip_wrap = A.DAPO_PREFIX, A.DAPO_SUFFIX, A.EVAL_SUFFIX, A.strip_wrap


# ------------------------------------------------------------------------------------------------ the JR rule (pure)
class JRError(ValueError):
    """A row without the DAPO wrapper, or a prompt that is not exactly one user message."""


def problem_text(dapo_content):
    """P of a DAPO content, verbatim. Refuses (JRError) a content without the exact DAPO wrapper."""
    if not isinstance(dapo_content, str):
        raise JRError(f"content is {type(dapo_content).__name__}, not str")
    if (len(dapo_content) < len(DAPO_PREFIX) + len(DAPO_SUFFIX) or not dapo_content.startswith(DAPO_PREFIX)
            or not dapo_content.endswith(DAPO_SUFFIX)):
        raise JRError(f"no DAPO wrapper: {dapo_content[:60]!r}...{dapo_content[-60:]!r}")
    p = strip_wrap(dapo_content, DAPO_PREFIX, DAPO_SUFFIX)
    if DAPO_PREFIX + p + DAPO_SUFFIX != dapo_content:             # explicit: survives python -O
        raise JRError("DAPO_PREFIX + P + DAPO_SUFFIX != content")
    return p


def jr_content(dapo_content):
    """The JR content of a DAPO content: P + EVAL_SUFFIX."""
    return problem_text(dapo_content) + EVAL_SUFFIX


def _plain(x):
    """parquet cell -> plain python (numpy arrays, tuples, nested dicts)."""
    if hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
        x = x.tolist()
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def user_message(prompt, i=None):
    """The single user message of a prompt cell, as a dict; refuses anything else."""
    if hasattr(prompt, "tolist") and not isinstance(prompt, (str, bytes)):
        prompt = prompt.tolist()
    msgs = [dict(m) for m in prompt]
    where = f"row {i}: " if i is not None else ""
    if len(msgs) != 1:
        raise JRError(f"{where}prompt has {len(msgs)} messages, not exactly one")
    if msgs[0].get("role") != "user":
        raise JRError(f"{where}the message role is {msgs[0].get('role')!r}, not 'user'")
    return msgs[0]


def jr_prompt(prompt, i=None):
    """The JR prompt cell: the same single message with only `content` replaced (every other key copied)."""
    m = dict(user_message(prompt, i))
    try:
        m["content"] = jr_content(m.get("content"))
    except JRError as e:
        raise JRError(f"row {i}: {e}" if i is not None else str(e))
    return [m]


def compare_rows(dapo_rows, jr_rows):
    """Row by row (row i against row i): the wrapper, the JR content, every other column and every other message key."""
    n = len(dapo_rows)
    res = dict(n_dapo=n, n_jr=len(jr_rows), wrapper_ok=0, jr_ok=0, other_columns_ok=0, message_keys_ok=0,
               first_bad=[])
    for i, (d, j) in enumerate(zip(dapo_rows, jr_rows)):
        try:
            dm = user_message(d["prompt"], i)
            p = problem_text(dm["content"])
            res["wrapper_ok"] += 1
        except JRError as e:
            if len(res["first_bad"]) < 5:
                res["first_bad"].append(dict(row=i, what="wrapper", err=str(e)[:200]))
            continue
        try:
            jm = user_message(j["prompt"], i)
            ok = jm.get("content") == p + EVAL_SUFFIX
        except JRError:
            ok, jm = False, {}
        res["jr_ok"] += int(ok)
        if not ok and len(res["first_bad"]) < 5:
            res["first_bad"].append(dict(row=i, what="jr_content"))
        res["message_keys_ok"] += int({k: v for k, v in jm.items() if k != "content"}
                                      == {k: v for k, v in dm.items() if k != "content"})
        res["other_columns_ok"] += int(sorted(d) == sorted(j) and all(_plain(d[k]) == _plain(j[k])
                                                                       for k in d if k != "prompt"))
    res["ok"] = (res["n_dapo"] == res["n_jr"] and res["wrapper_ok"] == res["jr_ok"] == res["other_columns_ok"]
                 == res["message_keys_ok"] == n)
    return res


# ------------------------------------------------------------------------------------------------ jr (pyarrow)
def build_table(dapo_table):
    """The JR file as an arrow table: the DAPO table with only the prompt column replaced, under the SAME arrow type."""
    import pyarrow as pa
    idx = dapo_table.schema.get_field_index("prompt")
    field = dapo_table.schema.field("prompt")
    new = [jr_prompt(p, i) for i, p in enumerate(dapo_table.column("prompt").to_pylist())]
    return dapo_table.set_column(idx, field, pa.array(new, type=field.type))


def prompt_lengths(contents, tokenizer_dir, builder=True):
    """Longest JR prompt through the text path of `gen` and through verl's continuous-token builder (training's)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    out = dict(tokenizer=os.path.realpath(tokenizer_dir), max_prompt=MAX_PROMPT)
    text_lens = []
    for c in contents:
        text = tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True)
        text_lens.append(len(tok(text, add_special_tokens=False)["input_ids"]))
    out["text_path"] = dict(max=max(text_lens), argmax=text_lens.index(max(text_lens)),
                            n_over=sum(x > MAX_PROMPT for x in text_lens))
    if builder:
        from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder
        b = create_continuous_token_builder(tok, model_family="deepseek", hf_model_type="qwen2")
        lens = [len(list(b.build_initial_tokens([{"role": "user", "content": c}]))) for c in contents]
        out["builder_path"] = dict(max=max(lens), argmax=lens.index(max(lens)),
                                   n_over=sum(x > MAX_PROMPT for x in lens),
                                   n_len_differs_from_text_path=sum(x != y for x, y in zip(lens, text_lens)))
    out["ok"] = out["text_path"]["n_over"] == 0 and (not builder or out["builder_path"]["n_over"] == 0)
    return out


def check_jr(dapo_path, jr_path, tokenizer_dir, builder):
    """Every check of a (DAPO parquet, JR file) pair. Returns the record (`ok` inside)."""
    import pyarrow.parquet as pq
    src_sha = sha256_file(dapo_path)
    rec = dict(schema=JR_SCHEMA, source=os.path.realpath(dapo_path), source_sha256=src_sha,
               source_sha256_expected=DAPO_SHA256, jr_sha256=sha256_file(jr_path),
               builder=os.path.abspath(__file__), builder_sha256=sha256_file(os.path.abspath(__file__)),
               t3_acc2=T3_PATH, t3_acc2_sha256=sha256_file(T3_PATH), checks={})
    c = rec["checks"]
    dt, jt = pq.read_table(dapo_path), pq.read_table(jr_path)
    c["schema_equal"] = jt.schema.remove_metadata().equals(dt.schema.remove_metadata())
    drows, jrows = dt.to_pylist(), jt.to_pylist()
    rec["n_rows"] = len(jrows)
    c["rows"] = compare_rows(drows, jrows)
    contents = None
    try:
        contents = [user_message(r["prompt"], i)["content"] for i, r in enumerate(jrows)]
    except JRError as e:                                  # already a failed rows check; record it
        c["parse_error"] = str(e)[:300]
    c["n_double_brace"] = None if contents is None else sum(DOUBLE_BRACE in x for x in contents)
    if tokenizer_dir and contents:
        c["lengths"] = prompt_lengths(contents, tokenizer_dir, builder=builder)
    else:
        c["lengths"] = "SKIPPED (--tokenizer '')" if not tokenizer_dir else None
    ok = dict(rows=c["rows"]["ok"], schema=c["schema_equal"], no_double_brace=c["n_double_brace"] == 0,
              lengths=(isinstance(c["lengths"], dict) and c["lengths"]["ok"]) if tokenizer_dir else True)
    rec["failed"] = sorted(k for k, v in ok.items() if v is not True)
    rec["ok"] = not rec["failed"]
    return rec


def cmd_jr(a):
    if os.path.exists(a.out):
        raise SystemExit(f"REFUSE: {a.out} exists")
    sha = sha256_file(a.dapo)
    if not is_dapo(a.dapo, sha) and not a.allow_other_data:
        raise SystemExit(f"REFUSE: {a.dapo} sha256 {sha} is not the DAPO parquet ({DAPO_SHA256}); "
                         "--allow-other-data to build from it anyway")
    import pyarrow.parquet as pq
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    jt = build_table(pq.read_table(a.dapo))
    tmp = a.out + ".tmp.parquet"
    pq.write_table(jt, tmp)
    rec = check_jr(a.dapo, tmp, a.tokenizer, builder=not a.no_builder)
    if rec["ok"]:
        os.replace(tmp, a.out)
        rec["jr"] = os.path.realpath(a.out)
        rec["jr_sha256_reference"] = JR_SHA256_REF
        rec["jr_equals_reference"] = rec["jr_sha256"] == JR_SHA256_REF
    else:
        failed = a.out[:-len(".parquet")] if a.out.endswith(".parquet") else a.out
        failed += ".FAILED.parquet"
        os.replace(tmp, failed)
        rec["jr"] = os.path.realpath(failed)
    _wj(rec["jr"] + ".json", rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "checks"}, indent=1, ensure_ascii=False))
    if not rec["ok"]:
        raise SystemExit(3)
    return rec


# ------------------------------------------------------------------------------------------------ order (torch)
def training_order(n_pool, seed=DATA_SEED, skip=SAMPLER_SKIP):
    """verl's training order: torch.Generator().manual_seed(seed), `skip` randperms discarded, then the next one."""
    import torch
    g = torch.Generator()
    g.manual_seed(seed)
    for _ in range(skip):
        torch.randperm(n_pool, generator=g)
    return torch.randperm(n_pool, generator=g).tolist()


def torchdata_order(n_pool, seed=DATA_SEED, skip=SAMPLER_SKIP):
    """The same order through torchdata's stateful RandomSampler (the class verl's create_rl_sampler builds), iterated
    skip + 1 times. None if torchdata is not installed."""
    try:
        import torch
        from torchdata.stateful_dataloader.sampler import RandomSampler
    except Exception:
        return None
    g = torch.Generator()
    g.manual_seed(seed)
    s = RandomSampler(data_source=list(range(n_pool)), generator=g)
    out = None
    for _ in range(skip + 1):
        out = list(iter(s))
    return out


def checked_order(n_pool):
    """(order, torchdata_sampler_equal); refuses an order the sampler class does not reproduce."""
    order = training_order(n_pool)
    td = torchdata_order(n_pool)
    td_eq = None if td is None else td == order
    if td_eq is False:
        raise SystemExit("REFUSE: torchdata's RandomSampler order differs from the randperm order")
    return order, td_eq


def cmd_order(a):
    import pyarrow.parquet as pq
    sha = sha256_file(a.data)
    if not is_dapo(a.data, sha) and not a.allow_other_data:
        raise SystemExit(f"REFUSE: {a.data} sha256 {sha} is not the DAPO parquet ({DAPO_SHA256})")
    n = pq.read_metadata(a.data).num_rows
    order, td_eq = checked_order(n)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w") as f:
        json.dump(order, f)
    os.replace(a.out + ".tmp", a.out)
    out = dict(out=os.path.realpath(a.out), sha256=sha256_file(a.out), n_pool=n, data_sha256=sha, seed=DATA_SEED,
               skip=SAMPLER_SKIP, first16=order[:16], torchdata_sampler_equal=td_eq)
    if td_eq is None:
        log("torchdata is not installed: the order was not cross-checked against the sampler class")
    print(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------------------------------------ gen (GPU)
def sample_seed(pos, s, base=SEED_BASE, group=GROUP):
    return base + group * pos + s


def sampling_kwargs(pos, s, seed_base):
    return dict(temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, repetition_penalty=REPETITION_PENALTY,
                max_tokens=MAX_RESP, seed=sample_seed(pos, s, seed_base))


def chunk_ranges(lo, hi, chunk):
    return [(s, min(s + chunk, hi)) for s in range(lo, hi, chunk)]


def chunk_path(out_dir, lo, hi):
    return os.path.join(out_dir, "chunks", f"c{lo:05d}_{hi:05d}.jsonl")


def block_path(out_dir, lo, hi):
    return os.path.join(out_dir, "blocks", f"b{lo:05d}_{hi:05d}.json")


def load_reward():
    """The training verifier (custom_reward.specopd_reward.reward_func) and the sha256 of its files."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from custom_reward.specopd_reward import reward_func
    files = ["custom_reward/specopd_reward.py"] + sorted(
        f"custom_reward/ttrl_math/{f}" for f in os.listdir(os.path.join(ROOT, "custom_reward", "ttrl_math"))
        if f.endswith(".py"))
    return reward_func, {f: sha256_file(os.path.join(ROOT, f)) for f in files}


def load_correct_threshold():
    """CORRECT_THRESHOLD of the training code's group metric (verl/verl/trainer/ppo/specopd_grpo_metrics.py), loaded by
    path so that verl itself is not imported."""
    path = os.path.join(ROOT, "verl", "verl", "trainer", "ppo", "specopd_grpo_metrics.py")
    spec = importlib.util.spec_from_file_location("specopd_grpo_metrics_rl_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return float(mod.CORRECT_THRESHOLD), sha256_file(path)


def prompt_record(pos, pool_index, msg, row, tok):
    """One prompt: its ids (chat template text, then tokenized without extra special tokens) and the grading fields."""
    text = tok.apply_chat_template([dict(msg)], tokenize=False, add_generation_prompt=True)
    pids = list(tok(text, add_special_tokens=False)["input_ids"])
    if len(pids) > MAX_PROMPT:
        raise ValueError(f"prompt pos {pos} (pool {pool_index}) has {len(pids)} > MAX_PROMPT {MAX_PROMPT} tokens: "
                         "training would truncate it; this pass does not reproduce that")
    rm = _plain(row["reward_model"])
    ei = row["extra_info"] if "extra_info" in row.index else None
    ei = _plain(ei) if isinstance(ei, dict) or hasattr(ei, "keys") else {}
    return dict(pos=pos, pool_index=int(pool_index), prompt_ids=pids, data_source=row["data_source"],
                ground_truth=rm["ground_truth"] if isinstance(rm, dict) else rm, extra_info=ei or {})


def prepare(a, lo, hi):
    """The prompts of order positions [lo, hi) of --data, the tokenizer, and the checks' record."""
    import pandas as pd
    from transformers import AutoTokenizer
    osha = sha256_file(a.order_data)
    if not is_dapo(a.order_data, osha) and not a.allow_other_data:
        raise SystemExit(f"REFUSE: --order-data {a.order_data} sha256 {osha} is not the DAPO parquet ({DAPO_SHA256})")
    dsha = sha256_file(a.data)
    dapo = pd.read_parquet(a.order_data)
    same = dsha == osha
    data = dapo if same else pd.read_parquet(a.data)
    if len(data) != len(dapo):
        raise SystemExit(f"REFUSE: --data has {len(data)} rows, --order-data {len(dapo)}: the files are not "
                         "row-aligned")
    cols = [c for c in dapo.columns if c != "prompt"]
    if sorted(cols) != sorted(c for c in data.columns if c != "prompt"):
        raise SystemExit("REFUSE: --data's columns differ from --order-data's")
    order, td_eq = checked_order(len(dapo))
    tok = AutoTokenizer.from_pretrained(a.student)
    prompts = []
    for pos in range(lo, hi):
        idx = int(order[pos])
        drow, row = dapo.iloc[idx], data.iloc[idx]
        msg = user_message(row["prompt"], idx)
        if not same and (msg.get("content") != jr_content(user_message(drow["prompt"], idx)["content"])
                         or any(_plain(drow[c]) != _plain(row[c]) for c in cols)):
            raise SystemExit(f"REFUSE: --data row {idx} is not the JR form of --order-data row {idx}")
        prompts.append(prompt_record(pos, idx, msg, row, tok))
    checks = dict(order_first16=order[:16], order_sha256=sha256_json(order), torchdata_sampler_equal=td_eq,
                  data_rule="identity" if same else "jr", n_rows_checked=hi - lo,
                  prompt_len=dict(max=max(len(p["prompt_ids"]) for p in prompts),
                                  mean=round(sum(len(p["prompt_ids"]) for p in prompts) / len(prompts), 1)))
    if td_eq is None:
        log("torchdata is not installed: the order was not cross-checked against the sampler class")
    return prompts, tok, dsha, osha, checks


def config_key(a, data_sha, order_sha, reward_files, threshold, metrics_sha):
    """Everything a sample depends on except its (pos, s). Blocks of one out dir must have equal keys."""
    return dict(student=os.path.realpath(a.student), data_sha256=data_sha, order_data_sha256=order_sha,
                data_seed=DATA_SEED, sampler_skip=SAMPLER_SKIP, group=GROUP, temperature=TEMPERATURE, top_p=TOP_P,
                top_k=TOP_K, repetition_penalty=REPETITION_PENALTY, max_tokens=MAX_RESP, max_prompt=MAX_PROMPT,
                seed_base=a.seed_base, chunk=a.chunk, grade="worker_thread", reward_sha256=reward_files,
                correct_threshold=threshold, grpo_metrics_sha256=metrics_sha)


def read_blocks(out_dir):
    d = os.path.join(out_dir, "blocks")
    if not os.path.isdir(d):
        return []
    return [_rj(os.path.join(d, f)) for f in sorted(os.listdir(d)) if BLOCK_RE.fullmatch(f)]


def check_blocks(out_dir, key, lo, hi):
    """Refuses a block whose configuration differs from, or whose positions overlap, another block of the out dir."""
    for b in read_blocks(out_dir):
        b0, b1 = b["positions"]
        if (b0, b1) == (lo, hi):
            continue
        if b0 < hi and lo < b1:
            raise SystemExit(f"REFUSE: positions [{lo}, {hi}) overlap the block [{b0}, {b1}) of {out_dir}")
        if b["config_key"] != key:
            raise SystemExit(f"REFUSE: the block [{b0}, {b1}) of {out_dir} has another configuration; "
                             "refusing to mix samples")


def make_thread_executor():
    """A one-worker executor, as verl's reward loop (loop.run_in_executor): every reward_func call runs off the main
    thread, where the verifier's SIGALRM timeout does not arm."""
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="verl_like_executor")
    worker_is_main = ex.submit(lambda: threading.current_thread() is threading.main_thread()).result()
    if worker_is_main or threading.current_thread() is not threading.main_thread():
        ex.shutdown()
        raise RuntimeError("the grading executor must run off the main thread, called from the main thread")
    return ex


def grade_rows(rows, prompts_by_pos, tok, reward_func, threshold):
    for r in rows:
        p = prompts_by_pos[r["pos"]]
        sol = tok.decode(r["response_ids"], skip_special_tokens=True)
        try:
            res = reward_func(p["data_source"], sol, p["ground_truth"], extra_info=dict(p["extra_info"]))
            score = float(res["score"]) if isinstance(res, dict) else float(res)
        except Exception as e:                      # training would raise; record it, score 0
            r["reward_error"] = repr(e)[:300]
            score = 0.0
        r["score"] = score
        r["correct"] = bool(score > threshold)


def make_engine(a, rec):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    import vllm
    rec["vllm"] = vllm.__version__
    llm = LLM(model=a.student, dtype="bfloat16", gpu_memory_utilization=a.gpu_util,
              max_model_len=MAX_PROMPT + MAX_RESP + 1, max_num_seqs=a.max_num_seqs, seed=0,
              enable_prefix_caching=True, enforce_eager=False)

    def run(reqs):
        outs = llm.generate([TokensPrompt(prompt_token_ids=pids) for pids, _ in reqs],
                            [SamplingParams(**kw) for _, kw in reqs], use_tqdm=False)
        return [(list(o.outputs[0].token_ids), o.outputs[0].finish_reason) for o in outs]
    return run


def cmd_gen(a):
    lo, hi = a.pos_lo, a.pos_hi
    if not 0 <= lo < hi <= TRAIN_POS_MAX:
        raise SystemExit(f"REFUSE: positions [{lo}, {hi}) are not inside [0, {TRAIN_POS_MAX}) (the training positions "
                         "bench/ho_select.py cleans its held-out set against)")
    if a.chunk < 1:
        raise SystemExit("REFUSE: --chunk must be positive")
    reward_func, reward_files = load_reward()
    threshold, metrics_sha = load_correct_threshold()
    prompts, tok, dsha, osha, checks = prepare(a, lo, hi)
    key = config_key(a, dsha, osha, reward_files, threshold, metrics_sha)
    check_blocks(a.out_dir, key, lo, hi)
    prompts_sha = sha256_json([p["prompt_ids"] for p in prompts])
    rec_p = block_path(a.out_dir, lo, hi)
    old = _rj(rec_p) if os.path.exists(rec_p) else None
    if old is not None and (old.get("config_key") != key or old.get("prompts_sha256") != prompts_sha):
        raise SystemExit(f"REFUSE: {rec_p} exists with another configuration or other prompts; refusing to mix samples")
    todo = [(c0, c1) for c0, c1 in chunk_ranges(lo, hi, a.chunk) if not os.path.exists(chunk_path(a.out_dir, c0, c1))]
    log(f"[{lo}, {hi}): {len(todo)} of {len(chunk_ranges(lo, hi, a.chunk))} chunks to generate (chunk = {a.chunk} "
        f"prompts x {GROUP} samples)")
    if a.dry_run:
        print(json.dumps(dict(positions=[lo, hi], config_key=key, checks=checks, prompts_sha256=prompts_sha,
                              todo=[dict(lo=c0, hi=c1, n_requests=(c1 - c0) * GROUP) for c0, c1 in todo],
                              first_request=sampling_kwargs(lo, 0, a.seed_base)), indent=1))
        return None
    os.makedirs(os.path.join(a.out_dir, "chunks"), exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "blocks"), exist_ok=True)
    rec = dict(schema=BLOCK_SCHEMA, positions=[lo, hi], config_key=key, prompts_sha256=prompts_sha,
               data=os.path.realpath(a.data), order_data=os.path.realpath(a.order_data), checks=checks,
               vllm=(old or {}).get("vllm"), chunks=(old or {}).get("chunks", []))
    _wj(rec_p, rec)
    by_pos = {p["pos"]: p for p in prompts}
    ex = make_thread_executor()
    try:
        if todo:
            t = time.time()
            run = make_engine(a, rec)
            log(f"engine ready in {time.time() - t:.0f}s (vLLM {rec['vllm']})")
            for c0, c1 in todo:
                reqs, keys = [], []
                for pos in range(c0, c1):
                    for s in range(GROUP):
                        kw = sampling_kwargs(pos, s, a.seed_base)
                        reqs.append((by_pos[pos]["prompt_ids"], kw))
                        keys.append((pos, s, kw["seed"]))
                t = time.time()
                outs = run(reqs)
                gen_s = time.time() - t
                if len(outs) != len(reqs):
                    raise RuntimeError(f"engine returned {len(outs)} outputs for {len(reqs)} requests")
                rows = []
                for (pos, s, sd), (ids, fr) in zip(keys, outs):
                    p = by_pos[pos]
                    rows.append(dict(pos=pos, s=s, pool_index=p["pool_index"], seed=sd,
                                     prompt_len=len(p["prompt_ids"]), resp_len=len(ids), finish_reason=fr,
                                     response_ids=ids))
                t = time.time()
                ex.submit(grade_rows, rows, by_pos, tok, reward_func, threshold).result()
                grade_s = time.time() - t
                dst = chunk_path(a.out_dir, c0, c1)
                with open(dst + ".tmp", "w") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                os.replace(dst + ".tmp", dst)
                ntok = sum(r["resp_len"] for r in rows)
                rec["chunks"].append(dict(lo=c0, hi=c1, n=len(rows), tokens=ntok, gen_s=round(gen_s, 1),
                                          grade_s=round(grade_s, 1), sha256=sha256_file(dst)))
                _wj(rec_p, rec)
                log(f"chunk {c0}-{c1}: {len(rows)} samples, {ntok} tokens, {gen_s:.0f}s generation, "
                    f"{grade_s:.0f}s grading, {sum(r['correct'] for r in rows)} correct, "
                    f"{sum('reward_error' in r for r in rows)} verifier errors")
    finally:
        ex.shutdown()
    log(f"DONE [{lo}, {hi}): every chunk of the block exists; run `summarize` once every block is done")
    return rec


# ------------------------------------------------------------------------------------------------ summarize (CPU)
def summarize_rows(rows, n_positions, group, seed_base, threshold):
    """rows: graded sample rows. Returns (summary dict, per-prompt rows). A position is COMPLETE when it has exactly the
    samples s = 0..group-1, each at seed seed_base + group * pos + s. Positions >= n_positions are counted, not used."""
    by_pos, beyond = {}, 0
    for r in rows:
        pos, s = int(r["pos"]), int(r["s"])
        if pos >= n_positions:
            beyond += 1
            continue
        if s in by_pos.setdefault(pos, {}):
            raise ValueError(f"pos {pos} sample {s} appears twice")
        by_pos[pos][s] = r
    per_prompt = []
    for pos in range(n_positions):
        g = by_pos.get(pos, {})
        if sorted(g) != list(range(group)):
            continue
        v = [g[s] for s in range(group)]
        for s, x in enumerate(v):
            if int(x["seed"]) != sample_seed(pos, s, seed_base, group):
                raise ValueError(f"pos {pos} sample {s}: seed {x['seed']} != {sample_seed(pos, s, seed_base, group)}")
            if bool(x["correct"]) != (float(x["score"]) > threshold):
                raise ValueError(f"pos {pos} sample {s}: stored 'correct' disagrees with score > {threshold}")
        pools = {int(x["pool_index"]) for x in v}
        if len(pools) != 1:
            raise ValueError(f"pos {pos}: samples from {len(pools)} different rows")
        nc = sum(bool(x["correct"]) for x in v)
        per_prompt.append(dict(pos=pos, pool_index=pools.pop(), n=group, n_correct=nc, pass_rate=nc / group,
                               mean_score=_mean([float(x["score"]) for x in v]),
                               n_finished=sum(x["finish_reason"] == "stop" for x in v),
                               n_truncated=sum(x["finish_reason"] == "length" for x in v),
                               mean_len=_mean([x["resp_len"] for x in v]),
                               n_reward_errors=sum("reward_error" in x for x in v)))
    done = {p["pos"] for p in per_prompt}
    kept = [p for p in per_prompt if KEEP[0] <= p["n_correct"] <= KEEP[1]]
    res = dict(schema=SUMMARY_SCHEMA, status="COMPLETE" if len(per_prompt) == n_positions else "PARTIAL",
               positions=[0, n_positions], n_positions_complete=len(per_prompt),
               n_positions_incomplete=sum(1 for p in by_pos if p not in done), group=group, seed_base=seed_base,
               correct_threshold=threshold, n_samples=sum(len(g) for g in by_pos.values()), n_samples_beyond=beyond,
               n_reward_errors=sum(p["n_reward_errors"] for p in per_prompt),
               n_truncated=sum(p["n_truncated"] for p in per_prompt),
               mean_pass_rate=_mean([p["pass_rate"] for p in per_prompt]),
               n_correct_histogram={str(k): sum(p["n_correct"] == k for p in per_prompt) for k in range(group + 1)},
               keep_rule=f"{KEEP[0]} <= #correct <= {KEEP[1]} of {group}", n_kept=len(kept))
    return res, per_prompt


def chunk_spans(out_dir):
    d = os.path.join(out_dir, "chunks")
    files = sorted(f for f in os.listdir(d) if CHUNK_RE.fullmatch(f)) if os.path.isdir(d) else []
    return [(f, int(CHUNK_RE.fullmatch(f).group(1)), int(CHUNK_RE.fullmatch(f).group(2))) for f in files]


def read_rows_light(out_dir, files):
    """The sample rows of the chunk files, without their response token ids (no summary reads them)."""
    rows = []
    for f in files:
        with open(os.path.join(out_dir, "chunks", f)) as fh:
            for line in fh:
                r = json.loads(line)
                r.pop("response_ids", None)
                rows.append(r)
    return rows


def cmd_summarize(a):
    if not 0 < a.n_positions <= TRAIN_POS_MAX:
        raise SystemExit(f"REFUSE: --n-positions must lie in (0, {TRAIN_POS_MAX}]")
    blocks = read_blocks(a.out_dir)
    if not blocks:
        raise SystemExit(f"REFUSE: {a.out_dir} holds no block records (blocks/b*.json): run `gen` first")
    key = blocks[0]["config_key"]
    if any(b["config_key"] != key for b in blocks):
        raise SystemExit(f"REFUSE: the blocks of {a.out_dir} differ in configuration")
    spans = sorted(tuple(b["positions"]) for b in blocks)
    for (p0, p1), (q0, q1) in zip(spans, spans[1:]):
        if q0 < p1:
            raise SystemExit(f"REFUSE: the blocks [{p0}, {p1}) and [{q0}, {q1}) overlap")
    chunks = chunk_spans(a.out_dir)
    for f, c0, c1 in chunks:
        if not any(b0 <= c0 < c1 <= b1 for b0, b1 in spans):
            raise SystemExit(f"REFUSE: chunk {f} lies in no block of {a.out_dir}")
    cs = sorted((c0, c1, f) for f, c0, c1 in chunks)
    for (p0, p1, f), (q0, q1, g) in zip(cs, cs[1:]):
        if q0 < p1:
            raise SystemExit(f"REFUSE: chunks {f} and {g} overlap")
    files = [f for f, _, _ in chunks]
    res, per_prompt = summarize_rows(read_rows_light(a.out_dir, files), a.n_positions, key["group"],
                                     key["seed_base"], key["correct_threshold"])
    csv_p = os.path.join(a.out_dir, "per_prompt.csv")
    with open(csv_p + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(PER_PROMPT_FIELDS))
        w.writeheader()
        w.writerows(per_prompt)
    os.replace(csv_p + ".tmp", csv_p)
    res.update(per_prompt_csv_sha256=sha256_file(csv_p), config_key=key,
               blocks=[dict(positions=b["positions"], vllm=b.get("vllm"), n_chunks=len(b.get("chunks", [])),
                            data=b.get("data")) for b in sorted(blocks, key=lambda b: b["positions"])],
               vllm_versions=sorted({str(b.get("vllm")) for b in blocks}),
               chunk_files=[dict(file=f, sha256=sha256_file(os.path.join(a.out_dir, "chunks", f))) for f in files])
    _wj(os.path.join(a.out_dir, "summary.json"), res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("chunk_files", "blocks", "config_key")}, indent=1))
    if len(res["vllm_versions"]) > 1:
        log(f"the blocks ran different vLLM versions: {res['vllm_versions']}")
    return res


# ------------------------------------------------------------------------------------------------ subset (CPU)
def select_subset(per_prompt, n_positions, batch=RL_BATCH, max_steps=MAX_STEPS, min_steps=MIN_STEPS, keep=KEEP):
    """per_prompt: dicts with pos, pool_index, n_correct covering [0, n_positions) exactly once. Keeps
    keep[0] <= n_correct <= keep[1] in increasing position; the subset is the first batch * S kept rows,
    S = min(max_steps, K // batch). Exits 4 if S < min_steps."""
    pp = sorted(per_prompt, key=lambda p: int(p["pos"]))
    if [int(p["pos"]) for p in pp] != list(range(n_positions)):
        raise SystemExit(f"REFUSE: the pass does not hold every position of [0, {n_positions}) exactly once")
    kept = [p for p in pp if keep[0] <= int(p["n_correct"]) <= keep[1]]
    k = len(kept)
    steps = min(max_steps, k // batch)
    if steps < min_steps:
        print(f"REFUSE: K = {k} kept positions in [0, {n_positions}) give {k // batch} steps of {batch}, fewer than "
              f"--min-steps {min_steps}; nothing is written", flush=True)
        raise SystemExit(4)
    sub = kept[:batch * steps]
    return dict(K=k, steps=steps, batch=batch, n_rows=len(sub), keep_rate=k / n_positions,
                positions=[int(p["pos"]) for p in sub], pool_indices=[int(p["pool_index"]) for p in sub],
                highest_position=int(sub[-1]["pos"]))


def cmd_subset(a):
    summ_p = os.path.join(a.pass_dir, "summary.json")
    csv_p = os.path.join(a.pass_dir, "per_prompt.csv")
    if not os.path.exists(summ_p):
        raise SystemExit(f"REFUSE: {summ_p} does not exist: run `summarize` first")
    res = _rj(summ_p)
    if res.get("schema") != SUMMARY_SCHEMA or res.get("status") != "COMPLETE" or res.get("group") != GROUP:
        raise SystemExit(f"REFUSE: {summ_p} is not a COMPLETE pass with {GROUP} samples per prompt "
                         f"(status={res.get('status')}, group={res.get('group')})")
    p0, n = res["positions"]
    if p0 != 0 or not 0 < n <= TRAIN_POS_MAX:
        raise SystemExit(f"REFUSE: the pass covers [{p0}, {n}), not [0, N) with N <= {TRAIN_POS_MAX}")
    if sha256_file(csv_p) != res.get("per_prompt_csv_sha256"):
        raise SystemExit(f"REFUSE: {csv_p} is not the per_prompt.csv that summarize wrote with {summ_p}")
    data_sha = sha256_file(a.data)
    if data_sha != res["config_key"]["data_sha256"]:
        raise SystemExit(f"REFUSE: --data sha256 {data_sha} is not the file this pass sampled "
                         f"({res['config_key']['data_sha256']})")
    if os.path.exists(a.out):
        raise SystemExit(f"REFUSE: {a.out} exists")
    with open(csv_p) as f:
        per_prompt = list(csv.DictReader(f))
    sel = select_subset(per_prompt, n, a.batch, a.max_steps, a.min_steps)
    import pandas as pd
    import pyarrow
    df = pd.read_parquet(a.data)
    sub = df.iloc[sel["pool_indices"]].reset_index(drop=True)
    contents = [user_message(sub.iloc[i]["prompt"], i)["content"] for i in range(len(sub))]
    seen, dup_rows = {}, 0
    for c in contents:
        dup_rows += int(c in seen)
        seen[c] = seen.get(c, 0) + 1
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sub.to_parquet(a.out + ".tmp.parquet", index=False)
    os.replace(a.out + ".tmp.parquet", a.out)
    sha = sha256_file(a.out)
    rec = dict(schema=SUBSET_SCHEMA, rule=f"{KEEP[0]} <= #correct <= {KEEP[1]} of {GROUP} base samples, increasing "
               f"order position; the first {a.batch} * S kept positions, S = min({a.max_steps}, floor(K / {a.batch}))",
               parquet=os.path.realpath(a.out), sha256=sha, sha256_reference=SUBSET_SHA256_REF,
               equals_reference=sha == SUBSET_SHA256_REF, n_rows=len(sub), steps=sel["steps"], batch=sel["batch"],
               K=sel["K"], n_positions=n, keep_rate=sel["keep_rate"], highest_position=sel["highest_position"],
               positions=sel["positions"], pool_indices=sel["pool_indices"],
               within_subset_duplicates=dict(rows_repeating_an_earlier_text=dup_rows,
                                             distinct_texts_repeated=sum(v > 1 for v in seen.values()),
                                             note="counted, never removed"),
               data=os.path.realpath(a.data), data_sha256=data_sha, per_prompt_csv_sha256=res["per_prompt_csv_sha256"],
               summary_sha256=sha256_file(summ_p), pandas=pd.__version__, pyarrow=pyarrow.__version__)
    _wj(a.out + ".json", rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("positions", "pool_indices")}, indent=1))
    if rec["steps"] < a.max_steps:
        log(f"the subset holds {rec['steps']} steps of {a.batch}, fewer than {a.max_steps}: train with "
            f"STEPS={rec['steps']}")
    return rec


# ------------------------------------------------------------------------------------------------ CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("jr", help="the JR prompt file from the DAPO parquet")
    p.add_argument("--dapo", default=DAPO)
    p.add_argument("--out", default=JR_DATA)
    p.add_argument("--tokenizer", default=STUDENT,
                   help="the student's tokenizer for the prompt-length check; '' skips it")
    p.add_argument("--no-builder", action="store_true", help="skip the length check through verl's continuous-token "
                                                             "builder (the text path is still checked)")
    p.add_argument("--allow-other-data", action="store_true")
    p = sub.add_parser("order", help="the training order (data.shuffle=True, data.seed=1) as a JSON list")
    p.add_argument("--data", default=DAPO)
    p.add_argument("--out", default=ORDER_JSON)
    p.add_argument("--allow-other-data", action="store_true")
    p = sub.add_parser("gen", help="GPU: the base student's samples for a block of order positions")
    p.add_argument("--out-dir", default=PASS_DIR)
    p.add_argument("--data", default=JR_DATA,
                   help="the parquet whose prompts are sampled (row-aligned with --order-data)")
    p.add_argument("--order-data", default=DAPO, help="the parquet the training order is computed on")
    p.add_argument("--allow-other-data", action="store_true")
    p.add_argument("--student", default=STUDENT)
    p.add_argument("--pos-lo", type=int, default=0, help="first order position of this block")
    p.add_argument("--pos-hi", type=int, default=N_POSITIONS, help="one past the last order position of this block")
    p.add_argument("--chunk", type=int, default=CHUNK, help="prompts per generate call (one chunk file each)")
    p.add_argument("--seed-base", type=int, default=SEED_BASE, help="sample seed = base + 8 * pos + s")
    p.add_argument("--gpu-util", type=float, default=0.90)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--dry-run", action="store_true", help="checks and prompts only: print the plan, write nothing")
    p = sub.add_parser("summarize", help="per_prompt.csv and summary.json from the chunk files")
    p.add_argument("--out-dir", default=PASS_DIR)
    p.add_argument("--n-positions", type=int, default=N_POSITIONS, help="order positions [0, N) to summarize")
    p = sub.add_parser("subset", help="the training subset parquet from a COMPLETE pass")
    p.add_argument("--pass-dir", default=PASS_DIR)
    p.add_argument("--data", default=JR_DATA, help="the parquet the pass sampled; the subset rows are taken from it")
    p.add_argument("--out", default=SUBSET)
    p.add_argument("--batch", type=int, default=RL_BATCH, help="prompts per RL step")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--min-steps", type=int, default=MIN_STEPS)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    return dict(jr=cmd_jr, order=cmd_order, gen=cmd_gen, summarize=cmd_summarize, subset=cmd_subset)[a.cmd](a)


if __name__ == "__main__":
    main()
