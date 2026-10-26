"""Held-out accuracy of a checkpoint (no speculation) on AIME24 x8 and MATH-500 x1.

    python t3_acc.py gen   --model <ckpt hf dir> --tokenizer <base student dir> --out-dir <dir> --temps 1.0:1.0,0.6:0.95
    python t3_acc.py grade --out-dir <dir>          (needs the verifier chain: run with venv-specopd)

gen: one vLLM 0.19.1 engine, NO speculative decoding. Prompts are the benchmarks' own chat lists, formatted with the
BASE student's chat template (identical token ids for every checkpoint; R1-Distill's template opens <think>), max 16384
new tokens. AIME24: 30 problems x 8 independent samples; MATH-500: 500 x 1. Every sample is its own request with seed
= seed + 1000 * bench_no + 8 * problem + sample, so the same (problem, sample) gets the same seed at every checkpoint
(prompt-paired design). Temperatures run one after the other (T=1.0 first, written to disk before the next starts).
grade: decode (skip special tokens) and score with custom_reward.ttrl_math.reward_func (this tree's math verifier,
last \\boxed{} answer); writes graded_<tag>.jsonl (no text) and summary_<tag>.json.
"""
import argparse
import json
import os
import sys
import time

BENCH = {"AIME24": 8, "MATH-500": 1}
DATA = os.environ.get("CLUSTER_ROOT", "/path/to/workdir") + "/workspace/datasets/test_data"


def load_bench(name):
    import pandas as pd
    df = pd.read_parquet(os.path.join(DATA, name, "test.parquet"))
    out = []
    for i, r in enumerate(df.to_dict("records")):
        rm = r["reward_model"]
        if isinstance(rm, str):
            import ast
            rm = ast.literal_eval(rm)
        out.append(dict(i=i, id=str(r.get("id", i)), prompt=[dict(m) for m in r["prompt"]], gt=str(rm["ground_truth"])))
    return out


def tag_of(temp, top_p):
    return f"T{temp:g}_p{top_p:g}"


def cmd_gen(a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    items = []
    for bno, (name, n) in enumerate(BENCH.items()):
        for p in load_bench(name):
            text = tok.apply_chat_template(p["prompt"], tokenize=False, add_generation_prompt=True)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            for s in range(n):
                items.append(dict(bench=name, i=p["i"], id=p["id"], s=s, gt=p["gt"], prompt_ids=ids,
                                  seed=a.seed + 1000 * bno + 8 * p["i"] + s))
    print(f"[acc] {len(items)} requests; first prompt: {tok.decode(items[0]['prompt_ids'])[-160:]!r}", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    llm = LLM(model=a.model, tokenizer=a.tokenizer, dtype="bfloat16", gpu_memory_utilization=a.gpu_util,
              max_model_len=a.max_tokens + 1024, max_num_seqs=a.max_num_seqs, seed=a.seed, enable_prefix_caching=True)
    init_s = time.time() - t0
    for tp in a.temps.split(","):
        temp, top_p = (float(x) for x in tp.split(":"))
        tag = tag_of(temp, top_p)
        path = os.path.join(a.out_dir, f"gen_{tag}.jsonl")
        if os.path.exists(path):
            print(f"[acc] {path} exists, skipping", flush=True)
            continue
        sps = [SamplingParams(temperature=temp, top_p=top_p, max_tokens=a.max_tokens, seed=it["seed"]) for it in items]
        t = time.time()
        outs = llm.generate([TokensPrompt(prompt_token_ids=it["prompt_ids"]) for it in items], sps, use_tqdm=False)
        wall = time.time() - t
        ntok = 0
        with open(path + ".tmp", "w") as f:
            for it, o in zip(items, outs):
                r = o.outputs[0]
                ntok += len(r.token_ids)
                f.write(json.dumps(dict(bench=it["bench"], i=it["i"], id=it["id"], s=it["s"], gt=it["gt"], seed=it["seed"],
                                        resp_len=len(r.token_ids), finish_reason=r.finish_reason,
                                        response_ids=list(r.token_ids))) + "\n")
        os.replace(path + ".tmp", path)
        meta = dict(model=a.model, tokenizer=a.tokenizer, temp=temp, top_p=top_p, max_tokens=a.max_tokens, n=len(items),
                    out_tokens=ntok, wall_s=round(wall, 1), init_s=round(init_s, 1), tok_per_s=round(ntok / wall, 1),
                    max_num_seqs=a.max_num_seqs, seed=a.seed, speculative=None)
        json.dump(meta, open(os.path.join(a.out_dir, f"gen_{tag}_meta.json"), "w"), indent=1)
        print(f"[acc] gen {tag} " + json.dumps(meta), flush=True)


def cmd_grade(a):
    import glob
    from transformers import AutoTokenizer
    from custom_reward.ttrl_math import reward_func
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    for path in sorted(glob.glob(os.path.join(a.out_dir, "gen_T*.jsonl"))):
        tag = os.path.basename(path)[4:-6]
        out = os.path.join(a.out_dir, f"graded_{tag}.jsonl")
        if os.path.exists(out):
            print(f"[acc] {out} exists, skipping", flush=True)
            continue
        rows = []
        t = time.time()
        for line in open(path):
            r = json.loads(line)
            text = tok.decode(r.pop("response_ids"), skip_special_tokens=True)
            try:
                res = reward_func(r["bench"], text, r["gt"])
                r["acc"] = float(res["acc"]) if isinstance(res, dict) else float(res > 0)
                r["pred"] = (res.get("pred") or "")[:200] if isinstance(res, dict) else ""
                r["has_boxed"] = "\\boxed" in text
            except Exception as e:  # noqa: BLE001 -- a verifier crash on one sample is a wrong answer, recorded
                r["acc"], r["pred"], r["grade_error"] = 0.0, "", repr(e)[:200]
            r["closed_think"] = "</think>" in text
            rows.append(r)
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summ = {}
        for b in BENCH:
            rs = [r for r in rows if r["bench"] == b]
            summ[b] = dict(n=len(rs), acc=sum(r["acc"] for r in rs) / len(rs),
                           trunc=sum(r["finish_reason"] == "length" for r in rs) / len(rs),
                           resp_len=sum(r["resp_len"] for r in rs) / len(rs),
                           grade_errors=sum("grade_error" in r for r in rs))
        summ["grade_s"] = round(time.time() - t, 1)
        json.dump(summ, open(os.path.join(a.out_dir, f"summary_{tag}.json"), "w"), indent=1)
        print(f"[acc] graded {tag} " + json.dumps(summ), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--model", required=True)
    g.add_argument("--tokenizer", required=True)
    g.add_argument("--out-dir", required=True)
    g.add_argument("--temps", default="1.0:1.0,0.6:0.95")
    g.add_argument("--max-tokens", type=int, default=16384)
    g.add_argument("--max-num-seqs", type=int, default=256)
    g.add_argument("--gpu-util", type=float, default=0.90)
    g.add_argument("--seed", type=int, default=2000)
    r = sub.add_parser("grade")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--tokenizer", required=True)
    a = ap.parse_args()
    {"gen": cmd_gen, "grade": cmd_grade}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
