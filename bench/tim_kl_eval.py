"""Held-out KL(p_theta || p_T) of a trained checkpoint, measured on its OWN samples.

The training log's actor/distillation/loss is a token mean over the SAMPLER's tokens (y ~ q), so under a mismatched
sampler it estimates KL(q||p_T) - KL(q||p_theta), not KL(p_theta||p_T): the sampler alone moves it. This script
measures the sampler-independent quantity instead. refresh/t3_acc.py gen samples each arm's trained checkpoint at
T=1.0 with NO speculation and seeds fixed per (problem, sample) (gen_T1_p1.jsonl: AIME24 30x8 + MATH-500 500x1);
here every one of those traces is TEACHER-FORCED through the checkpoint itself and through the teacher
(prompt_logprobs=0: the log-prob of the actual next token at every position), so per response token
    k1 = log p_theta(y_t | y_<t) - log p_T(y_t | y_<t),   y ~ p_theta  ->  E[k1] = KL(p_theta || p_T) per token.

    python bench/tim_kl_eval.py score   --model M --tokenizer TOK --gen D/gen_T1_p1.jsonl --out D/lp_<role>.npz
    python bench/tim_kl_eval.py combine --dir D        -> D/kl_teacher.jsonl (one row per sample) + kl_summary.json

score runs one vLLM engine IN-PROCESS on one GPU (one process per model: two engines in one process do not reliably
release memory). Prompts are built exactly as refresh/t3_acc.py builds them (benchmark chat list, BASE student's
chat template, add_generation_prompt, no special tokens), and the script checks the response ids round-trip.
combine is numpy-only. Per sample it writes n_tok, sum_k1 (token sum), sum_lp_student (its own NLL = the sampled
tokens' surprisal, an entropy proxy), and the per-problem token-mean k1 (for comparisons paired by problem).
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
DATA = os.environ.get("CLUSTER_ROOT", "/path/to/workdir") + "/workspace/datasets/test_data"


def prompts_for(tokenizer_dir):
    import pandas as pd
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    out = {}
    for b in ("AIME24", "MATH-500"):
        df = pd.read_parquet(os.path.join(DATA, b, "test.parquet"))
        for i, r in enumerate(df.to_dict("records")):
            text = tok.apply_chat_template([dict(m) for m in r["prompt"]], tokenize=False, add_generation_prompt=True)
            out[(b, i)] = tok(text, add_special_tokens=False)["input_ids"]
    return out


def cmd_score(a):
    import numpy as np
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    rows = [json.loads(l) for l in open(a.gen)]
    if a.limit:
        rows = rows[: a.limit]
    P = prompts_for(a.tokenizer)
    reqs, keys, nresp = [], [], []
    for r in rows:
        pids = P[(r["bench"], r["i"])]
        reqs.append(TokensPrompt(prompt_token_ids=pids + r["response_ids"]))
        keys.append((r["bench"], r["i"], r["s"], len(pids)))
        nresp.append(len(r["response_ids"]))
    maxlen = max(len(q["prompt_token_ids"]) for q in reqs)
    print(f"[kl] {len(reqs)} traces, {sum(nresp)} response tokens, longest {maxlen} tokens; model {a.model}", flush=True)
    t0 = time.time()
    llm = LLM(model=a.model, tokenizer=a.tokenizer, dtype="bfloat16", gpu_memory_utilization=a.gpu_util,
              max_model_len=maxlen + 16, enable_prefix_caching=False, seed=0, max_num_seqs=32,
              max_num_batched_tokens=a.batched_tokens)
    print(f"[kl] engine up in {time.time() - t0:.0f}s", flush=True)
    sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0)
    arrays = {}
    t = time.time()
    for c in range(0, len(reqs), a.chunk):  # chunks bound the host memory of prompt_logprobs dicts
        outs = llm.generate(reqs[c: c + a.chunk], [sp] * len(reqs[c: c + a.chunk]), use_tqdm=False)
        for k, (q, o) in enumerate(zip(reqs[c: c + a.chunk], outs)):
            ids = q["prompt_token_ids"]
            b, i, s, plen = keys[c + k]
            pl = o.prompt_logprobs
            arrays[f"{b}|{i}|{s}"] = np.asarray([pl[j][ids[j]].logprob for j in range(plen, len(ids))], dtype=np.float32)
        print(f"[kl] {min(c + a.chunk, len(reqs))}/{len(reqs)} scored ({time.time() - t:.0f}s)", flush=True)
    tmp = a.out + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, a.out)
    ntok = sum(x.size for x in arrays.values())
    print(f"[kl] wrote {a.out}: {len(arrays)} traces, {ntok} tokens, mean logp {sum(float(x.sum()) for x in arrays.values()) / ntok:.4f} "
          f"({time.time() - t:.0f}s scoring)", flush=True)


def cmd_combine(a):
    import numpy as np
    S = np.load(os.path.join(a.dir, "lp_student.npz"))
    T = np.load(os.path.join(a.dir, "lp_teacher.npz"))
    rows = []
    for k in S.files:
        s, t = S[k].astype(np.float64), T[k].astype(np.float64)
        assert s.shape == t.shape, (k, s.shape, t.shape)
        b, i, smp = k.split("|")
        k1 = s - t
        rows.append(dict(bench=b, i=int(i), s=int(smp), n_tok=int(s.size), sum_k1=float(k1.sum()),
                         sum_lp_student=float(s.sum()), sum_k3=float(np.sum(np.exp(-k1) - 1 + k1))))
    rows.sort(key=lambda r: (r["bench"], r["i"], r["s"]))
    with open(os.path.join(a.dir, "kl_teacher.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summ = {}
    for b in ("AIME24", "MATH-500", "ALL"):
        rs = [r for r in rows if b == "ALL" or r["bench"] == b]
        per = {}
        for r in rs:
            p = per.setdefault((r["bench"], r["i"]), [0.0, 0])
            p[0] += r["sum_k1"]
            p[1] += r["n_tok"]
        summ[b] = dict(n_samples=len(rs), n_problems=len(per), n_tok=sum(r["n_tok"] for r in rs),
                       kl_token_pooled=sum(r["sum_k1"] for r in rs) / max(1, sum(r["n_tok"] for r in rs)),
                       kl_problem_mean=float(np.mean([v[0] / v[1] for v in per.values() if v[1] > 0])),
                       k3_token_pooled=sum(r["sum_k3"] for r in rs) / max(1, sum(r["n_tok"] for r in rs)),
                       nll_token_pooled=-sum(r["sum_lp_student"] for r in rs) / max(1, sum(r["n_tok"] for r in rs)))
    json.dump(summ, open(os.path.join(a.dir, "kl_summary.json"), "w"), indent=1)
    print("[kl] " + json.dumps(summ), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--model", required=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--gen", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--gpu-util", type=float, default=0.60)
    s.add_argument("--batched-tokens", type=int, default=8192)
    s.add_argument("--chunk", type=int, default=128)
    s.add_argument("--limit", type=int, default=0)
    c = sub.add_parser("combine")
    c.add_argument("--dir", required=True)
    a = ap.parse_args()
    {"score": cmd_score, "combine": cmd_combine}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
