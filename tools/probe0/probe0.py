"""The exact gradient probe: the parameter-space gradients of reverse-KL OPD under a mismatched sampler, in closed
form.

One GH200, HF transformers + torch only, no verl. Models: student theta_0 = DeepSeek-R1-Distill-Qwen-7B (p), teacher
Skywork-OR1-Math-7B (r), and one sampler q_m per level m of --sigmas (name=sigma pairs, e.g. "nhigh=0.15"): the
student with the rollout engine's SamplerNoise (default seed 20261001) applied by the TRAINING code (specopd_tim.py, a
copy of verl/verl/utils/specopd_tim.py; --tim-ref compares the two). FP8 samplers are not probed.

  phase T   teacher forward over every trace -> final hidden states of the response rows (CPU bf16 cache)
  phase I   identity (fp32 student, then bf16): pulled-back closed-form G_KL vs vanilla HF autograd of the full-vocab
            KL, and vs the pulled-back autograd-in-logit-space gradient; logit-space max error
  phase Q   q_m hidden caches (noise applied to the student in place, record checked against a training run's
            step-0 [SPECOPD-TIM] line when --expected-step0 is given, then restored bitwise)
  phase A   part (a): exact expectations on the part-(a) traces (--a-traces); fields KL, bias_m, drift_m, andiff_m
            pulled back; Gram + <., n_hat>; per-row logit-space distributions; the K gate and the bias gate (read at
            the last --sigmas level)
  phase B   part (b): sampled k1 vs analytic on AIME24 problems x 8 base-student samples (--b-traces), split-half SNR
            within prompt

Output: <out>/probe0.json (all numbers, gate verdict), <out>/rows_a_<pass>.npz, <out>/rows_b.npz.
"""
import argparse
import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe_lib as PL  # noqa: E402

R = os.environ.get("CLUSTER_ROOT", "/path/to/workdir")
STUDENT = f"{R}/hf/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60"
TEACHER = f"{R}/hf/hub/models--Skywork--Skywork-OR1-Math-7B/snapshots/fcef970f1d10682f13c182b280146a194ed57d1e"
DATA = f"{R}/workspace/datasets/test_data"
STEP0_KEYS = ("d_eps", "d_n", "nnorm1", "n_tensors", "n_elems")
A_TRACES_HELP = (
    "part (a): JSONL of the base student's own responses, one per line with trace_id (unique), prompt_idx, prompt_ids, "
    "response_ids, finish_reason and seed; required unless --skip-a. No script in this repository writes it. Recipe: "
    "64 distinct DAPO-Math-17k prompts (a seeded uniform sample; prompt_idx = the row in "
    "$CLUSTER_ROOT/workspace/datasets/dapo-math-17k.parquet); prompt_ids = the base student's chat template applied "
    "to the row's prompt with add_generation_prompt=True, tokenized with add_special_tokens=False (so they end with "
    "the ids 151645 151648 198: the assistant tag, <think> and a newline); one response per prompt, sampled by vLLM "
    "from the base student in bf16 at T=1.0, top_p=1.0, at most 16384 new tokens, request seed 1000 + prompt_idx "
    "(the seed field), with EAGLE-3 speculative decoding and 3 draft tokens (lossless: plain sampling draws from the "
    "same distribution, though not the same tokens); response_ids and finish_reason ('stop' or 'length') as vLLM "
    "returns them (the final EOS token included)")
B_TRACES_HELP = (
    "part (b): the gen_T1_p1.jsonl that 'refresh/t3_acc.py gen --model <base student dir> --tokenizer <base student "
    "dir> --out-dir <dir> --temps 1.0:1.0' writes into <dir> (AIME24 x 8 and MATH-500 x 1 samples of the base "
    "student, no speculative decoding, T=1.0, top_p=1.0; at most 16384 new tokens and base seed 2000 are t3_acc.py's "
    "defaults); required unless --skip-b")

T0 = time.time()


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log(*a):
    print(f"[probe0 {time.strftime('%H:%M:%S')} +{time.time() - T0:7.0f}s]", *a, flush=True)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o if not isinstance(o, (np.floating, np.integer)) else o.item())
    os.replace(tmp, path)


class Trace:
    def __init__(self, key, prompt_ids, response_ids, meta):
        self.key = key
        self.ids = torch.tensor(list(prompt_ids) + list(response_ids), dtype=torch.long)
        self.n_prompt = len(prompt_ids)
        self.T = len(response_ids)
        self.meta = meta

    def truncated(self, n):
        t = Trace(self.key + f"@{n}", self.ids[:self.n_prompt].tolist(), self.ids[self.n_prompt:self.n_prompt + n].tolist(),
                  dict(self.meta, truncated_to=n))
        t.parent = self.key
        return t


def load_a(path, max_traces, max_resp, checks):
    rows = [json.loads(line) for line in open(path)]
    need = {"trace_id", "prompt_idx", "prompt_ids", "response_ids", "finish_reason", "seed"}
    assert all(need <= set(r) for r in rows), f"{path}: missing fields"
    checks["a_file"] = dict(path=path, n_lines=len(rows), n_prompts=len({r["prompt_idx"] for r in rows}),
                            finish=dict((k, sum(r["finish_reason"] == k for r in rows)) for k in ("stop", "length")),
                            total_response_tokens=sum(len(r["response_ids"]) for r in rows))
    if max_traces:
        rows = rows[:max_traces]
    out = []
    for r in rows:
        resp = r["response_ids"][:max_resp] if max_resp else r["response_ids"]
        out.append(Trace(r["trace_id"], r["prompt_ids"], resp, dict(prompt_idx=r["prompt_idx"],
                                                                       full_len=len(r["response_ids"]))))
    return out


def load_b(path, bench, problems, samples, half_a, max_resp, tok, checks, data_dir=DATA):
    import pandas as pd
    df = pd.read_parquet(os.path.join(data_dir, bench, "test.parquet"))
    prompts = [[dict(m) for m in r["prompt"]] for r in df.to_dict("records")]
    rows = [json.loads(line) for line in open(path)]
    rows = [r for r in rows if r["bench"] == bench and r["i"] in problems and r["s"] in samples]
    rows.sort(key=lambda r: (r["i"], r["s"]))
    assert len(rows) == len(problems) * len(samples), f"part (b): {len(rows)} rows, expected {len(problems) * len(samples)}"
    out = []
    for r in rows:
        # prompts built exactly as bench/tim_calibrate.py load_traces (= refresh/t3_acc.py)
        text = tok.apply_chat_template(prompts[r["i"]], tokenize=False, add_generation_prompt=True)
        pids = tok(text, add_special_tokens=False)["input_ids"]
        resp = r["response_ids"][:max_resp] if max_resp else r["response_ids"]
        t = Trace(f"{bench}_{r['i']}_s{r['s']}", pids, resp, dict(i=r["i"], s=r["s"], full_len=len(r["response_ids"]),
                                                                   finish=r["finish_reason"]))
        t.half = "A" if r["s"] in half_a else "B"
        out.append(t)
    checks["b_file"] = dict(path=path, bench=bench, problems=problems, samples=samples, half_a=sorted(half_a),
                            n=len(out), total_response_tokens=sum(t.T for t in out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    ap.add_argument("--student", default=STUDENT)
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--a-traces", default=None, help=A_TRACES_HELP)
    ap.add_argument("--a-max-traces", type=int, default=0)
    ap.add_argument("--max-resp", type=int, default=0, help="truncate every response to this many tokens (0 = no)")
    ap.add_argument("--b-traces", default=None, help=B_TRACES_HELP)
    ap.add_argument("--b-bench", default="AIME24")
    ap.add_argument("--b-problems", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--b-samples", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--b-half-a", default="0,1,2,3")
    ap.add_argument("--sigmas", required=True,
                    help='comma-separated name=sigma pairs, one noisy sampler per pair, e.g. "nhigh=0.15" (the bias '
                         "gate reads the last pair; bench/prefix_shift.py repro compares with the level at sigma 0.15)")
    ap.add_argument("--noise-seed", type=int, default=20261001)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--extra-ks", default="32,64,256,512")
    ap.add_argument("--chunk", type=int, default=512, help="response rows per lm_head / field chunk")
    ap.add_argument("--fdtype", choices=["float32", "float64"], default="float64", help="logit-space field dtype")
    ap.add_argument("--groups", choices=["level", "all"], default="level",
                    help="level: one pass per sampler level (KL kept; peak 4 CPU fp32 accumulators = 122 GB); "
                         "all: one pass with all 8 part-(a) fields (244 GB)")
    ap.add_argument("--id-traces", type=int, default=2)
    ap.add_argument("--id-len", type=int, default=1024)
    ap.add_argument("--id-fp32", type=int, default=1)
    ap.add_argument("--id-bf16", type=int, default=1)
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 for the real run; cpu only for the tiny-model dry run")
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--tim-ref", default=None,
                    help="the training code's specopd_tim.py: if given, it must be byte-identical to the probe's copy")
    ap.add_argument("--expected-step0", default=None,
                    help="JSON with d_eps, d_n, nnorm1, n_tensors, n_elems copied from the step-0 [SPECOPD-TIM] line "
                         "of a training run with the same student, noise seed and GPU type: the projection of the "
                         "CLEAN student on eps / n, i.e. a fingerprint of the noise vector and of the scales. Absent: "
                         "the noise record and n_hat are not checked")
    ap.add_argument("--allow-noise-mismatch", action="store_true")
    a = ap.parse_args()
    if not a.skip_a and not a.a_traces:
        ap.error("--a-traces is required unless --skip-a")
    if not a.skip_b and not a.b_traces:
        ap.error("--b-traces is required unless --skip-b")
    if a.mode == "smoke":
        a.a_max_traces = a.a_max_traces or 2
        a.max_resp = a.max_resp or 1024
        a.b_problems, a.b_samples = "0", "0,4"
        a.id_traces, a.id_len = 1, 256
    os.makedirs(a.out_dir, exist_ok=True)
    out_json = os.path.join(a.out_dir, "probe0.json")
    dev = torch.device(a.device)
    on_gpu = dev.type == "cuda"
    fd = getattr(torch, a.fdtype)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sigmas = [(s.split("=")[0], float(s.split("=")[1])) for s in a.sigmas.split(",") if s]
    extra_ks = [int(x) for x in a.extra_ks.split(",") if x]
    res = dict(args=vars(a), checks={}, timing={},
               sign_convention="every g is a LOSS gradient (sum_t J_t^T G_t / N); an SGD step moves theta by -lr*g, "
                               "so the movement along n_hat (positive = toward the sampler, as d_n) is -lr*<g, n_hat>",
               gpu=torch.cuda.get_device_name(0) if on_gpu else "cpu", torch=torch.__version__)
    import transformers
    res["transformers"] = transformers.__version__
    tm = PL.Timer()
    res["timing"] = tm.t

    def save():
        res["elapsed_s"] = round(time.time() - T0, 1)
        if on_gpu:
            res["max_mem_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        write_json(out_json, res)

    # ---------------- code identity -------------------------------------------------------------------------------
    tim_path = os.path.join(HERE, "specopd_tim.py")
    tim = PL.load_specopd_tim(tim_path)
    res["checks"]["specopd_tim_sha256"] = PL.sha256_file(tim_path)
    if a.tim_ref:
        assert PL.sha256_file(a.tim_ref) == res["checks"]["specopd_tim_sha256"], \
            f"{a.tim_ref} differs from the probe's copy"
        res["checks"]["specopd_tim_ref_equal"] = a.tim_ref
    else:
        log("no --tim-ref: skipping the byte comparison of the probe's specopd_tim.py with the training code's copy")
    EXP = None
    if a.expected_step0:
        with open(a.expected_step0) as f:
            EXP = json.load(f)
        missing = [k for k in STEP0_KEYS if k not in EXP]
        assert not missing, f"{a.expected_step0}: missing {missing}"
        EXP = {k: EXP[k] for k in STEP0_KEYS}
    else:
        log("no --expected-step0: skipping the check of the noise record and of n_hat against a training run's "
            "step-0 line")

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.student)
    cs, ct = AutoConfig.from_pretrained(a.student), AutoConfig.from_pretrained(a.teacher)
    tt = AutoTokenizer.from_pretrained(a.teacher)
    res["checks"]["vocab"] = dict(student_config=cs.vocab_size, teacher_config=ct.vocab_size, len_tokenizer=len(tok),
                                  tokenizers_equal=tok.get_vocab() == tt.get_vocab())
    assert cs.vocab_size == ct.vocab_size, "vocab sizes differ: logits would need slicing"
    assert res["checks"]["vocab"]["tokenizers_equal"]
    V = cs.vocab_size
    NTOK = len(tok)

    def load_model(path, dtype):
        m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation=a.attn)
        m.config.use_cache = False
        m = m.to(dev)
        assert m.get_output_embeddings().weight.shape[0] == V
        return m

    A = [] if a.skip_a else load_a(a.a_traces, a.a_max_traces, a.max_resp, res["checks"])
    B = [] if a.skip_b else load_b(a.b_traces, a.b_bench, [int(x) for x in a.b_problems.split(",")],
                                   [int(x) for x in a.b_samples.split(",")],
                                   set(int(x) for x in a.b_half_a.split(",")), a.max_resp, tok, res["checks"],
                                   data_dir=a.data_dir)
    if A:
        assert all(t.ids[t.n_prompt - 3:t.n_prompt].tolist() == [151645, 151648, 198] for t in A), "A prompt suffix"
    if B:
        assert all(t.ids[t.n_prompt - 3:t.n_prompt].tolist() == [151645, 151648, 198] for t in B), "B prompt suffix"
    ID = [t.truncated(min(a.id_len, t.T)) for t in A[:a.id_traces]] if A else []
    log(f"traces: A {len(A)} ({sum(t.T for t in A)} rows), B {len(B)} ({sum(t.T for t in B)} rows), identity {len(ID)}")
    save()

    # ---------------- phase T: teacher hidden-state cache ----------------------------------------------------------
    @torch.no_grad()
    def cache_rows(model, traces):
        model.eval()
        out = {}
        for t in traces:
            h = PL.forward_hidden(model, t.ids.to(dev), grad=False)
            out[t.key] = h[t.n_prompt - 1:t.n_prompt - 1 + t.T].to("cpu")
            del h
        return out

    with tm("T_teacher"):
        Tm = load_model(a.teacher, torch.bfloat16)
        W_T = Tm.get_output_embeddings().weight.detach().clone()
        hT = cache_rows(Tm, A + B)
        del Tm
        gc.collect(); empty_cache()
    for t in ID:
        hT[t.key] = hT[t.parent][:t.T]
    log("teacher cache done")

    def logr_rows(key, c0, c1):
        return torch.log_softmax(F.linear(hT[key][c0:c1].to(dev), W_T).to(fd), -1)

    # ---------------- phase I: identity ----------------------------------------------------------------------------
    def identity(model, tag):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        ref, cf, au = PL.FieldAcc(model), PL.FieldAcc(model), PL.FieldAcc(model)
        lerr = dict(max_abs_diff=0.0, max_abs=0.0, sum_sq_diff=0.0, sum_sq=0.0)
        kl_ref = 0.0
        for t in ID:
            ids = t.ids.to(dev)
            logr = logr_rows(t.key, 0, t.T)
            kl_ref += PL.reference_kl_grad(model, ids, t.n_prompt, logr, ref)

            def ff(z, c0, c1, logr=logr):
                out, _ = PL.level_fields(z, logr[c0:c1], None, k=a.k, want=("KL",))
                g = PL.kl_autograd_logit(z, logr[c0:c1])
                d = g - out["KL"]
                lerr["max_abs_diff"] = max(lerr["max_abs_diff"], float(d.abs().max()))
                lerr["max_abs"] = max(lerr["max_abs"], float(g.abs().max()))
                lerr["sum_sq_diff"] += float((d * d).sum())
                lerr["sum_sq"] += float((g * g).sum())
                return {"KL": out["KL"], "KLauto": g}, {}
            PL.pullback_trace(model, ids, t.n_prompt, ff, {"KL": cf, "KLauto": au}, a.chunk, fd)
        ro = PL.readout({"ref": ref.tensors(), "KL": cf.tensors(), "KLauto": au.tensors()}, dev,
                        diff_pairs=[("ref", "KL"), ("KLauto", "KL"), ("ref", "KLauto")])
        G = ro["gram"]
        o = dict(dtype=tag, n_traces=len(ID), rows=sum(t.T for t in ID), kl_sum=kl_ref,
                 logit_rel_max_err=lerr["max_abs_diff"] / max(lerr["max_abs"], 1e-300),
                 logit_rel_l2_err=math.sqrt(lerr["sum_sq_diff"] / max(lerr["sum_sq"], 1e-300)))
        for x, y in ro["diff"]:
            o[f"param_rel_l2_{x}_vs_{y}"] = math.sqrt(ro["diff"][(x, y)] / G[x][x])
            o[f"param_cos_{x}_vs_{y}"] = G[x][y] / math.sqrt(G[x][x] * G[y][y])
            o[f"param_rel_maxabs_{x}_vs_{y}"] = ro["maxabs"][(x, y)]
        o["norm_ref"] = math.sqrt(G["ref"]["ref"])
        del ref, cf, au, ro
        gc.collect(); empty_cache()
        return o

    if ID and a.id_fp32:
        with tm("I_identity_fp32"):
            S32 = load_model(a.student, torch.float32)
            res["identity_fp32"] = identity(S32, "float32")
            del S32
            gc.collect(); empty_cache()
        log("identity fp32:", json.dumps(res["identity_fp32"]))
        save()

    # ---------------- phase Q: samplers ----------------------------------------------------------------------------
    S = load_model(a.student, torch.bfloat16)
    W_S = S.get_output_embeddings().weight
    hQ = {}
    res["noise"] = {}
    for lvl, sigma in sigmas:
        if not A:
            break
        with tm(f"Q_{lvl}"):
            rec, clean = PL.apply_sampler_noise(S, sigma, a.noise_seed, tim)
            ok = None if EXP is None else (all(rec[k] == EXP[k] for k in ("n_tensors", "n_elems")) and all(
                abs(rec[k] - EXP[k]) <= 2e-6 * abs(EXP[k]) for k in ("d_eps", "d_n", "nnorm1")))
            rec = {k: v for k, v in rec.items() if k != "ms"}
            res["noise"][lvl] = dict(sigma=sigma, record=rec, expected_step0=EXP, matches_training_log=ok)
            log(f"noise {lvl}: {rec} matches_training_log={ok}")
            if ok is False and not a.allow_noise_mismatch:
                save()
                raise SystemExit(f"noise record for {lvl} does not match the step-0 line in --expected-step0")
            hQ[lvl] = cache_rows(S, A)
            PL.restore_clean(S, clean)
            del clean
            gc.collect(); empty_cache()
    save()

    if ID and a.id_bf16:
        with tm("I_identity_bf16"):
            res["identity_bf16"] = identity(S, "bfloat16")
        log("identity bf16:", json.dumps(res["identity_bf16"]))
        save()

    S.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    S.train()
    clean_params = dict(S.named_parameters())

    # ---------------- phase A: part (a) ----------------------------------------------------------------------------
    if A:
        N_a = sum(t.T for t in A)
        res["part_a"] = dict(N_rows=N_a, n_traces=len(A), levels={}, rows={})
        lvls = ["none"] + [l for l, _ in sigmas]
        if a.groups == "all":
            passes = [lvls]
        else:
            passes = [[l] for l in lvls]
        acc_KL = None
        for pl in passes:
            fields = []
            for l in pl:
                if l == "none":
                    fields += ["KL", "andiff_none"]
                else:
                    fields += [f"bias_{l}", f"drift_{l}", f"andiff_{l}"]
            accs = {}
            for f in fields:
                accs[f] = PL.FieldAcc(S)
                if f == "KL":
                    acc_KL = accs[f]
            assert acc_KL is not None, "the first pass must contain level none (KL)"
            rows = {}

            def sink(st, rows=rows):
                for k, v in st.items():
                    rows.setdefault(k, []).append(v.float().cpu().numpy())

            tag = "+".join(pl)
            with tm(f"A_pass_{tag}"):
                done = 0
                for ti, t in enumerate(A):
                    ids = t.ids.to(dev)

                    def ff(z, c0, c1, t=t):
                        logr = logr_rows(t.key, c0, c1)
                        out, st_all = {}, {}
                        for l in pl:
                            if l == "none":
                                logq, want = None, ("KL", "andiff")
                            else:
                                logq = torch.log_softmax(F.linear(hQ[l][t.key][c0:c1].to(dev), W_S).to(fd), -1)
                                want = ("bias", "drift", "andiff")
                            o, st = PL.level_fields(z, logr, logq, k=a.k, want=want, extra_ks=extra_ks)
                            for n, g in o.items():
                                out["KL" if n == "KL" else f"{n}_{l}"] = g
                            for n, v in st.items():
                                st_all[f"{l}/{n}"] = v
                            del logq
                        if "none" in pl:
                            st_all["none/pad_mass"] = (torch.logsumexp(z[:, NTOK:], -1) - torch.logsumexp(z, -1)).exp()
                            st_all["trace"] = torch.full((c1 - c0,), ti, dtype=torch.float32)
                            st_all["pos"] = torch.arange(c0, c1, dtype=torch.float32)
                        return out, st_all
                    PL.pullback_trace(S, ids, t.n_prompt, ff, accs, a.chunk, fd, stats_sink=sink)
                    done += t.T
                    if ti % 8 == 0 or ti == len(A) - 1:
                        log(f"part (a) pass {tag}: trace {ti + 1}/{len(A)} ({done}/{N_a} rows)")
            with tm(f"A_readout_{tag}"):
                vecs = {"KL": acc_KL.tensors()}
                for f in fields:
                    if f != "KL":
                        vecs[f] = accs[f].tensors()
                ro = PL.readout(vecs, dev, clean_params=clean_params, tim=tim, seed=a.noise_seed)
            res["part_a"].setdefault("readouts", {})[tag] = dict(
                gram=ro["gram"], gram_targets=ro["gram_t"], proj_raw=ro["proj_raw"], nnorm1=ro["nnorm1"],
                n_tensors=ro["n_tensors"], n_elems=ro["n_elems"])
            if EXP is not None:
                res["checks"].setdefault("nhat_nnorm1_matches_log", []).append(
                    abs(ro["nnorm1"] - EXP["nnorm1"]) <= 2e-6 * EXP["nnorm1"]
                    and ro["n_tensors"] == EXP["n_tensors"] and ro["n_elems"] == EXP["n_elems"])
            for l in pl:
                res["part_a"]["levels"][l] = PL.derive_level(ro, N_a, l, has_bias=(l != "none"))
            # per-row logit-space distributions
            arr = {k: np.concatenate(v) for k, v in rows.items()}
            np.savez_compressed(os.path.join(a.out_dir, f"rows_a_{tag}.npz"), **arr)
            for l in pl:
                d = {}
                for n in ("kl", "L", "P_tau", "student_head_mass", "sampler_head_mass", "teacher_head_mass",
                          "ratio_bias", "ratio_drift", "kl_qp", "kl_pq", "Eqa", "dEqa", "nG_KL", "nG_bias",
                          "nG_drift", "nG_andiff", "tail_term"):
                    if f"{l}/{n}" in arr:
                        d[n] = PL.quantiles(arr[f"{l}/{n}"])
                kl_sum = float(arr[f"{l}/kl"].astype(np.float64).sum())
                d["token_mean_kl"] = kl_sum / N_a
                d["L_over_KL"] = float(arr[f"{l}/L"].astype(np.float64).sum()) / kl_sum
                d["tail_share"] = float(arr[f"{l}/tail_term"].astype(np.float64).sum()) / float(
                    arr[f"{l}/L"].astype(np.float64).sum())
                for kk in extra_ks:
                    if f"{l}/L_k{kk}" in arr:
                        d[f"L_over_KL_k{kk}"] = float(arr[f"{l}/L_k{kk}"].astype(np.float64).sum()) / kl_sum
                if f"{l}/kl_qp" in arr:
                    d["token_mean_kl_qp"] = float(arr[f"{l}/kl_qp"].astype(np.float64).mean())
                    d["token_mean_Eqa"] = float(arr[f"{l}/Eqa"].astype(np.float64).mean())
                if l == "none":
                    d["pad_mass"] = PL.quantiles(arr["none/pad_mass"])
                res["part_a"]["rows"][l] = d
            for f in fields:
                if f != "KL":
                    del accs[f]
            del vecs, ro, accs, rows, arr
            gc.collect(); empty_cache()
            log(f"part (a) pass {tag} done:", json.dumps({l: res["part_a"]["levels"][l] for l in pl}))
            save()
        del acc_KL
        gc.collect()

        # ---- gates ----
        L = res["part_a"]["levels"]
        Rw = res["part_a"]["rows"]
        gate = {}
        hi = sigmas[-1][0]
        if hi in L:
            gate["r_bias_" + hi] = L[hi]["r_bias"]
            gate["bias_gate"] = (f"SMALL (r_bias({hi}) < 0.05: the bias gradient's norm is under 5% of the on-policy "
                                 "gradient's)") if L[hi]["r_bias"] < 0.05 else "NOT SMALL"
        kg = {}
        for l in L:
            c, lk = L[l]["cos_an_KL"], Rw[l]["L_over_KL"]
            kg[l] = dict(cos_an_KL=c, L_over_KL=lk, ok=bool(c >= 0.95 and lk >= 0.9))
        gate["k_gate"] = kg
        gate["k_gate_verdict"] = "PASS (K=128)" if all(v["ok"] for v in kg.values()) else "FAIL: raise K to 256"
        idn = res.get("identity_fp32") or res.get("identity_bf16")
        if idn:
            gate["identity"] = dict(source=idn["dtype"], logit_rel_max_err=idn["logit_rel_max_err"],
                                    param_rel_l2_ref_vs_KL=idn["param_rel_l2_ref_vs_KL"],
                                    ok=bool(idn["logit_rel_max_err"] < 1e-5 and (
                                        idn["param_rel_l2_ref_vs_KL"] < (1e-4 if idn["dtype"] == "float32" else 2e-2))))
        gate["noise_matches_training_log"] = None if EXP is None else all(
            v["matches_training_log"] for v in res["noise"].values())
        gate["nhat_matches_training_log"] = None if EXP is None else all(res["checks"]["nhat_nnorm1_matches_log"])
        res["gate"] = gate
        log("GATE:", json.dumps(gate))
        save()
    del hQ

    # ---------------- phase B: part (b) ----------------------------------------------------------------------------
    if B:
        accs = {f"{e}_{h}": PL.FieldAcc(S) for e in ("k1", "an") for h in ("A", "B")}
        Nh = {"A": 0, "B": 0}
        rows = {}

        def sinkb(st):
            for k, v in st.items():
                rows.setdefault(k, []).append(v.float().cpu().numpy())

        with tm("B_pass"):
            for ti, t in enumerate(B):
                ids = t.ids.to(dev)
                h = t.half

                def ff(z, c0, c1, t=t, h=h, ids=ids, ti=ti):
                    logr = logr_rows(t.key, c0, c1)
                    y = ids[t.n_prompt + c0:t.n_prompt + c1]
                    Gk, a_y = PL.k1s_field(z, logr, y)
                    Ga, Lr = PL.an_field(z, logr, z, a.k)
                    logp = torch.log_softmax(z, -1)
                    kl = (logp.exp() * (logp - logr)).sum(-1)
                    st = dict(nG_k1=Gk.norm(dim=-1), nG_an=Ga.norm(dim=-1), a_y=a_y, L=Lr, kl=kl,
                              trace=torch.full((c1 - c0,), ti, dtype=torch.float32))
                    return {f"k1_{h}": Gk, f"an_{h}": Ga}, st
                Nh[h] += PL.pullback_trace(S, ids, t.n_prompt, ff, {f"k1_{h}": accs[f"k1_{h}"], f"an_{h}": accs[f"an_{h}"]},
                                           a.chunk, fd, stats_sink=sinkb)
                if ti % 8 == 0 or ti == len(B) - 1:
                    log(f"part (b): trace {ti + 1}/{len(B)}")
        with tm("B_readout"):
            ro = PL.readout({k: v.tensors() for k, v in accs.items()}, dev)
        G = ro["gram"]
        pb = dict(N_A=Nh["A"], N_B=Nh["B"], gram=G)
        pb["k1"] = PL.split_half(ro, "k1", Nh["A"], Nh["B"])
        pb["an"] = PL.split_half(ro, "an", Nh["A"], Nh["B"])
        pb["noise_ratio_an_over_k1"] = pb["an"]["N2"] / pb["k1"]["N2"] if pb["k1"]["N2"] else float("nan")
        pb["signal_ratio_an_over_k1"] = pb["an"]["S2"] / pb["k1"]["S2"] if pb["k1"]["S2"] else float("nan")
        pb["cross_S2_k1A_anB"] = G["k1_A"]["an_B"] / (Nh["A"] * Nh["B"])
        pb["cross_S2_anA_k1B"] = G["an_A"]["k1_B"] / (Nh["A"] * Nh["B"])
        arr = {k: np.concatenate(v) for k, v in rows.items()}
        np.savez_compressed(os.path.join(a.out_dir, "rows_b.npz"), **arr)
        pb["rows"] = dict(nG_k1=PL.quantiles(arr["nG_k1"]), nG_an=PL.quantiles(arr["nG_an"]),
                          a_y=PL.quantiles(arr["a_y"]), kl=PL.quantiles(arr["kl"]),
                          mean_sq_nG_k1=float((arr["nG_k1"].astype(np.float64) ** 2).mean()),
                          mean_sq_nG_an=float((arr["nG_an"].astype(np.float64) ** 2).mean()),
                          L_over_KL=float(arr["L"].astype(np.float64).sum() / arr["kl"].astype(np.float64).sum()))
        res["part_b"] = pb
        log("part (b):", json.dumps({k: pb[k] for k in ("k1", "an", "noise_ratio_an_over_k1")}))
        del accs, ro
    save()
    log("DONE ->", out_json)


if __name__ == "__main__":
    main()
