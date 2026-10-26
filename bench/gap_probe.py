#!/usr/bin/env python3
"""Does the same sampler mismatch cause less gradient error when the student is closer to the teacher?  Offline, one
GPU, no training.

Notation. At one fixed prefix: p = the trainer's student (bf16 HF forward, as the FSDP trainer computes it), q = the
sampler (the same checkpoint with the training code's relative weight noise), tau = the teacher.  A = log tau - log p,
delta = q - p. Every vector below is a gradient with respect to the student's LOGITS z, in the ascent direction:
    g_p      = p * (A - p.A)                       exact on-policy update  = -d/dz KL(p || tau)
    g_q      = q * A - p (q.A)                     OPD's expected update when tokens come from q
    e_opd    = g_q - g_p = delta * A - p (delta.A)
    drift    = (q.A) (q - p)
    g_sc     = Cov_q(A, s) in logit form = g_q - drift = q * (A - q.A)          (derived here; checked by selfcheck)
    e_sc     = g_sc - g_p = delta * (A - q.A) - p (delta.A)
    g_is     = E_q[min(p/q, 2) A s] = m * A - p (m.A),  m = min(p, 2 q)        (capped importance weights)
    e_is     = g_is - g_p = (m - p) * A - p ((m - p).A)
    e_rl     = q - p                                   constant reward R = 1 (algebra only)
All of it is computed in float64 from the models' bf16 logits (log-softmax in float64), never by subtracting g_q and g_p.

Subcommands (one process each):
    build   the frozen prefix set (vLLM, base student): P held-out DAPO prompts (HO-DAPO, checked against its sha256),
            one T=1 response each (length capped), K cut positions per response drawn uniformly (seeded); written once
            and hashed (prefixes.json + prefix_record.json). Every later phase checks the hash.
    measure HF transformers + torch: teacher logits at every prefix (once), then per checkpoint (the base student and
            each trained checkpoint given as --ckpt-path LABEL=DIR): p's logits, the noise size tuned so that the mean
            KL(q||p) over the prefixes hits each target in {0.05, 0.2, 0.4} within +-10% (each try = noise on, one
            forward over the prefixes, noise off), the per-prefix measures at every try, and the parameter-level error
            on a subset at target 0.2 (the pull-back of the exact gradient probe (tools/probe0), probe0_lib).
    fp8     vLLM's real FP8 KV cache at scale 1 (the launcher's KV_DTYPE=fp8 path) as q at its own level, plus the
            same engine with the bf16 cache as the engine floor; full-vocabulary log-probs (max_logprobs=-1), so no
            top-k approximation is needed (the tail treatment for k < V is implemented as well). Descriptive.
    selfcheck  CPU only: closed forms vs autograd on random small vocabularies.

The sampler noise is the training code's SAMPLER_NOISE mechanism itself: verl/utils/specopd_tim.py,
SamplerNoise(sigma, seed 20261001).apply on the fp32 stream tensor of every transformer-block projection weight, on the
GPU, then copied into the bf16 parameter (the cast vLLM's load_weights makes). Per-tensor scales are frozen at the BASE
student's first sync (as in every noisy training run); the same noise vector is used for every checkpoint, only its
size sigma is tuned. For a trained checkpoint the stream tensor is its fp32 master weight
(actor/huggingface/model.safetensors is fp32), as the trainer streams it. With --expected-step0 (a JSON file with the
training log's step-0 [SPECOPD-TIM] fields d_eps, d_n, nnorm1, n_tensors, n_elems) the noise record is checked
against it; without it that check is skipped.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("CLUSTER_ROOT", "/path/to/workdir")
STUDENT = f"{R}/hf/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60"
TEACHER = f"{R}/hf/hub/models--Skywork--Skywork-OR1-Math-7B/snapshots/fcef970f1d10682f13c182b280146a194ed57d1e"
# label -> training step. Order = processing order: the primary pair (base, step 200) first. The base student is
# STUDENT; a trained checkpoint's HF directory (actor/huggingface of that step) is given as --ckpt-path LABEL=DIR.
CKPT_STEPS = {"base": 0, "s200": 200, "s50": 50, "s100": 100}
HO_DAPO = f"{R}/ho/ho_dapo.parquet"                                                     # bench/ho_select.py write
HO_DAPO_SHA256 = "504c0e1647f226edd5e42441701195737d126b3a029b320b3d8ff9565101b555"     # the held-out DAPO prompts
HO_TOKIDS_SHA256 = "980a54307d577f6ac5ddcba092c784bf5ba45ead173407ffd63b3ddf87686edc"   # all 1,000 prompts, base template
N_HO = 1000
NOISE_SEED = 20261001
STEP0_KEYS = ("d_eps", "d_n", "nnorm1", "n_tensors", "n_elems")   # --expected-step0: the step-0 [SPECOPD-TIM] fields
TARGETS = (0.05, 0.2, 0.4)
TOL = 0.10                 # the mean KL(q||p) must be within +-10% of the target
TOL_AIM = 0.03             # the tuner stops early inside +-3%
SIGMA_MAX = 2.0
IS_CAP = 2.0               # capped importance weights min(p/q, 2) (the trainer's RC_IS_THRESHOLD)
SUPPORT_MAIN = 1e-8        # q's support for max |A|
SUPPORT_THRESHOLDS = (1e-6, 1e-8, 1e-10, 1e-12)
PROMPT_SEED = 20261009     # which HO-DAPO rows
CUT_SEED = 20261010        # cut positions (per prompt: default_rng([CUT_SEED, k]))
GEN_SEED_BASE = 400000     # sampling seed of HO row k: 400,000 + k (disjoint from ho_eval's 100,000-115,999)
FORMAT = "gap_prefixes/v1"
MODES = {
    "main": dict(P=64, K=16, gen_cap=4096, ckpts=("base", "s200", "s50", "s100"), targets=TARGETS, max_evals=6,
                 param_n=64, param_target=0.2, fp8_ckpts=("base", "s200", "s50", "s100"), ck_min_s=1200,
                 param_min_s=900),
    "smoke": dict(P=4, K=4, gen_cap=256, ckpts=("base", "s200"), targets=TARGETS, max_evals=4, param_n=2,
                  param_target=0.2, fp8_ckpts=("base",), ck_min_s=240, param_min_s=120),
}
T0 = time.time()


def log(*a):
    print(f"[gap {time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------------------------- files
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def write_json(path, obj, canonical=False):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        if canonical:
            f.write(canonical_bytes(obj))
        else:
            f.write(json.dumps(obj, indent=1, sort_keys=True, default=_jsonable).encode())
    os.replace(tmp, path)


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    raise TypeError(f"not json-able: {type(o)}")


def read_json(path):
    with open(path) as f:
        return json.load(f)


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def specopd_tim_path():
    """The training code's specopd_tim.py: verl/verl/utils beside bench/ (or one directory further up)."""
    for root in (os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))):
        p = os.path.join(root, "verl", "verl", "utils", "specopd_tim.py")
        if os.path.isfile(p):
            return p
    raise SystemExit("[gap] REFUSE: verl/verl/utils/specopd_tim.py not found beside bench/")


def load_tim():
    return load_by_path("gap_specopd_tim", specopd_tim_path())


def load_probe_lib():
    return load_by_path("gap_probe0_lib", os.path.join(HERE, "probe0_lib.py"))


def checkpoint_paths(pairs):
    """label -> HF dir: 'base' is STUDENT; every --ckpt-path LABEL=DIR adds (or replaces) one label of CKPT_STEPS."""
    out = {"base": STUDENT}
    for x in pairs or ():
        label, sep, path = x.partition("=")
        if not sep or not path or label not in CKPT_STEPS:
            raise SystemExit(f"[gap] REFUSE: --ckpt-path {x!r}: expected LABEL=DIR with LABEL in {sorted(CKPT_STEPS)}")
        out[label] = path
    return out


def load_expected_step0(path, tag="[gap]"):
    """The training log's step-0 [SPECOPD-TIM] fields (STEP0_KEYS) from a JSON file, or None (no path: the check of a
    noise record against them is skipped, and one line says so)."""
    if not path:
        print(f"{tag} no --expected-step0: the noise record is not checked against the training log's step-0 line",
              flush=True)
        return None
    obj = read_json(path)
    missing = [k for k in STEP0_KEYS if k not in obj]
    if missing:
        raise SystemExit(f"{tag} REFUSE: {path} lacks {missing}")
    return {k: obj[k] for k in STEP0_KEYS}


# ----------------------------------------------------------------------------------------------------- prefix set
def tokids_sha256(ids):
    """ho_eval.py's formula: sha256 of the compact JSON of the list of prompt id lists."""
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def render_prompt_ids(tok, chats):
    """verl's / ho_eval.py's rendering: the student's chat template, add_generation_prompt, no special tokens."""
    out = []
    for chat in chats:
        text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        out.append(list(tok(text, add_special_tokens=False)["input_ids"]))
    return out


def load_ho_chats(path=HO_DAPO, sha256=HO_DAPO_SHA256):
    import pandas as pd
    got = sha256_file(path)
    if got != sha256:
        raise SystemExit(f"[gap] REFUSE: {path} sha256 {got} != the held-out prompts' {sha256}")
    df = pd.read_parquet(path)

    def plain(x):
        if hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
            x = x.tolist()
        if isinstance(x, dict):
            return {k: plain(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [plain(v) for v in x]
        return x
    return [[dict(m) for m in plain(df.iloc[k]["prompt"])] for k in range(len(df))]


def select_prompts(n_total, P, seed=PROMPT_SEED):
    import numpy as np
    return sorted(int(x) for x in np.random.default_rng(seed).choice(n_total, size=P, replace=False))


def draw_cuts(T, K, k, seed=CUT_SEED):
    """K distinct cut positions in [0, T) drawn uniformly (all T if T < K), in DRAW order. Position t means the prefix
    prompt + response[:t], whose next token is response[t]."""
    import numpy as np
    if T <= 0:
        return []
    n = min(K, T)
    return [int(x) for x in np.random.default_rng([seed, int(k)]).choice(T, size=n, replace=False)]


def prefix_rows(prompts):
    """Canonical row order: prompts in record order, each prompt's cuts in draw order. Returns (prompt_index, cut)."""
    return [(i, t) for i, pr in enumerate(prompts) for t in pr["cuts"]]


def subset_rows(prompts, n):
    """The parameter-level subset: the FIRST-drawn cut of each of the first n prompts (one prefix per prompt)."""
    rows, r = [], 0
    for i, pr in enumerate(prompts):
        if i < n and pr["cuts"]:
            rows.append(r)
        r += len(pr["cuts"])
    return rows


def load_prefixes(out_dir):
    """prefixes.json, checked against prefix_record.json's sha256. Returns (prompts, record)."""
    rec = read_json(os.path.join(out_dir, "prefix_record.json"))
    path = os.path.join(out_dir, "prefixes.json")
    got = sha256_file(path)
    if got != rec["sha256"]:
        raise SystemExit(f"[gap] REFUSE: {path} sha256 {got} != its record {rec['sha256']}")
    obj = read_json(path)
    if obj.get("format") != FORMAT:
        raise SystemExit(f"[gap] REFUSE: prefix format {obj.get('format')!r}")
    return obj["prompts"], rec


# ----------------------------------------------------------------------------------------------- logit-space math
def logit_measures(lp, lq, lt, is_cap=IS_CAP, thresholds=SUPPORT_THRESHOLDS):
    """lp, lq, lt: [R, V] float64 log-probabilities of p, q, tau at R prefixes. Returns name -> [R] tensor."""
    import torch
    p = lp.exp()
    q = lq.exp()
    A = lt - lp
    absA = A.abs()
    d = p * torch.expm1(lq - lp)                          # q - p without cancellation
    Ap = (p * A).sum(-1)                                  # = -KL(p||tau)
    Aq = (q * A).sum(-1)
    dA = (d * A).sum(-1)                                  # = Aq - Ap, directly
    out = dict(kl_qp=(q * (lq - lp)).sum(-1), kl_pq=(p * (lp - lq)).sum(-1), gap=(p * (lp - lt)).sum(-1),
               kl_q_tau=(q * (lq - lt)).sum(-1), Abar_q=Aq, Abar_p=Ap, dA=dA,
               meanabsA_q=(q * absA).sum(-1), meanabsA_p=(p * absA).sum(-1), maxabsA_all=absA.max(-1).values,
               p_max=p.max(-1).values, H_p=-(p * lp).sum(-1), q_sum=q.sum(-1))
    for th in thresholds:
        out[f"maxabsA_q{th:g}"] = absA.masked_fill(q <= th, 0.0).max(-1).values
    out["maxabsA_p1e-08"] = absA.masked_fill(p <= 1e-8, 0.0).max(-1).values
    lc = math.log(is_cap)
    dis = p * torch.expm1(torch.clamp(lc + lq - lp, max=0.0))   # min(p, c q) - p  (<= 0)
    disA = (dis * A).sum(-1)
    vecs = dict(
        e=d * A - p * dA[:, None],
        gp=p * (A - Ap[:, None]),
        drift=Aq[:, None] * d,
        esc=d * (A - Aq[:, None]) - p * dA[:, None],
        eis=dis * A - p * disA[:, None],
        d=d,
    )
    for name, v in vecs.items():
        out[f"{name}_L1"] = v.abs().sum(-1)
        out[f"{name}_L2"] = v.norm(dim=-1)
    out["gq_L2"] = (vecs["gp"] + vecs["e"]).norm(dim=-1)
    out["gsc_L2"] = (q * (A - Aq[:, None])).norm(dim=-1)
    out["sum_absd_absA"] = (d.abs() * absA).sum(-1)
    out["is_cap_share"] = q.masked_fill(~(lp > lc + lq), 0.0).sum(-1)   # sampled mass whose weight min(p/q, 2) is capped
    return out


def logit_fields(lp, lq, lt, is_cap=IS_CAP):
    """The parameter-level probe's logit fields (ascent direction): P = g_p, E = e_opd, D = drift, S = e_sc, I = e_is."""
    import torch
    p = lp.exp()
    q = lq.exp()
    A = lt - lp
    d = p * torch.expm1(lq - lp)
    Ap = (p * A).sum(-1, keepdim=True)
    Aq = (q * A).sum(-1, keepdim=True)
    dA = (d * A).sum(-1, keepdim=True)
    dis = p * torch.expm1(torch.clamp(math.log(is_cap) + lq - lp, max=0.0))
    disA = (dis * A).sum(-1, keepdim=True)
    return dict(P=p * (A - Ap), E=d * A - p * dA, D=Aq * d, S=d * (A - Aq) - p * dA, I=dis * A - p * disA)


def measures_from_logits(zp, zq, zt, chunk=64, is_cap=IS_CAP):
    """zp, zq, zt: [N, V] logits (any float dtype, same device). log-softmax in float64 per row chunk. -> numpy dict."""
    import numpy as np
    import torch
    acc = {}
    for c0 in range(0, zp.shape[0], chunk):
        c1 = min(zp.shape[0], c0 + chunk)
        lp = torch.log_softmax(zp[c0:c1].double(), -1)
        lq = torch.log_softmax(zq[c0:c1].double(), -1)
        lt = torch.log_softmax(zt[c0:c1].double(), -1)
        for k, v in logit_measures(lp, lq, lt, is_cap).items():
            acc.setdefault(k, []).append(v.cpu().numpy())
        del lp, lq, lt
    return {k: np.concatenate(v).astype(np.float64) for k, v in acc.items()}


def measures_from_logprobs(zp, lq_rows, zt, rows, is_cap=IS_CAP):
    """As measures_from_logits for the rows `rows` of zp / zt, with q given as float64 log-probs [len(rows), V]."""
    import torch
    lp = torch.log_softmax(zp[rows].double(), -1)
    lt = torch.log_softmax(zt[rows].double(), -1)
    out = {k: v.cpu().numpy() for k, v in logit_measures(lp, lq_rows, lt, is_cap).items()}
    del lp, lt
    return out


def autograd_reference(z, lq, lt, is_cap=IS_CAP):
    """The same logit-space quantities by torch autograd of each rule's surrogate at z (rows of logits; q, tau, A and
    the importance weights held fixed, as within one update). Returns name -> [R, V] (ascent direction)."""
    import torch
    z0 = z.detach().double()
    lp0 = torch.log_softmax(z0, -1)
    q = lq.exp()
    A = (lt - lp0).detach()
    Aq = (q * A).sum(-1, keepdim=True)
    w = torch.clamp(torch.exp(lp0 - lq), max=is_cap).detach()

    def grad(fn):
        zz = z0.clone().requires_grad_(True)
        lp = torch.log_softmax(zz, -1)
        (g,) = torch.autograd.grad(fn(lp).sum(), zz)
        return -g                                            # ascent = minus the loss gradient
    out = {}
    out["g_p"] = grad(lambda lp: (lp.exp() * (lp - lt)).sum(-1))                        # KL(p||tau)
    out["g_q"] = grad(lambda lp: -(q * A * lp).sum(-1))                                 # -E_q[sg(A) log p]
    out["g_q_sq"] = grad(lambda lp: 0.5 * (q * (lp - lt) ** 2).sum(-1))                 # E_q[(log p - log tau)^2] / 2
    out["g_sc"] = grad(lambda lp: -(q * A * (lp - (q * lp).sum(-1, keepdim=True))).sum(-1))   # E_q[A (s - E_q s)]
    out["g_sc_var"] = grad(lambda lp: 0.5 * ((q * (lp - lt) ** 2).sum(-1) - (q * (lp - lt)).sum(-1) ** 2))  # Var_q(lp - lt) / 2
    out["g_is"] = grad(lambda lp: -(q * w * A * lp).sum(-1))
    out["drift"] = grad(lambda lp: -(Aq * (q * lp).sum(-1, keepdim=True)).sum(-1))       # (E_q A) E_q[s]
    out["g_rl"] = grad(lambda lp: -(q * lp).sum(-1))                                    # constant reward R = 1
    return out


# --------------------------------------------------------------------------------------------------------- tuning
def _lerr(kl, T):
    return abs(math.log(kl / T)) if (kl > 0 and math.isfinite(kl)) else float("inf")


def sigma_first_guess(T):
    """The first sigma tried for a target mean KL(q||p) = T when nothing is known yet. For small noise the logits move
    in proportion to sigma and KL(q||p) is second order in that move, so KL grows about quadratically in sigma; the
    guess takes KL = 2 sigma^2, i.e. sigma = sqrt(T / 2). The constant only sets where the search starts: every later
    try interpolates between the measured points (propose_sigma)."""
    return math.sqrt(T / 2.0)


def propose_sigma(pts, T, sigma_max=SIGMA_MAX, sigma_min=1e-4):
    """Next sigma for target T from the tried points [(sigma, kl)], assuming kl grows with sigma: log-log interpolation
    between the nearest bracketing points (kept inside the middle 80% of the bracket), else extrapolation with the
    local log-log slope (clipped to [0.5, 4], default 2), at most x4 / /4 per step. None = nothing new to try."""
    good = [(s, k) for s, k in pts if k > 0 and math.isfinite(k)]
    below = [x for x in good if x[1] < T]
    above = [x for x in good if x[1] > T]
    tried = {round(s, 12) for s, _ in pts}

    def slope(a, b):
        if a[0] == b[0] or a[1] <= 0 or b[1] <= 0:
            return 2.0
        sl = math.log(b[1] / a[1]) / math.log(b[0] / a[0])
        return min(4.0, max(0.5, sl)) if math.isfinite(sl) and sl > 0 else 2.0
    if below and above:
        lo = max(below, key=lambda x: x[0])
        hi = min(above, key=lambda x: x[0])
        if lo[0] < hi[0]:
            x0, x1, y0, y1 = math.log(lo[0]), math.log(hi[0]), math.log(lo[1]), math.log(hi[1])
            x = x0 + (math.log(T) - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else 0.5 * (x0 + x1)
            x = min(max(x, x0 + 0.1 * (x1 - x0)), x1 - 0.1 * (x1 - x0))
        else:                                                # not monotone here: geometric middle of the two
            x = 0.5 * (math.log(lo[0]) + math.log(hi[0]))
        s = math.exp(x)
    elif below:
        srt = sorted(below)
        lo = srt[-1]
        b = slope(srt[-2], srt[-1]) if len(srt) >= 2 else 2.0
        s = min(lo[0] * (T / lo[1]) ** (1.0 / b), 4.0 * lo[0], sigma_max)
    elif above:
        srt = sorted(above)
        hi = srt[0]
        b = slope(srt[0], srt[1]) if len(srt) >= 2 else 2.0
        s = max(hi[0] * (T / hi[1]) ** (1.0 / b), hi[0] / 4.0, sigma_min)
    else:
        return None
    return None if round(s, 12) in tried else s


def tune_sigmas(eval_fn, targets, guess=sigma_first_guess, prior=None, tol_aim=TOL_AIM, tol=TOL, max_evals=6,
                sigma_max=SIGMA_MAX):
    """For each target (ascending): evaluate eval_fn(sigma) -> mean KL(q||p) until a tried point is within tol_aim of
    the target or max_evals tries for it were made. Points are shared across targets. First try for a target: the
    prior (e.g. the previous checkpoint's accepted sigma), else the bracket interpolation, else guess(T).
    Returns (accepted: T -> dict(sigma, kl, index, rel, ok, n_evals), points: list of dict(sigma, kl, target))."""
    pts = []
    accepted = {}
    for T in sorted(targets):
        n = 0
        first = True
        while True:
            if pts:
                j = min(range(len(pts)), key=lambda i: _lerr(pts[i]["kl"], T))
                if abs(pts[j]["kl"] / T - 1.0) <= tol_aim:
                    break
            if n >= max_evals:
                break
            s = None
            if first:
                xy = [(x["sigma"], x["kl"]) for x in pts]
                below = any(0 < k < T for _, k in xy)
                above = any(k > T for _, k in xy)
                if prior and T in prior:
                    s = prior[T]
                elif below and above:
                    s = propose_sigma(xy, T, sigma_max)
                else:
                    s = guess(T)
                if s is not None and any(abs(s - x["sigma"]) <= 1e-12 * max(1.0, s) for x in pts):
                    s = propose_sigma(xy, T, sigma_max)
                first = False
            else:
                s = propose_sigma([(x["sigma"], x["kl"]) for x in pts], T, sigma_max)
            if s is None:
                break
            s = min(float(s), sigma_max)
            kl = float(eval_fn(s))
            pts.append(dict(sigma=s, kl=kl, target=T))
            n += 1
        if pts:
            j = min(range(len(pts)), key=lambda i: _lerr(pts[i]["kl"], T))
            rel = pts[j]["kl"] / T - 1.0
            accepted[T] = dict(sigma=pts[j]["sigma"], kl=pts[j]["kl"], index=j, rel=rel, ok=bool(abs(rel) <= tol),
                               n_evals=n)
        else:
            accepted[T] = dict(sigma=None, kl=None, index=None, rel=None, ok=False, n_evals=0)
    return accepted, pts


# ------------------------------------------------------------------------------------------------- noise and models
def noise_apply(model, sigma, src, scales, tim, seed, dev):
    """q's weights: for every target parameter (tim.is_target), the training hook's SamplerNoise.apply on the fp32
    stream tensor src[name] moved to `dev` (the rollout GPU), copied into the bf16 parameter (vLLM's load_weights
    cast). `scales` pre-fills the frozen per-tensor scales (None = this call is the first sync and computes them, as
    training does on the initial student). Returns (finish_sync record, scales used)."""
    import torch
    sn = tim.SamplerNoise(sigma, seed)
    if scales:
        sn.scales.update(scales)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not tim.is_target(name, p):
                continue
            w = src[name].to(dev, dtype=torch.float32)
            ((_, wn),) = sn.apply([(name, w)])
            p.copy_(wn)
            del w, wn
    rec = sn.finish_sync()
    rec = {k: v for k, v in rec.items() if k != "ms"}
    return rec, dict(sn.scales)


def restore_clean(model, src, dev):
    import torch
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in src:
                p.copy_(src[name].to(dev))


def safetensors_map(path):
    """name -> file for an HF dir (sharded with an index json, or one model.safetensors)."""
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(idx):
        wm = read_json(idx)["weight_map"]
        return {k: os.path.join(path, v) for k, v in wm.items()}
    from safetensors import safe_open
    f1 = os.path.join(path, "model.safetensors")
    with safe_open(f1, framework="pt", device="cpu") as f:
        return {k: f1 for k in f.keys()}


def load_noise_source(model, path, tim):
    """The fp32 stream tensors of every noise target, read from the checkpoint's own safetensors (fp32 master weights
    for a trained checkpoint, bf16 -> exact fp32 for the base), on the CPU. Checks that bf16(source) equals the loaded
    parameter bit for bit (so restore_clean is exact and the source is this checkpoint's weights)."""
    import torch
    from safetensors import safe_open
    fmap = safetensors_map(path)
    names = [n for n, p in model.named_parameters() if tim.is_target(n, p)]
    missing = [n for n in names if n not in fmap]
    if missing:
        raise SystemExit(f"[gap] REFUSE: {len(missing)} noise targets absent from {path}: {missing[:3]}")
    by_file = {}
    for n in names:
        by_file.setdefault(fmap[n], []).append(n)
    src, dtypes = {}, {}
    for fpath, ns in by_file.items():
        with safe_open(fpath, framework="pt", device="cpu") as f:
            for n in ns:
                t = f.get_tensor(n)
                dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + 1
                src[n] = t.to(torch.float32)
    params = dict(model.named_parameters())
    with torch.no_grad():
        bad = [n for n in names if not torch.equal(params[n].detach(), src[n].to(params[n].device).to(params[n].dtype))]
    if bad:
        raise SystemExit(f"[gap] REFUSE: bf16(source) != loaded parameter for {len(bad)} tensors, e.g. {bad[:3]}")
    return src, dict(n_tensors=len(names), n_elems=int(sum(src[n].numel() for n in names)), source_dtypes=dtypes)


def load_model(path, dev, attn):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=attn)
    m.config.use_cache = False
    m = m.to(dev)
    m.eval()
    return m


def prefix_logits(model, prompts, dev, only=None):
    """bf16 logits [N, V] at every prefix, canonical row order (or the prompts in `only`, same order). One forward per
    prompt over prompt + response[:max cut]; the hidden state at index n_prompt - 1 + t gives the next-token logits of
    the prefix prompt + response[:t] (the trainer's logits at that response position)."""
    import torch
    import torch.nn.functional as F
    lm = model.get_output_embeddings().weight
    out = []
    with torch.no_grad():
        for i, pr in enumerate(prompts):
            if only is not None and i not in only:
                continue
            cuts = pr["cuts"]
            if not cuts:
                continue
            n = len(pr["prompt_ids"])
            ids = torch.tensor(pr["prompt_ids"] + pr["response_ids"][:max(cuts)], dtype=torch.long, device=dev)
            h = model.get_decoder()(input_ids=ids[None], use_cache=False).last_hidden_state[0]
            rows = torch.tensor([n - 1 + t for t in cuts], dtype=torch.long, device=dev)
            out.append(F.linear(h[rows], lm))
            del h
    return torch.cat(out)


def pullback_rows(model, PL, ids, rows, field_fn, accs, fdtype):
    """The exact gradient probe's pull-back (probe0_lib.pullback_trace) restricted to the listed sequence positions: g = sum_rows J^T G,
    i.e. the parameter gradient of sum_rows <stopgrad(G), z>, z = lm_head(h).to(fdtype); one body backward per field
    (gradient checkpointing). Returns the detached z [len(rows), V]."""
    import torch
    import torch.nn.functional as F
    body, lm = PL.param_list(model)
    body_p = [p for _, p in body]
    h = PL.forward_hidden(model, ids, grad=True)
    hr = h[torch.as_tensor(rows, dtype=torch.long, device=h.device)]
    hc = hr.detach().requires_grad_(True)
    z = F.linear(hc, lm).to(fdtype)
    with torch.no_grad():
        fields = field_fn(z.detach())
    names = [n for n in accs if n in fields]
    gh = {}
    for j, name in enumerate(names):
        ghc, gW = torch.autograd.grad(z, [hc, lm], grad_outputs=fields[name].to(z.dtype),
                                      retain_graph=(j < len(names) - 1))
        gh[name] = ghc
        accs[name].add_lm(gW)
        del gW
    for j, name in enumerate(names):
        grads = torch.autograd.grad(hr, body_p, grad_outputs=gh[name], retain_graph=(j < len(names) - 1),
                                    allow_unused=True)
        accs[name].add_body(grads)
        del grads
    zd = z.detach()
    del h, hr, hc, z, gh, fields
    return zd


def param_derived(G):
    """Parameter-level numbers from the Gram matrix of the pulled-back fields P (g_p), E (e_opd), D (drift), S (e_sc),
    I (e_is): error norms relative to ||g_p||, and the cosine / angle of each rule's expected update with g_p."""
    def nrm(x):
        return math.sqrt(max(x, 0.0))

    def cosang(dot, n1, n2):
        c = dot / (n1 * n2) if n1 > 0 and n2 > 0 else float("nan")
        return c, (math.degrees(math.acos(max(-1.0, min(1.0, c)))) if c == c else float("nan"))
    nP = nrm(G["P"]["P"])
    out = dict(norm_g_p=nP)
    for rule, err in (("opd", "E"), ("sc", "S"), ("is", "I")):
        nE = nrm(G[err][err])
        ng = nrm(G["P"]["P"] + 2 * G["P"][err] + G[err][err])
        c, ang = cosang(G["P"]["P"] + G["P"][err], nP, ng)
        out[f"rel_err_{rule}"] = nE / nP if nP > 0 else float("nan")
        out[f"cos_{rule}"] = c
        out[f"angle_deg_{rule}"] = ang
        out[f"norm_ratio_{rule}"] = ng / nP if nP > 0 else float("nan")
        out[f"cos_err_gp_{rule}"] = cosang(G["P"][err], nE, nP)[0]
    nD = nrm(G["D"]["D"])
    out["rel_drift"] = nD / nP if nP > 0 else float("nan")
    out["cos_drift_err_opd"] = cosang(G["D"]["E"], nD, nrm(G["E"]["E"]))[0]
    # the identity e_sc = e_opd - drift at the parameter level (a check of the pull-back's linearity)
    lhs = G["S"]["S"]
    rhs = G["E"]["E"] - 2 * G["E"]["D"] + G["D"]["D"]
    out["sc_identity_rel"] = abs(lhs - rhs) / max(abs(lhs), 1e-300)
    return out


# ------------------------------------------------------------------------------------------------ vLLM logprobs
def dense_logprobs(token_ids, logprobs, V, lp_fill=None):
    """Full-vocabulary float64 log q from a vLLM logprob list (token ids + log-probs; duplicates allowed). Ids not
    returned (top-k with k < V) get the remaining mass spread in proportion to p (lp_fill = log p of that row);
    with the full vocabulary returned nothing is filled. The row is renormalised in float64 (vLLM's fp32 log-softmax
    sums to 1 only to ~1e-6). Returns (log q [V], info dict)."""
    import torch
    lq = torch.full((V,), -math.inf, dtype=torch.float64)
    ids = torch.as_tensor(list(token_ids), dtype=torch.long)
    lq[ids] = torch.as_tensor(list(logprobs), dtype=torch.float64)
    have = torch.isfinite(lq)
    n_have = int(have.sum())
    head = float(torch.logsumexp(lq[have], 0).exp()) if n_have else 0.0
    info = dict(n_returned=n_have, head_mass=head, tail_mass=max(0.0, 1.0 - head), filled=0)
    if n_have < V:
        if lp_fill is None:
            raise ValueError("top-k logprobs without a fill distribution")
        lpf = lp_fill.to(torch.float64).cpu()
        miss = ~have
        lpt = torch.logsumexp(lpf[miss], 0)
        tail = max(1.0 - head, 1e-300)
        lq[miss] = math.log(tail) + lpf[miss] - lpt
        info["filled"] = int(miss.sum())
    lq = lq - torch.logsumexp(lq, 0)
    return lq, info


def flat_position0(lps):
    """(token ids, log-probs) of the first generated position from vLLM's FlatLogprobs or list-of-dicts logprobs."""
    if hasattr(lps, "start_indices"):
        a, b = lps.start_indices[0], lps.end_indices[0]
        return lps.token_ids[a:b], lps.logprobs[a:b]
    d = lps[0]
    ids = list(d.keys())
    return ids, [d[i].logprob for i in ids]


# ---------------------------------------------------------------------------------------------------- subcommands
def cmd_build(a):
    cfg = MODES[a.mode]
    D = a.out_dir
    os.makedirs(D, exist_ok=True)
    if os.path.exists(os.path.join(D, "prefix_record.json")):
        prompts, rec = load_prefixes(D)
        log(f"prefix set exists: sha256 {rec['sha256']} ({rec['n_prompts']} prompts, {rec['n_prefixes']} prefixes)")
        return 0
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STUDENT)
    chats = load_ho_chats()
    ids_all = render_prompt_ids(tok, chats)
    got = tokids_sha256(ids_all)
    if len(ids_all) != N_HO or got != HO_TOKIDS_SHA256:
        raise SystemExit(f"[gap] REFUSE: HO-DAPO token ids sha256 {got} != the expected {HO_TOKIDS_SHA256}")
    sel = select_prompts(N_HO, cfg["P"])
    log(f"HO-DAPO pins OK; {cfg['P']} prompts selected (seed {PROMPT_SEED}): {sel[:8]}...")
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    maxp = max(len(ids_all[k]) for k in sel)
    llm = LLM(model=STUDENT, dtype="bfloat16", gpu_memory_utilization=0.8, max_model_len=maxp + cfg["gen_cap"] + 1,
              seed=0, enable_prefix_caching=False, max_num_seqs=max(8, cfg["P"]))
    sps = [SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, repetition_penalty=1.0, max_tokens=cfg["gen_cap"],
                          seed=GEN_SEED_BASE + k, detokenize=False) for k in sel]
    t = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=ids_all[k]) for k in sel], sps, use_tqdm=False)
    gen_s = time.time() - t
    prompts = []
    for k, o in zip(sel, outs):
        c = o.outputs[0]
        resp = [int(x) for x in c.token_ids]
        prompts.append(dict(k=k, prompt_ids=ids_all[k], response_ids=resp, finish_reason=c.finish_reason,
                            gen_seed=GEN_SEED_BASE + k, cuts=draw_cuts(len(resp), cfg["K"], k)))
    obj = dict(format=FORMAT, mode=a.mode, P=cfg["P"], K=cfg["K"], gen_cap=cfg["gen_cap"], prompt_seed=PROMPT_SEED,
               cut_seed=CUT_SEED, gen_seed_base=GEN_SEED_BASE, ho_dapo=HO_DAPO, ho_dapo_sha256=HO_DAPO_SHA256,
               ho_tokids_sha256=HO_TOKIDS_SHA256, student=STUDENT, vllm=vllm.__version__,
               sampling=dict(temperature=1.0, top_p=1.0, top_k=-1, dtype="bfloat16", spec="none"), prompts=prompts)
    path = os.path.join(D, "prefixes.json")
    write_json(path, obj, canonical=True)
    lens = sorted(len(p["response_ids"]) for p in prompts)
    rec = dict(sha256=sha256_file(path), n_prompts=len(prompts), n_prefixes=sum(len(p["cuts"]) for p in prompts),
               response_len=dict(min=lens[0], median=lens[len(lens) // 2], max=lens[-1]),
               finish=dict((r, sum(p["finish_reason"] == r for p in prompts)) for r in ("stop", "length")),
               gen_s=round(gen_s, 1), mode=a.mode, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    write_json(os.path.join(D, "prefix_record.json"), rec)
    log(f"prefix set written: sha256 {rec['sha256']}; {json.dumps(rec)}")
    return 0


def _ck_done(D, ck):
    p = os.path.join(D, f"tune_{ck}.json")
    return os.path.exists(p) and read_json(p).get("status") == "complete"


def cmd_measure(a):
    import numpy as np
    import torch
    cfg = MODES[a.mode]
    D = a.out_dir
    prompts, prec = load_prefixes(D)
    rows = prefix_rows(prompts)
    N = len(rows)
    sub = subset_rows(prompts, cfg["param_n"])
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tim = load_tim()
    PL = load_probe_lib()
    paths = checkpoint_paths(a.ckpt_path)
    expected = load_expected_step0(a.expected_step0)
    deadline = a.deadline or float("inf")
    summ_path = os.path.join(D, "gap_measure.json")
    summ = read_json(summ_path) if os.path.exists(summ_path) else dict(checks={}, timing={}, skipped={})
    summ.update(prefix_sha256=prec["sha256"], N=N, P=len(prompts), mode=a.mode, attn=a.attn, targets=list(cfg["targets"]),
                tol=TOL, noise_seed=NOISE_SEED, is_cap=IS_CAP, support_main=SUPPORT_MAIN,
                specopd_tim=specopd_tim_path())
    import transformers
    summ["versions"] = dict(torch=torch.__version__, transformers=transformers.__version__)
    if dev.type == "cuda":
        summ["gpu"] = torch.cuda.get_device_name(0)

    def save_summ():
        summ["elapsed_s"] = round(time.time() - T0, 1)
        write_json(summ_path, summ)

    from transformers import AutoConfig, AutoTokenizer
    tok_s, tok_t = AutoTokenizer.from_pretrained(STUDENT), AutoTokenizer.from_pretrained(TEACHER)
    V = AutoConfig.from_pretrained(STUDENT).vocab_size
    assert AutoConfig.from_pretrained(TEACHER).vocab_size == V, "student / teacher vocab sizes differ"
    assert tok_s.get_vocab() == tok_t.get_vocab(), "student / teacher tokenizers differ"
    summ["checks"]["vocab"] = dict(V=V, tokenizers_equal=True)

    # ---- teacher (once) ----
    tau_path = os.path.join(D, "tau.pt")
    if os.path.exists(tau_path):
        blob = torch.load(tau_path, map_location="cpu")
        assert blob["prefix_sha256"] == prec["sha256"], "tau.pt is for another prefix set"
        zt = blob["logits"].to(dev)
        log("teacher logits loaded")
    else:
        t = time.time()
        Tm = load_model(TEACHER, dev, a.attn)
        zt = prefix_logits(Tm, prompts, dev)
        del Tm
        gc.collect()
        torch.cuda.empty_cache() if dev.type == "cuda" else None
        torch.save(dict(prefix_sha256=prec["sha256"], model=TEACHER, logits=zt.cpu()), tau_path + ".tmp")
        os.replace(tau_path + ".tmp", tau_path)
        summ["timing"]["teacher_s"] = round(time.time() - t, 1)
        log(f"teacher logits {tuple(zt.shape)} in {time.time() - t:.0f}s")
    assert zt.shape == (N, V), (zt.shape, N, V)
    save_summ()

    scales_path = os.path.join(D, "scales_base.json")
    priors = {}
    for ck in cfg["ckpts"]:
        tp = os.path.join(D, f"tune_{ck}.json")
        if _ck_done(D, ck):
            tj = read_json(tp)
            priors = {float(k): v["sigma"] for k, v in tj["accepted"].items() if v.get("sigma")}
            log(f"{ck}: done, skipped")
            continue
        if ck not in paths:
            summ["skipped"][ck] = "no --ckpt-path"
            log(f"{ck}: SKIPPED (no --ckpt-path {ck}=DIR)")
            save_summ()
            continue
        left = deadline - time.time()
        if left < cfg["ck_min_s"]:
            summ["skipped"][ck] = f"not started: {left:.0f}s left < {cfg['ck_min_s']}s"
            log(f"{ck}: SKIPPED ({summ['skipped'][ck]})")
            save_summ()
            continue
        if ck != "base" and not os.path.exists(scales_path):
            raise SystemExit(f"[gap] {ck}: the base student's noise scales ({scales_path}) are missing: run base first")
        step, path = CKPT_STEPS[ck], paths[ck]
        tck = time.time()
        log(f"== {ck} (step {step}): {path}")
        model = load_model(path, dev, a.attn)
        assert model.get_output_embeddings().weight.shape[0] == V
        src, src_info = load_noise_source(model, path, tim)
        log(f"{ck}: model + {src_info['n_tensors']} fp32 noise sources loaded ({time.time() - tck:.0f}s) {src_info}")
        zp = prefix_logits(model, prompts, dev)
        torch.save(dict(prefix_sha256=prec["sha256"], model=path, logits=zp.cpu()), os.path.join(D, f"p_{ck}.pt.tmp"))
        os.replace(os.path.join(D, f"p_{ck}.pt.tmp"), os.path.join(D, f"p_{ck}.pt"))
        first_prompt = [next(i for i, pr in enumerate(prompts) if pr["cuts"])]
        n0 = len(prompts[first_prompt[0]]["cuts"])
        r0 = sum(len(pr["cuts"]) for pr in prompts[:first_prompt[0]])
        ref0 = zp[r0:r0 + n0].clone()
        rep = prefix_logits(model, prompts, dev, only=set(first_prompt))
        checks = dict(source=src_info, determinism_max_abs_logit_diff=float((rep.float() - ref0.float()).abs().max()))
        clean_m = measures_from_logits(zp, zp, zt)
        checks["clean_kl_qp_max"] = float(np.abs(clean_m["kl_qp"]).max())
        scales = read_json(scales_path)["scales"] if ck != "base" else (
            read_json(scales_path)["scales"] if os.path.exists(scales_path) else None)
        evals, ms, qsubs = [], [], []

        def eval_sigma(s):
            nonlocal scales
            te = time.time()
            rec, used = noise_apply(model, s, src, scales, tim, NOISE_SEED, dev)
            if scales is None:                                   # the base student's first sync: training's scales
                scales = used
                write_json(scales_path, dict(scales=scales, source=path, seed=NOISE_SEED, record=rec))
            zq = prefix_logits(model, prompts, dev)
            restore_clean(model, src, dev)
            rep2 = prefix_logits(model, prompts, dev, only=set(first_prompt))
            rdiff = float((rep2.float() - ref0.float()).abs().max())
            m = measures_from_logits(zp, zq, zt)
            if not evals and ck == cfg["ckpts"][0]:
                # closed forms vs autograd on 4 real rows (float64, full vocabulary)
                rr = list(range(min(4, N)))
                lq4 = torch.log_softmax(zq[rr].double(), -1)
                lt4 = torch.log_softmax(zt[rr].double(), -1)
                ref = autograd_reference(zp[rr].double(), lq4, lt4)
                f = logit_fields(torch.log_softmax(zp[rr].double(), -1), lq4, lt4)
                errs = {}
                scale = float(ref["g_p"].norm())      # the natural scale (an error that is exactly 0, e.g. no capped
                for nm, (x, y) in dict(e_opd=(f["E"], ref["g_q"] - ref["g_p"]), e_sc=(f["S"], ref["g_sc"] - ref["g_p"]),
                                       e_is=(f["I"], ref["g_is"] - ref["g_p"]), drift=(f["D"], ref["drift"]),
                                       g_p=(f["P"], ref["g_p"])).items():   # weight, would else divide rounding by itself)
                    errs[nm] = float((x - y).norm() / max(float(y.norm()), scale, 1e-300))
                checks["autograd_rel_err_real_rows"] = errs
                del ref, f, lq4, lt4
            qsubs.append(zq[sub].cpu())
            del zq
            kl = float(m["kl_qp"].mean())
            evals.append(dict(sigma=s, kl=kl, record=rec, restore_max_abs_logit_diff=rdiff, s=round(time.time() - te, 1)))
            ms.append(m)
            log(f"{ck}: sigma {s:.5f} -> mean KL(q||p) {kl:.5f} (restore diff {rdiff:.2e}; {time.time() - te:.1f}s)")
            return kl

        accepted, pts = tune_sigmas(eval_sigma, cfg["targets"], sigma_first_guess, prior=priors or None,
                                    max_evals=cfg["max_evals"])
        priors = {T: v["sigma"] for T, v in accepted.items() if v["sigma"]}
        rec0 = evals[0]["record"]
        if expected is None:
            checks["noise_matches_training_log"] = "skipped: no --expected-step0"
        elif ck == "base":
            ok = all(rec0[k] == expected[k] for k in ("n_tensors", "n_elems")) and all(
                abs(rec0[k] - expected[k]) <= 2e-6 * abs(expected[k]) for k in ("d_eps", "d_n", "nnorm1"))
            checks["noise_matches_training_log"] = bool(ok)
        else:
            checks["noise_nnorm1_matches_base"] = bool(abs(rec0["nnorm1"] - expected["nnorm1"])
                                                       <= 2e-6 * expected["nnorm1"])
            checks["d_n_minus_base"] = rec0["d_n"] - expected["d_n"]
        checks["restore_max_abs_logit_diff"] = max(e["restore_max_abs_logit_diff"] for e in evals)
        arrays = {"evals_sigma": np.array([e["sigma"] for e in evals]), "evals_kl": np.array([e["kl"] for e in evals])}
        for i, m in enumerate(ms):
            for k, v in m.items():
                arrays[f"e{i}__{k}"] = v
        for k, v in clean_m.items():
            arrays[f"clean__{k}"] = v
        np.savez_compressed(os.path.join(D, f"measures_{ck}.npz.tmp.npz"), **arrays)
        os.replace(os.path.join(D, f"measures_{ck}.npz.tmp.npz"), os.path.join(D, f"measures_{ck}.npz"))
        pt = cfg["param_target"]
        ia = accepted[pt]["index"]
        tune = dict(ck=ck, step=step, path=path, evals=evals, points=pts,
                    accepted={str(T): v for T, v in accepted.items()}, checks=checks, status="tuned",
                    seconds=round(time.time() - tck, 1))
        write_json(tp, tune)
        log(f"{ck}: accepted {json.dumps({str(T): (round(v['sigma'], 5), round(v['kl'], 5), v['ok']) for T, v in accepted.items()})}")
        # ---- parameter level at the param target (descriptive; the exact gradient probe's pull-back) ----
        left = deadline - time.time()
        pp = os.path.join(D, f"param_{ck}.json")
        if ia is None:
            tune["param"] = "no accepted sigma at the parameter target"
        elif left < cfg["param_min_s"]:
            tune["param"] = f"skipped: {left:.0f}s left < {cfg['param_min_s']}s"
            summ["skipped"][f"param_{ck}"] = tune["param"]
        else:
            del src
            gc.collect()
            tparam = time.time()
            qsub = qsubs[ia].to(dev)
            torch.save(dict(prefix_sha256=prec["sha256"], rows=sub, sigma=accepted[pt]["sigma"], logits=qsubs[ia]),
                       os.path.join(D, f"qsub_{ck}.pt"))
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
            fields = ("P", "E", "D", "S", "I")
            accs = {f: PL.FieldAcc(model) for f in fields}
            zdiff, lerr = 0.0, []
            for j, r in enumerate(sub):
                i, t = rows[r]
                pr = prompts[i]
                ids = torch.tensor(pr["prompt_ids"] + pr["response_ids"][:t], dtype=torch.long, device=dev)
                lqj = torch.log_softmax(qsub[j:j + 1].double(), -1)
                ltj = torch.log_softmax(zt[r:r + 1].double(), -1)

                def ff(z, lqj=lqj, ltj=ltj):
                    return logit_fields(torch.log_softmax(z, -1), lqj, ltj)
                z = pullback_rows(model, PL, ids, [len(pr["prompt_ids"]) - 1 + t], ff, accs, torch.float64)
                zdiff = max(zdiff, float((z - zp[r:r + 1].double()).abs().max()))
                f1 = logit_fields(torch.log_softmax(z, -1), lqj, ltj)
                lerr.append([float(f1[k].norm()) for k in fields])
                if j % 16 == 0 or j == len(sub) - 1:
                    log(f"{ck}: parameter-level {j + 1}/{len(sub)} ({time.time() - tparam:.0f}s)")
            ro = PL.readout({f: accs[f].tensors() for f in fields}, dev)
            G = ro["gram"]
            der = param_derived(G)
            la = np.array(lerr)
            der.update(n_prefixes=len(sub), target=pt, sigma=accepted[pt]["sigma"], kl=accepted[pt]["kl"],
                       grad_forward_max_abs_logit_diff=zdiff,
                       logit_mean_rel_err=dict(opd=float((la[:, 1] / la[:, 0]).mean()), sc=float((la[:, 3] / la[:, 0]).mean()),
                                               is_=float((la[:, 4] / la[:, 0]).mean())),
                       gram=G, seconds=round(time.time() - tparam, 1))
            write_json(pp, der)
            tune["param"] = "done"
            log(f"{ck}: parameter level rel err opd {der['rel_err_opd']:.4f} sc {der['rel_err_sc']:.4f} "
                f"is {der['rel_err_is']:.4f}; angle opd {der['angle_deg_opd']:.2f} deg ({time.time() - tparam:.0f}s)")
            del accs, ro, qsub
            model.gradient_checkpointing_disable()
            model.eval()
        tune["status"] = "complete"
        tune["seconds"] = round(time.time() - tck, 1)
        write_json(tp, tune)
        summ["timing"][ck] = tune["seconds"]
        save_summ()
        del model, zp, ms, qsubs
        try:
            del src
        except NameError:                                     # already freed before the parameter-level part
            pass
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    save_summ()
    log("measure DONE")
    return 0


def _audit_worker(worker):
    """Runs inside the in-process vLLM worker (mo_measure.py's audit, summarised): the cache dtype the engine really
    runs, every attention layer's cache dtype / storage dtype / backend, and the k / v scales (1.0 = uncalibrated)."""
    out = {}
    try:
        from vllm.model_executor.layers.attention.attention import Attention
        mr = worker.model_runner
        cc = mr.vllm_config.cache_config
        out.update(cache_dtype=str(cc.cache_dtype), calculate_kv_scales=bool(getattr(cc, "calculate_kv_scales", False)))
        per = []
        for name, m in mr.get_model().named_modules():
            if isinstance(m, Attention):
                kv = getattr(m, "kv_cache", None)
                t = kv[0] if isinstance(kv, (list, tuple)) and kv else kv
                be = getattr(m, "attn_backend", None)
                per.append((str(m.kv_cache_dtype), str(getattr(t, "dtype", None)),
                            be.get_name() if hasattr(be, "get_name") else str(be),
                            getattr(m, "_k_scale_float", None), getattr(m, "_v_scale_float", None)))
        out["n_attention_layers"] = len(per)
        out["layer_kinds"] = sorted({json.dumps(x) for x in per})
    except Exception as e:  # noqa: BLE001 (an audit, never a gate)
        out["error"] = repr(e)
    return out


def cmd_fp8(a):
    import numpy as np
    import torch
    cfg = MODES[a.mode]
    D = a.out_dir
    ck, kv = a.ckpt, a.kv
    tag = f"fp8_{ck}_{kv}"
    if os.path.exists(os.path.join(D, f"{tag}.json")):
        log(f"{tag}: exists, skipped")
        return 0
    prompts, prec = load_prefixes(D)
    rows = prefix_rows(prompts)
    N = len(rows)
    blob_p = torch.load(os.path.join(D, f"p_{ck}.pt"), map_location="cpu")
    blob_t = torch.load(os.path.join(D, "tau.pt"), map_location="cpu")
    assert blob_p["prefix_sha256"] == prec["sha256"] == blob_t["prefix_sha256"]
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    paths = checkpoint_paths(a.ckpt_path)
    if ck not in paths:
        raise SystemExit(f"[gap] REFUSE: {tag}: no --ckpt-path {ck}=DIR")
    path = paths[ck]
    maxlen = max(len(prompts[i]["prompt_ids"]) + t for i, t in rows)
    t0 = time.time()
    llm = LLM(model=path, tokenizer=STUDENT, dtype="bfloat16", kv_cache_dtype=kv, gpu_memory_utilization=0.5,
              max_model_len=maxlen + 8, enable_prefix_caching=False, seed=0, max_logprobs=-1, max_num_seqs=32,
              max_num_batched_tokens=8192, compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"})
    t_engine = time.time() - t0
    audit = llm.collective_rpc(_audit_worker)
    audit = audit[0] if isinstance(audit, list) and audit else audit
    log(f"{tag}: engine up in {t_engine:.0f}s; audit {json.dumps(audit)}")
    if kv == "fp8" and "fp8" not in str(audit.get("cache_dtype", "")):
        raise SystemExit(f"[gap] {tag}: the engine's cache dtype is {audit.get('cache_dtype')}, not fp8")
    dev = torch.device("cuda")
    zp = blob_p["logits"].to(dev)
    zt = blob_t["logits"].to(dev)
    V = zp.shape[1]
    sp = SamplingParams(max_tokens=1, temperature=1.0, top_p=1.0, top_k=-1, logprobs=a.k, detokenize=False,
                        flat_logprobs=True, seed=0)
    acc, infos = {}, []
    t1 = time.time()
    for c0 in range(0, N, a.chunk):
        c1 = min(N, c0 + a.chunk)
        reqs = [TokensPrompt(prompt_token_ids=prompts[i]["prompt_ids"] + prompts[i]["response_ids"][:t])
                for i, t in rows[c0:c1]]
        outs = llm.generate(reqs, [sp] * len(reqs), use_tqdm=False)
        lps = []
        lp_rows = torch.log_softmax(zp[c0:c1].double(), -1)
        for j, o in enumerate(outs):
            ids, vals = flat_position0(o.outputs[0].logprobs)
            lq, info = dense_logprobs(ids, vals, V, lp_fill=lp_rows[j])
            lps.append(lq)
            infos.append(info)
        lq = torch.stack(lps).to(dev)
        m = measures_from_logprobs(zp, lq, zt, list(range(c0, c1)))
        for k, v in m.items():
            acc.setdefault(k, []).append(v)
        del lq, lps, lp_rows
        log(f"{tag}: {c1}/{N} prefixes ({time.time() - t1:.0f}s)")
    arrays = {k: np.concatenate(v) for k, v in acc.items()}
    np.savez_compressed(os.path.join(D, f"{tag}.npz.tmp.npz"), **arrays)
    os.replace(os.path.join(D, f"{tag}.npz.tmp.npz"), os.path.join(D, f"{tag}.npz"))
    tails = np.array([x["tail_mass"] for x in infos])
    out = dict(ck=ck, kv=kv, path=path, k=a.k, vllm=vllm.__version__, audit=audit, engine_s=round(t_engine, 1),
               score_s=round(time.time() - t1, 1), N=N, mean_kl_qp=float(arrays["kl_qp"].mean()),
               mean_e_L2=float(arrays["e_L2"].mean()), returned_min=int(min(x["n_returned"] for x in infos)),
               filled_max=int(max(x["filled"] for x in infos)), tail_mass_max=float(tails.max()),
               prefix_sha256=prec["sha256"])
    write_json(os.path.join(D, f"{tag}.json"), out)
    log(f"{tag}: mean KL(q||p) {out['mean_kl_qp']:.5f}, mean ||e_opd||_2 {out['mean_e_L2']:.3e}; "
        f"returned >= {out['returned_min']} of {V}")
    return 0


def cmd_selfcheck(a):
    """CPU: closed forms vs autograd on random small vocabularies."""
    import torch
    g = torch.Generator().manual_seed(0)
    worst = 0.0
    for V in (2, 3, 5, 9, 33):
        for scale in (0.5, 3.0, 8.0):
            z = torch.randn(6, V, generator=g, dtype=torch.float64) * scale
            lq = torch.log_softmax(z + 0.3 * torch.randn(6, V, generator=g, dtype=torch.float64) * scale, -1)
            lt = torch.log_softmax(torch.randn(6, V, generator=g, dtype=torch.float64) * scale, -1)
            ref = autograd_reference(z, lq, lt)
            f = logit_fields(torch.log_softmax(z, -1), lq, lt)
            for x, y in ((f["P"], ref["g_p"]), (f["E"], ref["g_q"] - ref["g_p"]), (f["S"], ref["g_sc"] - ref["g_p"]),
                         (f["I"], ref["g_is"] - ref["g_p"]), (f["D"], ref["drift"])):
                worst = max(worst, float((x - y).abs().max()))
    print(json.dumps(dict(selfcheck="closed forms vs autograd", max_abs_err=worst, ok=worst < 1e-10)))
    return 0 if worst < 1e-10 else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "measure", "fp8"):
        s = sub.add_parser(name)
        s.add_argument("--out-dir", required=True)
        s.add_argument("--mode", choices=sorted(MODES), required=True)
    sub.choices["measure"].add_argument("--deadline", type=float, default=0.0, help="epoch seconds; 0 = none")
    sub.choices["measure"].add_argument("--attn", default="flash_attention_2")
    sub.choices["measure"].add_argument("--device", default="cuda:0")
    sub.choices["measure"].add_argument("--expected-step0", default="",
                                        help="JSON with the training log's step-0 [SPECOPD-TIM] fields; absent = not checked")
    for name in ("measure", "fp8"):
        sub.choices[name].add_argument("--ckpt-path", action="append", default=[], metavar="LABEL=DIR",
                                       help=f"HF dir of a trained checkpoint, LABEL in {sorted(CKPT_STEPS)} (repeatable)")
    sub.choices["fp8"].add_argument("--ckpt", choices=sorted(CKPT_STEPS), required=True)
    sub.choices["fp8"].add_argument("--kv", choices=("fp8", "auto"), required=True)
    sub.choices["fp8"].add_argument("--k", type=int, default=-1, help="logprobs per prefix (-1 = full vocabulary)")
    sub.choices["fp8"].add_argument("--chunk", type=int, default=64)
    sub.add_parser("selfcheck")
    a = ap.parse_args(argv)
    return dict(build=cmd_build, measure=cmd_measure, fp8=cmd_fp8, selfcheck=cmd_selfcheck)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
