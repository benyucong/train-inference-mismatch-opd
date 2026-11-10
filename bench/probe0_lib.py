"""The exact gradient probe (tools/probe0): library.

Everything here is HF transformers + torch only. Notation:
  p = student (trainer) next-token distribution, r = teacher, q = sampler; a = log p - log r; KL = sum p a;
  C = sampler's top-K ids, tau = tail bucket; P_tau = sum_{v not in C} p, R_tau = sum_{v not in C} r;
  a~_v = a_v on C, a_tau = log P_tau - log R_tau off C;  L = sum_C p a + P_tau a_tau.

Logit-space vector fields per response row, part (a):
  G_KL     = p (a - KL)                              exact on-policy E_p[k1 PG] = grad_z KL(p||r)
  G_k1q    = q a - (q.a) p                           exact E_{y~q}[k1 PG]
  G_bias   = G_k1q - G_KL = (q - p) a - (q.a - KL) p (computed directly, no cancellation through G_k1q)
  G_drift  = (q.a) (q - p)                           score-centering drift part of the bias
  G_an     = p (a~ - L)                              analytic top-K + tail estimator, C = q's top-K
  G_andiff = G_an - G_KL = p ((a~ - a) - (L - KL))   (computed directly)
Part (b), on-policy, per row with the trace's own next token y:
  G_k1s    = a(y) (e_y - p)                          the sampled k1 PG logit gradient (verl k1 path, ratio 1)
  G_an     = as above with C = p's top-K (on-policy sampler = p)

Pull-back to parameter space: a field G becomes the parameter-space vector
  g = sum_t J_t^T G_t  (J_t = dz_t/dtheta),  i.e. the gradient of sum_t <stopgrad(G_t), z_t>,
computed with autograd through the student: z = lm_head(h).to(fdtype) per row chunk, h = the body's final hidden
states (gradient checkpointing), one body backward per field with retain_graph. The 1/N token mean is applied
at read-out time (linear).
"""
import hashlib
import importlib.util
import math
import os
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_specopd_tim(path=None):
    """The rollout engine's SamplerNoise module, loaded BY FILE (verl/verl/utils/specopd_tim.py or a copy of it;
    default: specopd_tim.py beside this file) so that verl/__init__ (ray, ...) is never imported."""
    path = path or os.path.join(HERE, "specopd_tim.py")
    spec = importlib.util.spec_from_file_location("probe_specopd_tim", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------------------------
# Logit-space formulas (one row = one response position; tensors [R, V] in fdtype)
# ----------------------------------------------------------------------------------------------------------------

def head_mask(sel_logits, k):
    """Boolean [R, V] mask of the top-k ids of sel_logits (the sampler's top-K = C_t). k >= V -> all True."""
    R, V = sel_logits.shape
    if k >= V:
        return torch.ones(R, V, dtype=torch.bool, device=sel_logits.device)
    idx = sel_logits.topk(k, dim=-1).indices
    m = torch.zeros(R, V, dtype=torch.bool, device=sel_logits.device)
    m.scatter_(1, idx, True)
    return m


def coarse_terms(logp, logr, a, mask):
    """L = sum_C p a + P_tau a_tau, a~ (tail entries replaced by a_tau), and the masses. Exact when the tail is
    empty (then L = KL, a~ = a). Tail masses from a masked logsumexp (no 1 - sum cancellation)."""
    p = logp.exp()
    head = (p * a).masked_fill(~mask, 0.0).sum(-1)
    tail_nonempty = (~mask).any(-1)
    logPt = torch.logsumexp(logp.masked_fill(mask, -math.inf), -1)
    logRt = torch.logsumexp(logr.masked_fill(mask, -math.inf), -1)
    a_tau = torch.where(tail_nonempty, logPt - logRt, torch.zeros_like(logPt))
    Pt = torch.where(tail_nonempty, logPt.exp(), torch.zeros_like(logPt))
    Rt = torch.where(tail_nonempty, logRt.exp(), torch.zeros_like(logRt))
    L = head + Pt * a_tau
    atilde = torch.where(mask, a, a_tau[:, None])
    return dict(L=L, atilde=atilde, Pt=Pt, Rt=Rt, a_tau=a_tau, head=head)


def level_fields(z, logr, logq=None, k=128, want=("KL", "bias", "drift", "andiff"), extra_ks=()):
    """Fields of part (a) for one sampler level. z = student logits [R, V] (fdtype), logr = teacher log-probs,
    logq = sampler log-probs (None = on-policy, q = p). Returns (fields dict, per-row stats dict)."""
    logp = torch.log_softmax(z, -1)
    p = logp.exp()
    a = logp - logr
    KL = (p * a).sum(-1)
    out, st = {}, {}
    st["kl"] = KL
    if logq is None:
        logq_ = logp
        q = p
    else:
        logq_ = logq
        q = logq.exp()
    G_KL = p * (a - KL[:, None])
    st["nG_KL"] = G_KL.norm(dim=-1)
    if "KL" in want:
        out["KL"] = G_KL
    Eqa = (q * a).sum(-1)
    st["Eqa"] = Eqa
    if logq is not None:
        st["kl_qp"] = (q * (logq - logp)).sum(-1)       # KL(q||p): the rollout_corr/kl direction
        st["kl_pq"] = (p * (logp - logq)).sum(-1)
        dq = q - p
        dEqa = (dq * a).sum(-1)                         # E_q[a] - KL summed directly (no cancellation)
        st["dEqa"] = dEqa
        G_bias = dq * a - dEqa[:, None] * p
        G_drift = Eqa[:, None] * dq
        st["nG_bias"] = G_bias.norm(dim=-1)
        st["nG_drift"] = G_drift.norm(dim=-1)
        if "bias" in want:
            out["bias"] = G_bias
        else:
            del G_bias
        if "drift" in want:
            out["drift"] = G_drift
        else:
            del G_drift
        del dq
    ks = sorted(set([k] + [int(x) for x in extra_ks]))
    kmax = min(max(ks), z.shape[-1])
    top_idx = logq_.topk(kmax, dim=-1).indices
    for kk in ks:
        if kk >= z.shape[-1]:
            mask = torch.ones_like(p, dtype=torch.bool)
        else:
            mask = torch.zeros_like(p, dtype=torch.bool)
            mask.scatter_(1, top_idx[:, :kk], True)
        ct = coarse_terms(logp, logr, a, mask)
        if kk == k:
            st["L"] = ct["L"]
            st["P_tau"] = ct["Pt"]
            st["student_head_mass"] = (p * mask).sum(-1)
            st["sampler_head_mass"] = (q * mask).sum(-1)
            st["teacher_head_mass"] = (logr.exp() * mask).sum(-1)
            st["tail_term"] = ct["Pt"] * ct["a_tau"]
            G_andiff = p * ((ct["atilde"] - a) - (ct["L"] - KL)[:, None])
            st["nG_andiff"] = G_andiff.norm(dim=-1)
            st["nG_an"] = (G_KL + G_andiff).norm(dim=-1)
            if "andiff" in want:
                out["andiff"] = G_andiff
            else:
                del G_andiff
        else:
            st[f"L_k{kk}"] = ct["L"]
        del ct, mask
    if logq is not None and "nG_bias" in st:
        st["ratio_bias"] = st["nG_bias"] / st["nG_KL"].clamp_min(1e-30)
        st["ratio_drift"] = st["nG_drift"] / st["nG_KL"].clamp_min(1e-30)
    if "KL" not in want:
        del G_KL
    return out, st


def an_field(z, logr, sel_logits, k):
    """G_an = p (a~ - L) with C = top-k of sel_logits (closed form)."""
    logp = torch.log_softmax(z, -1)
    p = logp.exp()
    a = logp - logr
    ct = coarse_terms(logp, logr, a, head_mask(sel_logits, k))
    return p * (ct["atilde"] - ct["L"][:, None]), ct["L"]


def k1s_field(z, logr, y):
    """G_k1s = a(y) (e_y - p): the logit gradient of the verl k1 PG loss a.detach() * log p(y) at ratio 1."""
    logp = torch.log_softmax(z, -1)
    p = logp.exp()
    a_y = (logp.gather(1, y[:, None]) - logr.gather(1, y[:, None]))[:, 0]
    G = -a_y[:, None] * p
    G.scatter_add_(1, y[:, None], a_y[:, None])
    return G, a_y


def kl_autograd_logit(z, logr):
    """grad_z of sum_rows KL(softmax(z)||r) by autograd (the identity reference in logit space)."""
    with torch.enable_grad():
        zz = z.detach().clone().requires_grad_(True)
        lp = torch.log_softmax(zz, -1)
        kl = (lp.exp() * (lp - logr)).sum()
        (g,) = torch.autograd.grad(kl, zz)
    return g


# ----------------------------------------------------------------------------------------------------------------
# Field accumulators and the pull-back
# ----------------------------------------------------------------------------------------------------------------

def param_list(model):
    """(name, param) for every trainable parameter; lm_head is handled separately (row-chunked)."""
    lm = model.get_output_embeddings().weight
    body = [(n, p) for n, p in model.named_parameters() if p is not lm]
    return body, lm


class FieldAcc:
    """Sum over traces of the pulled-back gradient of one field. Body parameters accumulate on acc_device in acc_dtype
    (default CPU fp32, 30 GB per field at 7B); lm_head accumulates on the compute device and is flushed at the end."""

    def __init__(self, model, acc_device="cpu", acc_dtype=torch.float32):
        body, lm = param_list(model)
        self.names = [n for n, _ in body] + ["lm_head.weight"]
        self.body = [torch.zeros(p.shape, dtype=acc_dtype, device=acc_device) for _, p in body]
        self.lm = torch.zeros(lm.shape, dtype=acc_dtype, device=lm.device)
        self.acc_dtype = acc_dtype
        self.flushed = None

    def add_body(self, grads):
        for a, g in zip(self.body, grads):
            if g is not None:
                a.add_(g.to(device=a.device, dtype=a.dtype))

    def add_lm(self, g):
        self.lm.add_(g.to(self.acc_dtype))

    def tensors(self):
        """name -> tensor (lm_head moved to the body's device once)."""
        if self.flushed is None:
            dev = self.body[0].device if self.body else self.lm.device
            self.flushed = self.lm.to(dev)
            self.lm = None
        return dict(zip(self.names, self.body + [self.flushed]))


def forward_hidden(model, ids, grad):
    """Final (normed) hidden states of the body for one sequence: [S, d]."""
    with torch.set_grad_enabled(grad):
        out = model.get_decoder()(input_ids=ids[None], use_cache=False)
    return out.last_hidden_state[0]


def pullback_trace(model, ids, n_prompt, field_fn, accs, chunk, fdtype, stats_sink=None):
    """One sequence (prompt + response). Response row t (predicting response token t) sits at position
    n_prompt - 1 + t. field_fn(z_chunk [R,V] fdtype detached, r0, r1) -> (dict name -> G, dict of per-row stats).
    accs: dict name -> FieldAcc (fields present in the returned dict but absent from accs are ignored).
    Returns the number of response rows."""
    body, lm = param_list(model)
    body_p = [p for _, p in body]
    h = forward_hidden(model, ids, grad=True)
    T = ids.numel() - n_prompt
    h_r = h[n_prompt - 1: n_prompt - 1 + T]
    gh = {name: torch.zeros_like(h_r) for name in accs}
    for c0 in range(0, T, chunk):
        c1 = min(T, c0 + chunk)
        hc = h_r[c0:c1].detach().requires_grad_(True)
        z = F.linear(hc, lm).to(fdtype)
        with torch.no_grad():
            fields, st = field_fn(z.detach(), c0, c1)
        if stats_sink is not None:
            stats_sink(st)
        names = [n for n in accs if n in fields]
        for j, name in enumerate(names):
            ghc, gW = torch.autograd.grad(z, [hc, lm], grad_outputs=fields[name].to(z.dtype),
                                          retain_graph=(j < len(names) - 1))
            gh[name][c0:c1] = ghc
            accs[name].add_lm(gW)
            del ghc, gW
        del z, hc, fields, st
    names = list(accs)
    for j, name in enumerate(names):
        grads = torch.autograd.grad(h_r, body_p, grad_outputs=gh[name], retain_graph=(j < len(names) - 1),
                                    allow_unused=True)
        accs[name].add_body(grads)
        del grads
    del h, h_r, gh
    return T


def reference_kl_grad(model, ids, n_prompt, logr, acc):
    """The identity reference: vanilla HF forward (model(...).logits, no chunking, no detached-hidden split), the
    full-vocab KL summed over the response rows in logr.dtype, one .backward(); the parameter gradients are added
    into the FieldAcc ``acc``. Returns the summed KL."""
    model.zero_grad(set_to_none=True)
    T = ids.numel() - n_prompt
    logits = model(input_ids=ids[None], use_cache=False).logits[0, n_prompt - 1: n_prompt - 1 + T]
    lp = torch.log_softmax(logits.to(logr.dtype), -1)
    kl = (lp.exp() * (lp - logr)).sum()
    kl.backward()
    kl = kl.detach()
    body, lm = param_list(model)
    acc.add_body([p.grad for _, p in body])
    acc.add_lm(lm.grad)
    model.zero_grad(set_to_none=True)
    return float(kl)


# ----------------------------------------------------------------------------------------------------------------
# Read-out: Gram matrix and the n_hat projection (one streaming pass over the parameters)
# ----------------------------------------------------------------------------------------------------------------

def _dot64(x, y, step=1 << 26):
    x = x.reshape(-1)
    y = y.reshape(-1)
    s = 0.0
    for i in range(0, x.numel(), step):
        s += float(torch.dot(x[i:i + step].double(), y[i:i + step].double()))
    return s


def readout(vecs, dev, clean_params=None, tim=None, seed=None, diff_pairs=()):
    """vecs: dict field -> dict name -> tensor (summed gradients). Returns
       gram[f][g] = <v_f, v_g> over all parameters,
       gram_t[f][g] = the same restricted to the noise targets (is_target),
       proj_raw[f] = sum_name s_name <v_f[name], eps_name>, n2 = sum_name s_name^2 ||eps_name||^2 (fp32 dot, as the
       training hook computes it), so <v_f, n_hat> = proj_raw[f] / sqrt(n2);
       diff[(f, g)] = ||v_f - v_g||^2 and maxabs[(f, g)] = max |v_f - v_g| computed directly (no Gram cancellation),
       plus n_tensors / n_elems of the target set. clean_params: name -> CLEAN weight tensor (s_name = std of its fp32
       copy, the step-0 scales); tim: the specopd_tim module (is_target, make_eps)."""
    fs = list(vecs)
    names = list(next(iter(vecs.values())).keys())
    for f in fs:
        assert list(vecs[f].keys()) == names, f"field {f}: parameter list differs"
    gram = {f: {g: 0.0 for g in fs} for f in fs}
    gram_t = {f: {g: 0.0 for g in fs} for f in fs}
    proj = {f: 0.0 for f in fs}
    n2 = 0.0
    diff = {pr: 0.0 for pr in diff_pairs}
    maxabs = {pr: 0.0 for pr in diff_pairs}
    n_t = 0
    n_e = 0
    for name in names:
        xs = {}
        for f in fs:  # float64 inputs (small checks) stay float64; everything else is read as fp32 (the accumulators)
            v = vecs[f][name]
            xs[f] = v.to(dev, torch.float64 if v.dtype == torch.float64 else torch.float32)
        is_t = tim is not None and clean_params is not None and tim.is_target(name, clean_params[name])
        for i, f in enumerate(fs):
            for g in fs[i:]:
                d = _dot64(xs[f], xs[g])
                gram[f][g] += d
                if is_t:
                    gram_t[f][g] += d
        for pr in diff_pairs:
            dd = xs[pr[0]] - xs[pr[1]]
            diff[pr] += _dot64(dd, dd)
            maxabs[pr] = max(maxabs[pr], float(dd.abs().max()))
            del dd
        if is_t:
            w0 = clean_params[name].to(dev, torch.float32)
            s = float(w0.std())
            eps = tim.make_eps(seed, name, w0.shape, w0.device)
            n2 += s * s * float(torch.dot(eps.reshape(-1), eps.reshape(-1)).double())
            for f in fs:
                proj[f] += s * _dot64(xs[f], eps)
            n_t += 1
            n_e += w0.numel()
            del w0, eps
        del xs
    for i, f in enumerate(fs):
        for g in fs[i:]:
            gram[g][f] = gram[f][g]
            gram_t[g][f] = gram_t[f][g]
    return dict(gram=gram, gram_t=gram_t, proj_raw=proj, n2=n2, diff=diff, maxabs=maxabs, nnorm1=math.sqrt(n2) if n2 > 0 else float("nan"),
                n_tensors=n_t, n_elems=n_e)


# ----------------------------------------------------------------------------------------------------------------
# Sampler noise (verbatim training code path)
# ----------------------------------------------------------------------------------------------------------------

def apply_sampler_noise(model, sigma, seed, tim, stream_dtype=torch.float32):
    """Build q = the noised student EXACTLY as the rollout engine does: every parameter, in named_parameters order
    (HF names = the actor stream's names), goes through tim.SamplerNoise(sigma, seed).apply as a 1-tensor bucket in
    the stream dtype (float32, as the trainer streams it), and the returned tensor is copy_'d into the engine's (bf16)
    parameter, i.e. the RNE cast vLLM's load_weights performs. Scales are std of the CLEAN weights at this call
    (= the first sync). Returns the finish_sync() record (d_eps, d_n, nnorm1, n_tensors, n_elems) and the CPU clean
    copies of the modified tensors (for restore_clean)."""
    sn = tim.SamplerNoise(sigma, seed)
    clean = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not tim.is_target(name, p):
                continue
            clean[name] = p.detach().to("cpu", copy=True)
            ((_, w_noisy),) = sn.apply([(name, p.detach().to(stream_dtype))])
            p.copy_(w_noisy)
            del w_noisy
    rec = sn.finish_sync()
    return rec, clean


def restore_clean(model, clean):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in clean:
                p.copy_(clean[name].to(p.device))
        for name, p in model.named_parameters():
            if name in clean:
                assert torch.equal(p.detach().cpu(), clean[name]), f"restore failed for {name}"


def nhat_reference(names_shapes_scales, seed, tim, device="cpu"):
    """Explicit n_hat = concat(s * eps) / ||concat(s * eps)|| (the noise direction's definition): a slow reference
    for checking the per-tensor projection in readout on small models."""
    parts = []
    for name, shape, s in names_shapes_scales:
        parts.append(s * tim.make_eps(seed, name, shape, device).reshape(-1).double())
    n = torch.cat(parts)
    return n / n.norm()


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, key):
        tm = self

        class _C:
            def __enter__(self_):
                self_.t0 = time.time()

            def __exit__(self_, *a):
                tm.t[key] = tm.t.get(key, 0.0) + time.time() - self_.t0
        return _C()


# ----------------------------------------------------------------------------------------------------------------
# Derived numbers (part (a) reports; part (b) split-half SNR)
# ----------------------------------------------------------------------------------------------------------------

def _nan_div(a, b):
    return a / b if b not in (0, 0.0) and b == b else float("nan")


def derive_level(ro, N, lvl, has_bias=True):
    """Part (a) numbers for one sampler level from a readout `ro` holding fields KL, andiff_<lvl> (and bias_<lvl>,
    drift_<lvl> unless lvl == 'none'). All vectors are SUMS over response rows; the token mean divides by N."""
    G, P = ro["gram"], ro["proj_raw"]
    sn = math.sqrt(ro["n2"])
    K = "KL"
    ad = f"andiff_{lvl}"
    nKL = math.sqrt(G[K][K])
    out = dict(level=lvl)
    out["norm_g_KL"] = nKL / N
    out["proj_g_KL_nhat"] = P[K] / sn / N
    # g_an = g_KL + g_andiff
    an2 = G[K][K] + 2 * G[K][ad] + G[ad][ad]
    out["norm_g_an"] = math.sqrt(max(an2, 0.0)) / N
    out["cos_an_KL"] = _nan_div(G[K][K] + G[K][ad], math.sqrt(max(an2, 0.0)) * nKL)
    out["r_andiff"] = math.sqrt(G[ad][ad]) / nKL
    out["cos_andiff_KL"] = _nan_div(G[K][ad], math.sqrt(G[ad][ad]) * nKL)
    out["proj_g_andiff_nhat"] = P[ad] / sn / N
    out["proj_g_an_nhat"] = (P[K] + P[ad]) / sn / N
    if has_bias:
        b, d = f"bias_{lvl}", f"drift_{lvl}"
        out["r_bias"] = math.sqrt(G[b][b]) / nKL
        out["r_drift"] = math.sqrt(G[d][d]) / nKL
        out["cos_bias_KL"] = _nan_div(G[b][K], math.sqrt(G[b][b]) * nKL)
        out["cos_drift_KL"] = _nan_div(G[d][K], math.sqrt(G[d][d]) * nKL)
        out["cos_bias_drift"] = _nan_div(G[b][d], math.sqrt(G[b][b] * G[d][d]))
        out["norm_g_bias"] = math.sqrt(G[b][b]) / N
        out["norm_g_drift"] = math.sqrt(G[d][d]) / N
        out["proj_g_bias_nhat"] = P[b] / sn / N
        out["proj_g_drift_nhat"] = P[d] / sn / N
        out["proj_g_k1q_nhat"] = (P[K] + P[b]) / sn / N
        out["drift_share_of_bias_along_nhat"] = _nan_div(P[d], P[b])
        # unit-vector projections (fraction of each gradient that lies along n_hat)
        out["cos_bias_nhat"] = _nan_div(P[b] / sn, math.sqrt(G[b][b]))
        out["cos_drift_nhat"] = _nan_div(P[d] / sn, math.sqrt(G[d][d]))
        # the same norms restricted to the 196 noise-target tensors
        Gt = ro["gram_t"]
        out["r_bias_targets_only"] = _nan_div(math.sqrt(Gt[b][b]), math.sqrt(Gt[K][K]))
        out["r_drift_targets_only"] = _nan_div(math.sqrt(Gt[d][d]), math.sqrt(Gt[K][K]))
    out["cos_KL_nhat"] = _nan_div(P[K] / sn, nKL)
    out["cos_andiff_nhat"] = _nan_div(P[ad] / sn, math.sqrt(G[ad][ad]))
    return out


def split_half(ro, est, NA, NB):
    """Part (b): fields <est>_A, <est>_B are SUMS over the half's response rows (rollout index 0-3 vs 4-7 of every
    prompt). g_h = sum_h / N_h (token mean per half), g = (sum_A + sum_B) / (N_A + N_B) (the full-batch estimate).
      S^2 = <g_A, g_B>               unbiased for ||mu||^2 given the prompts (halves independent given prompts)
      N^2 = ||g||^2 - S^2            the noise power of the full 64-trace estimate
    plus the per-half noise ||g_h||^2 - S^2 and SNR = S^2 / N^2."""
    G = ro["gram"]
    A, B = f"{est}_A", f"{est}_B"
    S2 = G[A][B] / (NA * NB)
    gA2 = G[A][A] / NA ** 2
    gB2 = G[B][B] / NB ** 2
    Nt = NA + NB
    g2 = (G[A][A] + 2 * G[A][B] + G[B][B]) / Nt ** 2
    return dict(S2=S2, norm2_g=g2, N2=g2 - S2, norm2_gA=gA2, norm2_gB=gB2, N2_half_A=gA2 - S2, N2_half_B=gB2 - S2,
                SNR=_nan_div(S2, g2 - S2), cos_A_B=_nan_div(G[A][B], math.sqrt(G[A][A] * G[B][B])))


def quantiles(x):
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {}
    q = np.percentile(x, [1, 10, 50, 90, 99])
    return dict(n=int(x.size), mean=float(x.mean()), min=float(x.min()), p1=float(q[0]), p10=float(q[1]),
                p50=float(q[2]), p90=float(q[3]), p99=float(q[4]), max=float(x.max()))
