#!/usr/bin/env python3
"""The prefix-shift reader: CPU, numpy. Reads what bench/prefix_shift.py wrote.

Parameter space. Every cell and term is a linear combination of the per-(prompt, half) sketches, so every norm, inner
product, cosine and share comes from the Gram matrix (gram.npz). Cells are token means:
    C^h(s, pi) = sum_k X(s, field, k, h) / sum_k T(s, k, h)        (h = response half A or B)
Cross-fitting: <X, Y>_cf = (<X^A, Y^B> + <X^B, Y^A>) / 2, ||X||^2_cf = <X^A, X^B> (may be negative; reported as is).
Plain values use both halves. The prompt is the resampling unit: 10,000 prompt resamples (seed 20261017), the same
resamples for every set and sampler of a checkpoint (paired), 95% percentile intervals.
    terms: G_ideal = C(std, p); Delta = T_q + S_p; T_q, S_p, T_p, S_q = Delta - T_p, I = T_q - T_p, G_actual,
    and the Shapley parts Tbar = (T_p + T_q) / 2, Sbar = (S_p + S_q) / 2 (Tbar + Sbar = Delta)
    r_X = ||X|| / ||G_ideal||, phi_T = ||T_q||^2 / ||Delta||^2, phi_S = ||S_p||^2 / ||Delta||^2, phi_x = 2<T_q,S_p>/||Delta||^2,
    the same three shares for the other order (phi_T_p, phi_S_q, phi_x_other) and the Shapley parts (phi_Tbar, phi_Sbar,
    phi_x_shapley), cosines, a_X = <X, G_ideal> / ||G_ideal||^2, the angle between G_actual and G_ideal, ||S_p|| / ||Delta||.
    FP8 token terms use the cut rows (weight T / 16) and S_p all rows; the all-cuts version (every term on cut rows) is
    reported beside it. Per-prefix sensitivity (each response counts once): r_S with response weights 1 / T (not
    computed when the hf pass's host-memory fallback kept G_p per response half; reading.txt says so).
Gates (any failure: the run is INVALID and nothing below is read):
    the CPU tests; the smoke (the reproduction of the exact gradient probe (tools/probe0), exact-against-sketch, every
    path ran); the standard engine against the trainer on P in [0.0003, 0.002]; vLLM-noisy against HF-noisy within
    +-0.002 nats; the placebo: every placebo ratio's interval contains 1 and the interval of ||S_0||^2_cf contains 0.
    The FP8 decode-against-prefill check only flags the FP8 token terms (S_p does not use q and stands).
Reported, NOT gating: the decode estimate of KL(q||p) within 5% of the exact mm(q); vLLM tail mass <= 1e-6;
    cut-weighted noise means inside the all-row interval; the prefix-caching check (a failed or skipped one rescored
    without caching); the noise restore and the teacher fingerprint (both refuse in the hf pass).
Readings: DETECTED, MATERIAL, SEPARATES; readings A, B, C; descriptive shift ratios. Fewer than 32 prompts at the
    parameter level at either checkpoint (a deadline-stopped hf pass): INCOMPLETE, not read.

    python bench/prefix_shift_read.py main  --theta0 D0 --theta200 D1 --smoke S [--json out.json]
    python bench/prefix_shift_read.py smoke --smoke S [--json S/reading.json]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prefix_shift as PS  # noqa: E402  (stdlib at import; torch only inside functions)

GP = PS.GP
B_BOOT = 10000
BOOT_SEED = 20261017
STD_TRAINER = (0.0003, 0.002)       # the standard engine against the trainer on P, nats per sampled token
VLLM_HF_MAX = 0.002                 # vLLM-noisy against HF-noisy, nats per sampled token
DEC_KL_REL = 0.05                   # the decode estimate of KL(q||p) within 5% of the exact mm(q)
FP8_DEC_FACTOR = 3.0                # FP8 decode-vs-prefill at most 3x the standard engine's
PHI_MATERIAL = 0.2                  # MATERIAL needs phi_S >= 0.2
ANGLE_DEG = 10.0                    # reading C: at least 10 degrees more turning
K_MIN = 32                          # the fallback: at least the first 32 prompts at the parameter level
CPU_TESTS_LINE = "CPU TESTS PASSED"
CKS = ("theta0", "theta200")


# ------------------------------------------------------------------------------------------------ small helpers
def summ(v):
    v = np.asarray(v, dtype=np.float64)
    bs = v[1:]
    ok = np.isfinite(bs)
    lo, hi = (np.percentile(bs[ok], [2.5, 97.5]) if ok.any() else (math.nan, math.nan))
    return dict(est=float(v[0]), lo=float(lo), hi=float(hi), n_nan=int((~ok).sum()))


def boot_mult(K, B=B_BOOT, seed=BOOT_SEED):
    """[B + 1, K] prompt multiplicities; row 0 = every prompt once (the point estimate)."""
    idx = np.random.default_rng(seed).integers(0, K, size=(B, K))
    m = np.zeros((B, K))
    np.add.at(m, (np.repeat(np.arange(B), K), idx.reshape(-1)), 1.0)
    return np.vstack([np.ones((1, K)), m])


def not3(x):
    return None if x is None else (not x)


def and3(xs):
    xs = list(xs)
    if any(x is False for x in xs):
        return False
    return None if any(x is None for x in xs) else True


def or3(xs):
    xs = list(xs)
    if any(x is True for x in xs):
        return True
    return None if any(x is None for x in xs) else False


def contains(s, x):
    return bool(s["lo"] <= x <= s["hi"])


# ------------------------------------------------------------------------------------------------ loading
def load_ck(D):
    hf = GP.read_json(os.path.join(D, "hf.json"))
    z = np.load(os.path.join(D, "gram.npz"))
    lay = json.loads(str(z["layout"]))
    n_done = int(z["n_done"])
    out = dict(D=D, hf=hf, G=z["G"], slot_pair=z["slot_pair"], slot_prompt=z["slot_prompt"], slot_unit=z["slot_unit"],
               layout=lay, K=n_done, T={s: z[f"T_{s}"][:n_done] for s in PS.SETS}, plan=PS.load_plan(D))
    out["rows"], out["rowsq"], out["sets"] = {}, {}, {}
    for s in PS.SETS:
        out["rows"][s] = dict(np.load(os.path.join(D, f"rows_{s}.npz")))
        out["sets"][s] = GP.read_json(os.path.join(D, "sets", f"{s}.json"))
        for qn in sum(PS.set_qs(s), ()):
            out["rowsq"][(s, qn)] = dict(np.load(os.path.join(D, f"rowsq_{s}_{qn}.npz")))
    out["scores"] = {e: GP.read_json(os.path.join(D, f"score_{e}.json")) for e in PS.SCORE_TARGETS}
    out["score_sets"] = {(e, s): GP.read_json(os.path.join(D, f"score_{e}_{s}.json"))
                         for e, ts in PS.SCORE_TARGETS.items() for s in ts}
    return out


# ---------------------------------------------------------------------------------------------- parameter space
class Param:
    """Quadratic forms of the per-(pair, prompt, half) sketch sums under prompt resampling."""

    def __init__(self, ck, m):
        lay = ck["layout"]
        self.pairs = {p["name"]: (i, p) for i, p in enumerate(lay["pairs"])}
        self.K, self.n, self.m = ck["K"], lay["n"], m
        npairs = len(lay["pairs"])
        nhu = npairs * self.K * 2
        G, sp, sk, su = ck["G"], ck["slot_pair"], ck["slot_prompt"], ck["slot_unit"]
        M = np.zeros((len(sp), nhu))
        self.has_pp = all(p["units"] == "response" for p in lay["pairs"] if p["field"] == "Gp")
        Mpp = np.zeros((len(sp), nhu)) if self.has_pp else None
        for r, (pi, k, u) in enumerate(zip(sp, sk, su)):
            p = lay["pairs"][pi]
            h = int(u) if p["units"] == "half" else PS.half_of(int(u), self.n)
            M[r, (pi * self.K + k) * 2 + h] = 1.0
            if self.has_pp and p["units"] == "response":
                T = ck["T"][p["set"]][k, u]
                Mpp[r, (pi * self.K + k) * 2 + h] = 1.0 / T if T > 0 else 0.0
        self.H = {"tok": M.T @ G @ M}
        if self.has_pp:
            self.H["pp"] = Mpp.T @ G @ Mpp
        self.Dn = {"tok": {}, "pp": {}}
        for s in PS.SETS:
            T = ck["T"][s]
            dt, dp = np.zeros((self.K, 2)), np.zeros((self.K, 2))
            for u in range(self.n):
                dt[:, PS.half_of(u, self.n)] += T[:, u]
                dp[:, PS.half_of(u, self.n)] += (T[:, u] > 0)
            self.Dn["tok"][s], self.Dn["pp"][s] = dt, dp
        self.cache = {}

    def idx(self, name, h):
        pi = self.pairs[name][0]
        return np.array([(pi * self.K + k) * 2 + h for k in range(self.K)])

    def Q(self, a, ha, b, hb, mode):
        key = (a, ha, b, hb, mode)
        if key not in self.cache:
            blk = self.H[mode][np.ix_(self.idx(a, ha), self.idx(b, hb))]
            self.cache[key] = np.einsum("bk,bk->b", self.m @ blk, self.m)
        return self.cache[key]

    def D(self, s, h, mode):
        return self.m @ self.Dn[mode][s][:, h]

    def ip(self, X, Y, ha, hb, mode="tok"):
        tot = 0.0
        for a, ca in X.items():
            for b, cb in Y.items():
                sa, sb = self.pairs[a][1]["set"], self.pairs[b][1]["set"]
                tot = tot + ca * cb * self.Q(a, ha, b, hb, mode) / (self.D(sa, ha, mode) * self.D(sb, hb, mode))
        return tot

    def cf(self, X, Y, mode="tok"):
        return 0.5 * (self.ip(X, Y, 0, 1, mode) + self.ip(X, Y, 1, 0, mode))

    def plain(self, X, Y, mode="tok"):
        tot = 0.0
        for a, ca in X.items():
            for b, cb in Y.items():
                sa, sb = self.pairs[a][1]["set"], self.pairs[b][1]["set"]
                num = sum(self.Q(a, ha, b, hb, mode) for ha in (0, 1) for hb in (0, 1))
                den = (self.D(sa, 0, mode) + self.D(sa, 1, mode)) * (self.D(sb, 0, mode) + self.D(sb, 1, mode))
                tot = tot + ca * cb * num / den
        return tot


def comb(*parts):
    out = {}
    for c, X in parts:
        for k, v in X.items():
            out[k] = out.get(k, 0.0) + c * v
    return {k: v for k, v in out.items() if v != 0.0}


def sampler_terms(q, version="mixed"):
    """The terms as linear combinations of (set:field) pairs. This order Delta = T_q + S_p; the other order
    Delta = T_p + S_q; the Shapley parts Tbar = (T_p + T_q) / 2 and Sbar = (S_p + S_q) / 2, with Tbar + Sbar = Delta."""
    S = PS.SAMPLER_SET[q]
    gp = "Gpc" if version == "all_cuts" else "Gp"
    Gi = {f"P:{gp}": 1.0}
    Tq = {f"{S}:E_{q}": 1.0}
    Tp = {f"P:E_{q}": 1.0}
    Sp = {f"{S}:{gp}": 1.0, f"P:{gp}": -1.0}
    D = comb((1, Tq), (1, Sp))
    Sq = comb((1, D), (-1, Tp))
    return dict(G_ideal=Gi, G_actual=comb((1, Gi), (1, D)), Delta=D, T_q=Tq, S_p=Sp, T_p=Tp, S_q=Sq,
                I=comb((1, Tq), (-1, Tp)), Tbar=comb((0.5, Tq), (0.5, Tp)), Sbar=comb((0.5, Sp), (0.5, Sq)))


# the three additive splits of Delta: (token part, shift part) -> share names (token, shift, twice the cross term)
SPLITS = {"this_order": ("T_q", "S_p", ("phi_T", "phi_S", "phi_x")),
          "other_order": ("T_p", "S_q", ("phi_T_p", "phi_S_q", "phi_x_other")),
          "shapley": ("Tbar", "Sbar", ("phi_Tbar", "phi_Sbar", "phi_x_shapley"))}


def sampler_readout(par, q, version="mixed"):
    T = sampler_terms(q, version)
    n2 = {k: par.cf(v, v) for k, v in T.items()}
    pl = {k: par.plain(v, v) for k, v in T.items()}
    gi = n2["G_ideal"]
    raw = {}
    for k in T:
        raw[f"norm2_cf_{k}"] = n2[k]
        raw[f"norm2_plain_{k}"] = pl[k]
        raw[f"noise_power_{k}"] = pl[k] - n2[k]
        raw[f"r2_{k}"] = n2[k] / gi
        raw[f"r_{k}"] = np.sqrt(np.maximum(n2[k] / gi, 0.0))
        raw[f"r_plain_{k}"] = np.sqrt(np.maximum(pl[k] / pl["G_ideal"], 0.0))
    for tok, sh, (pt, ps, px) in SPLITS.values():      # each split's three shares sum to 1 (cf is bilinear, symmetric)
        raw[pt], raw[ps], raw[px] = n2[tok] / n2["Delta"], n2[sh] / n2["Delta"], 2 * par.cf(T[tok], T[sh]) / n2["Delta"]
    names = ("G_ideal", "G_actual", "T_q", "S_p", "T_p", "Tbar", "Sbar")
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            den = n2[x] * n2[y]
            with np.errstate(invalid="ignore", divide="ignore"):
                raw[f"cos_{x}_{y}"] = np.where(den > 0, par.cf(T[x], T[y]) / np.sqrt(np.where(den > 0, den, 1.0)), np.nan)
    for x in ("Delta", "T_q", "S_p", "T_p", "S_q", "Tbar", "Sbar"):
        raw[f"a_{x}"] = par.cf(T[x], T["G_ideal"]) / gi
    c = raw["cos_G_ideal_G_actual"]
    raw["angle_deg"] = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
    raw["residual_after_token_correction"] = np.sqrt(np.maximum(raw["phi_S"], 0.0))
    return raw


def placebo_readout(par):
    Gi, S0 = {"P:Gp": 1.0}, {"P2:Gp": 1.0, "P:Gp": -1.0}
    n2 = par.cf(S0, S0)
    out = dict(norm2_cf_S_0=n2, norm2_plain_S_0=par.plain(S0, S0), r2_S_0=n2 / par.cf(Gi, Gi))
    if par.has_pp:
        out["norm2_cf_pp_S_0"] = par.cf(S0, S0, "pp")
        out["r2_pp_S_0"] = out["norm2_cf_pp_S_0"] / par.cf(Gi, Gi, "pp")
    return out


def prefix_sensitivity(par, q):
    """Per-prefix weighting (each response counts once, rows weighted 1 / T): S_p and its r^2, all rows."""
    S = PS.SAMPLER_SET[q]
    Gi, Sp = {"P:Gp": 1.0}, {f"{S}:Gp": 1.0, "P:Gp": -1.0}
    n2 = par.cf(Sp, Sp, "pp")
    return dict(norm2_cf_pp_S_p=n2, r2_pp_S_p=n2 / par.cf(Gi, Gi, "pp"))


# ---------------------------------------------------------------------------------------------------- logit level
class Logit:
    def __init__(self, ck, m):
        self.ck, self.m, self.K = ck, m, ck["K"]

    def mean(self, x, w, pid):
        x, w, pid = (np.asarray(v) for v in (x, w, pid))
        ok = np.isfinite(x) & (w != 0)
        num = np.bincount(pid[ok], weights=(w * x)[ok], minlength=self.K)[:self.K]
        den = np.bincount(pid[ok], weights=w[ok], minlength=self.K)[:self.K]
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.m @ num) / (self.m @ den)

    def weights(self, z, cut, weighting):
        T = z["T"].astype(np.float64)
        if cut:
            return z["w_cut"] if weighting == "token" else np.where(z["w_cut"] > 0, z["w_cut"] / T, 0.0)
        return np.ones_like(T) if weighting == "token" else 1.0 / T

    def base(self, s, key, weighting="token", cut=False):
        z = self.ck["rows"][s]
        return self.mean(z[key], self.weights(z, cut, weighting), z["i"].astype(np.int64))

    def q(self, s, qn, key, weighting="token"):
        z = self.ck["rowsq"][(s, qn)]
        cut = qn in ("f8all", "f8s0", "floor")
        return self.mean(z[key], self.weights(z, cut, weighting), z["i"].astype(np.int64))


QKEYS = ("kl_qp", "kl_pq", "Abar_q", "gq_L2", "e_L2", "cos_gq_gp", "drift_L2", "eis_L2", "gp_L2", "gap")


def logit_tables(L):
    out = {}
    for s in PS.SETS:
        for w in ("token", "per_prefix"):
            out[f"{s}|base|{w}"] = {k: summ(L.base(s, k, w)) for k in ("gap", "kl_Tp", "gp_L2", "H_p")}
        for qn in sum(PS.set_qs(s), ()):
            for w in ("token", "per_prefix"):
                d = {k: summ(L.q(s, qn, k, w)) for k in QKEYS}
                d["rho"] = summ(L.q(s, qn, "e_L2", w) / L.q(s, qn, "gp_L2", w))
                out[f"{s}|{qn}|{w}"] = d
        z = L.ck["rows"][s]
        if np.isfinite(z["lq_dec"]).any():
            out[f"{s}|sampled_d"] = summ(L.mean(z["lq_dec"] - z["lp_y"], np.ones(len(z["t"])), z["i"].astype(np.int64)))
    return out


def shift_ratios(L):
    out = {}
    pl_gap = L.base("P2", "gap") / L.base("P", "gap")
    pl_n = L.base("P2", "gp_L2") / L.base("P", "gp_L2")
    out["placebo"] = dict(KL_gap=summ(pl_gap), n_p=summ(pl_n))
    for q in PS.SAMPLERS:
        S = PS.SAMPLER_SET[q]
        out[q] = dict(KL_gap=summ(L.base(S, "gap") / L.base("P", "gap")),
                      n_p=summ(L.base(S, "gp_L2") / L.base("P", "gp_L2")),
                      e=summ(L.q(S, q, "e_L2") / L.q("P", q, "e_L2")),
                      mm=summ(L.q(S, q, "kl_qp") / L.q("P", q, "kl_qp")))
        kg, pg = out[q]["KL_gap"], out["placebo"]["KL_gap"]
        out[q]["teacher_gap_shift"] = bool((kg["lo"] > 1 or kg["hi"] < 1) and (kg["lo"] > pg["hi"] or kg["hi"] < pg["lo"]))
    return out


# ------------------------------------------------------------------------------------------------------- gates
def fp8_reference(ck):
    """The standard engine's decode-vs-prefill value, the FP8 check's reference: (value, source).
    theta200: P's decode log q (the standard generation engine) against the bf16-cache floor scoring engine's prefill at
    P's cuts, the same two engines' structure as the FP8 side (FP8 generation decode against FP8 scoring prefill).
    theta0: a documented stand-in, because theta0's P (the saved held-out samples) has no per-token decode log q: the
    sigma 0.15 noisy standard engine (Qn015) teacher-forcing its own responses, within one generation engine.
    A missing value is NaN, and NaN flags (fails closed)."""
    if ck["plan"]["ckpt"] == "theta200":
        dp = ck["score_sets"].get(("floor", "P"), {}).get("decode_prefill") or {}
        return float(dp.get("mean_abs", math.nan)), ("theta200: set P, decode log q of the standard generation engine "
                                                     "against the floor scoring engine's prefill at P's cuts")
    ref = ck["sets"]["Qn015"].get("decode_prefill_ref") or {}
    return float(ref.get("mean_abs_cut", math.nan)), ("theta0 stand-in: set Qn015 (sigma 0.15 noisy standard engine) "
                                                      "teacher-forced in its own generation engine, at its cuts; P has "
                                                      "no per-token decode log q at theta0")


def fp8_flags(ck):
    """The FP8 decode-against-prefill check flags the FP8 token terms (it does not gate): flagged unless the mean
    |delta log q(y)| at the sampler's own cuts is at most 3 times the reference. Missing values flag."""
    r_, src = fp8_reference(ck)
    flags = {}
    for f in ("f8all", "f8s0"):
        dp = ck["score_sets"].get((f, PS.SAMPLER_SET[f]), {}).get("decode_prefill") or {}
        val = float(dp.get("mean_abs", math.nan))
        flags[f] = dict(mean_abs=val, standard_engine=r_, reference_source=src, factor=FP8_DEC_FACTOR,
                        flagged=bool(not (val <= FP8_DEC_FACTOR * r_)))
    return flags


# Checks reported in reading.json / reading.txt, never gating. (The noise restore and the teacher fingerprint also
# refuse inside the hf pass itself, so a run that fails them has no gram.npz.)
NON_GATING = ("decode_kl_vs_exact_mm_n015", "decode_kl_vs_exact_mm_n0325", "tail_mass_le_1e-6", "cut_vs_all_rows",
              "noise_restored_bitwise", "teacher_unchanged", "prefix_caching")


def gate_values(ck, L, par):
    """The in-run gates for one checkpoint, plus the other checks (reported, not gating)."""
    g, chk = {}, {}
    zP = ck["rows"]["P"]
    if ck["plan"]["ckpt"] == "theta0":
        nll = np.asarray(ck["rows"]["P"]["resp_nll_sum"], np.float64)
        Tr = np.asarray(ck["rows"]["P"]["resp_T"], np.float64)
        v = (-np.sum(nll) - np.sum(zP["lp_y"])) / np.sum(Tr) if np.isfinite(nll).all() else math.nan
        src = "nll_sum of the saved held-out samples"
    else:
        v = float(np.mean(zP["lq_dec"] - zP["lp_y"]))
        src = "decode log q of the generated P"
    g["std_vs_trainer"] = dict(value=float(v), range=list(STD_TRAINER), source=src,
                               ok=bool(STD_TRAINER[0] <= v <= STD_TRAINER[1]))
    for q in ("n015", "n0325"):
        S = PS.SAMPLER_SET[q]
        zb, zq = ck["rows"][S], ck["rowsq"][(S, q)]
        if not (np.array_equal(zb["t"], zq["t"]) and np.array_equal(zb["resp"], zq["resp"])):
            raise SystemExit(f"[read] {S}: base and {q} rows are not aligned")
        diff = float(np.mean(zb["lq_dec"] - zq["lq_y"]))
        d_dec = float(np.mean(zb["lq_dec"] - zb["lp_y"]))
        mm = float(np.mean(zq["kl_qp"]))
        # The gate is the +-0.002 nats alone. The decode estimate of KL(q||p) within 5% of the exact mm(q) is
        # reported below as a check and does not gate.
        g[f"vllm_vs_hf_{q}"] = dict(mean_diff=diff, max_abs=VLLM_HF_MAX, ok=bool(abs(diff) <= VLLM_HF_MAX))
        chk[f"decode_kl_vs_exact_mm_{q}"] = dict(decode_kl=d_dec, exact_mm=mm, rel=abs(d_dec / mm - 1),
                                                 rel_max=DEC_KL_REL, ok=bool(abs(d_dec / mm - 1) <= DEC_KL_REL))
    sr = shift_ratios(L)
    pl = placebo_readout(par)
    s0 = summ(pl["norm2_cf_S_0"])
    g["placebo"] = dict(KL_gap=sr["placebo"]["KL_gap"], n_p=sr["placebo"]["n_p"], norm2_cf_S_0=s0,
                        ok=bool(contains(sr["placebo"]["KL_gap"], 1.0) and contains(sr["placebo"]["n_p"], 1.0)
                                and contains(s0, 0.0)))
    flags = fp8_flags(ck)
    g["fp8_decode_prefill"] = dict(flags=flags, gating=False, ok=True)
    tails = {f"{e}:{s}": v.get("tail_mass_max") for (e, s), v in ck["score_sets"].items()}
    chk["tail_mass_le_1e-6"] = dict(values=tails, ok=bool(all(t is not None and t <= 1e-6 for t in tails.values())))
    chk["noise_restored_bitwise"] = bool(all(v.get("restored_bitwise") for v in ck["hf"]["checks"]["noise"].values()))
    chk["teacher_unchanged"] = bool(ck["hf"]["checks"].get("teacher_unchanged"))
    cut = {}
    for q in ("n015", "n0325"):
        S = PS.SAMPLER_SET[q]
        z = ck["rowsq"][(S, q)]
        pid = z["i"].astype(np.int64)
        for key in ("kl_qp", "gap", "e_L2", "gp_L2"):
            allr = summ(L.mean(z[key], np.ones(len(pid)), pid))
            cw = float(L.mean(z[key], z["w_cut"], pid)[0])
            cut[f"{S}:{key}"] = dict(cut_weighted=cw, all_rows=allr, ok=contains(allr, cw))
    chk["cut_vs_all_rows"] = dict(items=cut, ok=bool(all(v["ok"] for v in cut.values())))
    caching = {e: dict(caching=v.get("caching"), check=v.get("check")) for e, v in ck["scores"].items()}
    chk["prefix_caching"] = caching
    return g, chk, flags, sr, pl


def cpu_tests_ok(D):
    p = os.path.join(D, "cpu_tests.txt")
    return bool(os.path.exists(p) and any(line.strip() == CPU_TESTS_LINE for line in open(p)))


# ---------------------------------------------------------------------------------------------------- the rules
def sampler_rule(mx, s0, flagged):
    """mx: one sampler's summaries (the mixed version); s0: the placebo's ||S_0||^2_cf summary.
    DETECTED: the lower end of ||S_p||^2_cf is above the upper end of the placebo's ||S_0||^2_cf.
    MATERIAL: detected and phi_S >= 0.2. phi_S = ||S_p||^2 / ||Delta||^2 and Delta holds the FP8 token term, so when
    the FP8 token terms are flagged and S is detected, MATERIAL is undetermined (None); S_p itself stands."""
    det = bool(mx["norm2_cf_S_p"]["lo"] > s0["hi"])
    mat = (det and mx["phi_S"]["est"] >= PHI_MATERIAL) if not (flagged and det) else None
    return dict(DETECTED=det, MATERIAL=mat, token_terms_flagged=bool(flagged), phi_S=mx["phi_S"]["est"],
                S_p_lo=mx["norm2_cf_S_p"]["lo"], placebo_hi=s0["hi"])


def family_rule(fa, pa, flagged):
    """fa, pa: the raw arrays of the family's sampler with the larger and with the smaller perturbation (index 0 = the
    point estimate, 1.. = the prompt resamples, paired). SEPARATES_S: the paired interval of r_S(larger) - r_S(smaller)
    excludes 0, and r_S(larger)/r_S(smaller) > r_T(larger)/r_T(smaller), written cross-multiplied (r >= 0).
    SEPARATES_T: the mirror. Direction (reading C): the paired interval of angle(larger) - angle(smaller) has its lower
    end at or above 10 degrees, or the upper end of a_Delta(larger) is below 0. Flagged FP8 token terms: all three
    undetermined (None)."""
    dS, dT = summ(fa["r_S_p"] - pa["r_S_p"]), summ(fa["r_T_q"] - pa["r_T_q"])
    rSf, rSp, rTf, rTp = (float(x[0]) for x in (fa["r_S_p"], pa["r_S_p"], fa["r_T_q"], pa["r_T_q"]))
    exS, exT = bool(dS["lo"] > 0 or dS["hi"] < 0), bool(dT["lo"] > 0 or dT["hi"] < 0)
    sepS = bool(exS and rSf * rTp > rTf * rSp)
    sepT = bool(exT and rTf * rSp > rSf * rTp)
    ang = summ(fa["angle_deg"] - pa["angle_deg"])
    aD = summ(fa["a_Delta"])
    direction = bool(ang["lo"] >= ANGLE_DEG or aD["hi"] < 0)
    return dict(r_S_diff=dS, r_T_diff=dT, r_S=dict(larger=rSf, smaller=rSp), r_T=dict(larger=rTf, smaller=rTp),
                SEPARATES_S=None if flagged else sepS, SEPARATES_T=None if flagged else sepT, angle_diff_deg=ang,
                a_Delta_larger=aD, direction=None if flagged else direction, token_terms_flagged=bool(flagged),
                S_sign=("larger > smaller" if dS["est"] > 0 else "larger <= smaller"))


# ----------------------------------------------------------------------------------------------- one checkpoint
def read_ck(D, B=B_BOOT):
    ck = load_ck(D)
    m = boot_mult(ck["K"], B)
    par, L = Param(ck, m), Logit(ck, m)
    gates, checks, flags, sr, pl = gate_values(ck, L, par)
    res = dict(D=D, ckpt=ck["plan"]["ckpt"], K=ck["K"], K_planned=len(ck["plan"]["prompts"]), status=ck["hf"]["status"],
               gates=gates, checks=checks, layout=dict(gp_units=ck["layout"]["gp_units"], all_cuts=ck["layout"]["all_cuts"]),
               host_memory_fallback=ck["hf"].get("host_memory_fallback"), non_gating_checks=list(NON_GATING))
    res["placebo"] = {k: summ(v) for k, v in pl.items()}
    res["samplers"] = {}
    s0 = res["placebo"]["norm2_cf_S_0"]
    for q in PS.SAMPLERS:
        versions = ("mixed", "all_cuts") if (q in ("f8all", "f8s0") and ck["layout"]["all_cuts"]) else ("mixed",)
        d = {}
        for ver in versions:
            raw = sampler_readout(par, q, ver)
            d[ver] = {k: summ(v) for k, v in raw.items()}
            d[ver]["_raw"] = raw
        if par.has_pp:
            d["per_prefix"] = {k: summ(v) for k, v in prefix_sensitivity(par, q).items()}
            d["per_prefix"]["DETECTED"] = bool(d["per_prefix"]["norm2_cf_pp_S_p"]["lo"] > res["placebo"]["norm2_cf_pp_S_0"]["hi"])
        flagged = bool(flags.get(q, {}).get("flagged", False))
        d["rule"] = sampler_rule(d["mixed"], s0, flagged)
        res["samplers"][q] = d
    res["families"] = {}
    for fam, fs in PS.FAMILIES.items():
        fa, pa = res["samplers"][fs["larger"]]["mixed"]["_raw"], res["samplers"][fs["smaller"]]["mixed"]["_raw"]
        flagged = any(res["samplers"][x]["rule"]["token_terms_flagged"] for x in (fs["larger"], fs["smaller"]))
        res["families"][fam] = dict(smaller=fs["smaller"], larger=fs["larger"], **family_rule(fa, pa, flagged))
    for q in res["samplers"]:
        for ver in list(res["samplers"][q]):
            if isinstance(res["samplers"][q][ver], dict):
                res["samplers"][q][ver].pop("_raw", None)
    res["logit"] = logit_tables(L)
    res["shift_ratios"] = sr
    res["valid"] = bool(all(v["ok"] for v in gates.values()))
    return res


def readings(per):
    """Readings A, B, C over both checkpoints (three-valued: True / False / None = undetermined)."""
    A_items, B_items, C_items, dir_items = [], [], [], []
    for ck, r in per.items():
        for fam, f in r["families"].items():
            A_items.append(and3([r["samplers"][f["larger"]]["rule"]["MATERIAL"], f["SEPARATES_S"]]))
            C_items.append(and3([not3(f["SEPARATES_S"]), not3(f["SEPARATES_T"])]))
            dir_items.append(f["direction"])
        for q, s in r["samplers"].items():
            B_items.append(not3(s["rule"]["MATERIAL"]))
    A, B, C = or3(A_items), and3(B_items), and3(C_items)
    sub = None
    if C:
        sub = "direction, not size" if any(x is True for x in dir_items) else (
            "not visible at a fixed point" if all(x is False for x in dir_items) else "undetermined (flagged token terms)")
    return dict(A=A, B=B, C=C, C_reading=sub,
                share_per_token_corrections_cannot_remove={ck: {q: r["samplers"][q]["mixed"]["residual_after_token_correction"]
                                                                for q in r["samplers"]} for ck, r in per.items()})


# ----------------------------------------------------------------------------------------------------- printing
def fmt(s, nd=4):
    return f"{s['est']:.{nd}g} [{s['lo']:.{nd}g}, {s['hi']:.{nd}g}]"


def text_ck(r):
    L = [f"== {r['ckpt']} ({r['D']}): {r['K']} of {r['K_planned']} prompts, hf status {r['status']}, "
         f"G_p units {r['layout']['gp_units']}"]
    if r["layout"]["gp_units"] != "response":
        L.append("NOTE: the per-prefix weighting (each response counts once) was NOT computed: the hf pass's host-memory "
                 f"fallback kept G_p per response half ({json.dumps(r.get('host_memory_fallback'))})")
    L.append("gates:")
    for k, v in r["gates"].items():
        L.append(f"  {'ok  ' if v['ok'] else 'FAIL'} {k}: " + json.dumps({a: b for a, b in v.items() if a != 'ok'}, default=str)[:300])
    L.append("checks (reported, NOT gating): "
             + json.dumps({k: (v["ok"] if isinstance(v, dict) and "ok" in v else v) for k, v in r["checks"].items()},
                          default=str)[:400])
    L.append(f"placebo ||S_0||^2_cf {fmt(r['placebo']['norm2_cf_S_0'])}, r2 {fmt(r['placebo']['r2_S_0'])}")
    L.append(f"{'sampler':>7} {'r_T':>26} {'r_S':>26} {'phi_S':>26} {'angle':>8} {'DET':>5} {'MAT':>5}")
    for q, s in r["samplers"].items():
        m = s["mixed"]
        L.append(f"{q:>7} {fmt(m['r_T_q']):>26} {fmt(m['r_S_p']):>26} {fmt(m['phi_S']):>26} {m['angle_deg']['est']:8.3f} "
                 f"{str(s['rule']['DETECTED']):>5} {str(s['rule']['MATERIAL']):>5}")
    L.append("other order (T_p + S_q) and Shapley parts (Tbar + Sbar), descriptive:")
    for q, s in r["samplers"].items():
        m = s["mixed"]
        L.append(f"{q:>7} r_T_p {fmt(m['r_T_p'])} r_S_q {fmt(m['r_S_q'])} | r_Tbar {fmt(m['r_Tbar'])} r_Sbar "
                 f"{fmt(m['r_Sbar'])} phi_Tbar {fmt(m['phi_Tbar'])} phi_Sbar {fmt(m['phi_Sbar'])}")
    for fam, f in r["families"].items():
        L.append(f"family {fam}: SEPARATES_S {f['SEPARATES_S']} SEPARATES_T {f['SEPARATES_T']} (r_S diff {fmt(f['r_S_diff'])}, "
                 f"r_T diff {fmt(f['r_T_diff'])}); direction {f['direction']} (angle diff {fmt(f['angle_diff_deg'])}, "
                 f"a_Delta(larger) {fmt(f['a_Delta_larger'])})")
    for q, v in r["shift_ratios"].items():
        L.append(f"shift ratios {q}: " + ", ".join(f"{k} {fmt(x)}" for k, x in v.items() if isinstance(x, dict)))
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------------- commands
def cmd_main(a):
    out = dict(experiment="prefix shift", B=a.boot, seed=BOOT_SEED, gates={}, checkpoints={})
    smoke_ok = False
    sp = os.path.join(a.smoke, "reading.json") if a.smoke else ""
    if sp and os.path.exists(sp):
        smoke_ok = GP.read_json(sp).get("status") == "SMOKE PASS"
    out["gates"]["smoke"] = dict(path=sp, ok=bool(smoke_ok))
    dirs = {ck: d for ck, d in (("theta0", a.theta0), ("theta200", a.theta200)) if d}
    out["gates"]["cpu_tests"] = dict(ok=bool(dirs and all(cpu_tests_ok(d) for d in dirs.values())))
    per = {}
    for ck, d in dirs.items():
        if not os.path.exists(os.path.join(d, "gram.npz")):
            out["checkpoints"][ck] = dict(D=d, missing="gram.npz")
            continue
        per[ck] = read_ck(d, a.boot)
        out["checkpoints"][ck] = per[ck]
    valid = bool(out["gates"]["smoke"]["ok"] and out["gates"]["cpu_tests"]["ok"] and per
                 and all(r["valid"] for r in per.values()))
    complete = set(per) == set(CKS)
    partial = any(r["K"] < r["K_planned"] for r in per.values())
    few = {ck: r["K"] for ck, r in per.items() if r["K"] < K_MIN}
    counts = ", ".join(f"{ck} {r['K']} of {r['K_planned']} prompts" for ck, r in per.items())
    out["prompts"] = {ck: dict(K=r["K"], K_planned=r["K_planned"]) for ck, r in per.items()}
    if not valid:
        out["status"] = "INVALID"
        for ck, r in per.items():          # stop before reading: only the gates are kept
            out["checkpoints"][ck] = dict(D=r["D"], gates=r["gates"], checks=r["checks"])
    elif not complete:
        out["status"] = "INCOMPLETE (readings A, B, C need both checkpoints)"
    elif few:
        # the fallback is the first 32 prompts at the parameter level; a deadline-stopped hf pass with fewer is not
        # read (it cannot be resumed: its sketches are not on disk)
        out["status"] = f"INCOMPLETE (fewer than {K_MIN} prompts at the parameter level: {few})"
    else:
        out["status"] = "VALID" + (f" (PARTIAL: fewer prompts than planned at the parameter level: {counts})"
                                   if partial else "")
        out["readings"] = readings(per)
        out["readings"]["prompts"] = out["prompts"]
    txt = [f"prefix-shift reading: {out['status']}", f"prompts at the parameter level: {counts or 'none'}",
           "gates: " + json.dumps(out["gates"]),
           "not gating (reported per checkpoint): " + ", ".join(NON_GATING)]
    for ck, r in per.items():
        if valid:
            txt.append(text_ck(r))
        else:
            txt.append(f"== {ck}: gates " + json.dumps({k: v["ok"] for k, v in r["gates"].items()}))
    if "readings" in out:
        rd = out["readings"]
        txt.append(f"Reading A (the prefix shift is material for the larger perturbation and separates it from the "
                   f"smaller): {rd['A']}   [{counts}]")
        txt.append(f"Reading B (the fixed-prefix error suffices): {rd['B']}   [{counts}]")
        txt.append(f"Reading C (neither size separates): {rd['C']}" + (f" -> {rd['C_reading']}" if rd["C_reading"] else "")
                   + f"   [{counts}]")
    print("\n".join(txt))
    if a.json:
        GP.write_json(a.json, out)
    return 0 if (valid and complete and not few) else (3 if not valid else 5)


def cmd_smoke(a):
    S = a.smoke
    out = dict(experiment="prefix-shift smoke", gates={})
    rp = os.path.join(S, "repro", "repro.json")
    rep = GP.read_json(rp) if os.path.exists(rp) else {}
    out["gates"]["probe_reproduction"] = dict(ok=bool(rep.get("ok")) and not rep.get("max_traces"),
                                                  compare=rep.get("compare"), noise=rep.get("noise", {}).get("check"))
    ep = os.path.join(S, "theta0", "exact_check.json")
    ex = GP.read_json(ep) if os.path.exists(ep) else {}
    out["gates"]["exact_vs_sketch"] = dict(ok=bool(ex.get("ok")), r_T=ex.get("r_T"), r_S=ex.get("r_S"),
                                           entries_ok=ex.get("entries_ok"))
    paths, miss = {}, []
    for ck in CKS:
        D = os.path.join(S, ck)
        need = [f"sets/{s}.json" for s in PS.SETS] + [f"score_{e}.json" for e in PS.SCORE_TARGETS] + \
               ["gram.npz", "hf.json"] + [f"rows_{s}.npz" for s in PS.SETS]
        miss += [f"{ck}/{f}" for f in need if not os.path.exists(os.path.join(D, f))]
        for e in PS.SCORE_TARGETS:
            p = os.path.join(D, f"score_{e}.json")
            if os.path.exists(p) and GP.read_json(p).get("status") != "complete":
                miss.append(f"{ck}/score_{e}.json (status {GP.read_json(p).get('status')})")
        p = os.path.join(D, "hf.json")
        if os.path.exists(p) and GP.read_json(p).get("status") != "complete":
            miss.append(f"{ck}/hf.json (status {GP.read_json(p).get('status')})")
        # the f8s0 engines ran with the FA3 skipped-layer patch: generation and scoring
        fa3 = {}
        p = os.path.join(D, "sets", "Qf8s0.json")
        if os.path.exists(p):
            fa3["gen"] = (GP.read_json(p).get("fa3_kvskip") or {}).get("after_set")
        p = os.path.join(D, "score_f8s0.json")
        if os.path.exists(p):
            sj = GP.read_json(p)
            fa3["score"] = (sj.get("fa3_kvskip_off") or sj.get("fa3_kvskip_on") or {}).get("now")
        for k in ("gen", "score"):
            if not (isinstance(fa3.get(k), dict) and fa3[k].get("installed") is True):
                miss.append(f"{ck} f8s0 {k}: FA3 skipped-layer patch record ({fa3.get(k)})")
        paths[ck] = fa3
    out["fa3_kvskip"] = paths
    hf0 = GP.read_json(os.path.join(S, "theta0", "hf.json")) if os.path.exists(os.path.join(S, "theta0", "hf.json")) else {}
    mp = hf0.get("mem_probe", {})
    if not mp.get("ran"):
        miss.append("theta0 mem probe")
    out["gates"]["every_path_ran"] = dict(ok=not miss, missing=miss)
    out["gates"]["cpu_tests"] = dict(ok=cpu_tests_ok(S))
    out["mem_probe"] = mp
    ok = all(v["ok"] for v in out["gates"].values())
    out["status"] = "SMOKE PASS" if ok else "SMOKE FAIL"
    info = {}
    for ck in CKS:
        D = os.path.join(S, ck)
        if os.path.exists(os.path.join(D, "gram.npz")):
            try:
                r = read_ck(D, a.boot)
                info[ck] = dict(gates_on_smoke_data=r["gates"], checks=r["checks"],
                                samplers={q: v["rule"] for q, v in r["samplers"].items()})
            except Exception as e:  # noqa: BLE001 (informational only: the smoke has 4 prompts)
                info[ck] = dict(error=repr(e))
    out["informational"] = info
    rows = hf0.get("timing", {}).get("responses", [])
    if rows:
        rate = sum(x[5] for x in rows) / sum(x[3] * (x[4] + 1) for x in rows)
        mpj = os.path.join(S, "theta0", "sets", "MEMPROBE.jsonl")
        main_tok = json.loads(open(mpj).readline()).get("main_P_tokens") if os.path.exists(mpj) else None
        fields = sum(len(PS.set_fields(s)) + 1 for s in PS.SETS)
        out["planning_estimate"] = dict(
            s_per_token_per_pass=rate, main_P_tokens=main_tok,
            hf_loop_s_if_every_set_had_P_lengths=(rate * fields * main_tok) if main_tok else None,
            note="planning only (not a reading): smoke rate x (fields + 1 forward) x main P tokens, per set")
    print(json.dumps(dict(status=out["status"], gates={k: v["ok"] for k, v in out["gates"].items()},
                          missing=out["gates"]["every_path_ran"]["missing"], planning=out.get("planning_estimate")), indent=1))
    if a.json:
        GP.write_json(a.json, out)
    return 0 if ok else 3


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("main")
    m.add_argument("--theta0", default="")
    m.add_argument("--theta200", default="")
    m.add_argument("--smoke", default="")
    m.add_argument("--json", default="")
    m.add_argument("--boot", type=int, default=B_BOOT)
    s = sub.add_parser("smoke")
    s.add_argument("--smoke", required=True)
    s.add_argument("--json", default="")
    s.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(argv)
    return cmd_main(a) if a.cmd == "main" else cmd_smoke(a)


if __name__ == "__main__":
    sys.exit(main())
