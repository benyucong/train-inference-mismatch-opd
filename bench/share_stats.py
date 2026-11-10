#!/usr/bin/env python3
"""First-layer share of the offline mismatch: the statistics (CPU; numpy + stdlib). Used by bench/mo_formats_read.py.

    $PYTHON_BIN bench/share_stats.py --selftest

Input per configuration: tim_calibrate's decode outputs dec_<label>.json and dec_<label>.npz (per sampled token
log q from the configuration's engine and log p from a bf16 vLLM prefill, the response lengths, the prompt key of every
response). A configuration's mismatch is the token-pooled mean of d = log q(y) - log p(y) over its sampled tokens
(tim_calibrate's kl_samp_train).

The first-layer share of a configuration A is (A - B) / (A - F), with B = A with the first decoder layer's KV cache in
bf16 and F = the bf16 floor. Its 95% interval comes from a PAIRED bootstrap over prompts: N_BOOT resamples of the prompts
with replacement (numpy default_rng(SEED)), the SAME draw for every configuration, each configuration's value in a
resample being its token-pooled mean over the drawn prompts' samples; percentile interval. The draw is
tim_calibrate._cluster_ci's (same generator, seed and shape), so each configuration's marginal interval equals the
kl_samp_train_ci95 of its dec json. decide() reads the interval:
    - the denominator's interval (A - F, same resamples) must lie above 0; otherwise the share is UNDEFINED;
    - NOT FIRST-LAYER DOMINATED  iff the share's upper bound < NOT_DOM_HI;
    - FIRST-LAYER DOMINATED      iff the share's lower bound > DOM_LO;
    - otherwise INCONCLUSIVE.
"""
import json
import os
import sys

N_BOOT, SEED = 2000, 0
NOT_DOM_HI, DOM_LO = 0.5, 0.8


def load_dec(d, label, with_json=True):
    """dec_<label>.json + per-sample d = lq - lp (float64) and prompt keys from dec_<label>.npz."""
    import numpy as np
    js = json.load(open(os.path.join(d, f"dec_{label}.json"))) if with_json else None
    z = np.load(os.path.join(d, f"dec_{label}.npz"))
    lq, lp, lens = z["lq"].astype(np.float64), z["lp"].astype(np.float64), z["lens"].astype(np.int64)
    off = np.concatenate([[0], np.cumsum(lens)])
    ds = [lq[off[k]:off[k + 1]] - lp[off[k]:off[k + 1]] for k in range(len(lens))]
    finish = [str(x) for x in z["finish"]] if "finish" in z.files else []
    return dict(json=js, d=ds, groups=[str(g) for g in z["groups"]], finish=finish)


def per_key(dec, keys):
    """Per prompt key: the sum of d over the key's sampled tokens and their count."""
    import numpy as np
    s = {k: 0.0 for k in keys}
    c = {k: 0 for k in keys}
    for x, g in zip(dec["d"], dec["groups"]):
        s[g] += float(x.sum())
        c[g] += int(x.size)
    return np.array([s[k] for k in keys]), np.array([c[k] for k in keys], dtype=np.float64)


def pooled(dec):
    """The token-pooled mean of d (NaN without tokens)."""
    import numpy as np
    cat = np.concatenate(dec["d"]) if dec["d"] else np.zeros(0)
    return float(cat.mean()) if cat.size else float("nan")


def paired_boot(decs, n_boot=N_BOOT, seed=SEED):
    """One draw of prompt indices shared by every configuration (tim_calibrate._cluster_ci's draw: the same rng, seed and
    shape, so each configuration's marginal interval equals its dec json's kl_samp_train_ci95)."""
    import numpy as np
    keys = sorted(set().union(*[set(x["groups"]) for x in decs.values()]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), (n_boot, len(keys)))
    out = {}
    for name, dec in decs.items():
        s, c = per_key(dec, keys)
        num, den = s[idx].sum(1), c[idx].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return out, keys


def pct(x):
    """The 2.5% / 97.5% percentiles of the finite resamples."""
    import numpy as np
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if x.size else [float("nan")] * 2


def ratio_reading(pt, bs, num_hi, num_lo, den_hi, den_lo):
    """(pt[num_hi] - pt[num_lo]) / (pt[den_hi] - pt[den_lo]) with the paired-bootstrap interval; UNDEFINED when the
    denominator's interval does not lie above 0. Resamples with a denominator <= 0 are dropped (fewer than 2.5% of them
    whenever the ratio is defined) and counted."""
    import numpy as np
    den_b = bs[den_hi] - bs[den_lo]
    den_ci = pct(den_b)
    point_den = pt[den_hi] - pt[den_lo]
    rec = dict(denominator=point_den, denominator_ci95=den_ci)
    if not (den_ci[0] > 0):
        rec.update(defined=False, share=float("nan"), ci95=[float("nan")] * 2, n_dropped=int(np.sum(~(den_b > 0))))
        return rec
    ok = den_b > 0
    r_b = (bs[num_hi][ok] - bs[num_lo][ok]) / den_b[ok]
    rec.update(defined=True, share=(pt[num_hi] - pt[num_lo]) / point_den, ci95=pct(r_b), n_dropped=int(np.sum(~ok)))
    return rec


def decide(rec):
    """The reading of a ratio_reading record (see the module docstring)."""
    if not rec["defined"]:
        return "UNDEFINED"
    lo, hi = rec["ci95"]
    if hi < NOT_DOM_HI:
        return "NOT FIRST-LAYER DOMINATED"
    if lo > DOM_LO:
        return "FIRST-LAYER DOMINATED"
    return "INCONCLUSIVE"


# ---------------------------------------------------------------------------------------------------------------------
# selftest (synthetic decode outputs; numpy)
# ---------------------------------------------------------------------------------------------------------------------
_LAB = dict(F="f_floor", A="a_all_layers", B="b_first_layer_bf16")


def _fake_dir(root, means, seed=1, as_floor=False):
    """dec_<label>.npz + dec_<label>.json of F, A and B: 2 samples x 16 prompts with a prompt-level heterogeneity shared
    by the configurations; each json holds the pooled point and the configuration's own bootstrap interval, as
    tim_calibrate writes them. as_floor: A and B are bit-identical to the floor (nothing to attribute)."""
    import numpy as np
    d = os.path.join(root, f"D{seed}_{abs(hash(str(means))) % 10**6}_{'floor' if as_floor else 'ok'}")
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(seed)
    keys = [f"{b}:{i}" for b in ("AIME24", "MATH-500") for i in range(8)]
    from_prompt = rng.normal(0, 0.3, len(keys))
    floor_arrays = None
    for k, lab in _LAB.items():
        lq, lp, lens, groups = [], [], [], []
        if as_floor and k in ("A", "B") and floor_arrays is not None:
            lq, lp, lens, groups = (list(x) for x in floor_arrays)
        for gi, g in enumerate(keys if not lens else []):
            for _ in range(2):
                n = int(rng.integers(200, 800))
                dd = np.abs(rng.normal(means[k] * (1 + from_prompt[gi]), means[k] * 0.5 + 1e-6, n))
                p_ = rng.normal(-1.0, 0.3, n)
                lq.append((p_ + dd).astype(np.float32))
                lp.append(p_.astype(np.float32))
                lens.append(n)
                groups.append(g)
        if k == "F":
            floor_arrays = (list(lq), list(lp), list(lens), list(groups))
        np.savez(os.path.join(d, f"dec_{lab}.npz"), lq=np.concatenate(lq), lp=np.concatenate(lp),
                 lens=np.array(lens), groups=np.array(groups), finish=np.array(["stop"] * len(lens)))
        dec = load_dec(d, lab, with_json=False)
        bs, _ = paired_boot({k: dec})
        json.dump(dict(kl_samp_train=pooled(dec), kl_samp_train_ci95=pct(bs[k])),
                  open(os.path.join(d, f"dec_{lab}.json"), "w"))
    return d


def _share(d):
    decs = {k: load_dec(d, lab) for k, lab in _LAB.items()}
    pt = {k: pooled(v) for k, v in decs.items()}
    bs, keys = paired_boot(decs)
    rr = ratio_reading(pt, bs, "A", "B", "A", "F")
    return decide(rr), rr, decs, pt, bs, keys


def selftest():
    import tempfile
    root = tempfile.mkdtemp()
    dom = dict(F=0.001, A=0.5, B=0.02)                   # a first-layer-dominated shape (synthetic)
    # 1. the first layer dominates; the joint draw reproduces each configuration's own point and interval
    dec, rr, decs, pt, bs, keys = _share(_fake_dir(root, dom))
    assert dec == "FIRST-LAYER DOMINATED", (dec, rr)
    assert len(keys) == 16 and rr["defined"] and rr["n_dropped"] == 0, rr
    for k, x in decs.items():
        j = x["json"]
        assert abs(pt[k] - j["kl_samp_train"]) <= 1e-12 * max(1.0, abs(j["kl_samp_train"])), (k, pt[k], j)
        assert pct(bs[k]) == j["kl_samp_train_ci95"], (k, pct(bs[k]), j["kl_samp_train_ci95"])
        assert len(x["d"]) == len(x["groups"]) == 32 and x["finish"] == ["stop"] * 32
    # 2. removing the first layer's K/V removes little (B close to A)
    dec, rr, *_ = _share(_fake_dir(root, dict(F=0.001, A=0.30, B=0.27)))
    assert dec == "NOT FIRST-LAYER DOMINATED", (dec, rr)
    # 3. half-way: inconclusive
    dec, rr, *_ = _share(_fake_dir(root, dict(F=0.001, A=0.30, B=0.12)))
    assert dec == "INCONCLUSIVE", (dec, rr)
    # 4. nothing to attribute: A at the floor -> undefined
    dec, rr, *_ = _share(_fake_dir(root, dict(F=0.001, A=0.001, B=0.001), seed=3, as_floor=True))
    assert dec == "UNDEFINED" and not rr["defined"] and rr["n_dropped"] == N_BOOT, (dec, rr)
    # 4b. a tiny but positive mismatch gives a wide interval: inconclusive, never a verdict
    dec, rr, *_ = _share(_fake_dir(root, dict(F=0.0006, A=0.0007, B=0.0006), seed=4))
    assert dec in ("UNDEFINED", "INCONCLUSIVE"), (dec, rr)
    # 5. decide() on hand-made records: the bounds are strict
    assert decide(dict(defined=True, ci95=[0.1, 0.4999])) == "NOT FIRST-LAYER DOMINATED"
    assert decide(dict(defined=True, ci95=[0.1, 0.5])) == "INCONCLUSIVE"
    assert decide(dict(defined=True, ci95=[0.8001, 1.2])) == "FIRST-LAYER DOMINATED"
    assert decide(dict(defined=True, ci95=[0.8, 1.2])) == "INCONCLUSIVE"
    assert decide(dict(defined=False, ci95=[float("nan")] * 2)) == "UNDEFINED"
    print("[share_stats] SELFTEST PASS")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] != ["--selftest"]:
        raise SystemExit("usage: $PYTHON_BIN bench/share_stats.py --selftest (a library of bench/mo_formats_read.py)")
    sys.exit(selftest())
