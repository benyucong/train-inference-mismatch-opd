#!/usr/bin/env python3
"""Reader of the Model Optimizer formats job (descriptive; CPU, numpy + stdlib).

    $PYTHON_BIN bench/mo_formats_read.py --out-dir D [--json D/reading.json] [--anchors A.json]
    $PYTHON_BIN bench/mo_formats_read.py --selftest

Per engine configuration (mo_common.FORMATS): the offline mismatch (tim_calibrate's decode kl_samp_train, i.e. the
token-pooled mean of log q - log p over the sampled tokens, 95% bootstrap over prompts), k3, the 0-1k bin, mean length,
truncated fraction, its ratio to the unit-scale FP8 cache (mo_kv8_unit), and the first-layer share of each pair in
mo_common.SHARE_PAIRS ((A - B) / (A - F) with a paired bootstrap over prompts and its reading: bench/share_stats.py).

A configuration is read only if it is VALID: exit 0, 32 samples, the engine audit (engine_<label>.json, taken after the
build and again after the decode) shows exactly the configuration it claims:
  - cache: every attention layer FP8 (kv_cache_dtype fp8/fp8_e4m3, storage uint8/float8_e4m3fn) except the expected
    bf16 layers (auto/bfloat16, storage bfloat16); kv 'auto' = every layer bf16;
  - scales on every FP8 layer: 'unit' -> the _k_scale and _v_scale tensors are 1.0; 'calibrated' -> they equal the
    export's per-layer k_scale / v_scale as vLLM serves them (rounded to bf16: vLLM's k_scale / v_scale checkpoint
    parameters take the model dtype; mo_common.bf16) and layer 0's (when FP8) is not 1.0. A run that silently
    fell back to scale 1.0 is INVALID. The float mirrors, _q_scale and _prob_scale are reported (queries at 1.0 is
    vLLM's behaviour without a q_scale in the checkpoint);
  - the same scales after the decode as after the build;
  - every decoder linear layer has the expected quant method (UnquantizedLinearMethod on the expected bf16 layers);
    FP8 weights are float8_e4m3fn; the NVFP4 backend is the expected one;
  - the backend of every attention layer is FLASH_ATTN; the short generation is not empty.
Emulated NVFP4 configurations (mo_common.NVFP4_EMU; kv 'static'): the k / v tensors equal the static file's (the
engine's SPECOPD_SAMPLER_QUANT_STATIC, recorded in engine_<label>.json), the decoder linear layers carry the expected
method (the NVFP4 activation quantizer for a=nvfp4), the sampler-quant attach record shows every decoder linear layer
patched with a static amax and no LM head, the activation op ran on bf16 only, the weight re-stream quantized the
expected number of HF weights with the expected spec and the live weights lie on the NVFP4 grid (off-grid <= 1e-3), and
the engine is compiled with CUDA graphs (FULL_AND_PIECEWISE). A configuration absent from every directory read is
"not run" (listed, not a row).
The job's anchors are validity checks of the whole reading: the floor (mo_z_none) and the unit-scale FP8 cache
(mo_kv8_unit) must be VALID and, with --anchors (a JSON file {label: [lo, hi]}: reference 95% intervals measured with
the same harness), each must read inside its reference interval. If either fails, nothing is read (the kernel or the
software stack changed against the reference). Without a reference interval the interval check is skipped and the
report says so.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mo_common as C  # noqa: E402

FP8_KV = ({"fp8", "fp8_e4m3"}, {"torch.uint8", "torch.float8_e4m3fn"})
BF16_KV = ({"auto", "bfloat16"}, {"torch.bfloat16"})


def _share_lib():
    import share_stats
    return share_stats


def _rd(p):
    with open(p) as f:
        return json.load(f)


def kv_class(a):
    if a["kv_cache_dtype"] in FP8_KV[0] and a["storage"] in FP8_KV[1]:
        return "fp8"
    if a["kv_cache_dtype"] in BF16_KV[0] and a["storage"] in BF16_KV[1]:
        return "bf16"
    return None


def f32(x):
    import struct
    return struct.unpack("f", struct.pack("f", float(x)))[0]


def check_audit(aud, cfg, export_kv, tag):
    """Problems of one engine audit against the configuration (empty = ok)."""
    bad = []
    e = cfg["expect"]
    bf16_layers = set(e.get("kv_bf16_layers", ()))
    attn = sorted(aud.get("attn", []), key=lambda r: r["layer"])
    if [r["layer"] for r in attn] != list(range(C.N_LAYERS)):
        bad.append(f"{tag}: attention layers {[r['layer'] for r in attn][:4]}... != 0..{C.N_LAYERS - 1}")
    for r in attn:
        li = r["layer"]
        want = "bf16" if (e["kv"] == "auto" or li in bf16_layers) else "fp8"
        got = kv_class(r)
        if got != want:
            bad.append(f"{tag}: layer {li} cache {r['kv_cache_dtype']}/{r['storage']} is {got}, want {want}")
            continue
        if r["backend"] != "FLASH_ATTN":
            bad.append(f"{tag}: layer {li} backend {r['backend']}")
        if want != "fp8":
            continue
        k, v = r.get("k") or [], r.get("v") or []
        if e["kv"] == "unit":
            if k != [1.0] or v != [1.0]:
                bad.append(f"{tag}: layer {li} k/v scale {k}/{v}, want 1.0")
        elif e["kv"] in ("calibrated", "static", "per_head"):
            ref = (export_kv or {}).get(str(li))
            if not ref:
                bad.append(f"{tag}: layer {li}: no exported k/v scale")
                continue
            wk, wv = ([C.bf16(x) for x in _as_list(ref[q])] for q in ("k", "v"))
            n_want = None if e["kv"] != "per_head" else len(wk)
            if e["kv"] == "per_head" and (n_want < 2 or len(k) != n_want):
                bad.append(f"{tag}: layer {li} has {len(k)} k scale(s); per-head export has {n_want} (not loaded per head)")
            elif e["kv"] != "per_head" and len(wk) != 1:
                bad.append(f"{tag}: layer {li}: the export has {len(wk)} k scales, want one per tensor")
            elif k != wk or v != wv:
                bad.append(f"{tag}: layer {li} k/v scale {k}/{v} != exported {ref['k']}/{ref['v']} as vLLM serves it "
                           f"(bf16: {wk}/{wv})")
            if li == 0 and k and all(x == 1.0 for x in k):
                bad.append(f"{tag}: layer 0 k scale is 1.0 (fell back to unit scale)")
        elif e["kv"] == "computed":
            if not k or not v or k == [1.0] or v == [1.0] or not all(math.isfinite(x) and x > 0 for x in k + v):
                bad.append(f"{tag}: layer {li} k/v scale {k}/{v}: not computed by vLLM (want finite, positive, != 1.0)")
    if e.get("cudagraph_mode"):
        comp = aud.get("compilation") or {}
        if comp.get("cudagraph_mode") != e["cudagraph_mode"] or comp.get("enforce_eager"):
            bad.append(f"{tag}: compilation {comp}, want cudagraph_mode {e['cudagraph_mode']} with torch.compile on")
    lin_bf16 = set(e.get("linear_bf16_layers", ()))
    lins = aud.get("linear", [])
    if not lins:
        bad.append(f"{tag}: no decoder linear layers in the audit")
    for r in lins:
        want = "UnquantizedLinearMethod" if r["layer"] in lin_bf16 else e["linear"]
        if r["method"] != want:
            bad.append(f"{tag}: {r['name']} method {r['method']}, want {want}")
        elif want == "ModelOptFp8LinearMethod" and r.get("weight_dtype") != "torch.float8_e4m3fn":
            bad.append(f"{tag}: {r['name']} weight {r.get('weight_dtype')}, want float8_e4m3fn")
        elif want == "ModelOptNvFp4LinearMethod" and e.get("nvfp4_backend") and \
                e["nvfp4_backend"].upper() not in str(r.get("backend")).upper():
            bad.append(f"{tag}: {r['name']} NVFP4 backend {r.get('backend')}, want {e['nvfp4_backend']}")
    return bad


def scales_of(aud):
    return {r["layer"]: (r.get("k"), r.get("v"), r.get("q"), r.get("prob")) for r in aud.get("attn", [])}


def _bases(base):
    if not base:
        return []
    return [x for x in (base if isinstance(base, (list, tuple)) else str(base).split(",")) if x]


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _find_export(name, roots):
    """The export record of ckpt `name` in the first root that has it, or None."""
    for r in roots:
        p = os.path.join(r, name, "export_record.json")
        if os.path.exists(p):
            return _rd(p)
    return None


def label_dir(d, base, label):
    """The directory holding this configuration's outputs: the run's own (d) if it has them, else the first base run
    (base: one directory, a comma-separated list or a list) that has them; None if none has."""
    for x in [d] + _bases(base):
        if os.path.exists(os.path.join(x, f"rc_{label}.txt")):
            return x
    return None


def check_emulated(eng, cfg):
    """Problems of an emulated NVFP4 configuration's sampler-quant records (see the module docstring)."""
    e, bad = cfg["expect"], []
    qs = eng.get("quant_sync") or {}
    if qs.get("spec") != e["sampler_quant"]:
        bad.append(f"sampler-quant spec {qs.get('spec')!r}, want {e['sampler_quant']}")
    if qs.get("n_tensors") != e["n_weight_tensors"]:
        bad.append(f"weight re-stream quantized {qs.get('n_tensors')} HF weights, want {e['n_weight_tensors']}")
    qa = eng.get("quant_audit") or {}
    for k in ("layers.0.self_attn.qkv_proj", "layers.0.self_attn.o_proj", "layers.0.mlp.gate_up_proj", "mlp.down_proj"):
        if k not in qa or qa[k] > 1e-3:
            bad.append(f"live weights {k} off the NVFP4 grid: {qa.get(k)}")
    for tag in ("audit_built", "audit_after_decode"):
        aud = eng.get(tag) or {}
        sp = aud.get("sampler_quant_patched") or {}
        if e.get("act_attach", "a=" in e["sampler_quant"]):
            if sp.get("n_linear") != 4 * C.N_LAYERS or sp.get("lm_head") != 0 or any(
                    sp.get(k) for k in ("nvfp4_missing", "nvfp4_unused", "smooth_missing", "smooth_unused")):
                bad.append(f"{tag}: activation attach {dict((k, sp.get(k)) for k in ('n_linear', 'lm_head'))}")
            seen = aud.get(e.get("op_seen", "nvfp4_op_seen")) or {}
            if set(seen) != {e["op_dtype"]}:
                bad.append(f"{tag}: the NVFP4 activation op ran on {seen}, want only {e['op_dtype']}")
        if e["kv"] == "static" and not (sp.get("static_kv") or {}).get("ok_at_load"):
            bad.append(f"{tag}: static KV scales not attached ({sp.get('static_kv')})")
    return bad


def validity(d, label, cfg, ckpt_root):
    roots = _as_list(ckpt_root)
    bad = []
    rc = os.path.join(d, f"rc_{label}.txt")
    if not os.path.exists(rc) or open(rc).read().strip() != "0":
        bad.append(f"rc {open(rc).read().strip() if os.path.exists(rc) else 'missing'}")
    js = os.path.join(d, f"dec_{label}.json")
    if not os.path.exists(js):
        return bad + ["no dec json (not scored)"], None
    dec = _rd(js)
    if dec.get("n_samples", 0) + dec.get("n_empty", 0) != C.N_SAMPLES:
        bad.append(f"samples {dec.get('n_samples')} + empty {dec.get('n_empty')} != {C.N_SAMPLES}")
    ep = os.path.join(d, f"engine_{label}.json")
    if not os.path.exists(ep):
        return bad + ["no engine audit"], dec
    eng = _rd(ep)
    export_kv = None
    if cfg["expect"]["kv"] == "static" or cfg["env"].get("SPECOPD_SAMPLER_QUANT_STATIC"):
        sp = (eng.get("env") or {}).get("SPECOPD_SAMPLER_QUANT_STATIC")
        if not sp or not os.path.exists(sp):
            bad.append(f"static file {sp!r} not readable")
        else:
            export_kv = _rd(sp).get("kv")
        bad += check_emulated(eng, cfg)
    if cfg["ckpt"] or cfg.get("kv_from"):
        er = _find_export(cfg["ckpt"] or cfg["kv_from"], roots)
        export_kv = er.get("kv_scales") if er else None
        if export_kv is None:
            bad.append("no export record")
    if cfg.get("kv_from"):
        kw = eng.get("kv_written") or {}
        if not kw.get("written"):
            bad.append(f"kv_from {cfg['kv_from']}: no scales written ({kw})")
    for tag in ("audit_built", "audit_after_decode"):
        if tag not in eng:
            bad.append(f"missing {tag}")
            continue
        bad += check_audit(eng[tag], cfg, export_kv, tag)
    if "audit_built" in eng and "audit_after_decode" in eng and \
            scales_of(eng["audit_built"]) != scales_of(eng["audit_after_decode"]):
        bad.append("scales changed during the decode")
    if cfg["expect"].get("verl_nvfp4_patch") is False and not (eng.get("verl_nvfp4_patch") or {}).get("installed_stub"):
        bad.append("verl's NVFP4 patch was not replaced by the recorder")
    if cfg["expect"].get("fa3_kvskip_patch") and not (eng.get("fa3_kvskip_patch") or {}).get("installed"):
        bad.append("the FA3 kv-skip patch was expected but not installed")
    if not (eng.get("short_generation") or {}).get("token_ids"):
        bad.append("short generation empty")
    return bad, dec


def read(d, ckpt_root=None, base=None, anchors=None):
    """d: the run's out dir; base: other runs' out dirs whose outputs fill in the configurations d does not have (the
    NVFP4 job reads the formats job's floor and unit cache); anchors: {label: (lo, hi)} reference intervals of the floor
    and the unit-scale cache (mo_common.load_anchors), None or {} = the interval check is skipped."""
    import numpy as np
    q = _share_lib()
    ckpt_root = _as_list(ckpt_root) if ckpt_root else []
    ckpt_root = ckpt_root + [os.path.join(x, "ckpt") for x in [d] + _bases(base) if os.path.isdir(os.path.join(x, "ckpt"))]
    rows, valid, decs, not_run = [], {}, {}, []
    for label, cfg in C.FORMATS + C.NVFP4_EMU + C.CAL1 + C.SMOOTH_EMU:
        dl = label_dir(d, base, label)
        if dl is None and (label in C.NVFP4_EMU_LABELS or label in C.CAL1_LABELS or label in C.SMOOTH_EMU_LABELS):
            not_run.append(label)
            valid[label] = False
            continue
        dl = dl or d
        bad, dec = validity(dl, label, cfg, ckpt_root)
        row = dict(label=label, kind=cfg["kind"], ckpt=cfg["ckpt"], args=cfg["args"], env=cfg["env"],
                   valid=not bad, problems=bad, dir=dl)
        if dec is not None:
            row.update(mismatch=dec["kl_samp_train"], ci95=dec["kl_samp_train_ci95"], k3=dec.get("kl_k3"),
                       bin_0_1k=(dec.get("bins", {}).get("0-1024") or {}).get("kl_samp_train"),
                       mean_len=dec.get("mean_len"), frac_truncated=dec.get("frac_truncated"),
                       n_empty=dec.get("n_empty"))
        ep = os.path.join(dl, f"engine_{label}.json")
        if os.path.exists(ep):
            eng = _rd(ep)
            row["compilation"] = (eng.get("audit_built") or {}).get("compilation")
            a0 = {r["layer"]: r for r in eng.get("audit_built", {}).get("attn", [])}.get(0)
            if a0:
                row["layer0_scales"] = dict(k=a0.get("k"), v=a0.get("v"), q=a0.get("q"), prob=a0.get("prob"),
                                            kf=a0.get("kf"), vf=a0.get("vf"), qf=a0.get("qf"))
            row["short_generation"] = (eng.get("short_generation") or {}).get("text", "")[:200]
            if eng.get("short_generation_after_sync"):
                row["short_generation_after_sync"] = eng["short_generation_after_sync"].get("text", "")[:200]
        rows.append(row)
        valid[label] = not bad
        if not bad:
            decs[label] = q.load_dec(dl, label)
    ref = anchors or {}
    anchors = {}
    for lab in (C.FLOOR, C.UNIT):
        r = next(x for x in rows if x["label"] == lab)
        m = r.get("mismatch")
        ci = ref.get(lab)
        anchors[lab] = dict(value=m, reference_ci=list(ci) if ci else None,
                            ok=bool(valid[lab] and m is not None and (ci is None or ci[0] <= m <= ci[1])))
    out = dict(anchors=anchors, rows=rows, not_run=not_run)
    if not all(a["ok"] for a in anchors.values()):
        out.update(reading="INVALID", why="an anchor failed: the floor or the unit-scale cache is invalid or outside "
                                          "its reference interval (the kernel or the stack changed)")
        return out
    unit = next(x for x in rows if x["label"] == C.UNIT)["mismatch"]
    for r in rows:
        if r["valid"] and r.get("mismatch") is not None:
            r["ratio_to_unit_cache"] = r["mismatch"] / unit if unit else float("nan")
    shares = []
    for a, b in C.SHARE_PAIRS:
        rec = dict(A=a, B=b, F=C.FLOOR)
        if a in not_run or b in not_run:
            continue
        if not (valid[a] and valid[b]):
            rec.update(reading="not read (a configuration is invalid)")
        else:
            sub = {"F": decs[C.FLOOR], "A": decs[a], "B": decs[b]}
            bs, _ = q.paired_boot(sub)
            pt = {k: q.pooled(v) for k, v in sub.items()}
            rr = q.ratio_reading(pt, bs, "A", "B", "A", "F")
            rec.update(rr, reading=q.decide(rr))
        shares.append(rec)
    out.update(reading="READ", shares=shares, unit_cache_mismatch=unit, cal1=cal1_summary(rows, shares, not_run), sleepwake=sleepwake(next((x for x in [d] + _bases(base)
                                                if os.path.exists(os.path.join(x, "sleepwake.json"))), d)))
    del np
    return out


def cal1_summary(rows, shares, not_run):
    """The KV-cache calibration comparison (mo_common.CAL1_MAP): every configuration's mismatch and ratio to
    configuration 1 and the first-layer shares of configurations 2 and 5. None when no configuration of mo_common.CAL1
    was in the directories read."""
    if all(lab in not_run for lab in C.CAL1_LABELS):
        return None
    by = {r["label"]: r for r in rows}
    unit = (by.get(C.UNIT) or {}).get("mismatch")
    cfgs = []
    for num, lab in C.CAL1_MAP:
        r = by.get(lab)
        if r is None:
            cfgs.append(dict(cal=num, label=lab, status="not run"))
            continue
        m = r.get("mismatch") if r["valid"] else None
        cfgs.append(dict(cal=num, label=lab, status="read" if r["valid"] else "INVALID", mismatch=m, ci95=r.get("ci95"),
                         ratio_to_1=(m / unit) if (m is not None and unit) else None,
                         problems=None if r["valid"] else r["problems"][:3]))
    sh = {(x["A"], x["B"]): x for x in shares}
    return dict(configurations=cfgs,
                shares=[dict(A=a, B=b, **{k: v for k, v in (sh.get((a, b)) or {}).items() if k not in ("A", "B")})
                        for a, b in C.CAL1_SHARES])


def sleepwake(d):
    """Descriptive: do the calibrated K/V scale TENSORS survive verl's sleep(2) -> wake(weights) ->
    wake(kv_cache) order, and does a static activation scale survive the level-2 sleep?"""
    p = os.path.join(d, "sleepwake.json")
    if not os.path.exists(p):
        return None
    rec = _rd(p)
    kv = rec.get("export_kv_scales") or {}
    out = {}
    for tag in ("built", "after_wake_weights", "after_wake_kv_cache"):
        aud = rec.get(tag) or {}
        fp8 = [r for r in aud.get("attn", []) if kv_class(r) == "fp8"]
        eq = [r["layer"] for r in fp8 if str(r["layer"]) in kv and r.get("k") == [C.bf16(kv[str(r["layer"])]["k"])]
              and r.get("v") == [C.bf16(kv[str(r["layer"])]["v"])]]
        unit = [r["layer"] for r in fp8 if r.get("k") == [1.0] and r.get("v") == [1.0]]
        floats_eq = [r["layer"] for r in fp8 if str(r["layer"]) in kv and r.get("kf") == C.bf16(kv[str(r["layer"])]["k"])]
        q0 = next((r for r in aud.get("linear", []) if r["layer"] == 0 and r["name"].endswith("qkv_proj")), {})
        out[tag] = dict(n_fp8_layers=len(fp8), tensors_equal_export=len(eq), tensors_unit=len(unit),
                        floats_equal_export=len(floats_eq), layer0_qkv_input_scale=q0.get("input_scale"))
    return out


def report(out):
    lines = [f"MO formats job: {out['reading']}" + (f" -- {out.get('why')}" if out.get("why") else "")]
    no_ref = [lab for lab, a in out["anchors"].items() if a["reference_ci"] is None]
    if no_ref:
        lines.append(f"  anchor interval check skipped for {', '.join(no_ref)}: no reference interval (--anchors)")
    for lab, a in out["anchors"].items():
        where = f" in reference {a['reference_ci']}" if a["reference_ci"] is not None else ""
        lines.append(f"  anchor {lab}: {a['value']}{where}: {'ok' if a['ok'] else 'FAIL'}")
    lines.append(f"{'configuration':>22} {'mismatch':>9} {'95% CI':>20} {'/unit':>6} {'0-1k':>8} {'len':>6} {'trunc':>5}"
                 f"  layer-0 k/v/q   kind")
    for r in out["rows"]:
        m = r.get("mismatch")
        ci = r.get("ci95") or [float("nan")] * 2
        l0 = r.get("layer0_scales") or {}
        sc = "/".join(f"{(l0.get(x) or [float('nan')])[0]:.3g}" for x in ("k", "v", "q")) if l0 else "-"
        tail = "" if r["valid"] else "  INVALID: " + "; ".join(r["problems"][:3])
        lines.append(f"{r['label']:>22} {m if m is not None else float('nan'):9.5f} "
                     f"[{ci[0]:.4f}, {ci[1]:.4f}] {r.get('ratio_to_unit_cache', float('nan')):6.3f} "
                     f"{r.get('bin_0_1k') or float('nan'):8.4f} {r.get('mean_len') or 0:6.0f} "
                     f"{r.get('frac_truncated') or 0:5.2f}  {sc:>15}   {r['kind']}{tail}")
    for s in out.get("shares", []):
        if "share" in s:
            lines.append(f"  first-layer share ({s['A']} vs {s['B']}): {s['share']:.3f} "
                         f"[{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] -> {s['reading']}")
        else:
            lines.append(f"  first-layer share ({s['A']} vs {s['B']}): {s['reading']}")
    if out.get("not_run"):
        lines.append(f"  not in these runs: {', '.join(out['not_run'])}")
    c1 = out.get("cal1")
    if c1:
        lines.append("KV-cache calibration comparison:")
        for c in c1["configurations"]:
            if c.get("mismatch") is not None:
                lines.append(f"  configuration {c['cal']:>18} {c['label']:>16}: {c['mismatch']:.5f} "
                             f"[{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}]  x{c['ratio_to_1']:.3f} of configuration 1")
            else:
                lines.append(f"  configuration {c['cal']:>18} {c['label']:>16}: {c['status']} {c.get('problems') or ''}")
        for x in c1["shares"]:
            lines.append(f"  first-layer share {x['A']} vs {x['B']}: "
                         + (f"{x['share']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}] -> {x['reading']}"
                            if "share" in x else str(x.get("reading", "not read"))))
    for tag, v in (out.get("sleepwake") or {}).items():
        lines.append(f"  sleep/wake probe (mo_fp8_kv) {tag}: of {v['n_fp8_layers']} FP8 layers, k/v tensors = export "
                     f"{v['tensors_equal_export']}, = 1.0 {v['tensors_unit']}; k floats = export {v['floats_equal_export']}; "
                     f"layer-0 qkv input_scale {v['layer0_qkv_input_scale']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------- test
def _fake(root, means, mutate=None):
    import numpy as np
    d = os.path.join(root, f"D_{mutate or 'ok'}")
    os.makedirs(os.path.join(d, "ckpt"), exist_ok=True)
    rng = np.random.default_rng(1)
    keys = [f"{b}:{i}" for b in ("AIME24", "MATH-500") for i in range(8)]
    het = rng.normal(0, 0.3, len(keys))
    kv = {str(i): dict(k=0.05 + 0.01 * i, v=0.02) for i in range(C.N_LAYERS)}
    kv_fl = {str(i): v for i, v in kv.items() if i not in ("0", "27")}
    for name in C.EXPORTS:
        os.makedirs(os.path.join(d, "ckpt", name), exist_ok=True)
        json.dump(dict(kv_scales=kv_fl if name == "fp8_kv_fl" else kv),
                  open(os.path.join(d, "ckpt", name, "export_record.json"), "w"))
    static = os.path.join(d, "static.json")
    json.dump(dict(kv={k: dict(k=C.bf16(v["k"]), v=C.bf16(v["v"])) for k, v in kv.items()}), open(static, "w"))
    for name in C.LLMC_EXPORTS:
        os.makedirs(os.path.join(d, "ckpt", name), exist_ok=True)
        per = name == "llmc_kv_head"
        json.dump(dict(kv_scales={k: dict(k=[C.bf16(v["k"] * (1 + 0.1 * h)) for h in range(4)] if per else [C.bf16(v["k"])],
                                          v=[C.bf16(v["v"])] * (4 if per else 1)) for k, v in kv.items()}),
                  open(os.path.join(d, "ckpt", name, "export_record.json"), "w"))
    llmc_kv = {n: _rd(os.path.join(d, "ckpt", n, "export_record.json"))["kv_scales"] for n in C.LLMC_EXPORTS}
    for label, cfg in C.FORMATS + (C.NVFP4_EMU + C.SMOOTH_EMU if mutate in ("emu", "emu_badop") else ()) + \
            (C.CAL1 if mutate in ("cal", "cal_perhead_lost") else ()):
        e = cfg["expect"]
        lq, lp, lens, groups = [], [], [], []
        for gi, g in enumerate(keys):
            for _ in range(2):
                n = int(rng.integers(200, 600))
                dd = np.full(n, means[label] * (1 + 0.1 * het[gi]))
                p_ = rng.normal(-1.0, 0.3, n)
                lq.append((p_ + dd).astype(np.float32))
                lp.append(p_.astype(np.float32))
                lens.append(n)
                groups.append(g)
        np.savez(os.path.join(d, f"dec_{label}.npz"), lq=np.concatenate(lq), lp=np.concatenate(lp),
                 lens=np.array(lens), groups=np.array(groups), finish=np.array(["stop"] * len(lens)))
        cat = np.concatenate([a - b for a, b in zip(lq, lp)]).astype(np.float64)
        json.dump(dict(kl_samp_train=float(cat.mean()), kl_samp_train_ci95=[float(cat.mean()) * 0.9,
                                                                             float(cat.mean()) * 1.1],
                       kl_k3=0.0, n_samples=32, n_empty=0, mean_len=float(np.mean(lens)), frac_truncated=0.0,
                       bins={"0-1024": dict(kl_samp_train=float(cat.mean()))}),
                  open(os.path.join(d, f"dec_{label}.json"), "w"))
        open(os.path.join(d, f"rc_{label}.txt"), "w").write("0\n")
        bf = set(e.get("kv_bf16_layers", ()))
        attn = []
        for i in range(C.N_LAYERS):
            fp8 = e["kv"] != "auto" and i not in bf
            if not fp8:
                k = v = [1.0]
            elif e["kv"] == "unit":
                k = v = [1.0]
            elif cfg["ckpt"] in C.LLMC_EXPORTS:
                k, v = llmc_kv[cfg["ckpt"]][str(i)]["k"], llmc_kv[cfg["ckpt"]][str(i)]["v"]
                if mutate == "cal_perhead_lost" and label == "cal_c3":
                    k, v = [max(k)], [max(v)]
            elif e["kv"] == "computed":
                k, v = [0.02 + 0.001 * i], [0.004]
            else:   # calibrated / static: the same bf16-served values
                k, v = [C.bf16(kv[str(i)]["k"])], [C.bf16(kv[str(i)]["v"])]
            if mutate == "unit_fallback" and label == "mo_fp8_kv" and fp8:
                k = v = [1.0]
            attn.append(dict(layer=i, kv_cache_dtype="fp8" if fp8 else "auto",
                             storage="torch.uint8" if fp8 else "torch.bfloat16", backend="FLASH_ATTN", k=k, v=v,
                             q=[1.0], prob=[1.0], kf=k[0], vf=v[0], qf=1.0))
        lin = []
        for i in range(C.N_LAYERS):
            for nm in ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"):
                meth = "UnquantizedLinearMethod" if i in set(e.get("linear_bf16_layers", ())) else e["linear"]
                if mutate == "wrong_backend" and label == "mo_nvfp4_kv_emu":
                    be = "NvFp4LinearBackend.MARLIN"
                else:
                    be = f"NvFp4LinearBackend.{e.get('nvfp4_backend', 'x').upper()}"
                lin.append(dict(layer=i, name=f"model.layers.{i}.{nm}", method=meth, backend=be,
                                weight_dtype="torch.float8_e4m3fn" if meth == "ModelOptFp8LinearMethod"
                                else "torch.bfloat16"))
        aud = dict(attn=attn, linear=lin, compilation=dict(mode="3", enforce_eager=False,
                                                           cudagraph_mode=e.get("cudagraph_mode", "FULL_AND_PIECEWISE")))
        eng = dict(audit_built=aud, audit_after_decode=aud, short_generation=dict(token_ids=[1, 2], text="ok"),
                   verl_nvfp4_patch=dict(installed_stub=True) if e.get("verl_nvfp4_patch") is False else "n/a",
                   fa3_kvskip_patch=dict(installed=True, fallback=0) if e.get("fa3_kvskip_patch") else None)
        if cfg.get("kv_from"):
            eng["kv_written"] = dict(written=[i for i in range(C.N_LAYERS) if i not in set(e.get("kv_bf16_layers", ()))])
        if label in C.NVFP4_EMU_LABELS + C.SMOOTH_EMU_LABELS:
            a_on = e.get("act_attach", "a=" in e["sampler_quant"])
            sp = dict(n_linear=4 * C.N_LAYERS if a_on else 0, lm_head=0, nvfp4_missing=[], nvfp4_unused=[],
                      static_kv=dict(ok_at_load=True) if e["kv"] == "static" else None)
            seen = {e["op_dtype"]: 7} if a_on else {}
            if mutate == "emu_badop" and label == "mo_nvfp4emu_kv":
                seen = {"torch.float32": 7}
            aud = dict(aud, sampler_quant_patched=sp, **{e.get("op_seen", "nvfp4_op_seen"): seen})
            eng.update(audit_built=aud, audit_after_decode=aud, env={"SPECOPD_SAMPLER_QUANT_STATIC": static},
                       quant_sync=dict(spec=e["sampler_quant"], n_tensors=e["n_weight_tensors"]),
                       quant_audit={k: 0.0 for k in ("layers.0.self_attn.qkv_proj", "layers.0.self_attn.o_proj",
                                                     "layers.0.mlp.gate_up_proj", "mlp.down_proj")})
        json.dump(eng, open(os.path.join(d, f"engine_{label}.json"), "w"))
    return d


def selftest():
    import tempfile
    root = tempfile.mkdtemp()
    means = dict(mo_z_none=0.001, mo_kv8_unit=0.5, mo_fp8_kv=0.05, mo_fp8_kv_fl=0.02,
                 mo_nvfp4_kv_emu=0.3, mo_nvfp4_kv_marlin=0.2, mo_fp8_kv_s0=0.01, mo_nvfp4_kv_emu_s0=0.28,
                 mo_kv8_unit_s0=0.004,
                 mo_nvfp4emu_kv=0.6, mo_nvfp4emu_kv_s0=0.004, mo_nvfp4emu_bf16kv=0.04, mo_nvfp4emu_w_kv=0.55,
                 mo_sqemu=0.03, mo_awqemu=0.05, mo_sqemu_kv=0.55, mo_sqemu_kv_s0=0.04, mo_awqemu_kv=0.56,
                 mo_awqemu_kv_s0=0.06,
                 cal_c2=0.5, cal_c3=0.2, cal_c2_s0=0.004, cal_c8=0.003, cal_c4=0.45, cal_c2_llmc=0.5)
    labels = set(C.formats_dict()) - set(C.GRID_LABELS) - set(C.BF16KV_LABELS)
    assert set(means) == labels, set(means) ^ labels
    ref = {C.FLOOR: (0.0005, 0.002), C.UNIT: (0.4, 0.6)}     # synthetic reference intervals around the means above
    d_ok = _fake(root, means)
    o = read(d_ok)
    assert o["reading"] == "READ", o
    assert all(r["valid"] for r in o["rows"]), [r["problems"] for r in o["rows"] if not r["valid"]]
    assert all(a["ok"] and a["reference_ci"] is None for a in o["anchors"].values()), o["anchors"]
    assert "anchor interval check skipped" in report(o)
    o = read(d_ok, anchors=ref)
    assert o["reading"] == "READ" and all(a["ok"] and a["reference_ci"] for a in o["anchors"].values()), o["anchors"]
    sh = {s["A"]: s for s in o["shares"]}
    assert sh["mo_kv8_unit"]["reading"] == "FIRST-LAYER DOMINATED", sh["mo_kv8_unit"]
    assert sh["mo_nvfp4_kv_emu"]["reading"] == "NOT FIRST-LAYER DOMINATED", sh["mo_nvfp4_kv_emu"]
    o = read(_fake(root, means, "unit_fallback"))
    r = next(x for x in o["rows"] if x["label"] == "mo_fp8_kv")
    assert not r["valid"] and any("fell back" in p for p in r["problems"]), r["problems"]
    o = read(_fake(root, means, "wrong_backend"))
    r = next(x for x in o["rows"] if x["label"] == "mo_nvfp4_kv_emu")
    assert not r["valid"] and any("NVFP4 backend" in p for p in r["problems"])
    # the NVFP4 job: its own dir holds only the NVFP4 configurations, the base run the rest
    import shutil
    base = _fake(root, means, "base")
    d2 = os.path.join(root, "D_nvfp4")
    os.makedirs(d2, exist_ok=True)
    for lab in C.NVFP4_LABELS:
        for f in (f"dec_{lab}.json", f"dec_{lab}.npz", f"engine_{lab}.json", f"rc_{lab}.txt"):
            shutil.move(os.path.join(base, f), os.path.join(d2, f))
    o = read(d2, base=base)
    assert o["reading"] == "READ" and all(r["valid"] for r in o["rows"]), [r["problems"] for r in o["rows"]]
    assert {r["label"]: r["dir"] for r in o["rows"]}["mo_nvfp4_kv_emu"] == d2
    eng = _rd(os.path.join(d2, "engine_mo_nvfp4_kv_emu.json"))
    eng["audit_built"]["compilation"]["cudagraph_mode"] = "FULL_AND_PIECEWISE"
    json.dump(eng, open(os.path.join(d2, "engine_mo_nvfp4_kv_emu.json"), "w"))
    r = next(x for x in read(d2, base=base)["rows"] if x["label"] == "mo_nvfp4_kv_emu")
    assert not r["valid"] and any("cudagraph_mode" in p for p in r["problems"])
    bad = dict(means, mo_kv8_unit=0.9)
    da = _fake(root, bad, "anchor")
    o = read(da, anchors=ref)
    assert o["reading"] == "INVALID" and not o["anchors"][C.UNIT]["ok"], o["reading"]
    assert read(da)["reading"] == "READ"      # without a reference interval only validity is checked
    os.remove(os.path.join(da, f"dec_{C.FLOOR}.json"))
    assert read(da)["reading"] == "INVALID"   # an invalid floor fails the anchors with or without a reference
    dp = _fake(root, means, "print")
    eng = _rd(os.path.join(dp, "engine_mo_fp8_kv.json"))["audit_built"]
    reset = json.loads(json.dumps(eng))
    for r in reset["attn"]:
        r["k"] = r["v"] = [1.0]
    json.dump(dict(export_kv_scales=_rd(os.path.join(dp, "ckpt", "fp8_kv", "export_record.json"))["kv_scales"],
                   built=eng, after_wake_weights=eng, after_wake_kv_cache=reset), open(os.path.join(dp, "sleepwake.json"), "w"))
    o = read(dp)
    assert o["sleepwake"]["built"]["tensors_equal_export"] == 28 and o["sleepwake"]["after_wake_kv_cache"]["tensors_unit"] == 28
    assert o["sleepwake"]["after_wake_kv_cache"]["floats_equal_export"] == 28   # vLLM leaves the float mirrors
    assert o["not_run"] == list(C.NVFP4_EMU_LABELS + C.CAL1_LABELS + C.SMOOTH_EMU_LABELS) and len(o["rows"]) == len(C.FORMATS)
    # the emulated NVFP4 job: its own dir has only the emulated configurations, two base runs the rest
    base2 = _fake(root, means, "base2")
    d3 = os.path.join(root, "D_nvfp4_b")   # a fresh NVFP4-job dir (d2's emu audit was mutated above)
    os.makedirs(d3, exist_ok=True)
    for lab in C.NVFP4_LABELS:
        for f in (f"dec_{lab}.json", f"dec_{lab}.npz", f"engine_{lab}.json", f"rc_{lab}.txt"):
            shutil.move(os.path.join(base2, f), os.path.join(d3, f))
    de = _fake(root, means, "emu")
    for label, _ in C.FORMATS:
        for f in (f"dec_{label}.json", f"dec_{label}.npz", f"engine_{label}.json", f"rc_{label}.txt"):
            os.remove(os.path.join(de, f))
    o = read(de, base=f"{d3},{base2}")
    assert o["reading"] == "READ" and all(r["valid"] for r in o["rows"]), [(r["label"], r["problems"]) for r in o["rows"] if not r["valid"]]
    dirs = {r["label"]: r["dir"] for r in o["rows"]}
    assert dirs["mo_nvfp4emu_kv"] == de and dirs["mo_nvfp4_kv_emu"] == d3 and dirs["mo_z_none"] == base2
    sh = {s["A"]: s for s in o["shares"]}
    assert sh["mo_nvfp4emu_kv"]["reading"] == "FIRST-LAYER DOMINATED", sh["mo_nvfp4emu_kv"]
    assert sh["mo_sqemu_kv"]["reading"] == sh["mo_awqemu_kv"]["reading"] == "FIRST-LAYER DOMINATED", sh["mo_sqemu_kv"]
    # the calibration comparison: its own dir holds only the CAL1 configurations (+ the llm-compressor exports), the
    # formats job the rest
    dc = _fake(root, means, "cal")
    for label, _ in C.FORMATS:
        for f in (f"dec_{label}.json", f"dec_{label}.npz", f"engine_{label}.json", f"rc_{label}.txt"):
            os.remove(os.path.join(dc, f))
    o = read(dc, os.path.join(base2, "ckpt"), base=base2)
    bad = [(r["label"], r["problems"]) for r in o["rows"] if not r["valid"] and r["label"] not in C.NVFP4_LABELS]
    assert o["reading"] == "READ" and not bad, bad
    c1 = o["cal1"]
    st = {c["cal"]: c["status"] for c in c1["configurations"]}
    assert st["2"] == st["3"] == st["4"] == st["8"] == st["5"] == "read", st
    assert next(x for x in c1["shares"] if x["A"] == "cal_c2")["reading"] == "FIRST-LAYER DOMINATED"
    dl = _fake(root, means, "cal_perhead_lost")
    r = next(x for x in read(dl)["rows"] if x["label"] == "cal_c3")
    assert not r["valid"] and any("not loaded per head" in p for p in r["problems"]), r["problems"]
    assert next(c for c in read(dl)["cal1"]["configurations"] if c["cal"] == "3")["status"] == "INVALID"
    db = _fake(root, means, "emu_badop")
    o = read(db, base=base2)
    r = next(x for x in o["rows"] if x["label"] == "mo_nvfp4emu_kv")
    assert not r["valid"] and any("ran on" in p for p in r["problems"]), r["problems"]
    print(report(o))
    print("[mo_formats_read] SELFTEST PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--base-dir", default="", help="other jobs' out dirs (comma-separated, first wins) filling in the "
                    "configurations --out-dir lacks")
    ap.add_argument("--ckpt-root", default="")
    ap.add_argument("--anchors", default="", help="JSON {label: [lo, hi]}: reference 95%% intervals of mo_z_none and "
                    "mo_kv8_unit measured with the same harness; absent: the anchor interval check is skipped")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    out = read(a.out_dir, a.ckpt_root or None, a.base_dir or None, C.load_anchors(a.anchors))
    if a.json:
        C.write_json(a.json, out)
    print(report(out))
    return 0 if out["reading"] == "READ" else 6


if __name__ == "__main__":
    sys.exit(main())
