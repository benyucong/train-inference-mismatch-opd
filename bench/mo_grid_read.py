"""Reader of the weight / activation grid job (mo_common.GRID): is each configuration what it claims, and what did it
measure? (stdlib only)

A configuration is VALID only if its engine audit (engine_<label>.json, written by mo_measure.py q) shows:
  - the cache: kv 'auto' -> every attention layer bf16; 'calibrated' -> every layer FP8 except kv_bf16_layers (bf16),
    and every FP8 layer's k / v scale = the kv_from export's served scale (kv_written lists exactly the FP8 layers);
  - every decoder linear layer has the expected quant method, and for W4A8 the compressed-tensors scheme
    CompressedTensorsW4A8Fp8; W8A16 serves bf16 weights with no quant config;
  - the engine loaded the expected checkpoint (model_path), and both audits (built, after decoding) agree on the cache;
  - load check: each grid format's bf16-cache engine starts its greedy short generation with the floor's first
    LOAD_PREFIX tokens (a quantized vLLM engine does not fail on weights the checkpoint never supplied).
The controls (mo_common.GRID_CONTROLS, measured again in the grid job) must be VALID and, with --anchors (a JSON file
{label: [lo, hi]}: reference 95% intervals measured with the same harness), read inside their reference interval; a
control without one is checked for validity only, and the skipped interval check is printed. Prints one line per label
and writes <D>/grid_reading.json. Exit 0 iff every configuration and control is VALID.
Usage: python3 bench/mo_grid_read.py <D> [<ckpt_root> [<labels> [<controls>]]] [--anchors A.json]
       python3 bench/mo_grid_read.py --selftest   (a synthetic directory)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mo_common as C  # noqa: E402


def check_engine(eng, cfg, ckpt_path):
    """Problems (list of str) with one configuration's engine record; empty = valid."""
    bad = []
    e = cfg["expect"]
    for which in ("audit_built", "audit_after_decode"):
        ab = eng.get(which)
        if not ab:
            bad.append(f"{which} missing")
            continue
        attn = {a["layer"]: a for a in ab["attn"]}
        if sorted(attn) != list(range(C.N_LAYERS)):
            bad.append(f"{which}: attention layers {sorted(attn)}")
            continue
        skip = set(e.get("kv_bf16_layers", ()))
        for i, a in attn.items():
            fp8 = str(a["kv_cache_dtype"]).startswith("fp8")
            want_fp8 = e["kv"] == "calibrated" and i not in skip
            if fp8 != want_fp8:
                bad.append(f"{which}: layer {i} cache {a['kv_cache_dtype']} / {a['storage']} (want "
                           f"{'fp8' if want_fp8 else 'bf16'})")
            if not fp8 and "bfloat16" not in str(a["storage"]):
                bad.append(f"{which}: layer {i} bf16 cache stored as {a['storage']}")
        if e["kv"] == "calibrated":
            served = {int(k): v for k, v in (eng.get("kv_from_served") or {}).items()}
            for i, a in attn.items():
                if i in skip:
                    continue
                if not a["k"] or not a["v"] or a["k"][0] != served[i][0] or a["v"][0] != served[i][1] \
                        or a["kf"] != served[i][0] or a["vf"] != served[i][1]:
                    bad.append(f"{which}: layer {i} k/v scale {a['k']}/{a['v']} != served {served[i]}")
        lin = [x for x in ab["linear"] if x["layer"] >= 0]
        if len(lin) != 4 * C.N_LAYERS:
            bad.append(f"{which}: {len(lin)} decoder linear layers (want {4 * C.N_LAYERS})")
        for x in lin:
            if x["method"] != e["linear"]:
                bad.append(f"{which}: {x['name']} method {x['method']}")
            if e.get("scheme") and x.get("scheme") != e["scheme"]:
                bad.append(f"{which}: {x['name']} scheme {x.get('scheme')}")
            if e["linear"] == "UnquantizedLinearMethod" and x.get("weight_dtype") != "torch.bfloat16":
                bad.append(f"{which}: {x['name']} weight {x.get('weight_dtype')}")
        if ckpt_path is not None and os.path.realpath(ab["model_path"]) != os.path.realpath(ckpt_path):
            bad.append(f"{which}: model_path {ab['model_path']} != {ckpt_path}")
        if e["linear"] == "UnquantizedLinearMethod" and ab["quant_config"] != "NoneType":
            bad.append(f"{which}: quant_config {ab['quant_config']}")
    if cfg.get("vllm_w4a8_view_fix"):
        fx = eng.get("w4a8_view_fix")
        if not isinstance(fx, dict) or fx.get("calls") != 4 * C.N_LAYERS:
            bad.append(f"w4a8 view fix: {fx} (want {4 * C.N_LAYERS} calls)")
    if e["kv"] == "calibrated":
        kw = eng.get("kv_written") or {}
        want = sorted(set(range(C.N_LAYERS)) - set(e.get("kv_bf16_layers", ())))
        if kw.get("written") != want or kw.get("unused") != sorted(e.get("kv_bf16_layers", ())):
            bad.append(f"kv_written {kw} (want written {want})")
    return bad[:12]


LOAD_PREFIX = 6   # greedy tokens a bf16-cache grid engine must share with the floor (catches weights never loaded)


def read(d, ckpt_root, labels, controls, anchors=None):
    """controls: labels of mo_common.GRID_CONTROLS; anchors: {label: (lo, hi)} reference intervals of the controls
    (mo_common.load_anchors), a control without one is checked for validity only."""
    fd = C.formats_dict()
    anchors = anchors or {}
    out, ok = {}, True
    floor_ids = None
    fp = os.path.join(d, f"engine_{C.FLOOR}.json")
    if os.path.exists(fp):
        floor_ids = ((json.load(open(fp)).get("short_generation") or {}).get("token_ids") or [])[:LOAD_PREFIX]
    no_ref = [c for c in controls if c not in anchors]
    if no_ref:
        print(f"[grid] control interval check skipped for {', '.join(no_ref)}: no reference interval (--anchors)")
    for lab in list(labels) + list(controls):
        cfg = fd[lab]
        ep, dp = os.path.join(d, f"engine_{lab}.json"), os.path.join(d, f"dec_{lab}.json")
        if not (os.path.exists(ep) and os.path.exists(dp)):
            out[lab] = dict(valid=False, problems=["engine or dec file missing"])
            ok = False
            print(f"[grid] {lab}: INVALID (engine or dec file missing)")
            continue
        eng, dec = json.load(open(ep)), json.load(open(dp))
        ckpt = os.path.join(ckpt_root, cfg["ckpt"]) if cfg["ckpt"] else C.STUDENT
        bad = check_engine(eng, cfg, ckpt)
        if lab in C.GRID_LABELS and cfg["expect"]["kv"] == "auto":
            ids = ((eng.get("short_generation") or {}).get("token_ids") or [])[:LOAD_PREFIX]
            if not floor_ids or ids != floor_ids:
                bad.append(f"load check: first {LOAD_PREFIX} greedy tokens {ids} != the floor's {floor_ids}")
        val, ci = dec["kl_samp_train"], dec["kl_samp_train_ci95"]
        ref = anchors.get(lab) if lab in controls else None
        if ref is not None and not (ref[0] <= val <= ref[1]):
            bad.append(f"control {val:.5f} outside the reference interval {list(ref)}")
        out[lab] = dict(valid=not bad, problems=bad, kl_samp_train=val, ci95=ci, n_tok=dec["n_tok"],
                        kind=cfg["kind"], control_reference_ci=list(ref) if ref is not None else None,
                        short_generation=(eng.get("short_generation") or {}).get("text", "")[:200])
        ok &= not bad
        print(f"[grid] {lab}: {'VALID' if not bad else 'INVALID'}  {val:.5f} [{ci[0]:.5f}, {ci[1]:.5f}]  "
              f"n_tok={dec['n_tok']}" + (f"  (reference {list(ref)})" if ref is not None else "")
              + ("" if not bad else f"  {bad}"))
    return ok, out


# --------------------------------------------------------------------------------------------------------------- test
def _fake(root, means, mutate=None):
    """A synthetic grid directory: every GRID configuration and control recorded as mo_measure.py q records it."""
    d = os.path.join(root, f"G_{mutate or 'ok'}")
    ckpt_root = os.path.join(d, "ckpt")
    os.makedirs(ckpt_root, exist_ok=True)
    fd = C.formats_dict()
    served = {str(i): [C.bf16(0.05 + 0.01 * i), C.bf16(0.02)] for i in range(C.N_LAYERS)}
    names = ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")
    for lab in C.GRID_LABELS + C.GRID_CONTROLS:
        cfg = fd[lab]
        e = cfg["expect"]
        skip = set(e.get("kv_bf16_layers", ()))
        attn = []
        for i in range(C.N_LAYERS):
            fp8 = e["kv"] == "calibrated" and i not in skip
            k, v = served[str(i)] if fp8 else (1.0, 1.0)
            if mutate == "kv_scale" and lab == "cal_c2" and i == 0:
                k = 1.0
            attn.append(dict(layer=i, kv_cache_dtype="fp8" if fp8 else "auto",
                             storage="torch.uint8" if fp8 else "torch.bfloat16", k=[k], v=[v], kf=k, vf=v))
        unq = e["linear"] == "UnquantizedLinearMethod"
        lin = [dict(layer=i, name=f"model.layers.{i}.{nm}", method=e["linear"], scheme=e.get("scheme"),
                    weight_dtype="torch.bfloat16" if unq else "torch.int32") for i in range(C.N_LAYERS) for nm in names]
        lin.append(dict(layer=-1, name="lm_head", method="UnquantizedLinearMethod", weight_dtype="torch.bfloat16"))
        aud = dict(attn=attn, linear=lin, model_path=os.path.join(ckpt_root, cfg["ckpt"]) if cfg["ckpt"] else C.STUDENT,
                   quant_config="NoneType" if unq else "CompressedTensorsConfig")
        ids = list(range(11, 18))
        if mutate == "unloaded" and lab == "mo_w8a16_bf16kv":
            ids[2] = 99
        eng = dict(audit_built=aud, audit_after_decode=aud, short_generation=dict(token_ids=ids, text="ok"))
        if e["kv"] == "calibrated":
            eng.update(kv_from_served=served, kv_written=dict(written=sorted(set(range(C.N_LAYERS)) - skip),
                                                              unused=sorted(skip)))
        if cfg.get("vllm_w4a8_view_fix"):
            eng["w4a8_view_fix"] = dict(calls=4 * C.N_LAYERS - (1 if mutate == "view_fix" else 0))
        with open(os.path.join(d, f"engine_{lab}.json"), "w") as f:
            json.dump(eng, f)
        m = means[lab]
        with open(os.path.join(d, f"dec_{lab}.json"), "w") as f:
            json.dump(dict(kl_samp_train=m, kl_samp_train_ci95=[0.9 * m, 1.1 * m], n_tok=1000), f)
    return d, ckpt_root


def selftest():
    import tempfile
    root = tempfile.mkdtemp()
    means = dict({lab: 0.1 for lab in C.GRID_LABELS}, mo_z_none=0.001, cal_c2=0.5)
    ap = os.path.join(root, "anchors.json")
    with open(ap, "w") as f:
        json.dump({"mo_z_none": [0.0005, 0.002], "cal_c2": [0.4, 0.6]}, f)     # synthetic reference intervals
    ref = C.load_anchors(ap)
    d, ck = _fake(root, means)
    ok, out = read(d, ck, C.GRID_LABELS, C.GRID_CONTROLS, ref)
    assert ok and all(r["valid"] for r in out.values()) and out["cal_c2"]["control_reference_ci"] == [0.4, 0.6], out
    ok, out = read(d, ck, C.GRID_LABELS, C.GRID_CONTROLS)                    # no reference: validity only
    assert ok and out["cal_c2"]["control_reference_ci"] is None, out
    ok, out = read(d, ck, C.GRID_LABELS, C.GRID_CONTROLS, dict(ref, cal_c2=(0.6, 0.7)))
    assert not ok and any("outside the reference interval" in p for p in out["cal_c2"]["problems"]), out["cal_c2"]
    for mut, lab, what in (("kv_scale", "cal_c2", "k/v scale"), ("unloaded", "mo_w8a16_bf16kv", "load check"),
                           ("view_fix", "mo_w4a8_kv", "w4a8 view fix")):
        d2, ck2 = _fake(root, means, mut)
        ok, out = read(d2, ck2, C.GRID_LABELS, C.GRID_CONTROLS, ref)
        assert not ok and any(what in p for p in out[lab]["problems"]), (mut, out[lab])
    os.remove(os.path.join(d, f"dec_{C.GRID_LABELS[0]}.json"))
    ok, out = read(d, ck, C.GRID_LABELS, C.GRID_CONTROLS, ref)
    assert not ok and out[C.GRID_LABELS[0]]["problems"] == ["engine or dec file missing"], out
    print("[mo_grid_read] SELFTEST PASS")
    return 0


def main(argv):
    if argv[:1] == ["--selftest"]:
        return selftest()
    anchors = ""
    if "--anchors" in argv:
        i = argv.index("--anchors")
        anchors, argv = argv[i + 1], argv[:i] + argv[i + 2:]
    d = argv[0]
    ckpt_root = argv[1] if len(argv) > 1 else os.path.join(d, "ckpt")
    labels = argv[2].split(",") if len(argv) > 2 else C.GRID_LABELS   # a comma-separated subset of mo_common.GRID_LABELS
    ctl = [c for c in C.GRID_CONTROLS if len(argv) <= 3 or c in argv[3].split(",")]
    assert all(lab in C.GRID_LABELS for lab in labels)
    ok, out = read(d, ckpt_root, labels, ctl, C.load_anchors(anchors))
    C.write_json(os.path.join(d, "grid_reading.json"), dict(all_valid=ok, configurations=out))
    print("[grid] ALL VALID" if ok else "[grid] NOT ALL VALID")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
