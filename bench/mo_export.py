#!/usr/bin/env python3
"""Model Optimizer post-training quantization of our student and its export in the format vLLM loads (venv-modelopt).

One export per process: a built-in Model Optimizer PTQ recipe (modelopt.recipe.load_recipe, unchanged), optionally with
every quantizer of some decoder layers disabled (mo_common.EXPORTS), calibrated with max calibration on the
calibration set (mo_calset.py: 512 prompt + response sequences, each one full sequence, batch 1, no KV cache), then
modelopt.torch.export.export_hf_checkpoint. The student's tokenizer files are copied next to the weights, so vLLM can
load the export directory as a model.

    $R/venv-modelopt/bin/python bench/mo_export.py export --name fp8_kv --calset-dir C --out-dir D/ckpt/fp8_kv
    $R/venv-modelopt/bin/python bench/mo_export.py tiny   --name fp8_kv --out-dir /tmp/x   # CPU: random 4-layer Qwen2
    $R/venv-modelopt/bin/python bench/mo_export.py amax   --name nvfp4_kv --calset-dir C --export-dir E \
        --static-out D/nvfp4_static.json --capture-out D/nvfp4_capture.pt                 # GPU: the emulated sampler's inputs
    python3 bench/mo_export.py selftest                                                     # stdlib only

smooth (the emulated INT8 SmoothQuant / INT4 AWQ samplers): Model Optimizer's preset
(mtq.INT8_SMOOTHQUANT_CFG / mtq.INT4_AWQ_CFG, unchanged) calibrated on the same set with the same loop, no export; the
float32 weight factor each module's smoothing applied (model_calib._apply_weight_pre_quant_scale's argument, recorded
by wrapping that function) and the bf16 activation factor (input_quantizer.pre_quant_scale) per decoder linear layer
-> a safetensors file, plus the static file (grid int8sq | int4awq; SmoothQuant's per-tensor activation amax) and the
capture for bench/mo_smooth_check.py (as amax's, with the smoothed weight and Model Optimizer's own quantizers).

    $R/venv-modelopt/bin/python bench/mo_export.py smooth --recipe int8_sq --calset-dir C --static-out D/sq_static.json \
        --tensors-out D/sq_static.safetensors --capture-out D/sq_capture.pt

amax (the emulated NVFP4 sampler): the SAME calibration (same model load, calibration set, loop)
WITHOUT an export; writes the static file of specopd_quant's w=nvfp4 grid (format specopd_quant_static/v1: every decoder
linear layer's calibrated activation amax, the KV scales of the formats job's export E as vLLM serves them, provenance and
a cross-check of this calibration against E: input_scale / weight_scale_2 recomputed the export's way, K/V maxima), and
the capture for bench/mo_nvfp4_check.py: for decoder layers 0, 13, 27 and each of their seven linear layers, the input X
the calibrated model gives it on one calibration sequence (bf16, as Model Optimizer's forward feeds its quantizer), and
Model Optimizer's own fake quantization of X (bf16 and float32), of W (first 256 rows + sha256 of the whole tensor), of
W's first rows in float32, of a perturbed W' with its re-calibrated amax, and the module's output.

Writes <out>/export_record.json: recipe and its resolved quant_cfg, versions, calibration-set hash and token count, the
exported quantization config, per decoder layer the exported k_scale / v_scale and every linear layer's input_scale /
weight_scale(_2) summary (the step-0 static scales a training sampler keeps), the observed K/V maxima, md5 per file.
"""
import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mo_common as C  # noqa: E402

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json")


def log(*a):
    print(f"[mo_export {time.strftime('%H:%M:%S')}]", *a, flush=True)


def disable_entries(layers):
    """Quantizer-config entries that switch off every quantizer of these decoder layers (appended LAST, so they win)."""
    return [{"quantizer_name": f"*layers.{i}.*", "enable": False} for i in layers]


def quant_cfg(name):
    from modelopt.recipe import load_recipe
    ex = C.EXPORTS[name]
    rec = load_recipe(ex["recipe"])
    cfg = rec.quantize.model_dump(exclude_unset=True)
    cfg = dict(cfg, quant_cfg=list(cfg["quant_cfg"]) + disable_entries(ex["disable_layers"]))
    return cfg


def observed_maxima(model):
    """Per decoder layer: the calibrated K / V maxima Model Optimizer observed (post-RoPE keys, values)."""
    out = {}
    for name, m in model.named_modules():
        if hasattr(m, "k_bmm_quantizer") and ".layers." in name:
            li = int(name.split(".layers.")[1].split(".")[0])
            rec = {}
            for q in ("k_bmm_quantizer", "v_bmm_quantizer"):
                tq = getattr(m, q)
                amax = getattr(tq, "amax", None)
                rec[q] = dict(enabled=bool(tq.is_enabled),
                              amax=None if amax is None else [float(x) for x in amax.float().reshape(-1).tolist()],
                              amax_dtype=None if amax is None else str(amax.dtype))
            out[str(li)] = rec
    return out


def read_export(out_dir):
    """Every scale tensor of the export, by name (float lists), plus the exported configs."""
    from safetensors import safe_open
    scales, n_tensors, dtypes = {}, 0, {}
    for f in sorted(os.listdir(out_dir)):
        if not f.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(out_dir, f), "pt") as fh:
            for k in fh.keys():
                n_tensors += 1
                t = fh.get_tensor(k)
                dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + 1
                if k.endswith(("_scale", "_scale_2")) and t.numel() <= 64:
                    scales[k] = [float(x) for x in t.float().reshape(-1).tolist()]
    cfg = json.load(open(os.path.join(out_dir, "config.json")))
    hq = os.path.join(out_dir, "hf_quant_config.json")
    return dict(scales=scales, n_tensors=n_tensors, tensor_dtypes=dtypes,
                quantization_config=cfg.get("quantization_config"),
                hf_quant_config=json.load(open(hq)) if os.path.exists(hq) else None)


def kv_table(scales):
    """{layer: {k: x, v: y}} from the exported names (model.layers.<i>.self_attn.{k,v}_proj.{k,v}_scale)."""
    out = {}
    for k, v in scales.items():
        for part in ("k", "v"):
            suf = f".self_attn.{part}_proj.{part}_scale"
            if k.endswith(suf) and ".layers." in k:
                li = int(k.split(".layers.")[1].split(".")[0])
                out.setdefault(str(li), {})[part] = v[0]
    return out


def calibrate(model, name, seqs, device):
    """Model Optimizer's max calibration of the recipe on the calibration set: each sequence once, batch 1, no KV cache.
    Returns (resolved quant cfg, seconds)."""
    import torch

    import modelopt.torch.quantization as mtq
    cfg = quant_cfg(name)

    def loop(m):
        with torch.no_grad():
            for k, s in enumerate(seqs):
                ids = torch.tensor([s], dtype=torch.long, device=device)
                m(input_ids=ids, use_cache=False, logits_to_keep=1)
                if k % 64 == 0:
                    log(f"  calibration {k}/{len(seqs)}")

    t = time.time()
    mtq.quantize(model, cfg, forward_loop=loop)
    return cfg, time.time() - t


def run(model, tok_src, name, seqs, out_dir, device):
    import modelopt
    import torch
    import transformers

    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise SystemExit(f"REFUSE: {out_dir} exists and is not empty")
    n_tok = sum(len(s) for s in seqs)
    cfg, t_cal = calibrate(model, name, seqs, device)
    try:
        mtq.print_quant_summary(model)
    except Exception as e:  # noqa: BLE001
        log(f"print_quant_summary unavailable: {e!r}")
    maxima = observed_maxima(model)
    t = time.time()
    os.makedirs(out_dir, exist_ok=True)
    export_hf_checkpoint(model, export_dir=out_dir)
    t_exp = time.time() - t
    if tok_src:
        for f in TOKENIZER_FILES:
            src = os.path.join(tok_src, f)
            if os.path.exists(src) and not os.path.exists(os.path.join(out_dir, f)):
                shutil.copy2(src, out_dir)
    ex = read_export(out_dir)
    rec = dict(name=name, recipe=C.EXPORTS[name]["recipe"], disable_layers=list(C.EXPORTS[name]["disable_layers"]),
               quant_cfg=cfg, versions=dict(modelopt=modelopt.__version__, torch=torch.__version__,
                                            transformers=transformers.__version__),
               n_calibration_sequences=len(seqs), n_calibration_tokens=n_tok, seconds_calibrate=round(t_cal, 1),
               seconds_export=round(t_exp, 1), observed_maxima=maxima, kv_scales=kv_table(ex["scales"]), **ex,
               files={f: C.md5_file(os.path.join(out_dir, f)) for f in sorted(os.listdir(out_dir))
                      if os.path.isfile(os.path.join(out_dir, f))})
    C.write_json(os.path.join(out_dir, "export_record.json"), rec)
    kv0 = rec["kv_scales"].get("0", {})
    log(f"{name}: exported {ex['n_tensors']} tensors {ex['tensor_dtypes']}; quant_algo "
        f"{(ex['quantization_config'] or {}).get('quant_algo')}; layer-0 k/v scale {kv0}; "
        f"calibrate {t_cal:.0f}s on {n_tok} tokens, export {t_exp:.0f}s -> {out_dir}")
    return rec


def cmd_export(a):
    import torch
    from transformers import AutoModelForCausalLM
    import mo_calset
    rows, crec = mo_calset.load_calset(a.calset_dir)
    seqs = [r["prompt_ids"] + r["response_ids"] for r in rows]
    if a.max_seqs:
        seqs = seqs[: a.max_seqs]
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(C.STUDENT, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map="cuda").eval()
    rec = run(model, C.STUDENT, a.name, seqs, a.out_dir, "cuda")
    rec["calset_sha256"] = crec["calset_sha256"]
    C.write_json(os.path.join(a.out_dir, "export_record.json"), rec)
    return 0


CAPTURE_LAYERS = (0, 13, 27)
CAPTURE_MODULES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
                   "mlp.up_proj", "mlp.down_proj")
CAPTURE_ROWS = 256
CAPTURE_TOKENS = 256
CAPTURE_SEED = 20261008
NVFP4_DIV = 6.0 * 448.0     # Model Optimizer: global scale = amax / (maxbound 6 * E4M3 max 448)


def quantizer_amax(model):
    """{HF linear module: {input|weight: {enabled, dtype, amax}}} of the decoder layers, as calibrated."""
    out = {}
    for name, m in model.named_modules():
        if ".layers." not in name or not (hasattr(m, "input_quantizer") and hasattr(m, "weight_quantizer")):
            continue
        rec = {}
        for tag in ("input", "weight"):
            tq = getattr(m, tag + "_quantizer")
            amax = getattr(tq, "amax", None)
            rec[tag] = dict(enabled=bool(tq.is_enabled), dtype=None if amax is None else str(amax.dtype),
                            amax=None if amax is None else [float(x) for x in amax.float().reshape(-1).tolist()])
        out[name] = rec
    return out


FUSED = (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), ("mlp.gate_proj", "mlp.up_proj"))


def export_cross_check(lin, maxima, erec, device):
    """This calibration against the formats job's export E (same recipe, same set): the export's input_scale /
    weight_scale_2 recomputed from this calibration's amax with the export's expression on the same device (a float32
    tensor divided by 6 * 448; a fused group's shards take the group maximum, as the export unifies them), and the K/V
    maxima. Equal everywhere = the calibration reproduced E's."""
    import torch

    def gs(a):
        return float(torch.tensor(a, dtype=torch.float32, device=device) / NVFP4_DIV)

    sc = erec.get("scales") or {}
    group = {}
    for n in lin:
        for g in FUSED:
            for part in g:
                if n.endswith("." + part):
                    group[n] = [n[: -len(part)] + q for q in g]
    res = dict(input=[0, 0], weight=[0, 0], kv=[0, 0], first_diffs=[])
    for n, r in sorted(lin.items()):
        names = group.get(n, [n])
        for tag, key in (("input", "input_scale"), ("weight", "weight_scale_2")):
            ex = sc.get(f"{n}.{key}")
            if not r[tag]["enabled"] or ex is None:
                continue
            want = max(lin[x][tag]["amax"][0] for x in names)
            ok = [gs(want)] == [float(v) for v in ex]
            res[tag][0] += int(ok)
            res[tag][1] += 1
            if not ok and len(res["first_diffs"]) < 6:
                res["first_diffs"].append(dict(module=n, what=key, ours=gs(want), export=ex))
    for li, rec in sorted((erec.get("observed_maxima") or {}).items()):
        ours = maxima.get(li)
        for q in ("k_bmm_quantizer", "v_bmm_quantizer"):
            ok = ours is not None and ours[q]["amax"] == rec[q]["amax"]
            res["kv"][0] += int(ok)
            res["kv"][1] += 1
    res["reproduced"] = all(v[0] == v[1] and v[1] > 0 for k, v in res.items() if k in ("input", "weight", "kv"))
    return res


def static_record(name, lin, erec, xcheck, crec, n_seqs, n_tok, t_cal, export_dir):
    """The specopd_quant static file (format specopd_quant_static/v1)."""
    import modelopt
    import torch
    import transformers
    act, wamax = {}, {}
    for n, r in sorted(lin.items()):
        for tag, dst in (("input", act), ("weight", wamax)):
            if r[tag]["enabled"]:
                a = r[tag]["amax"]
                if len(a) != 1:
                    raise SystemExit(f"{n}.{tag}_quantizer amax has {len(a)} elements (per-tensor expected)")
                dst[n] = a[0]
    kv_export = erec["kv_scales"]
    return dict(
        format="specopd_quant_static/v1", grid="nvfp4", recipe=C.EXPORTS[name]["recipe"], model=C.STUDENT,
        act_amax=act, weight_amax_step0=wamax,
        kv={li: dict(k=C.bf16(v["k"]), v=C.bf16(v["v"])) for li, v in sorted(kv_export.items(), key=lambda x: int(x[0]))},
        kv_export=kv_export,
        kv_rule="the formats job's NVFP4 export's k_scale / v_scale as vLLM serves a checkpoint's (rounded to bf16: "
                "vLLM's k / v scale parameters take the model dtype); the same values the formats job's NVFP4 "
                "configurations ran with",
        stale="fixed from step 0: activation amax (global scale of every linear input) and the KV scales; fresh at "
              "every sync: each weight's amax and all block scales",
        export_cross_check=xcheck,
        provenance=dict(calset_sha256=crec["calset_sha256"], n_calibration_sequences=n_seqs, n_calibration_tokens=n_tok,
                        seconds_calibrate=round(t_cal, 1), export_dir=export_dir,
                        export_record_md5=C.md5_file(os.path.join(export_dir, "export_record.json")),
                        versions=dict(modelopt=modelopt.__version__, torch=torch.__version__,
                                      transformers=transformers.__version__)))


def _triton_version():
    try:
        import triton
        return triton.__version__
    except Exception:  # noqa: BLE001
        return None


def capture(model, seq, out_path, device):
    """Inputs and Model Optimizer's own fake-quantized outputs for bench/mo_nvfp4_check.py (see the module docstring)."""
    import hashlib

    import torch

    def sha(t):
        return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

    targets = [f"model.layers.{i}.{m}" for i in CAPTURE_LAYERS for m in CAPTURE_MODULES]
    mods = dict((n, m) for n, m in model.named_modules() if n in set(targets))
    if sorted(mods) != sorted(targets):
        raise SystemExit(f"capture: modules {sorted(set(targets) - set(mods))[:3]} not found")
    got = {}
    hooks = [m.register_forward_pre_hook(lambda mod, args, n=n: got.setdefault(n, args[0].detach().clone()))
             for n, m in mods.items()]
    ids = torch.tensor([seq[:CAPTURE_TOKENS]], dtype=torch.long, device=device)
    with torch.no_grad():
        model(input_ids=ids, use_cache=False, logits_to_keep=1)
    for h in hooks:
        h.remove()
    g = torch.Generator(device=device).manual_seed(CAPTURE_SEED)
    out, R = {}, CAPTURE_ROWS
    with torch.no_grad():
        for n in targets:
            m, X = mods[n], got[n]
            iq, wq = m.input_quantizer, m.weight_quantizer
            W = m.weight.detach()
            w_mo = wq(W)
            rec = dict(x=X.cpu(), x_dtype=str(X.dtype), amax_in=iq.amax.detach().float().cpu(), x_mo=iq(X).cpu(),
                       x32_mo=iq(X.float()).cpu(), amax_w=wq.amax.detach().float().cpu(), w_sha256=sha(W),
                       w_rows=W[:R].cpu(), w_mo_rows=w_mo[:R].cpu(), w_mo_sha256=sha(w_mo),
                       w32_mo_rows=wq(W[:R].float()).cpu())
            noise = torch.randn(W.shape, generator=g, device=W.device, dtype=torch.float32)
            W2 = (W.float() + noise * float(W.float().std()) * 0.01).to(W.dtype)   # a simulated training update
            old = wq.amax.detach().clone()
            wq.amax = W2.abs().amax().to(old.dtype).reshape(old.shape)               # the weight quantizer re-calibrated
            w2_mo = wq(W2)
            rec.update(amax_w2=wq.amax.detach().float().cpu(), w2_rows=W2[:R].cpu(), w2_mo_rows=w2_mo[:R].cpu(),
                       w2_mo_sha256=sha(w2_mo), w2_sha256=sha(W2))
            wq.amax = old
            rec.update(y_mo=m(X).cpu(), bias=None if m.bias is None else m.bias.detach().cpu())
            out[n] = rec
            del w_mo, W2, w2_mo
    torch.save(dict(modules=out, meta=dict(layers=CAPTURE_LAYERS, rows=R, tokens=int(ids.shape[1]),
                                           seed=CAPTURE_SEED, student=C.STUDENT, torch=torch.__version__,
                                           triton=_triton_version(), gpu=torch.cuda.get_device_name())), out_path)
    log(f"capture: {len(out)} modules, {int(ids.shape[1])} tokens -> {out_path}")


def cmd_amax(a):
    import torch
    from transformers import AutoModelForCausalLM
    import mo_calset
    if not str(C.EXPORTS[a.name]["recipe"]).endswith("nvfp4_default-kv_fp8"):
        raise SystemExit(f"amax: {a.name} is not the NVFP4 recipe")
    for p in (a.static_out, a.capture_out):
        if p and os.path.exists(p):
            raise SystemExit(f"REFUSE: {p} exists")
    rows, crec = mo_calset.load_calset(a.calset_dir)
    erec = json.load(open(os.path.join(a.export_dir, "export_record.json")))
    if erec.get("calset_sha256") != crec["calset_sha256"] or erec.get("recipe") != C.EXPORTS[a.name]["recipe"]:
        raise SystemExit("amax: the export was made from another calibration set or recipe")
    seqs = [r["prompt_ids"] + r["response_ids"] for r in rows]
    if a.max_seqs:
        seqs = seqs[: a.max_seqs]
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(C.STUDENT, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map="cuda").eval()
    _, t_cal = calibrate(model, a.name, seqs, "cuda")
    lin, maxima = quantizer_amax(model), observed_maxima(model)
    xcheck = export_cross_check(lin, maxima, erec, "cuda")
    rec = static_record(a.name, lin, erec, xcheck, crec, len(seqs), sum(len(s) for s in seqs), t_cal, a.export_dir)
    C.write_json(a.static_out, rec)
    log(f"static file: {len(rec['act_amax'])} activation amax, {len(rec['kv'])} KV layers; cross-check vs the export: "
        f"input {xcheck['input']}, weight {xcheck['weight']}, kv {xcheck['kv']} -> reproduced={xcheck['reproduced']} "
        f"{xcheck['first_diffs'][:2]} -> {a.static_out}")
    if a.capture_out:
        capture(model, seqs[0], a.capture_out, "cuda")
    return 0


SMOOTH = {"int8_sq": dict(grid="int8sq", cfg="INT8_SMOOTHQUANT_CFG"), "int4_awq": dict(grid="int4awq", cfg="INT4_AWQ_CFG")}


def smooth_calibrate(model, recipe, seqs, device):
    """Model Optimizer's preset on the calibration set (the same loop as calibrate()), recording the float32 weight
    factor of every smoothing. Returns (cfg, seconds, {module name: inv float32 [in]})."""
    import copy

    import torch

    import modelopt.torch.quantization as mtq
    import modelopt.torch.quantization.model_calib as mc
    cfg = copy.deepcopy(getattr(mtq, SMOOTH[recipe]["cfg"]))
    names = {id(m): n for n, m in model.named_modules()}
    inv = {}
    orig = mc._apply_weight_pre_quant_scale

    def record(linear, pre_quant_scale):
        inv[names.get(id(linear), "?")] = pre_quant_scale.detach().to(torch.float32).reshape(-1).clone().cpu()
        return orig(linear, pre_quant_scale)

    mc._apply_weight_pre_quant_scale = record

    def loop(m):
        with torch.no_grad():
            for k, s in enumerate(seqs):
                ids = torch.tensor([s], dtype=torch.long, device=device)
                m(input_ids=ids, use_cache=False, logits_to_keep=1)
                if k % 64 == 0:
                    log(f"  calibration {k}/{len(seqs)}")

    t = time.time()
    try:
        mtq.quantize(model, cfg, forward_loop=loop)
    finally:
        mc._apply_weight_pre_quant_scale = orig
    return cfg, time.time() - t, inv


def smooth_capture(model, seq, out_path, device, grid):
    """Inputs and Model Optimizer's own outputs for bench/mo_smooth_check.py: per captured linear layer the raw input X,
    the input quantizer on X (x * pqs, + INT8 for SmoothQuant), the original and the smoothed weight (sha256 + rows),
    the weight quantizer on the smoothed weight, a perturbed weight smoothed and quantized with the weight quantizer
    re-calibrated (fresh scales), and the module's output."""
    import hashlib

    import torch

    from modelopt.torch.quantization import extensions as ext
    from modelopt.torch.quantization import tensor_quant as TQ
    from modelopt.torch.quantization.model_calib import max_calibrate

    def sha(t):
        return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

    def same(a, b):
        return bool(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8)))

    cx = ext.get_cuda_ext() if str(device).startswith("cuda") else None

    def two_paths(t, amax, q, mo_out):
        """Model Optimizer's two integer fake-quant paths on the same tensor: the torch one (_tensor_quant) and the
        CUDA extension (per tensor / per axis; static blocks have none), and which of them its quantizer ran."""
        bits, nr = int(q._num_bits), bool(q._narrow_range)
        leg = TQ._tensor_quant(t, amax, bits, False, nr)
        r = dict(mo_eq_legacy=same(mo_out, leg), cuda=None, cuda_eq_legacy=None, mo_eq_cuda=None)
        if cx is not None and (amax.numel() == 1 or amax.numel() in amax.shape):
            cu = (cx.fake_tensor_quant(t, amax, bits, False, nr) if amax.numel() == 1 else
                  cx.fake_tensor_quant_with_axis(t, amax.squeeze(), list(amax.shape).index(amax.numel()), bits, False, nr))
            r.update(cuda=True, cuda_eq_legacy=same(cu, leg), mo_eq_cuda=same(mo_out, cu))
        elif cx is not None:
            r["cuda"] = "none for this amax shape (static blocks: Model Optimizer falls back to _tensor_quant)"
        return r

    nl = int(model.config.num_hidden_layers)
    targets = [f"model.layers.{i}.{m}" for i in CAPTURE_LAYERS if i < nl for m in CAPTURE_MODULES]
    mods = dict((n, m) for n, m in model.named_modules() if n in set(targets))
    got = {}
    hooks = [m.register_forward_pre_hook(lambda mod, args, n=n: got.setdefault(n, args[0].detach().clone()))
             for n, m in mods.items()]
    ids = torch.tensor([seq[:CAPTURE_TOKENS]], dtype=torch.long, device=device)
    with torch.no_grad():
        model(input_ids=ids, use_cache=False, logits_to_keep=1)
    for h in hooks:
        h.remove()
    g = torch.Generator(device=device).manual_seed(CAPTURE_SEED)
    out, R = {}, CAPTURE_ROWS
    with torch.no_grad():
        for n in targets:
            m, X = mods[n], got[n]
            iq, wq = m.input_quantizer, m.weight_quantizer
            Wp = m.weight.detach().clone()                    # the smoothed weight (W * inv)[bf16]
            w_mo = wq(Wp)
            rec = dict(x=X.cpu(), x_dtype=str(X.dtype), x_mo=iq(X).cpu(), pqs=iq.pre_quant_scale.detach().cpu(),
                       act_amax=None if grid != "int8sq" else iq.amax.detach().float().cpu(),
                       wp_sha256=sha(Wp), wp_rows=Wp[:R].cpu(), w_mo_rows=w_mo[:R].cpu(), w_mo_sha256=sha(w_mo),
                       w_amax_dtype=str(wq.amax.dtype) if wq.amax is not None else None)
            rec["paths"] = dict(weight=two_paths(Wp if wq.block_sizes is None else wq._process_for_blockquant(Wp),
                                                 wq.amax, wq, w_mo if wq.block_sizes is None
                                                 else wq._process_for_blockquant(w_mo)))
            if grid == "int8sq":
                rec["paths"]["act"] = two_paths(X * iq.pre_quant_scale, iq.amax, iq, rec["x_mo"].to(X.device))
            noise = torch.randn(Wp.shape, generator=g, device=Wp.device, dtype=torch.float32)
            W2 = (Wp.float() + noise * float(Wp.float().std()) * 0.01).to(Wp.dtype)   # a smoothed weight, perturbed
            old_w, old_a = m.weight.data.clone(), wq.amax.detach().clone()
            old_ia = iq.amax.detach().clone() if iq.amax is not None else None
            m.weight.data.copy_(W2)
            wq.reset_amax()
            max_calibrate(m, lambda mod: mod.weight_quantizer(mod.weight))         # re-calibrated: fresh scales
            w2_mo = wq(W2)
            rec.update(w2p_rows=W2[:R].cpu(), w2_mo_rows=w2_mo[:R].cpu())
            m.weight.data.copy_(old_w)
            wq.amax = old_a
            if old_ia is not None:
                iq.amax = old_ia
            rec.update(y_mo=m(X).cpu(), bias=None if m.bias is None else m.bias.detach().cpu())
            out[n] = rec
    torch.save(dict(modules=out, meta=dict(layers=CAPTURE_LAYERS, rows=R, tokens=int(ids.shape[1]), seed=CAPTURE_SEED,
                                           student=C.STUDENT, grid=grid, torch=torch.__version__,
                                           gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu")),
               out_path)
    log(f"capture: {len(out)} modules -> {out_path}")


def cmd_smooth(a):
    import hashlib

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    import modelopt
    import mo_calset
    from modelopt.torch.quantization import extensions as ext
    grid = SMOOTH[a.recipe]["grid"]
    for p in (a.static_out, a.tensors_out, a.capture_out):
        if p and os.path.exists(p):
            raise SystemExit(f"REFUSE: {p} exists")
    torch.manual_seed(0)
    dev = "cpu" if a.tiny else "cuda"
    if a.tiny:   # login-node CPU check of the same calls on a random 2-layer Qwen2 (no checkpoint, random sequences)
        from transformers import Qwen2Config, Qwen2ForCausalLM
        qc = Qwen2Config(hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=512, max_position_embeddings=256, tie_word_embeddings=False,
                         architectures=["Qwen2ForCausalLM"])
        model = Qwen2ForCausalLM(qc).to(torch.bfloat16).eval()
        g = torch.Generator().manual_seed(1)
        seqs = [torch.randint(0, 512, (40 + 8 * i,), generator=g).tolist() for i in range(4)]
        crec = dict(calset_sha256="tiny")
        orig_w = {n: m.weight.detach().clone() for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    else:
        rows, crec = mo_calset.load_calset(a.calset_dir)
        seqs = [r["prompt_ids"] + r["response_ids"] for r in rows]
        if a.max_seqs:
            seqs = seqs[: a.max_seqs]
        model = AutoModelForCausalLM.from_pretrained(C.STUDENT, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                     device_map="cuda").eval()
    cfg, t_cal, inv = smooth_calibrate(model, a.recipe, seqs, dev)
    cuda_ext = (not a.tiny) and ext.get_cuda_ext() is not None   # which integer fake-quant path Model Optimizer ran
    tens, act, wmax, problems, smoothed_eq, narrow = {}, {}, {}, [], 0, set()
    for name, m in model.named_modules():
        if ".layers." not in name or not hasattr(m, "input_quantizer"):
            continue
        iq = m.input_quantizer
        pqs = getattr(iq, "pre_quant_scale", None)
        if pqs is None or name not in inv:
            problems.append(f"{name}: no smoothing (pqs {pqs is not None}, inv {name in inv})")
            continue
        narrow.add((bool(iq._narrow_range), bool(m.weight_quantizer._narrow_range)))
        tens[f"{name}.pqs"] = pqs.detach().to(torch.bfloat16).reshape(-1).contiguous().cpu()
        tens[f"{name}.inv"] = inv[name].contiguous()
        if grid == "int8sq":
            act[name] = float(iq.amax.float())
        w0 = orig_w[name] if a.tiny else C.student_tensor(name + ".weight", "cuda")
        wp = (w0 * inv[name].to(w0.device).reshape(1, -1)).to(w0.dtype)
        smoothed_eq += int(torch.equal(wp.view(torch.int16), m.weight.detach().view(torch.int16)))
        wmax[name] = dict(dtype=str(m.weight_quantizer.amax.dtype), shape=list(m.weight_quantizer.amax.shape))
    if len(narrow) != 1:
        problems.append(f"narrow_range differs across modules: {sorted(narrow)}")
    if problems:
        raise SystemExit(f"smooth: {problems[:4]}")
    (nr_in, nr_w), = narrow
    os.makedirs(os.path.dirname(os.path.abspath(a.tensors_out)), exist_ok=True)
    save_file(tens, a.tensors_out)
    tsha = hashlib.sha256(open(a.tensors_out, "rb").read()).hexdigest()
    rec = dict(format="specopd_quant_static/v1", grid=grid, recipe=f"mtq.{SMOOTH[a.recipe]['cfg']}", model=C.STUDENT,
               act_amax=act or None, tensors=dict(path=os.path.basename(a.tensors_out), sha256=tsha),
               n_linear=len(tens) // 2, smoothed_weight_equal=smoothed_eq, weight_amax=wmax,
               narrow_range=dict(input=nr_in, weight=nr_w),
               modelopt_int_path="cuda_ext" if cuda_ext else "torch (legacy _tensor_quant)",
               stale="fixed from step 0: each module's activation smoothing pqs (bf16), its weight factor inv "
                     "(float32)" + (" and the per-tensor activation amax" if grid == "int8sq" else "") +
                     "; fresh at every sync: the smoothed weight W' = (W * inv)[bf16] and its "
                     + ("per-output-channel INT8 scales" if grid == "int8sq" else "per-block (128) INT4 scales"),
               quant_cfg=cfg,
               provenance=dict(calset_sha256=crec["calset_sha256"], n_calibration_sequences=len(seqs),
                               n_calibration_tokens=sum(len(s) for s in seqs), seconds_calibrate=round(t_cal, 1),
                               versions=dict(modelopt=modelopt.__version__, torch=torch.__version__,
                                             transformers=__import__("transformers").__version__)))
    C.write_json(a.static_out, rec)
    log(f"smooth {a.recipe}: {rec['n_linear']} linear layers, smoothed weights = (W * inv)[bf16] in "
        f"{smoothed_eq}/{rec['n_linear']}; Model Optimizer's integer path: {rec['modelopt_int_path']}; "
        f"calibrated {t_cal:.0f}s -> {a.static_out}, {a.tensors_out}")
    if smoothed_eq != rec["n_linear"]:
        raise SystemExit("smooth: the recorded weight factor does not reproduce Model Optimizer's smoothed weights")
    if a.capture_out:
        smooth_capture(model, seqs[0], a.capture_out, dev, grid)
    return 0


def smoothkv_record(srec, spath, erec, export_dir):
    """A smoothed static file (mo_export smooth) + Model Optimizer's calibrated FP8 cache: the k / v scales of the
    formats job's fp8_kv export as vLLM serves a checkpoint's (rounded to bf16) -- the calibrated cache of the
    calibrated-cache configurations; the export's KV scales are the same in all of that job's exports (their
    calibration forward is unquantized). The tensors file stays the smoothemu job's (absolute path, sha256 unchanged);
    nothing else of the smoothing changes."""
    import copy
    import hashlib
    if srec.get("format") != "specopd_quant_static/v1" or srec.get("grid") not in ("int8sq", "int4awq") or srec.get("kv"):
        raise SystemExit(f"smoothkv: {spath} is not a smoothed static file without a cache (grid {srec.get('grid')}, "
                         f"kv {bool(srec.get('kv'))})")
    kv_export = erec["kv_scales"]
    if len(kv_export) != C.N_LAYERS:
        raise SystemExit(f"smoothkv: the export has KV scales for {len(kv_export)} layers, want {C.N_LAYERS}")
    rec = copy.deepcopy(srec)
    tp = rec["tensors"]["path"]
    rec["tensors"]["path"] = tp if os.path.isabs(tp) else os.path.join(os.path.dirname(os.path.abspath(spath)), tp)
    with open(spath, "rb") as f:
        ssha = hashlib.sha256(f.read()).hexdigest()
    rec.update(kv={li: dict(k=C.bf16(v["k"]), v=C.bf16(v["v"])) for li, v in sorted(kv_export.items(), key=lambda x: int(x[0]))},
               kv_export=kv_export,
               kv_rule="Model Optimizer's calibrated FP8 cache: the formats job's fp8_kv export's k_scale / v_scale as "
                       "vLLM serves a checkpoint's (rounded to bf16)",
               stale=srec["stale"].replace("fixed from step 0: ", "fixed from step 0: the FP8 cache's k / v scales, "),
               derived_from=dict(static=os.path.abspath(spath), static_sha256=ssha, export_dir=export_dir,
                                 export_record_md5=C.md5_file(os.path.join(export_dir, "export_record.json"))))
    return rec


def cmd_smoothkv(a):
    for p in (a.static_out,):
        if os.path.exists(p):
            raise SystemExit(f"REFUSE: {p} exists")
    srec = json.load(open(a.static))
    erec = json.load(open(os.path.join(a.export_dir, "export_record.json")))
    rec = smoothkv_record(srec, a.static, erec, a.export_dir)
    C.write_json(a.static_out, rec)
    log(f"smoothkv: {a.static} + the cache of {a.export_dir} ({len(rec['kv'])} layers, layer 0 k/v "
        f"{rec['kv']['0']['k']}/{rec['kv']['0']['v']}) -> {a.static_out}")
    return 0


def cmd_tiny(a):
    """CPU test of the same path on a random 4-layer Qwen2 (login node, seconds)."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(hidden_size=128, intermediate_size=256, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=512, max_position_embeddings=512, tie_word_embeddings=False,
                      architectures=["Qwen2ForCausalLM"], torch_dtype="bfloat16")
    model = Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()
    g = torch.Generator().manual_seed(1)
    seqs = [torch.randint(0, 512, (48,), generator=g).tolist() for _ in range(4)]
    saved = dict(C.EXPORTS)
    if a.name == "fp8_kv_fl":   # first and last of the TINY model
        C.EXPORTS[a.name] = dict(C.EXPORTS[a.name], disable_layers=(0, 3))
    try:
        run(model, None, a.name, seqs, a.out_dir, "cpu")
    finally:
        C.EXPORTS.clear()
        C.EXPORTS.update(saved)
    return 0


def selftest():
    e = disable_entries((0, 27))
    assert e == [{"quantizer_name": "*layers.0.*", "enable": False}, {"quantizer_name": "*layers.27.*", "enable": False}]
    assert kv_table({"model.layers.3.self_attn.k_proj.k_scale": [0.5], "model.layers.3.self_attn.v_proj.v_scale": [0.25],
                     "model.layers.3.mlp.down_proj.input_scale": [1.0]}) == {"3": {"k": 0.5, "v": 0.25}}
    lin = {"model.layers.0.self_attn.q_proj": dict(input=dict(enabled=True, amax=[2.0]), weight=dict(enabled=True, amax=[0.5])),
           "model.layers.0.self_attn.k_proj": dict(input=dict(enabled=True, amax=[2.0]), weight=dict(enabled=True, amax=[0.25])),
           "model.layers.0.self_attn.v_proj": dict(input=dict(enabled=True, amax=[2.0]), weight=dict(enabled=True, amax=[1.0])),
           "model.layers.0.self_attn.o_proj": dict(input=dict(enabled=True, amax=[3.0]), weight=dict(enabled=True, amax=[0.75]))}
    try:
        import torch  # noqa: F401
        have_torch = True
    except ImportError:
        have_torch = False
    if have_torch:
        import struct
        f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]  # noqa: E731
        sc = {f"model.layers.0.self_attn.{p}_proj.input_scale": [f32(2.0 / NVFP4_DIV)] for p in "qkv"}
        sc.update({f"model.layers.0.self_attn.{p}_proj.weight_scale_2": [f32(1.0 / NVFP4_DIV)] for p in "qkv"})
        sc.update({"model.layers.0.self_attn.o_proj.input_scale": [f32(3.0 / NVFP4_DIV)],
                   "model.layers.0.self_attn.o_proj.weight_scale_2": [f32(0.75 / NVFP4_DIV)]})
        mx = {"0": {q: dict(amax=[1.5]) for q in ("k_bmm_quantizer", "v_bmm_quantizer")}}
        x = export_cross_check(lin, mx, dict(scales=sc, observed_maxima=mx), "cpu")
        assert x["input"] == [4, 4] and x["weight"] == [4, 4] and x["kv"] == [2, 2] and x["reproduced"], x
        sc["model.layers.0.self_attn.k_proj.weight_scale_2"] = [f32(0.25 / NVFP4_DIV)]   # an export WITHOUT unification
        assert not export_cross_check(lin, mx, dict(scales=sc, observed_maxima=mx), "cpu")["reproduced"]
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        sp = os.path.join(td, "static.json")
        srec = dict(format="specopd_quant_static/v1", grid="int8sq", tensors=dict(path="static.safetensors", sha256="x"),
                    stale="fixed from step 0: each module's smoothing", narrow_range=dict(input=False, weight=False))
        json.dump(srec, open(sp, "w"))
        ed = os.path.join(td, "fp8_kv")
        os.makedirs(ed)
        kvs = {str(i): dict(k=0.0123456 + i, v=0.5) for i in range(C.N_LAYERS)}
        json.dump(dict(kv_scales=kvs), open(os.path.join(ed, "export_record.json"), "w"))
        r = smoothkv_record(srec, sp, dict(kv_scales=kvs), ed)
        assert r["tensors"]["path"] == os.path.join(td, "static.safetensors") and r["narrow_range"] == srec["narrow_range"]
        assert r["kv"]["0"] == dict(k=C.bf16(0.0123456), v=0.5) and len(r["kv"]) == C.N_LAYERS
        assert r["stale"].startswith("fixed from step 0: the FP8 cache's k / v scales, each module's smoothing")
        for bad in (dict(srec, kv={"0": {}}), dict(srec, grid="nvfp4")):
            try:
                smoothkv_record(bad, sp, dict(kv_scales=kvs), ed)
                raise AssertionError("smoothkv accepted a bad static file")
            except SystemExit:
                pass
    assert C.selftest() == 0
    print("[mo_export] SELFTEST PASS")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("export", "tiny", "amax", "smooth", "smoothkv"))
    ap.add_argument("--static", default="", help="smoothkv: the smoothemu job's static file")
    ap.add_argument("--name", default="", choices=[""] + sorted(C.EXPORTS))
    ap.add_argument("--recipe", default="", choices=["", "int8_sq", "int4_awq"], help="smooth: the preset")
    ap.add_argument("--tensors-out", default="", help="smooth: the safetensors file of pqs / inv")
    ap.add_argument("--tiny", action="store_true", help="smooth: CPU check on a random 2-layer Qwen2")
    ap.add_argument("--calset-dir", default=C.CALSET_DIR)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--max-seqs", type=int, default=0, help="testing only: the first N calibration sequences")
    ap.add_argument("--export-dir", default="", help="amax: the formats job's export of this recipe (cross-check, KV)")
    ap.add_argument("--static-out", default="", help="amax: the static file to write")
    ap.add_argument("--capture-out", default="", help="amax: the capture for bench/mo_nvfp4_check.py")
    a = ap.parse_args(argv)
    if a.cmd == "smooth":
        if not (a.recipe and a.static_out and a.tensors_out):
            ap.error("smooth needs --recipe, --static-out and --tensors-out")
        return cmd_smooth(a)
    if a.cmd == "smoothkv":
        if not (a.static and a.export_dir and a.static_out):
            ap.error("smoothkv needs --static, --export-dir and --static-out")
        return cmd_smoothkv(a)
    if not a.name:
        ap.error("--name is required")
    if a.cmd == "amax":
        if not (a.export_dir and a.static_out):
            ap.error("amax needs --export-dir and --static-out")
        return cmd_amax(a)
    if not a.out_dir:
        ap.error("--out-dir is required")
    return cmd_export(a) if a.cmd == "export" else cmd_tiny(a)


if __name__ == "__main__":
    sys.exit(main())
