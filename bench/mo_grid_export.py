"""Two more weight / activation combinations, W8A16 (FP8-rounded weights, bf16 activations) and W4A8 (INT4 weights,
FP8 activations), as checkpoints the harness can serve; and a copy of a Model Optimizer export that declares no
KV-cache quantization (nokv).

  dequant  (venv-specopd, one GPU): the SERVED weights of the W8A8 FP8 configuration (export fp8_kv) in bf16. vLLM
           0.19.1 serves the formats job's Model Optimizer export fp8_kv by re-rounding each fused group (q/k/v,
           gate/up) whose shard scales differ onto the group's largest scale
           (ModelOptFp8LinearMethod.process_weights_after_loading -> requantize_with_max_scale); this step calls that
           same vLLM function on the same tensors, then writes bf16(fp8 x scale) per shard (o_proj, down_proj: their own
           scale). Served unquantized, the engine runs those FP8-rounded weights with bf16 activations: W8A16, differing
           from the W8A8 FP8 configuration only in the activations. Checks: each written weight re-quantizes (bf16 /
           scale -> float8_e4m3fn) to exactly the FP8 bytes vLLM serves; every other tensor equals the bf16 student's
           bit for bit; names and shapes equal the student's.
  w4afp8   (venv-llmc, one GPU): llm-compressor's AWQ (transform) then its W4AFP8 preset (INT4 weights, symmetric, groups
           of 128; FP8 activations, dynamic per token) on the calibration set, lm_head unquantized, saved in the
           pack-quantized format that vLLM's CompressedTensorsW4A8Fp8 loads (weight_packed / weight_scale / weight_shape;
           compressed-tensors would otherwise pick int-quantized for a scheme with activation quantization). Checks: the
           format, and those three tensors for all 196 decoder projections with no bare .weight left.
  tiny     (venv-llmc, CPU): the w4afp8 recipe, save call and checks on a random two-layer Qwen2 (a quick CPU test).
  nokv     (stdlib, CPU): a copy of a Model Optimizer export whose configs declare no KV-cache quantization, for the
           bf16-cache configuration mo_fp8_bf16kv (bench/mo_bf16kv.sbatch): with kv_cache_dtype 'auto' vLLM serves
           the cache dtype the checkpoint's quantization config declares, so the export itself would get its FP8
           cache. Every file is copied unchanged except two: config.json loses kv_cache_scheme (and
           kv_cache_quant_algo, if present) from its quantization_config, and hf_quant_config.json's
           quantization.kv_cache_quant_algo becomes null, as Model Optimizer writes both for a recipe that leaves the
           cache unquantized. Checks: the source files equal the source record's md5s, every copied file equals its
           source, the two configs differ from the source's in exactly the removed entries, and no KV-cache entry
           with a value is left. The record is the source's with the new configs and md5s and a `nokv` section.
  selftest (stdlib, CPU): nokv on a synthetic export.

Each writes <out>/export_record.json (files and md5s, inputs and their hashes, versions, checks) and refuses a
non-empty <out>.
Usage:
  $PYTHON_BIN bench/mo_grid_export.py dequant --src $BASE/ckpt/fp8_kv --out $D/ckpt/w8a16_fp8
  env -u PYTHONPATH $CLUSTER_ROOT/venv-llmc/bin/python -I bench/mo_grid_export.py w4afp8 --calset-dir $CALSET \
      --out $D/ckpt/w4a8_awq
  env -u PYTHONPATH $CLUSTER_ROOT/venv-llmc/bin/python -I bench/mo_grid_export.py tiny --out /tmp/x
  python3 bench/mo_grid_export.py nokv --src $BASE/ckpt/fp8_kv --out $CKPT2/fp8_kv
  python3 bench/mo_grid_export.py selftest
"""
import argparse
import glob
import json
import os

import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mo_common as C  # noqa: E402

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json")
AWQ_N_SAMPLES = 128         # calibration rows: j = 0 of every even k (128 of the 512 calibration responses)
AWQ_MAX_TOKENS = 2048       # prompt + response tokens per row, truncated (llm-compressor's usual AWQ budget)
FUSED = (("self_attn", ("q_proj", "k_proj", "v_proj")), ("mlp", ("gate_proj", "up_proj")))   # vLLM's qkv_proj, gate_up_proj
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def log(*a):
    print("[mo_grid_export]", *a, flush=True)


def _fresh(out):
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit(f"REFUSE: {out} exists and is not empty")
    os.makedirs(out, exist_ok=True)


def _files_record(out):
    return {f: C.md5_file(os.path.join(out, f)) for f in sorted(os.listdir(out))
            if os.path.isfile(os.path.join(out, f)) and f != "export_record.json"}


def _student_tensors():
    from safetensors import safe_open
    out = {}
    for p in sorted(glob.glob(os.path.join(C.STUDENT, "*.safetensors"))):
        f = safe_open(p, "pt")
        for k in f.keys():
            out[k] = f
    return out


def cmd_dequant(a):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import requantize_with_max_scale

    _fresh(a.out)
    src_rec = json.load(open(os.path.join(a.src, "export_record.json")))
    bad = [f for f, h in src_rec["files"].items() if C.md5_file(os.path.join(a.src, f)) != h]
    assert not bad, f"source export files differ from its record: {bad}"
    sf = safe_open(os.path.join(a.src, "model.safetensors"), "pt")
    keys = set(sf.keys())
    stu = _student_tensors()
    dev = torch.device("cuda")
    # 1. the FP8 weights and scales vLLM serves: per fused group, its own requantize_with_max_scale when scales differ
    served = {}          # weight key -> (fp8 tensor on cpu, fp32 scalar scale)
    groups = []
    for i in range(C.N_LAYERS):
        for mod, shards in FUSED:
            ks = [f"model.layers.{i}.{mod}.{s}.weight" for s in shards]
            w = [sf.get_tensor(k) for k in ks]
            s = [sf.get_tensor(k[: -len(".weight")] + ".weight_scale").float().reshape(()) for k in ks]
            assert all(t.dtype == torch.float8_e4m3fn for t in w), ks
            scales = torch.stack(s).to(dev)
            fused = torch.cat(w, 0).to(dev)
            widths = [t.shape[0] for t in w]
            if not bool((scales == scales[0]).all()):     # ModelOptFp8LinearMethod's own condition
                smax, fused = requantize_with_max_scale(fused, scales, widths)
                requant = True
            else:
                smax, requant = scales.max(), False
            smax = smax.float().reshape(()).cpu()
            r0 = 0
            for k, n, t in zip(ks, widths, w):
                q = fused[r0:r0 + n].cpu()
                served[k] = (q, smax)
                r0 += n
            groups.append(dict(layer=i, module=mod, scales=[float(x) for x in s], served_scale=float(smax),
                               requantized=requant))
        for p in ("self_attn.o_proj", "mlp.down_proj"):
            k = f"model.layers.{i}.{p}.weight"
            served[k] = (sf.get_tensor(k), sf.get_tensor(f"model.layers.{i}.{p}.weight_scale").float().reshape(()))
    assert len(served) == 7 * C.N_LAYERS
    # 2. write bf16(fp8 x scale) for the served weights, every other tensor as the export (= the student) has it
    scale_suffixes = (".weight_scale", ".input_scale", ".k_scale", ".v_scale")
    out, n_copy, max_rel, worst = {}, 0, 0.0, None
    for k in sorted(keys):
        if k.endswith(scale_suffixes):
            continue
        if k in served:
            q, s = served[k]
            deq = (q.float() * s).to(torch.bfloat16)
            back = (deq.float() / s).to(torch.float8_e4m3fn)
            assert torch.equal(back.view(torch.uint8), q.view(torch.uint8)), f"{k}: bf16 round trip changes FP8 bytes"
            ref = stu[k].get_tensor(k)
            assert ref.shape == deq.shape, k
            rel = float(((deq.float() - ref.float()).norm() / ref.float().norm()).item())
            if rel > max_rel:
                max_rel, worst = rel, k
            out[k] = deq.contiguous()
        else:
            t = sf.get_tensor(k)
            ref = stu[k].get_tensor(k)
            assert t.dtype == ref.dtype == torch.bfloat16 and torch.equal(t, ref), f"{k}: differs from the student"
            out[k] = t.contiguous()
            n_copy += 1
    assert set(out) == set(stu), f"tensor names differ from the student: {sorted(set(out) ^ set(stu))[:8]}"
    save_file(out, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    cfg.pop("quantization_config", None)
    cfg["torch_dtype"] = "bfloat16"
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    for f in TOKENIZER_FILES:
        shutil.copy2(os.path.join(a.src, f), os.path.join(a.out, f))
    import vllm
    rec = dict(format="mo_grid_export/dequant/v2", source=os.path.abspath(a.src),
               source_record_md5=C.md5_file(os.path.join(a.src, "export_record.json")),
               source_recipe=src_rec.get("recipe"), calset_sha256=src_rec.get("calset_sha256"),
               n_weights=len(served), n_groups_requantized=sum(g["requantized"] for g in groups), groups=groups,
               n_copied=n_copy, round_trip_exact=True, copied_equal_student=True,
               max_rel_err_vs_student=max_rel, worst_tensor=worst, quantization_config=None,
               torch=torch.__version__, vllm=vllm.__version__, written=time.strftime("%Y-%m-%dT%H:%M:%S"))
    rec["files"] = _files_record(a.out)
    C.write_json(os.path.join(a.out, "export_record.json"), rec)
    log(f"dequant: {len(served)} served FP8 weights -> bf16 (round trip exact; {rec['n_groups_requantized']} of "
        f"{len(groups)} fused groups re-rounded by vLLM's requantize_with_max_scale), {n_copy} tensors equal the "
        f"student's; largest relative error against the student {max_rel:.4f} ({worst}) -> {a.out}")
    return 0


def _w4afp8_recipe():
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier
    return [AWQModifier(duo_scaling="both"),
            QuantizationModifier(targets=["Linear"], scheme="W4AFP8", ignore=["lm_head"])]


def _save_and_check(model, out, n_layers):
    """Save pack-quantized and check what vLLM's CompressedTensorsW4A8Fp8 will look for; returns the quantization_config."""
    from safetensors import safe_open
    model.save_pretrained(out, save_compressed=True, quantization_format="pack-quantized")
    qc = json.load(open(os.path.join(out, "config.json"))).get("quantization_config")
    assert qc and qc.get("format") == "pack-quantized", f"saved format {qc and qc.get('format')}"
    names = {}
    for p in glob.glob(os.path.join(out, "*.safetensors")):
        f = safe_open(p, "pt")
        for k in f.keys():
            names[k] = f
    bad = []
    for i in range(n_layers):
        for proj in PROJ:
            mod = "self_attn" if proj in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
            base = f"model.layers.{i}.{mod}.{proj}"
            for suf in (".weight_packed", ".weight_scale", ".weight_shape"):
                if base + suf not in names:
                    bad.append(base + suf + " missing")
            if base + ".weight" in names:
                bad.append(base + ".weight present (unpacked)")
            if base + ".weight_packed" in names:
                t = names[base + ".weight_packed"].get_slice(base + ".weight_packed")
                if t.get_dtype() not in ("I32", "int32"):
                    bad.append(f"{base}.weight_packed dtype {t.get_dtype()}")
    assert not bad, f"pack-quantized check: {bad[:6]} ({len(bad)} problems)"
    return qc


def cmd_w4afp8(a):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import compressed_tensors
    import llmcompressor
    from llmcompressor import oneshot

    import mo_calset

    _fresh(a.out)
    rows, crec = mo_calset.load_calset(a.calset_dir)
    pick = [r for r in rows if r["j"] == 0 and r["k"] % 2 == 0]
    assert len(pick) == AWQ_N_SAMPLES, len(pick)
    ids = [(r["prompt_ids"] + r["response_ids"])[:AWQ_MAX_TOKENS] for r in pick]
    ds = Dataset.from_dict({"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]})
    tok = AutoTokenizer.from_pretrained(C.STUDENT)
    model = AutoModelForCausalLM.from_pretrained(C.STUDENT, torch_dtype=torch.bfloat16, device_map="cuda")
    t0 = time.time()
    oneshot(model=model, tokenizer=tok, dataset=ds, recipe=_w4afp8_recipe(), num_calibration_samples=len(ids),
            max_seq_length=AWQ_MAX_TOKENS)
    t_cal = time.time() - t0
    qc = _save_and_check(model, a.out, C.N_LAYERS)
    tok.save_pretrained(a.out)
    shutil.copy2(os.path.join(C.STUDENT, "generation_config.json"), os.path.join(a.out, "generation_config.json"))
    rec = dict(format="mo_grid_export/w4afp8/v2", student=C.STUDENT, student_rev=C.STUDENT_REV,
               recipe="llmcompressor AWQModifier(duo_scaling='both') + QuantizationModifier(scheme='W4AFP8', "
                      "targets=['Linear'], ignore=['lm_head']); saved pack-quantized",
               calset_sha256=crec["calset_sha256"], n_samples=len(ids), max_tokens=AWQ_MAX_TOKENS,
               n_tokens=sum(len(x) for x in ids), calibrate_s=round(t_cal, 1), quantization_config=qc,
               pack_check="196 projections: weight_packed (int32), weight_scale, weight_shape; no bare .weight",
               llmcompressor=llmcompressor.__version__, compressed_tensors=compressed_tensors.__version__,
               torch=torch.__version__, written=time.strftime("%Y-%m-%dT%H:%M:%S"))
    rec["files"] = _files_record(a.out)
    C.write_json(os.path.join(a.out, "export_record.json"), rec)
    log(f"w4afp8: {len(ids)} rows, {rec['n_tokens']} tokens, calibration {t_cal:.0f}s, pack-quantized -> {a.out}")
    return 0


def cmd_tiny(a):
    """The w4afp8 path end to end on a random two-layer Qwen2 on CPU: recipe, save call, format and tensor checks."""
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    from llmcompressor import oneshot

    _fresh(a.out)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    model = Qwen2ForCausalLM(cfg).to(torch.bfloat16)
    g = torch.Generator().manual_seed(1)
    ids = [torch.randint(0, 512, (128,), generator=g).tolist() for _ in range(8)]
    ds = Dataset.from_dict({"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]})
    tok = AutoTokenizer.from_pretrained(C.STUDENT)   # any tokenizer: the rows are already token ids
    oneshot(model=model, tokenizer=tok, dataset=ds, recipe=_w4afp8_recipe(), num_calibration_samples=len(ids),
            max_seq_length=128)
    qc = _save_and_check(model, a.out, 2)
    groups = qc.get("config_groups", {})
    log(f"tiny: PASS (format {qc['format']}; groups {json.dumps(groups)[:300]})")
    return 0


# The KV-cache quantization entries of a Model Optimizer export's configs. vLLM reads config.json's quantization_config
# first (Model Optimizer writes it in the compressed-tensors style: kv_cache_scheme) and hf_quant_config.json otherwise
# (its quantization section: kv_cache_quant_algo); with kv_cache_dtype 'auto' it serves the cache dtype they declare.
KV_KEYS = ("kv_cache_scheme", "kv_cache_quant_algo")
NOKV_CONFIGS = ("config.json", "hf_quant_config.json")


def strip_kv(cfg, name):
    """(cfg without its KV-cache quantization entries, {path: removed value}). name = config.json: the KV_KEYS are
    deleted from quantization_config (and from a nested quantization section); hf_quant_config.json: its
    quantization.kv_cache_quant_algo is set to None and kv_cache_scheme deleted."""
    out = json.loads(json.dumps(cfg))
    secs = []
    if name == "config.json":
        qc = out.get("quantization_config")
        if isinstance(qc, dict):
            secs.append(("quantization_config", qc))
            if isinstance(qc.get("quantization"), dict):
                secs.append(("quantization_config.quantization", qc["quantization"]))
    elif name == "hf_quant_config.json":
        if isinstance(out.get("quantization"), dict):
            secs.append(("quantization", out["quantization"]))
    else:
        raise ValueError(name)
    removed = {}
    for path, sec in secs:
        for k in KV_KEYS:
            if sec.get(k) is not None:
                removed[f"{path}.{k}"] = sec[k]
                if name == "hf_quant_config.json" and k == "kv_cache_quant_algo":
                    sec[k] = None
                else:
                    del sec[k]
    return out, removed


def _leaf_diff(a, b, path=""):
    """The paths at which two JSON values differ (dicts compared key by key)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else k
            out += [p] if (k not in a or k not in b) else _leaf_diff(a[k], b[k], p)
        return out
    return [] if a == b else [path]


def _kv_left(x, path=""):
    """Paths of keys mentioning kv_cache that still carry a value."""
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            p = f"{path}.{k}" if path else k
            if "kv_cache" in k and v is not None:
                out.append(p)
            out += _kv_left(v, p)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out += _kv_left(v, f"{path}[{i}]")
    return out


def cmd_nokv(a):
    src_rec_path = os.path.join(a.src, "export_record.json")
    src_rec = json.load(open(src_rec_path))
    files = src_rec["files"]
    bad = [f for f, h in files.items() if C.md5_file(os.path.join(a.src, f)) != h]
    assert not bad, f"source export files differ from its record: {bad}"
    extra = sorted(f for f in os.listdir(a.src) if os.path.isfile(os.path.join(a.src, f))
                   and f != "export_record.json" and f not in files)
    assert not extra, f"source files not in its record: {extra}"
    assert "config.json" in files, "the source has no config.json"
    new, removed = {}, {}       # new: the configs that lose an entry; every other file is copied unchanged
    for f in NOKV_CONFIGS:
        if f in files:
            orig = json.load(open(os.path.join(a.src, f)))
            stripped, rm = strip_kv(orig, f)
            diff = _leaf_diff(orig, stripped)
            assert sorted(diff) == sorted(rm), f"{f}: changed {diff}, want only {sorted(rm)}"
            left = _kv_left(stripped)
            assert not left, f"{f}: KV-cache entries left {left}"
            if rm:
                new[f] = stripped
                removed.update({f"{f}:{k}": v for k, v in rm.items()})
    if not removed:
        raise SystemExit(f"REFUSE: {a.src} declares no KV-cache quantization (nothing to remove)")
    _fresh(a.out)
    for f in sorted(files):
        dst = os.path.join(a.out, f)
        if f in new:
            with open(dst, "w") as fh:
                json.dump(new[f], fh, indent=2 if f == "config.json" else 4)
                fh.write("\n")
        else:
            shutil.copy2(os.path.join(a.src, f), dst)
            assert C.md5_file(dst) == files[f], f"{f}: the copy differs from the source"
    for f in new:   # read back what was written: equal to the stripped config
        assert json.load(open(os.path.join(a.out, f))) == new[f], f
    rec = dict(src_rec)
    if "config.json" in new:
        rec["quantization_config"] = new["config.json"].get("quantization_config")
    if "hf_quant_config.json" in new:
        rec["hf_quant_config"] = new["hf_quant_config.json"]
    rec["files"] = _files_record(a.out)
    assert {f for f in files if rec["files"][f] != files[f]} == set(new), "files other than the configs changed"
    rec["nokv"] = dict(format="mo_grid_export/nokv/v1", source=os.path.abspath(a.src),
                       source_record_md5=C.md5_file(src_rec_path), removed=removed, changed_files=sorted(new),
                       written=time.strftime("%Y-%m-%dT%H:%M:%S"))
    C.write_json(os.path.join(a.out, "export_record.json"), rec)
    log(f"nokv: {len(files)} files of {a.src}, {len(files) - len(new)} copied unchanged; removed {sorted(removed)} "
        f"-> {a.out}")
    return 0


def cmd_selftest(_a=None):
    """nokv on a synthetic export (stdlib only)."""
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        src = os.path.join(root, "src")
        os.makedirs(src)
        qc = dict(config_groups={"group_0": dict(weights=dict(num_bits=8, type="float"), targets=["Linear"])},
                  ignore=["lm_head"], quant_algo="FP8", kv_cache_scheme=dict(dynamic=False, num_bits=8, type="float"),
                  producer=dict(name="modelopt", version="x"), quant_method="modelopt")
        cfg = dict(model_type="qwen2", num_hidden_layers=2, quantization_config=qc)
        hq = dict(producer=dict(name="modelopt", version="x"),
                  quantization=dict(quant_algo="FP8", kv_cache_quant_algo="FP8", exclude_modules=["lm_head"]))
        json.dump(cfg, open(os.path.join(src, "config.json"), "w"), indent=2)
        json.dump(hq, open(os.path.join(src, "hf_quant_config.json"), "w"), indent=4)
        open(os.path.join(src, "model.safetensors"), "wb").write(bytes(range(256)) * 64)
        open(os.path.join(src, "tokenizer.json"), "w").write("{}")
        C.write_json(os.path.join(src, "export_record.json"), dict(name="fp8_kv", kv_scales={"0": dict(k=0.1, v=0.2)},
                                                                    files=_files_record(src)))
        out = os.path.join(root, "out", "fp8_kv")
        assert cmd_nokv(argparse.Namespace(src=src, out=out)) == 0
        c2 = json.load(open(os.path.join(out, "config.json")))
        h2 = json.load(open(os.path.join(out, "hf_quant_config.json")))
        assert "kv_cache_scheme" not in c2["quantization_config"] and h2["quantization"]["kv_cache_quant_algo"] is None
        assert _leaf_diff(cfg, c2) == ["quantization_config.kv_cache_scheme"]
        assert _leaf_diff(hq, h2) == ["quantization.kv_cache_quant_algo"]
        rec = json.load(open(os.path.join(out, "export_record.json")))
        assert rec["kv_scales"] == {"0": dict(k=0.1, v=0.2)} and rec["quantization_config"] == c2["quantization_config"]
        assert rec["files"]["model.safetensors"] == C.md5_file(os.path.join(src, "model.safetensors"))
        assert rec["nokv"]["changed_files"] == ["config.json", "hf_quant_config.json"]
        assert {f for f, h in rec["files"].items() if C.md5_file(os.path.join(out, f)) != h} == set()
        refused = ((dict(src=src, out=out), "not empty"), (dict(src=out, out=os.path.join(root, "o2")), "declares no"))
        for args, what in refused:
            try:
                cmd_nokv(argparse.Namespace(**args))
                raise AssertionError(f"nokv accepted {args}")
            except SystemExit as e:
                assert what in str(e), e
        open(os.path.join(src, "model.safetensors"), "ab").write(b"x")
        try:
            cmd_nokv(argparse.Namespace(src=src, out=os.path.join(root, "o3")))
            raise AssertionError("nokv accepted a source that differs from its record")
        except AssertionError as e:
            assert "differ from its record" in str(e), e
    log("selftest: PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("dequant")
    d.add_argument("--src", required=True)
    d.add_argument("--out", required=True)
    w = sp.add_parser("w4afp8")
    w.add_argument("--calset-dir", required=True)
    w.add_argument("--out", required=True)
    t = sp.add_parser("tiny")
    t.add_argument("--out", required=True)
    n = sp.add_parser("nokv")
    n.add_argument("--src", required=True, help="the Model Optimizer export (e.g. the formats job's D/ckpt/fp8_kv)")
    n.add_argument("--out", required=True, help="the copy (e.g. $CKPT2/fp8_kv); refused unless absent or empty")
    sp.add_parser("selftest")
    a = ap.parse_args(argv)
    return {"dequant": cmd_dequant, "w4afp8": cmd_w4afp8, "tiny": cmd_tiny, "nokv": cmd_nokv,
            "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
