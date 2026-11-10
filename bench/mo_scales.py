#!/usr/bin/env python3
"""The step-0 scale file of a Model Optimizer sampler (verl/utils/specopd_modelopt.py, format specopd_modelopt/v1), made
from a Model Optimizer export (bench/mo_export.py). CPU; venv-specopd (safetensors + torch).

    $PYTHON_BIN bench/mo_scales.py make --export-dir <D>/ckpt/fp8_kv --recipe fp8 --out $CLUSTER_ROOT/mo/scales/fp8.json
    $PYTHON_BIN bench/mo_scales.py make --export-dir <D>/ckpt/fp8_kv_fl --recipe fp8fl \
        --out $CLUSTER_ROOT/mo/scales/fp8fl.json
    python3     bench/mo_scales.py selftest

<D> = the formats job's D (bench/mo_formats.sbatch). $CLUSTER_ROOT/mo/scales/<recipe>.json is the launcher's default
SAMPLER_MODELOPT_SCALES for SAMPLER_MODELOPT=<recipe> (scripts/cluster_opd_sync.sh).

Contents: the export's quantization_config (config.json), for every fused linear layer of vLLM's Qwen2 (qkv_proj <-
q/k/v, gate_up_proj <- gate/up, o_proj, down_proj) the static input scale vLLM serves (ModelOptFp8LinearMethod takes the
max over the shards' exported input_scale), every layer's k_scale / v_scale as vLLM serves them from the export
(`kv`: rounded to bf16, because vLLM's KV-scale checkpoint parameters take the model dtype; the float32 export values are
kept in `kv_export`), and the sha256 of every export file. Read straight from the safetensors (float32 values, written as JSON floats: exact). Refuses to overwrite.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mo_common as C  # noqa: E402

FORMAT = "specopd_modelopt/v1"
PACKED = {"qkv_proj": ("q_proj", "k_proj", "v_proj"), "gate_up_proj": ("gate_proj", "up_proj")}
UNPACKED = ("o_proj", "down_proj")
SHARD_TO_FUSED = {s: f for f, ss in PACKED.items() for s in ss}
_NAME = re.compile(r"^(model\.layers\.(\d+)\.(?:self_attn|mlp))\.(\w+)\.(input_scale|k_scale|v_scale)$")


def build(scalars, quantization_config, recipe, n_layers=None):
    """scalars: {tensor name: float} of the export's 0-dim scale tensors. n_layers: the model's decoder layers (the
    layers without exported k / v scales are listed as kv_bf16_layers: the recipe keeps their cache in bf16)."""
    inputs, kv = {}, {}
    for name, val in scalars.items():
        m = _NAME.match(name)
        if not m:
            continue
        prefix, li, mod, kind = m.group(1), m.group(2), m.group(3), m.group(4)
        if kind == "input_scale":
            fused = SHARD_TO_FUSED.get(mod, mod if mod in UNPACKED else None)
            if fused is None:
                raise ValueError(f"unexpected linear {name}")
            key = f"{prefix}.{fused}"
            inputs.setdefault(key, {})[mod] = float(val)
        elif kind == "k_scale" and mod == "k_proj":
            kv.setdefault(li, {})["k"] = float(val)
        elif kind == "v_scale" and mod == "v_proj":
            kv.setdefault(li, {})["v"] = float(val)
    fused = {}
    for key, shards in inputs.items():
        want = PACKED.get(key.rsplit(".", 1)[1], (key.rsplit(".", 1)[1],))
        if sorted(shards) != sorted(want):
            raise ValueError(f"{key}: input scales for {sorted(shards)}, want {sorted(want)}")
        fused[key] = max(shards.values())
    for li, d in kv.items():
        if sorted(d) != ["k", "v"]:
            raise ValueError(f"layer {li}: k/v scale incomplete {d}")
    # served = what vLLM 0.19.1 serves from the export: its k_scale / v_scale parameters are bf16 (mo_common.bf16)
    served = {li: dict(k=C.bf16(d["k"]), v=C.bf16(d["v"])) for li, d in kv.items()}
    kv_bf16 = sorted(i for i in range(n_layers) if str(i) not in kv) if n_layers else []
    if recipe == "fp8" and kv_bf16:
        raise ValueError(f"recipe fp8 but no k/v scales for layers {kv_bf16}")
    if recipe == "fp8fl" and n_layers and kv_bf16 != [0, n_layers - 1]:
        raise ValueError(f"recipe fp8fl: layers without k/v scales {kv_bf16}, want [0, {n_layers - 1}]")
    return dict(format=FORMAT, recipe=recipe, quantization_config=quantization_config, input_scale=fused, kv=served,
                kv_export=kv, kv_bf16_layers=kv_bf16,
                input_scale_shards=inputs, n_quantized_linear=len(fused), n_kv_layers=len(kv))


def cmd_make(a):
    import torch
    from safetensors import safe_open
    if os.path.exists(a.out):
        raise SystemExit(f"REFUSE: {a.out} exists")
    cfg = json.load(open(os.path.join(a.export_dir, "config.json")))
    qc = cfg.get("quantization_config")
    if not qc or qc.get("quant_method") != "modelopt":
        raise SystemExit("REFUSE: the export has no Model Optimizer quantization_config")
    if a.recipe in ("fp8", "fp8fl") and str(qc.get("quant_algo")).upper() != "FP8":
        raise SystemExit(f"REFUSE: recipe fp8 but quant_algo {qc.get('quant_algo')}")
    scalars, files = {}, {}
    for f in sorted(os.listdir(a.export_dir)):
        p = os.path.join(a.export_dir, f)
        if os.path.isfile(p):
            files[f] = C.sha256_file(p)
        if f.endswith(".safetensors"):
            with safe_open(p, "pt") as fh:
                for k in fh.keys():
                    if k.endswith(("input_scale", "k_scale", "v_scale")):
                        t = fh.get_tensor(k)
                        assert t.numel() == 1 and t.dtype == torch.float32, (k, t.shape, t.dtype)
                        scalars[k] = float(t.reshape(()))
    rec = build(scalars, qc, a.recipe, int(cfg["num_hidden_layers"]))
    rec.update(export_dir=os.path.abspath(a.export_dir), export_files_sha256=files)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    C.write_json(a.out, rec)
    print(f"[mo_scales] {a.out}: {rec['n_quantized_linear']} quantized linear layers, {rec['n_kv_layers']} FP8 cache "
          f"layers; layer-0 k/v {rec['kv'].get('0')}; sha256 {C.sha256_file(a.out)}")
    return 0


def selftest():
    sc = {}
    for i in range(2):
        for m in ("q_proj", "k_proj", "v_proj"):
            sc[f"model.layers.{i}.self_attn.{m}.input_scale"] = 0.5 if m != "v_proj" else 0.25
        sc[f"model.layers.{i}.self_attn.o_proj.input_scale"] = 0.1
        sc[f"model.layers.{i}.mlp.gate_proj.input_scale"] = 0.2
        sc[f"model.layers.{i}.mlp.up_proj.input_scale"] = 0.3
        sc[f"model.layers.{i}.mlp.down_proj.input_scale"] = 0.4
        sc[f"model.layers.{i}.self_attn.k_proj.k_scale"] = 0.05
        sc[f"model.layers.{i}.self_attn.v_proj.v_scale"] = 0.01
    sc["model.layers.0.self_attn.q_proj.weight_scale"] = 9.0      # ignored
    r = build(sc, {"quant_method": "modelopt", "quant_algo": "FP8"}, "fp8")
    assert r["n_quantized_linear"] == 8 and r["n_kv_layers"] == 2
    assert r["input_scale"]["model.layers.1.self_attn.qkv_proj"] == 0.5
    assert r["input_scale"]["model.layers.0.mlp.gate_up_proj"] == 0.3
    assert r["kv_export"]["1"] == {"k": 0.05, "v": 0.01}
    assert r["kv"]["1"] == {"k": C.bf16(0.05), "v": C.bf16(0.01)} and C.bf16(0.05) != 0.05
    assert build(sc, {}, "fp8", 2)["kv_bf16_layers"] == []
    sc3 = {k.replace("layers.1.", "layers.2."): v for k, v in sc.items()}
    sc3.update({k: v for k, v in sc.items() if "layers.0." in k})
    sc3 = {k: v for k, v in sc3.items() if not (("layers.0." in k or "layers.2." in k) and k.endswith(("k_scale", "v_scale")))}
    sc3.update({"model.layers.1.self_attn.k_proj.k_scale": 0.05, "model.layers.1.self_attn.v_proj.v_scale": 0.01})
    assert build(sc3, {}, "fp8fl", 3)["kv_bf16_layers"] == [0, 2]
    try:
        build(sc3, {}, "fp8", 3)
        raise AssertionError("recipe fp8 with bf16-cache layers was accepted")
    except ValueError:
        pass
    del sc["model.layers.1.mlp.up_proj.input_scale"]
    try:
        build(sc, {}, "fp8")
        raise AssertionError("an incomplete fused layer was accepted")
    except ValueError:
        pass
    print("[mo_scales] SELFTEST PASS")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("make",))
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--recipe", default="fp8", choices=("fp8", "fp8fl"))
    ap.add_argument("--out", required=True)
    return cmd_make(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
