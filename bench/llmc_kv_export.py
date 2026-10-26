#!/usr/bin/env python3
"""llm-compressor KV-cache-only FP8 calibration of our student and its export (venv-llmc): the per-head FP8 cache
(key / value scales per KV head; Model Optimizer offers per-tensor only) and a per-tensor cross-check of Model
Optimizer's calibrated FP8 cache.

    $R/venv-llmc/bin/python -I bench/llmc_kv_export.py export --strategy attn_head --calset-dir C --out-dir D/ckpt/llmc_kv_head
    $R/venv-llmc/bin/python -I bench/llmc_kv_export.py tiny --strategy attn_head --out-dir /tmp/x     # CPU, the same calls
    python3 bench/llmc_kv_export.py selftest                                                     # stdlib

llm-compressor's QuantizationModifier with only a kv_cache_scheme (FP8 E4M3, static, symmetric, observer
memoryless_minmax, strategy tensor | attn_head), oneshot on the SAME calibration set as Model Optimizer's exports
(mo_calset.py: 512 prompt + response sequences, each one full sequence, batch 1), saved with save_compressed (bf16
weights unchanged + k_scale / v_scale per layer); the student's tokenizer files are copied. Writes export_record.json
in the formats job's shape (kv_scales per layer as lists: 1 value per tensor, one per KV head for attn_head;
quantization_config; md5 per file; calibration-set hash and token count; versions; seconds).
"""
import argparse
import json
import os
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mo_common as C  # noqa: E402

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json")
_KV = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.([kv])_scale$")


def log(*a):
    print(f"[llmc_kv_export {time.strftime('%H:%M:%S')}]", *a, flush=True)


def kv_table(scales):
    """{layer: {k: [..], v: [..]}} from the exported tensor names model.layers.<i>.self_attn.{k,v}_scale."""
    out = {}
    for name, vals in scales.items():
        m = _KV.search(name)
        if m:
            out.setdefault(m.group(1), {})[m.group(2)] = list(vals)
    return out


def scheme(strategy):
    return {"num_bits": 8, "type": "float", "strategy": strategy, "dynamic": False, "symmetric": True}


def cmd_export(a):
    import compressed_tensors
    import llmcompressor
    import torch
    import transformers
    from datasets import Dataset
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    import mo_calset
    if os.path.exists(a.out_dir) and os.listdir(a.out_dir):
        raise SystemExit(f"REFUSE: {a.out_dir} exists and is not empty")
    tok = AutoTokenizer.from_pretrained(C.STUDENT)
    torch.manual_seed(0)
    if a.cmd == "tiny":   # CPU check of the same calls on a random 2-layer Qwen2 (login node)
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, vocab_size=512, max_position_embeddings=256, tie_word_embeddings=False,
                          architectures=["Qwen2ForCausalLM"])
        model = Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()
        g = torch.Generator().manual_seed(1)
        seqs = [torch.randint(0, 512, (40 + 8 * i,), generator=g).tolist() for i in range(3)]
        crec = dict(calset_sha256="tiny")
    else:
        rows, crec = mo_calset.load_calset(a.calset_dir)
        seqs = [r["prompt_ids"] + r["response_ids"] for r in rows]
        if a.max_seqs:
            seqs = seqs[: a.max_seqs]
        model = AutoModelForCausalLM.from_pretrained(C.STUDENT, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                     device_map="cuda").eval()
    max_len = max(len(s) for s in seqs)
    ds = Dataset.from_list([{"input_ids": s, "attention_mask": [1] * len(s)} for s in seqs])
    t = time.time()
    oneshot(model=model, processor=tok, dataset=ds, recipe=[QuantizationModifier(kv_cache_scheme=scheme(a.strategy))],
            num_calibration_samples=len(seqs), max_seq_length=max_len, pipeline="basic", shuffle_calibration_samples=False)
    t_cal = time.time() - t
    os.makedirs(a.out_dir, exist_ok=True)
    model.save_pretrained(a.out_dir, save_compressed=True)
    for f in TOKENIZER_FILES:
        src = os.path.join(C.STUDENT, f)
        if os.path.exists(src) and not os.path.exists(os.path.join(a.out_dir, f)):
            shutil.copy2(src, a.out_dir)
    scales, dtypes = {}, {}
    for f in sorted(os.listdir(a.out_dir)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(a.out_dir, f), "pt") as fh:
                for k in fh.keys():
                    if k.endswith(("k_scale", "v_scale")):
                        x = fh.get_tensor(k)
                        scales[k] = [float(v) for v in x.float().reshape(-1).tolist()]
                        dtypes[k] = f"{x.dtype} {tuple(x.shape)}"
    cfg = json.load(open(os.path.join(a.out_dir, "config.json")))
    kv = kv_table(scales)
    rec = dict(name=f"llmc_kv_{'head' if a.strategy == 'attn_head' else 'tensor'}", tool="llm-compressor",
               strategy=a.strategy, kv_cache_scheme=scheme(a.strategy), kv_scales=kv,
               kv_scale_dtype=sorted(set(dtypes.values())), quantization_config=cfg.get("quantization_config"),
               calset_sha256=crec["calset_sha256"], n_calibration_sequences=len(seqs),
               n_calibration_tokens=sum(len(s) for s in seqs), max_seq_length=max_len, seconds_calibrate=round(t_cal, 1),
               versions=dict(llmcompressor=llmcompressor.__version__, compressed_tensors=compressed_tensors.__version__,
                             torch=torch.__version__, transformers=transformers.__version__),
               files={f: C.md5_file(os.path.join(a.out_dir, f)) for f in sorted(os.listdir(a.out_dir))
                      if os.path.isfile(os.path.join(a.out_dir, f))})
    C.write_json(os.path.join(a.out_dir, "export_record.json"), rec)
    n_layers = model.config.num_hidden_layers
    if len(kv) != n_layers or any(len(v.get("k", [])) != len(v.get("v", [])) for v in kv.values()):
        raise SystemExit(f"export has KV scales for {len(kv)} layers (want {n_layers}): {sorted(kv)[:4]}")
    log(f"{rec['name']}: {len(kv)} layers, layer-0 k {kv['0']['k']} v {kv['0']['v']} ({rec['kv_scale_dtype']}); "
        f"calibrated {t_cal:.0f}s on {rec['n_calibration_tokens']} tokens -> {a.out_dir}")
    return 0


def selftest():
    t = kv_table({"model.layers.3.self_attn.k_scale": [0.5, 0.25], "model.layers.3.self_attn.v_scale": [0.1, 0.2],
                  "model.layers.3.self_attn.q_proj.weight": [1.0], "model.layers.10.self_attn.k_scale": [2.0]})
    assert t == {"3": {"k": [0.5, 0.25], "v": [0.1, 0.2]}, "10": {"k": [2.0]}}, t
    assert scheme("attn_head")["strategy"] == "attn_head" and scheme("tensor")["dynamic"] is False
    assert C.selftest() == 0
    print("[llmc_kv_export] SELFTEST PASS")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("export", "tiny"))
    ap.add_argument("--strategy", required=True, choices=("tensor", "attn_head"))
    ap.add_argument("--calset-dir", default=C.CALSET_DIR)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-seqs", type=int, default=0, help="testing only: the first N calibration sequences")
    a = ap.parse_args(argv)
    return cmd_export(a)


if __name__ == "__main__":
    sys.exit(main())
