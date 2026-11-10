#!/usr/bin/env python3
"""Bit-for-bit checks of the sampler-side Model Optimizer numerics against Model Optimizer's own code (venv-modelopt).

    $R/venv-modelopt/bin/python -I bench/mo_numerics_check.py fp8 [--device cpu|cuda] [--real]   -> exit 0 = all equal

fp8: verl/utils/specopd_modelopt.mo_fp8_weight (what the rollout worker runs at every sync) against
modelopt.torch.export.unified_export_hf._export_quantized_weight -- the per-module step of export_hf_checkpoint -- on the
same bf16 weight tensors, same device: a one-layer nn.Linear holding W is quantized with the recipe's weight quantizer
(per-tensor FP8 E4M3, max calibration, mtq.quantize) and exported; the FP8 bytes and the weight scale must be identical.
Inputs: random bf16 tensors of several shapes and magnitudes, and (--real) the student's own weight tensors of decoder
layers 0, 13 and 27 (q/k/v/o, gate/up/down) read from its safetensors. Both sides receive the identical bf16 tensor
(no fp32 intermediate on either side). The end-to-end check against the exported 7B checkpoint inside vLLM is
bench/mo_refit_check.py (GPU).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import mo_common as C  # noqa: E402

# verl/ is looked for next to bench/ (the repository root) and one directory above it
SPECOPD_MODELOPT = next((p for p in (os.path.join(r, "verl", "verl", "utils", "specopd_modelopt.py")
                                     for r in (ROOT, os.path.dirname(ROOT))) if os.path.exists(p)),
                        os.path.join(ROOT, "verl", "verl", "utils", "specopd_modelopt.py"))


def load_sampler_numerics():
    """verl/utils/specopd_modelopt.py loaded BY PATH (stdlib at import): the verl package itself needs venv-specopd."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("specopd_modelopt", SPECOPD_MODELOPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def modelopt_fp8_export(w):
    """Model Optimizer's export of one weight through its own code path. Returns (fp8 weight, float32 scale)."""
    import torch
    import torch.nn as nn

    import modelopt.torch.quantization as mtq
    from modelopt.torch.export.unified_export_hf import _export_quantized_weight
    lin = nn.Linear(w.shape[1], w.shape[0], bias=False, dtype=torch.bfloat16, device=w.device)
    with torch.no_grad():
        lin.weight.copy_(w)
    cfg = {"quant_cfg": [{"quantizer_name": "*", "enable": False},
                         {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": (4, 3), "axis": None}}],
           "algorithm": "max"}
    model = nn.Sequential(lin)
    x = torch.zeros(1, w.shape[1], dtype=torch.bfloat16, device=w.device)
    mtq.quantize(model, cfg, forward_loop=lambda m: m(x))
    _export_quantized_weight(model[0], torch.bfloat16, "weight")
    return model[0].weight.detach(), model[0].weight_scale.detach()


def tensors(real, device):
    import torch
    g = torch.Generator().manual_seed(0)
    out = []
    for shape, mag in (((64, 128), 1.0), ((512, 384), 0.02), ((256, 1024), 37.0), ((3, 17), 1e-3)):
        out.append((f"random{shape}x{mag}", (torch.randn(shape, generator=g) * mag).to(torch.bfloat16)))
    w = torch.randn(128, 64, generator=g).to(torch.bfloat16)
    w[5, 7] = 448.0 * 3                                       # one outlier: everything else lands near zero
    out.append(("random_outlier", w))
    if real:
        want = [f"model.layers.{i}.{m}.weight" for i in (0, 13, 27) for m in
                ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
        for name in want:   # the MO_PAIR student's own weights (sharded or single-file checkpoint)
            out.append((name, C.student_tensor(name)))
    return [(n, t.to(device)) for n, t in out]


def cmd_fp8(a):
    import torch
    mo_fp8_weight = load_sampler_numerics().mo_fp8_weight
    rows, bad = [], 0
    for name, w in tensors(a.real, a.device):
        q0, s0 = modelopt_fp8_export(w.clone())
        q1, s1 = mo_fp8_weight(w.clone())
        same_q = q0.dtype == q1.dtype == torch.float8_e4m3fn and q0.shape == q1.shape and \
            bool(torch.equal(q0.view(torch.uint8), q1.view(torch.uint8)))
        same_s = s0.dtype == s1.dtype == torch.float32 and bool(torch.equal(s0.reshape(()).cpu(), s1.reshape(()).cpu()))
        ok = same_q and same_s
        bad += not ok
        nd = int((q0.view(torch.uint8) != q1.view(torch.uint8)).sum()) if q0.shape == q1.shape else -1
        rows.append(dict(name=name, shape=list(w.shape), ok=ok, scale_modelopt=float(s0), scale_ours=float(s1),
                         n_bytes_differ=nd))
        print(f"[mo_numerics] fp8 {name:42s} {str(tuple(w.shape)):>14} scale {float(s0):.9g} vs {float(s1):.9g} "
              f"bytes differing {nd} -> {'EQUAL' if ok else 'DIFFERENT'}", flush=True)
    if a.json:
        C.write_json(a.json, dict(check="fp8", device=a.device, rows=rows, n_bad=bad,
                                  sampler_code_md5=C.md5_file(SPECOPD_MODELOPT)))
    print(f"[mo_numerics] fp8: {len(rows) - bad}/{len(rows)} tensors bit-identical on {a.device}")
    return 0 if bad == 0 else 7


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("check", choices=("fp8",))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--real", action="store_true", help="also the student's own weight tensors (layers 0, 13, 27)")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    return cmd_fp8(a)


if __name__ == "__main__":
    sys.exit(main())
