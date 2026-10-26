# Copyright 2026 Anonymous Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NVIDIA Model Optimizer's INT8 SmoothQuant and INT4 AWQ fake quantization, re-implemented for the emulated samplers
(verl/utils/specopd_quant.py grids int8sq / int4awq). The rollout environment need not have `modelopt` installed; its
arithmetic (modelopt 0.47.0) is:

  * integer fake quantization (tensor_quant.py `_tensor_quant`, the torch path, and kernels/quantization/gemm/
    tensor_quant_gpu.cu `fake_tensor_quant_device`, the CUDA-extension path -- the same IEEE float32 arithmetic):
        bound = 2^(bits-1) - 1; scale = bound / amax (float32 division); q = clamp(rint(float32(x) * scale), lo,
        bound) with lo = -bound - 1 (FULL range: QuantizerAttributeConfig.narrow_range defaults to False and neither
        preset sets it, so an activation beyond -amax reaches -128 / 127; a weight with its own max-calibrated amax
        never leaves [-bound, bound]); out = q / scale (float32 division), cast to x's dtype; amax <= 2^-24 -> 0;
  * SmoothQuant (model_calib.smoothquant, alpha 1.0): per input channel pre_quant_scale s (bf16) on the activation,
    x' = x * s (in x's dtype), INT8 per-tensor static on x'; weights W' = (W * inv)[bf16] with the float32 inv = 1 / s
    of the calibration, INT8 per output channel (axis 0) on W';
  * INT4 AWQ (awq_lite): the same smoothing (s = 1 / best_scale per module), the activation NOT quantized, weights INT4
    in static blocks of 128 along the input axis (one amax per block) on W'.
Fixed from step 0 (the calibration): s, inv, and SmoothQuant's activation amax. Fresh at every sync: W' from the
current weight, and its per-channel (INT8) or per-block (INT4) amax = max |W'| (the weight quantizer re-calibrated).

The activation side runs as the custom ops specopd::smooth_mul(x, s) and specopd::smooth_int8(x, s, amax): opaque to
Inductor, so the compiled graph materializes x in bf16 before the multiply, as Model Optimizer's eager forward does;
plain eager torch ops inside (capturable in CUDA graphs). Bit-identity with Model Optimizer on identical tensors is
bench/mo_smooth_check.py's (GPU).
"""

import torch

EPS = 1.0 / (1 << 24)
NARROW = False   # Model Optimizer's default (both presets): the static file records its quantizers' value
_DEFINED = False
SEEN: dict = {}  # input dtypes the ops' eager bodies ran on (profile / warm-up / non-graph runs; not graph replays)


def _lo(bound, narrow):
    return -bound if narrow else -bound - 1.0


def int_qdq(x, amax, bits, block=None, narrow=NARROW):
    """Model Optimizer's integer fake quantization of x (any float dtype) with amax broadcastable to it (float32 0-dim
    for per-tensor, [rows, 1] per row, or [rows, nblocks, 1] after the block reshape). Returns x's dtype."""
    dt = x.dtype
    xf = x.float()
    a = amax.float()
    bound = torch.full((), float(2 ** (bits - 1) - 1), dtype=torch.float32, device=xf.device)
    scale = bound / a                                    # float32 tensor / tensor: IEEE division
    zero = a <= EPS
    if bool(zero.any()):
        scale = torch.where(zero, torch.zeros_like(scale), scale)
    q = torch.clamp(torch.round(xf * scale), _lo(bound, narrow), bound)
    if bool(zero.any()):
        scale = torch.where(zero, torch.ones_like(scale), scale)
    return (q / scale).to(dt)


def int_qdq_static(x, amax, bits, narrow=NARROW):
    """The same with a per-tensor amax and no data-dependent branch (the engine's activation path: CUDA-graph safe).
    amax must be > 2^-24 (checked when attached)."""
    dt = x.dtype
    bound = torch.full((), float(2 ** (bits - 1) - 1), dtype=torch.float32, device=x.device)
    scale = bound / amax.float()
    return (torch.clamp(torch.round(x.float() * scale), _lo(bound, narrow), bound) / scale).to(dt)


def smooth_weight(w, inv):
    """W' = (W * inv)[W's dtype]: model_calib._apply_weight_pre_quant_scale (inv float32, one per input channel)."""
    return (w * inv.to(w.device).reshape(1, -1)).to(w.dtype)


def qdq_weight_int8_rows(wp):
    """INT8 per output channel (axis 0) of the smoothed weight: amax = max |W'| per row."""
    return int_qdq(wp, wp.abs().amax(dim=1, keepdim=True), 8)


def qdq_weight_int4_blocks(wp, block=128):
    """INT4 static blocks of `block` along the input axis: amax = max |W'| per block."""
    out_f, in_f = wp.shape
    if in_f % block:
        raise ValueError(f"int4 blocks: input width {in_f} is not a multiple of {block}")
    b = wp.reshape(out_f, in_f // block, block)
    return int_qdq(b, b.abs().amax(dim=-1, keepdim=True), 4).reshape(out_f, in_f)


def define_ops():
    """Register specopd::smooth_mul and specopd::smooth_int8 once per process (idempotent)."""
    global _DEFINED
    if _DEFINED:
        return torch.ops.specopd.smooth_mul, torch.ops.specopd.smooth_int8

    @torch.library.custom_op("specopd::smooth_mul", mutates_args=())
    def smooth_mul(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        SEEN[str(x.dtype)] = SEEN.get(str(x.dtype), 0) + 1
        return x * s

    @smooth_mul.register_fake
    def _(x, s):
        return torch.empty_like(x)

    @torch.library.custom_op("specopd::smooth_int8", mutates_args=())
    def smooth_int8(x: torch.Tensor, s: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
        SEEN[str(x.dtype)] = SEEN.get(str(x.dtype), 0) + 1
        return int_qdq_static(x * s, amax, 8)

    @smooth_int8.register_fake
    def _(x, s, amax):
        return torch.empty_like(x)

    _DEFINED = True
    return torch.ops.specopd.smooth_mul, torch.ops.specopd.smooth_int8
