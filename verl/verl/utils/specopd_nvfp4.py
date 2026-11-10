# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NVIDIA Model Optimizer's NVFP4 fake quantization, VENDORED for the emulated NVFP4 sampler (verl/utils/specopd_quant.py
grid nvfp4). The rollout environment need not have `modelopt` installed, so the kernel that
Model Optimizer 0.47.0 itself runs for NVFP4 fake quantization on Hopper is copied here VERBATIM:

  * fp4_fake_quant_kernel / fp4_fake_quant_block   <- modelopt/torch/kernels/quantization/gemm/fp4_kernel_hopper.py
  * fp4_round_magnitude / fp8_quantize_scale       <- modelopt/torch/kernels/quantization/common/nvfp4_quant.py
  * _TORCH_TO_TL_DTYPE / _torch_dtype_to_tl        <- modelopt/torch/kernels/quantization/gemm/fp4_kernel.py

Model Optimizer's TensorQuantizer forward for its NVFP4 recipe (num_bits e2m1, block_sizes {-1: 16, type: dynamic,
scale_bits e4m3}) calls exactly fp4_fake_quant_block(x, amax) on SM >= 8.9 (tensor_quant._dynamic_block_quantize_impl),
with amax = the quantizer's calibrated maximum: for weights the weight's own max |W|, for activations the maximum of the
calibration run (a STATIC per-tensor global scale; the per-16 block scales are always computed from the data).
Bit-identity with Model Optimizer on identical inputs is checked by bench/mo_nvfp4_check.py (GPU).

The kernel is exposed as the custom op ``specopd::nvfp4_fake_quant(x, amax: float)``. Inside a torch.compile'd vLLM graph a
custom op is opaque, so its input is MATERIALIZED in its own dtype: the quantizer sees the bf16-rounded activation, as
Model Optimizer's eager forward does (Inductor cannot elide the bf16 cast in front of an opaque op; checked by
bench/mo_nvfp4_check.py: compiled == eager bit for bit). It launches a fill and one Triton kernel and allocates its output
with the caching allocator: capturable in CUDA graphs (no host-device copy). Triton specializes the kernel on the row
count (1 / multiple of 16 / other), so the sampler warms all three variants per input width before any graph capture.
"""

import torch

_LIB_DEFINED = False
SEEN: dict = {}  # input dtypes the op's eager body ran on (profile / warm-up / non-graph runs; not graph replays)


def _kernels():
    import triton
    import triton.language as tl

    _TORCH_TO_TL_DTYPE = {
        torch.float32: tl.float32,
        torch.float: tl.float32,
        torch.float16: tl.float16,
        torch.half: tl.float16,
        torch.bfloat16: tl.bfloat16,
    }

    def _torch_dtype_to_tl(dtype: torch.dtype):
        if dtype not in _TORCH_TO_TL_DTYPE:
            raise ValueError(f"Unsupported dtype for fp4 fake quantization: {dtype}")
        return _TORCH_TO_TL_DTYPE[dtype]

    @triton.jit
    def fp4_round_magnitude(abs_scaled):
        return tl.where(
            abs_scaled <= 0.25,
            0.0,
            tl.where(
                abs_scaled < 0.75,
                0.5,
                tl.where(
                    abs_scaled <= 1.25,
                    1.0,
                    tl.where(
                        abs_scaled < 1.75,
                        1.5,
                        tl.where(
                            abs_scaled <= 2.5,
                            2.0,
                            tl.where(abs_scaled < 3.5, 3.0, tl.where(abs_scaled <= 5.0, 4.0, 6.0)),
                        ),
                    ),
                ),
            ),
        )

    @triton.jit
    def fp8_quantize_scale(block_amax, global_scale):
        FP8_E4M3_MAX: tl.constexpr = 448.0
        scale_in_fp8_range = block_amax / (6.0 * global_scale)
        scale_saturated = tl.minimum(scale_in_fp8_range, FP8_E4M3_MAX)
        return scale_saturated.to(tl.float8e4nv).to(tl.float32) * global_scale

    @triton.jit
    def fp4_fake_quant_kernel(
        x_ptr,
        y_ptr,
        M,
        N,
        global_scale_ptr,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        BLOCK_SIZE: tl.constexpr,
        TILE_M: tl.constexpr,
        TILE_N: tl.constexpr,
        NUM_FP4_BLOCKS: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        row_start = pid_m * TILE_M
        col_start = pid_n * TILE_N

        x_block_ptr = tl.make_block_ptr(
            base=x_ptr,
            shape=(M, N),
            strides=(stride_xm, stride_xn),
            offsets=(row_start, col_start),
            block_shape=(TILE_M, TILE_N),
            order=(1, 0),
        )
        y_block_ptr = tl.make_block_ptr(
            base=y_ptr,
            shape=(M, N),
            strides=(stride_ym, stride_yn),
            offsets=(row_start, col_start),
            block_shape=(TILE_M, TILE_N),
            order=(1, 0),
        )

        global_scale = tl.load(global_scale_ptr).to(tl.float32)
        global_scale_safe = tl.where(global_scale > 0.0, global_scale, 1e-12)

        tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

        tile_reshaped = tl.reshape(tile, (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE))
        x_abs = tl.abs(tile_reshaped)

        block_max = tl.max(x_abs, axis=2, keep_dims=True)

        block_max_quant = fp8_quantize_scale(block_max, global_scale_safe)
        block_max_quant = tl.where(block_max_quant >= 1e-5, block_max_quant, 1.0)

        block_max_quant_broadcast = tl.broadcast_to(block_max_quant, (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE))

        abs_scaled = x_abs / block_max_quant_broadcast

        q_val = fp4_round_magnitude(abs_scaled)

        x_rescaled = q_val * block_max_quant_broadcast
        x_rescaled = tl.where(tile_reshaped >= 0, x_rescaled, -x_rescaled)

        tile_quant = tl.reshape(x_rescaled, (TILE_M, TILE_N))

        tl.store(y_block_ptr, tile_quant.to(OUT_DTYPE), boundary_check=(0, 1))

    def fp4_fake_quant_block(x, global_amax, block_size=16, tile_rows=16, tile_cols=64, num_warps=None,
                             num_stages=None):
        x_shape = x.shape
        x_dtype = x.dtype
        x = x.reshape(-1, x_shape[-1]).contiguous()

        M, N = x.shape
        y = torch.empty_like(x)

        stride_xm, stride_xn = x.stride()
        stride_ym, stride_yn = y.stride()

        tile_cols = max(tile_cols, block_size)
        tile_cols_aligned = ((tile_cols + block_size - 1) // block_size) * block_size
        num_fp4_blocks = tile_cols_aligned // block_size

        global_scale = (global_amax.float() / (6.0 * 448.0)).to(x.device)

        grid = lambda *_: (triton.cdiv(M, tile_rows), triton.cdiv(N, tile_cols_aligned))  # noqa: E731

        launch_kwargs = {
            "BLOCK_SIZE": block_size,
            "TILE_M": tile_rows,
            "TILE_N": tile_cols_aligned,
            "NUM_FP4_BLOCKS": num_fp4_blocks,
            "OUT_DTYPE": _torch_dtype_to_tl(x_dtype),
        }
        if num_warps is not None:
            launch_kwargs["num_warps"] = num_warps
        if num_stages is not None:
            launch_kwargs["num_stages"] = num_stages
        with torch.cuda.device(x.device):
            fp4_fake_quant_kernel[grid](
                x,
                y,
                M,
                N,
                global_scale,
                stride_xm,
                stride_xn,
                stride_ym,
                stride_yn,
                **launch_kwargs,
            )

        y = y.view(*x_shape)
        return y

    return fp4_fake_quant_block


_FQ = None


def fp4_fake_quant_block(x, global_amax):
    """Model Optimizer's fp4_fake_quant_block (vendored, see the module docstring; default tiling); lazily compiled.
    global_amax: a float32 tensor (0-dim or 1 element) on x's device."""
    global _FQ
    if _FQ is None:
        _FQ = _kernels()
    return _FQ(x, global_amax)


def define_op():
    """Register specopd::nvfp4_fake_quant(Tensor x, float amax) -> Tensor once per process (idempotent). The amax is
    the static per-tensor global maximum (a Python float holding a float32 value exactly): a constant of the traced
    graph, filled into a 0-dim float32 tensor on x's device inside the op (a fill kernel: no host-device copy, capturable
    in CUDA graphs), then exactly Model Optimizer's fp4_fake_quant_block(x, amax)."""
    global _LIB_DEFINED
    if _LIB_DEFINED:
        return torch.ops.specopd.nvfp4_fake_quant

    @torch.library.custom_op("specopd::nvfp4_fake_quant", mutates_args=())
    def nvfp4_fake_quant(x: torch.Tensor, amax: float) -> torch.Tensor:
        if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError(f"nvfp4_fake_quant: input dtype {x.dtype}")
        if x.shape[-1] % 16:
            raise ValueError(f"nvfp4_fake_quant: last dim {x.shape[-1]} is not a multiple of 16")
        SEEN[str(x.dtype)] = SEEN.get(str(x.dtype), 0) + 1
        return fp4_fake_quant_block(x, torch.full((), amax, dtype=torch.float32, device=x.device))

    @nvfp4_fake_quant.register_fake
    def _(x, amax):
        return torch.empty_like(x)

    _LIB_DEFINED = True
    return torch.ops.specopd.nvfp4_fake_quant


def nvfp4_qdq(x, amax: float):
    """NVFP4 fake quantization of x (any shape, last dim a multiple of 16) with the static per-tensor global maximum
    ``amax`` (Python float), through the custom op (the engine's activation quantizer)."""
    return define_op()(x, float(amax))


def nvfp4_qdq_weight(w):
    """Model Optimizer's weight fake quantization of one [out, in] weight (CUDA): its weight quantizer's amax is the
    max calibrator's max |W| over the whole tensor (in W's dtype, exact in float32), recomputed here from the CURRENT
    weight at every sync (a re-calibration of the weight quantizer); the per-16 block scales are always from the data.
    Returns a new tensor in W's dtype."""
    if not w.is_cuda:
        raise RuntimeError(f"nvfp4_qdq_weight: Model Optimizer's Triton kernel needs a CUDA tensor (got {w.device})")
    return fp4_fake_quant_block(w, w.abs().amax().to(torch.float32))


# ------------------------------------------------------------------------------------------- pure-torch reference
_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def fp4_round_magnitude_ref(a):
    """fp4_round_magnitude in torch (same thresholds, same <= / < pattern)."""
    t = torch
    return t.where(a <= 0.25, 0.0, t.where(a < 0.75, 0.5, t.where(a <= 1.25, 1.0, t.where(a < 1.75, 1.5, t.where(
        a <= 2.5, 2.0, t.where(a < 3.5, 3.0, t.where(a <= 5.0, 4.0, 6.0)))))))


def fp4_fake_quant_ref(x, global_amax, block_size=16):
    """The kernel's arithmetic in plain torch (any device). NOT the proof object: Triton lowers the kernel's two fp32
    divisions to its own division instruction and torch on the CPU divides exactly, so rare elements can differ by one
    grid step; bench/mo_nvfp4_check.py counts them on the GPU. Used by the CPU tests of the plumbing."""
    shape, dt = x.shape, x.dtype
    xf = x.reshape(-1, shape[-1]).to(torch.float32)
    m, n = xf.shape
    gs = global_amax.to(torch.float32) / (6.0 * 448.0)
    gs = torch.where(gs > 0.0, gs, torch.full_like(gs, 1e-12))
    b = xf.reshape(m, n // block_size, block_size)
    ax = b.abs()
    bm = ax.amax(-1, keepdim=True)
    s = torch.clamp(bm / (6.0 * gs), max=448.0).to(torch.float8_e4m3fn).to(torch.float32) * gs
    s = torch.where(s >= 1e-5, s, torch.ones_like(s))
    q = fp4_round_magnitude_ref(ax / s) * s
    return torch.where(b >= 0, q, -q).reshape(shape).to(dt)


def on_grid_fraction_off(w, block_size=16, tol=2.0 ** -7):
    """Audit of LIVE dequantized NVFP4 values (bf16 storage): per 16-block, the largest |value| is 6 x its block scale
    (the block maximum always rounds to the top level: block max / block scale lies in [5.65, 6.4] after the FP8
    rounding of the scale), so value / (block max / 6) must be a level of the E2M1 grid up to bf16 storage error.
    Returns the fraction of elements off the grid (0 = quantized; an unquantized tensor shows most elements off)."""
    f = w.detach().to(torch.float32).reshape(-1, block_size)
    s = (f.abs().amax(-1, keepdim=True) / 6.0).clamp_min(1e-30)
    q = (f / s).abs()
    lv = torch.tensor(_E2M1_LEVELS, dtype=torch.float32, device=f.device)
    near = lv[(q.unsqueeze(-1) - lv).abs().argmin(-1)]
    return float(((q - near).abs() > q * tol + 1e-6).float().mean())
