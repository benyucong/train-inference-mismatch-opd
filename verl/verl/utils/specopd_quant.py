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
"""Sampler-only quantization exactly as in Score Centering (arXiv 2609.20807; github.com/martin-marek/score-centering
@7c56e9ee, models/quant.py + models/model.py L118-175), on the student's ROLLOUT engine only.

Opt-in: ``SPECOPD_SAMPLER_QUANT=<spec>`` (+ ``SPECOPD_SAMPLER_QUANT_MODEL=<student path>``, required). FAIL-CLOSED: an
engine whose model is not that path RAISES (a sampler whose path was spelled differently must never run silently
unquantized) unless ``SPECOPD_SAMPLER_QUANT_ALLOW_SKIP=1``, which skips it instead (test / leak checks only). The teacher
and reward engines never see the variable: vLLMReplica blanks it for them (replica_env_overrides). Unset = nothing is
imported or patched: byte-identical behaviour.

SPEC: comma-separated ``key=value``, every key optional:
    w=<grid>     weights, re-quantized at EVERY weight sync (update_weights_from_ipc) from a bf16 copy of the actor
                 stream (the FSDP stream is fp32 masters; Score Centering's copy_sampler_weights casts to bf16 first)
    a=<grid>     activations: dynamic per-token absmax at the input of every matmul (all LinearBase layers + lm_head)
                 (a TIED lm_head = embed_tokens: its logits GEMM input only, never the embedding lookup)
    kv=<kvgrid>  every KV-cache write, per (token, KV head) over head_dim, AFTER bias and RoPE (Q and softmax P are
                 never quantized; the cache stays bf16, kv_cache_dtype must be auto)
    kvpart=kv|k|v   which of K / V is snapped (default kv; k / v = diagnostics, not Score Centering settings)
    kvskip=<i:j:..> decoder layers whose K/V are NOT snapped (colon-separated indices; default none): the K/V those
                 layers write stay BF16 bit for bit; their weights and activations still follow w / a.
    kvl0heads=<h:h:..>  LAYER-0 KV-HEAD SUBSET (the settings between kvskip=0 and Score Centering's all-layer snap; NOT a
                 Score Centering setting). The SET S of layer-0 KV heads (vLLM's layer-0 KV-head order,
                 the global head index) whose K/V ARE snapped; layer 0's other heads stay bf16 bit for bit, every
                 other layer is unchanged, kvpart applies to the heads in S too. Written WITHOUT 0 in kvskip (layer
                 0 is head-masked, not exempt: 0 in kvskip together with kvl0heads is refused). ONE spelling per
                 subset: the heads strictly ascending (2:0 or 0:0:2 is refused, write 0:2), so a setting has exactly
                 one canonical string and compile hash. S = {} is NOT written kvl0heads= (refused): that
                 setting is plain kvskip=0. S = all heads is accepted and snaps layer 0 exactly as the spec without
                 kvl0heads (bit for bit: an equivalence check). Every head of S must exist in the model (checked at
                 attach time); rollout TP must be 1.
                 S -> spec (R1-Distill-Qwen-7B, 4 KV heads, Score Centering's FP8 sampler):
                     S = {}         w=e4m3,a=e4m3,kv=e4m3_unit,kvskip=0
                     S = {0,1,2,3}  w=e4m3,a=e4m3,kv=e4m3_unit
                     proper S       w=e4m3,a=e4m3,kv=e4m3_unit,kvl0heads=<S ascending>, e.g. kvl0heads=1:3
  grids for w / a :  int8 (alias 8), e4m3 (alias fp8); w alone also intN (N=2..16), e2m1 (fp4) = weight-only
  grids for kv    :  e4m3_unit (alias fp8: e4m3 grid at UNIT scale = vLLM kv_cache_dtype=fp8 default k/v_scale=1),
                     e2m1 (fp4) / e3m2 (fp6): absmax/gmax scale per (token, head); intN (alias N): absmax/(2^(N-1)-1)
  Score Centering settings:  FP8 sampler         w=e4m3,a=e4m3,kv=e4m3_unit      (sweeps/30b_math.yaml L55)
                   FP8 + FP4 KV        w=e4m3,a=e4m3,kv=e2m1           (L56)
                   INT8 W/A + INT4 KV  w=int8,a=int8,kv=int4           (L57)
                   0.6B int8 sampler   w=int8,a=int8,kv=int8           (06b_quant_stale.yaml L55)
  w and a together = Score Centering's W8A8 sampler (dtypes.sampler=int8|float8_e4m3fn): both grids must then be equal.

NVIDIA MODEL OPTIMIZER NVFP4, EMULATED (NOT a Score Centering setting): grid nvfp4
for w and a = Model Optimizer's NVFP4 recipe (general/ptq/nvfp4_default-kv_fp8: E2M1 values in blocks of 16 along the
input axis, one E4M3 scale per block, one float32 global scale per tensor = amax / (6 * 448)) computed with Model
Optimizer's OWN fake-quantization kernel, vendored in verl/utils/specopd_nvfp4.py (Hopper has no FP4 tensor cores).
    w=nvfp4,a=nvfp4   W4A4 (the recipe's linear layers)      w=nvfp4   weight-only (a diagnostic, not a recipe)
  * targets: the recipe's -- every decoder linear layer (q/k/v/o, gate/up/down); NOT embed_tokens and NOT lm_head (the
    recipe disables *lm_head*; embeddings are not linear layers);
  * weights: at every sync each streamed HF weight (bf16 copy) is fake-quantized on its own, with amax = max |W| of the
    current weight (Model Optimizer's weight quantizer re-calibrated: its amax is the max calibrator's per-tensor max),
    exactly as Model Optimizer's fake-quantized HF model does it -- per HF tensor, so q / k / v (gate / up) keep their own
    global scales inside vLLM's fused layers (Model Optimizer's EXPORT instead unifies them to the largest; a recorded
    difference between its fake quantization and its export);
  * activations: the input of every decoder linear layer, through the custom op specopd::nvfp4_fake_quant with the
    STATIC per-tensor amax Model Optimizer's calibration found at step 0 (fixed for the whole run: stale by design, as a
    served checkpoint's input_scale), from SPECOPD_SAMPLER_QUANT_STATIC=<json> (format specopd_quant_static/v1, written
    by bench/mo_export.py amax): act_amax per HF linear name; a fused layer's shards must agree (they see one input);
  * KV cache (KV_DTYPE=fp8, the recipe's FP8 cache): vLLM's REAL FP8 cache with the file's per-layer k / v scales,
    written after load and again after every KV wake-up (vLLM resets them to 1.0 there); the query and
    probability scales stay vLLM's 1.0. A bf16 cache (KV_DTYPE=auto) is a diagnostic. kv= grids are refused.
  The amax constants are baked into the compiled graph, so the compile cache must be off (as for every grid here).

NVIDIA MODEL OPTIMIZER INT8 SMOOTHQUANT / INT4 AWQ, EMULATED (vLLM 0.19.1 refuses both
checkpoints): grids int8sq (w=int8sq,a=int8sq = Model Optimizer's INT8_SMOOTHQUANT_CFG, alpha 1.0) and int4awq
(w=int4awq = INT4_AWQ_CFG, awq_lite; its activation smoothing is part of the recipe and always applied), with Model
Optimizer's arithmetic re-implemented in verl/utils/specopd_smooth.py. Targets: the decoder linear layers (no lm_head,
no embed_tokens). The step-0 calibration (SPECOPD_SAMPLER_QUANT_STATIC: format specopd_quant_static/v1, grid
int8sq|int4awq, its `tensors` file = per HF linear `<name>.pqs` (bf16, the activation smoothing) and `<name>.inv`
(float32, the weights' 1 / smoothing as the calibration applied it), and for int8sq `act_amax` per HF linear) is FIXED;
at every sync each weight is smoothed W' = (W * inv)[bf16] and re-quantized with FRESH scales (INT8 per output channel
/ INT4 per block of 128); the activation of each decoder linear layer is x * pqs (custom op specopd::smooth_mul) and,
for int8sq, INT8 per tensor with the fixed amax (specopd::smooth_int8). A fused layer whose shards carry different
smoothing (AWQ searches per module) runs one GEMM per shard (the HF model's separate projections); equal smoothing
(SmoothQuant at alpha 1.0: 1 / act_amax, one input) runs fused. KV: bf16 (the recipes quantize no cache), or vLLM's
FP8 cache with static scales from the file as for nvfp4.

What is quantized -- the arithmetic of Score Centering's quant.py, mirrored operation for operation where the
frameworks allow (whether XLA executes their absmax / qmax as an IEEE division is not verified):
  * WEIGHTS (quant.quantize / fake_quantize_tree): every tensor with ndim >= 2 whose HF name contains neither "norm" nor
    "bias" and does not end in "mlp.gate.weight" -- q/k/v/o, gate/up/down, embed_tokens AND lm_head. Symmetric absmax,
    one fp32 scale per OUTPUT ROW (HF [out, in] layout: Score Centering's o_proj contract_dims=2 over (heads, head_dim) is
    the same row), scale = max(absmax, 1e-30) / qmax, int: round half-to-even then clip +-qmax; float: grid snap with
    subnormals and saturation (== clip + RNE cast). vLLM's qkv / gate_up fusion only concatenates rows, so per-row
    scales are exact in the fused parameters. Dequantized as bf16(q * scale) -- except embed_tokens in W8A8 mode,
    dequantized as bf16(q) * bf16(scale) in bf16 exactly as Score Centering's quant.take().
  * ACTIVATIONS (quant.qeinsum): x -> q = x / (max(absmax_token, 1e-30) / qmax) -> grid -> bf16(q * scale) before the
    GEMM, per token over the contraction axis (TP=1 here, so the full axis, as in Score Centering).
  * KV (quant.kv_fake_quantize): K and V as passed to vLLM's Attention.forward = after the QKV bias and RoPE (Qwen2
    applies both before self.attn), snapped in fp32 and stored in the bf16 cache; prompt tokens too (Score Centering feeds
    the prompt through the same decode path). Identical in prefill and decode.
Known deviations: Score Centering runs real 8-bit GEMMs with fp32/int32 accumulation and rescales the fp32 accumulator; here
the dequantized operands are bf16 and the GEMM is cuBLAS bf16 (fp32 accumulate) -- each operand carries one extra bf16
rounding of bf16(q * s): relative <= 2^-9, i.e. <= 1/16 of a step on the e4m3 grid (step 2^-4 relative) but up to ~1/4
of an int8 step for |q| near 127 (every row's / token's absmax element has |q| = 127); its error power is still far below
the int8 noise power s^2/12. Score Centering's gate_proj output / down_proj input is fp32 (preferred_element_type) where
vLLM's SiluAndMul emits bf16. Under torch.compile Inductor elides intermediate bf16 casts, so inside the fused graph
a quantizer can see the unrounded fp32 RMSNorm / RoPE / SiLU output that eager rounds to bf16 first: boundary
elements flip one grid step (same error statistics; XLA has the same class of behaviour) -- compiled and eager
engines therefore differ like two noise draws, and every comparison must keep both sides compiled. Every
sampler-quant engine runs Inductor with emulate_divison_rounding (IEEE fp32 division in its compiled graphs, a
process-wide flag: fp32-ulp-level side effects elsewhere, e.g. SiLU's sigmoid, that a clean engine does not share).

Engine side: the patches are installed in the vLLM worker process (vLLMColocateWorkerExtension.__new__, before the
model is built) and attached AFTER GPUModelRunner.load_model to the TARGET model's modules only -- never the
speculative drafter's (EAGLE-3) layers, which are counted and checked untouched -- before the first forward, i.e. before
torch.compile traces and CUDA graphs capture. vLLM 0.19.1 on torch >= 2.10 loads AOT-compiled graphs from its cache by a
key that does not see this module, so a patched engine REQUIRES VLLM_DISABLE_COMPILE_CACHE=1 (set here if possible,
else a hard error); the spec is also added to additional_config so the compile hash differs from a clean engine's.
Log lines (WARNING, one per engine / per sync):
    [SPECOPD-QUANT-CFG] ...   what this engine patched (or SKIP for a non-sampler engine)
    [SPECOPD-QUANT] sync=..   per weight sync: tensors / elements quantized, relative RMS weight error, ms
    [SPECOPD-QUANT-AUDIT] ..  after the first sync: live parameters read back and checked to lie on the grid
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

ENV = "SPECOPD_SAMPLER_QUANT"
ENV_MODEL = "SPECOPD_SAMPLER_QUANT_MODEL"
ENV_ALLOW_SKIP = "SPECOPD_SAMPLER_QUANT_ALLOW_SKIP"
ENV_STATIC = "SPECOPD_SAMPLER_QUANT_STATIC"  # nvfp4 / int8sq / int4awq: Model Optimizer's step-0 calibration
STATIC_GRIDS = ("nvfp4", "int8sq", "int4awq")
STATIC_FORMAT = "specopd_quant_static/v1"

# Mini-float grids: (mantissa bits, max value, min normal exponent) -- quant.py _FLOAT_GRIDS verbatim.
_FLOAT_GRIDS = {"e4m3": (3, 448.0, -6), "e3m2": (2, 28.0, -2), "e2m1": (1, 6.0, 0)}
_FLOAT_ALIASES = {"fp8": "e4m3", "fp6": "e3m2", "fp4": "e2m1"}


@dataclass(frozen=True)
class Grid:
    """One quantization grid. kind 'int': symmetric 2^(bits-1)-1 levels; 'float': a mini-float (subnormals, saturating).
    unit=True (KV only): the grid at scale 1 (no absmax scaling) -- Score Centering's kv 'fp8'."""

    name: str
    kind: str
    qmax: float
    bits: int = 0
    man_bits: int = 0
    min_normal_exp: int = 0
    unit: bool = False


def _int_grid(bits: int) -> Grid:
    if not 2 <= bits <= 16:
        raise ValueError(f"int grid needs 2..16 bits, got {bits}")
    return Grid(name=f"int{bits}", kind="int", qmax=float(2 ** (bits - 1) - 1), bits=bits)


NVFP4 = Grid(name="nvfp4", kind="nvfp4", qmax=6.0, bits=4)  # Model Optimizer's NVFP4 (specopd_nvfp4.py)
INT8SQ = Grid(name="int8sq", kind="int8sq", qmax=127.0, bits=8)  # Model Optimizer's INT8 SmoothQuant (specopd_smooth.py)
INT4AWQ = Grid(name="int4awq", kind="int4awq", qmax=7.0, bits=4)  # Model Optimizer's INT4 AWQ, blocks of 128


def _float_grid(name: str, unit: bool = False) -> Grid:
    man, gmax, emin = _FLOAT_GRIDS[name]
    return Grid(name=name + ("_unit" if unit else ""), kind="float", qmax=gmax, man_bits=man, min_normal_exp=emin,
                unit=unit)


def parse_grid(raw: str, role: str) -> Grid:
    """role 'w' | 'a' | 'kv' -> Grid. Accepts Score Centering's spellings (fp8/fp4/fp6/8/4) and explicit names."""
    s = raw.strip().lower()
    if s == "nvfp4":
        if role == "kv":
            raise ValueError(
                "kv=nvfp4: Model Optimizer's NVFP4 recipe keeps a REAL FP8 cache with static scales (KV_DTYPE=fp8 + "
                f"{ENV_STATIC}), not a fake-quantized cache"
            )
        return NVFP4
    if s in ("int8sq", "int4awq"):
        if role == "kv":
            raise ValueError(f"kv={s}: Model Optimizer's {s} recipe quantizes no cache")
        if s == "int4awq" and role == "a":
            raise ValueError("a=int4awq: AWQ quantizes no activation (its smoothing comes with w=int4awq)")
        return INT8SQ if s == "int8sq" else INT4AWQ
    if role == "kv":
        if s in ("fp8", "e4m3_unit"):
            return _float_grid("e4m3", unit=True)
        if s in ("e4m3",):
            raise ValueError("kv=e4m3 is ambiguous: Score Centering's fp8 KV is UNIT scale -> write kv=e4m3_unit")
    s = _FLOAT_ALIASES.get(s, s)
    if s in _FLOAT_GRIDS:
        if role == "kv" and s == "e4m3":
            raise ValueError("kv=e4m3 is ambiguous: use kv=e4m3_unit")
        g = _float_grid(s)
    else:
        m = re.fullmatch(r"(?:int)?(\d+)", s)
        if not m:
            raise ValueError(f"unknown {role} grid {raw!r}")
        g = _int_grid(int(m.group(1)))
    if role == "w" and g.name == "e3m2":
        # quant.WEIGHT_QUANT_DTYPES has int8 / fp8 / int4 / fp4 (+ intN); fp6 exists for the KV cache only.
        raise ValueError(f"w={raw}: Score Centering has no fp6 weight scheme (weight_quant_spec rejects it)")
    if role == "a" and g.name not in ("int8", "e4m3"):
        # Score Centering quantizes activations only in its W8A8 sampler (QArray, itemsize-1 dtypes).
        raise ValueError(f"a={raw}: Score Centering's activation quantization exists for int8 and e4m3 only")
    return g


@dataclass(frozen=True)
class QuantSpec:
    w: Optional[Grid] = None
    a: Optional[Grid] = None
    kv: Optional[Grid] = None
    kv_part: str = "kv"
    kv_skip: tuple = ()
    kv_l0_heads: tuple = ()  # () = off; else the layer-0 KV heads that ARE snapped (strictly ascending)

    @classmethod
    def parse(cls, raw: str) -> "QuantSpec":
        kw: dict = {}
        seen: set = set()
        for item in [x for x in (raw or "").split(",") if x.strip()]:
            if "=" not in item:
                raise ValueError(f"{ENV}: item {item!r} is not key=value (spec {raw!r})")
            k, v = (t.strip() for t in item.split("=", 1))
            k = k.lower()
            # by the SPEC key (kw is keyed by field name for kvpart / kvskip / kvl0*, so `k in kw` missed those)
            if k in seen:
                raise ValueError(f"{ENV}: duplicate key {k!r} in {raw!r}")
            seen.add(k)
            if k in ("w", "a", "kv"):
                kw[k] = None if v.lower() in ("", "none", "off", "bf16") else parse_grid(v, k)
            elif k == "kvpart":
                if v.lower() not in ("kv", "k", "v"):
                    raise ValueError(f"{ENV}: kvpart must be kv, k or v (got {v!r})")
                kw["kv_part"] = v.lower()
            elif k == "kvskip":
                try:
                    ids = tuple(sorted({int(x) for x in v.split(":") if x.strip()}))
                except ValueError:
                    raise ValueError(f"{ENV}: kvskip takes colon-separated layer indices (got {v!r})") from None
                if not ids or min(ids) < 0:
                    raise ValueError(f"{ENV}: kvskip needs >= 1 non-negative layer index (got {v!r})")
                kw["kv_skip"] = ids
            elif k == "kvl0heads":
                if v == "":
                    raise ValueError(f"{ENV}: kvl0heads= (S = {{}}) is not a head subset: write kvskip=0 ({raw!r})")
                parts = v.split(":")
                if not all(re.fullmatch(r"\d+", x) for x in parts):
                    raise ValueError(f"{ENV}: kvl0heads takes colon-separated layer-0 KV-head indices (got {v!r})")
                heads = tuple(int(x) for x in parts)
                if list(heads) != sorted(set(heads)):
                    # one spelling per subset -> one canonical string / compile hash per setting
                    want = ":".join(map(str, sorted(set(heads))))
                    raise ValueError(f"{ENV}: kvl0heads={v}: write the heads once each, ascending (kvl0heads={want})")
                kw["kv_l0_heads"] = heads
            else:
                raise ValueError(f"{ENV}: unknown key {k!r} in {raw!r} (keys: w, a, kv, kvpart, kvskip, kvl0heads)")
        spec = cls(**kw)
        if spec.w is not None and spec.a is not None and spec.w.name != spec.a.name:
            raise ValueError(f"{ENV}: w and a together are Score Centering's W8A8 sampler -> same grid (got {raw!r})")
        if spec.a is not None and spec.w is None:
            raise ValueError(f"{ENV}: a= without w= has no Score Centering counterpart (W8A8 quantizes both) ({raw!r})")
        if spec.smooth == "int8sq" and spec.a is None:
            raise ValueError(f"{ENV}: w=int8sq is the W8A8 SmoothQuant recipe: write w=int8sq,a=int8sq ({raw!r})")
        if spec.smooth and spec.kv is not None:
            raise ValueError(f"{ENV}: w={spec.smooth}: no kv= grid (bf16 cache, or KV_DTYPE=fp8 with static scales) ({raw!r})")
        if spec.nvfp4 and spec.kv is not None:
            raise ValueError(
                f"{ENV}: w=nvfp4 is Model Optimizer's recipe, whose cache is vLLM's real FP8 cache (KV_DTYPE=fp8): "
                f"no kv= grid ({raw!r})"
            )
        if "kv_part" in kw and spec.kv is None:
            raise ValueError(f"{ENV}: kvpart without kv ({raw!r})")
        if "kv_skip" in kw and spec.kv is None:
            raise ValueError(f"{ENV}: kvskip without kv ({raw!r})")
        if "kv_l0_heads" in kw:
            if spec.kv is None:
                raise ValueError(f"{ENV}: kvl0heads without kv ({raw!r})")
            if 0 in spec.kv_skip:
                raise ValueError(
                    f"{ENV}: kvl0heads sets which layer-0 heads are snapped; layer 0 must not also be in kvskip "
                    f"(S = {{}} is kvskip=0 alone) ({raw!r})"
                )
        return spec

    @property
    def l0_heads(self) -> tuple:
        """The layer-0 KV heads that are snapped under kvl0heads (ascending), () when kvl0heads is off."""
        return self.kv_l0_heads

    @property
    def empty(self) -> bool:
        return self.w is None and self.a is None and self.kv is None

    @property
    def w8a8(self) -> bool:
        return self.w is not None and self.a is not None and not self.nvfp4 and not self.smooth

    @property
    def smooth(self) -> Optional[str]:
        """'int8sq' / 'int4awq' (Model Optimizer's smoothed integer recipes, emulated), else None."""
        return self.w.kind if self.w is not None and self.w.kind in ("int8sq", "int4awq") else None

    @property
    def static_grid(self) -> bool:
        """A grid whose step-0 calibration comes from ENV_STATIC."""
        return self.nvfp4 or bool(self.smooth)

    @property
    def nvfp4(self) -> bool:
        """Model Optimizer's NVFP4 (emulated): its targets, its kernel, static activation amax from ENV_STATIC."""
        return self.w is not None and self.w.kind == "nvfp4"

    def canonical(self) -> str:
        parts = []
        if self.w is not None:
            parts.append(f"w={self.w.name}")
        if self.a is not None:
            parts.append(f"a={self.a.name}")
        if self.kv is not None:
            parts.append(f"kv={self.kv.name}")
            if self.kv_part != "kv":
                parts.append(f"kvpart={self.kv_part}")
            if self.kv_skip:
                parts.append("kvskip=" + ":".join(map(str, self.kv_skip)))
            if self.kv_l0_heads:
                parts.append("kvl0heads=" + ":".join(map(str, self.kv_l0_heads)))
        return ",".join(parts) or "none"


# ----------------------------------------------------------------------------------------------------------------------
# Grid arithmetic. Pure torch, device-agnostic, torch.compile-able (no data-dependent control flow, no log2/exp2, no
# int bitcasts), bit-identical on CPU, CUDA eager and Inductor/Triton.
# ----------------------------------------------------------------------------------------------------------------------


def snap_float(x: torch.Tensor, g: Grid) -> torch.Tensor:
    """quant._snap_float: round fp32 ``x`` onto the mini-float grid (round-half-even, subnormals, saturate at qmax).
    The binade's quantum 2^(e - man) is picked by exact comparisons against the powers of two up to the grid's top
    binade (above it everything saturates anyway), so there is no log2/exp2 (accuracy under Triton) and no int bitcast
    (Inductor widens such int math to int64 for some grids); scaling by a power of two is exact."""
    ax = x.abs()
    e_lo, e_hi = g.min_normal_exp, int(math.floor(math.log2(g.qmax)))
    quantum = torch.full_like(ax, 2.0 ** (e_lo - g.man_bits))
    inv_quantum = torch.full_like(ax, 2.0 ** (g.man_bits - e_lo))
    for k in range(e_lo + 1, e_hi + 1):
        ge = ax >= 2.0**k
        quantum = torch.where(ge, 2.0 ** (k - g.man_bits), quantum)
        inv_quantum = torch.where(ge, 2.0 ** (g.man_bits - k), inv_quantum)
    snapped = torch.round(ax * inv_quantum) * quantum
    return torch.sign(x) * snapped.clamp(max=g.qmax)


def _div(t: torch.Tensor, c: float) -> torch.Tensor:
    """t / c, IEEE-rounded, the literal ``absmax / qmax`` of Score Centering's code. Both PyTorch's CUDA kernel (Python-scalar
    divisor) and Inductor on GPU (constant divisor, lowering.div_prim) replace it by t * fp32(1/c), which moves the scale
    by 1 ulp on some rows. The divisor is therefore built from t (t * 0 + c: finite t, not a constant to Inductor), so
    the division stays a tensor-by-tensor IEEE division -- div_rn in Triton once emulate_divison_rounding is on."""
    return t / (t * 0.0 + c)


def _to_grid(q: torch.Tensor, g: Grid) -> torch.Tensor:
    """fp32 values already divided by their scale -> grid values (quant.quantize: int round+clip; float clip+cast)."""
    if g.kind == "int":
        return torch.round(q).clamp(-g.qmax, g.qmax)
    return snap_float(q, g)


def qdq_lastdim(x: torch.Tensor, g: Grid) -> torch.Tensor:
    """Per-row (last axis) symmetric absmax quantize -> dequantize, in x.dtype: quant.quantize / fake_quantize with
    contract_dims=1, scale = max(absmax, 1e-30) / qmax in fp32. Used for weights (rows = output channels) and for
    activations (rows = tokens)."""
    xf = x.float()
    scale = _div(xf.abs().amax(-1, keepdim=True).clamp_min(1e-30), g.qmax)
    return (_to_grid(xf / scale, g) * scale).to(x.dtype)


def qdq_embed_rows(x: torch.Tensor, g: Grid) -> torch.Tensor:
    """embed_tokens in the W8A8 sampler: quant.take() dequantizes the gathered rows as bf16(q) * bf16(scale) in bf16."""
    xf = x.float()
    scale = _div(xf.abs().amax(-1, keepdim=True).clamp_min(1e-30), g.qmax)
    q = _to_grid(xf / scale, g)
    return (q.to(torch.bfloat16) * scale.to(torch.bfloat16)).to(x.dtype)


def kv_qdq(u: torch.Tensor, g: Grid) -> torch.Tensor:
    """quant.kv_fake_quantize over the last axis (head_dim) of u = [..., heads, head_dim]: unit-scale e4m3, scaled
    mini-float (scale = max(absmax / gmax, 1e-9)) or int-N (scale = max(absmax / (2^(N-1)-1), 1e-9), round-half-even,
    no clip). Computed in fp32, returned in u.dtype (the bf16 cache container)."""
    uf = u.float()
    if g.kind == "float" and g.unit:
        return snap_float(uf, g).to(u.dtype)
    scale = _div(uf.abs().amax(-1, keepdim=True), g.qmax).clamp_min(1e-9)
    if g.kind == "float":
        return (snap_float(uf / scale, g) * scale).to(u.dtype)
    return (torch.round(uf / scale) * scale).to(u.dtype)


def kv_qdq_heads(u: torch.Tensor, g: Grid, heads) -> torch.Tensor:
    """kv_qdq on the KV heads ``heads`` of u = [..., kv_heads, head_dim] only; every other head is returned bit for bit
    unchanged. Bitwise BY CONSTRUCTION: the snap is kv_qdq on the WHOLE tensor u -- the very
    expression the unmasked path computes, so the compiler sees the same program for it -- and the heads are then
    selected with torch.where(head_mask, snapped, u), a pure per-element select that cannot change any value's rounding.
    So heads = all is kv_qdq(u) bit for bit, and an unselected head is u bit for bit. (A per-head slice + cat form
    would give Inductor a different program to fuse, so it need not be bitwise equal to kv_qdq under compile.)
    The mask is built from a static arange and Python-int comparisons: no data-dependent control flow, no state on the
    module (nothing for a level-2 sleep to wipe), CUDA-graph safe."""
    sel = sorted(set(int(h) for h in heads))
    n = u.shape[-2]
    if not sel or sel[0] < 0 or sel[-1] >= n:
        raise ValueError(f"kv_qdq_heads: heads {sel} not within the {n} KV heads")
    idx = torch.arange(n, device=u.device).unsqueeze(-1)  # [kv_heads, 1], broadcasts over [..., kv_heads, head_dim]
    mask = idx == sel[0]
    for h in sel[1:]:
        mask = mask | (idx == h)
    return torch.where(mask, kv_qdq(u, g), u)


def is_weight_target(name: str, t: torch.Tensor) -> bool:
    """quant.quantize_tree / fake_quantize_tree skip rule, on HF names: ndim <= 1, 'norm', 'bias', MoE router -> skip."""
    return (
        t.dim() >= 2
        and t.is_floating_point()
        and "norm" not in name
        and "bias" not in name
        and not name.endswith("mlp.gate.weight")
    )


_MO_LINEAR = re.compile(r"(?:^|\.)layers\.\d+\.(?:self_attn|mlp)\.(?:q|k|v|o|gate|up|down)_proj\.weight$")


def is_mo_linear_weight(name: str, t: torch.Tensor) -> bool:
    """Model Optimizer's NVFP4 recipe targets (HF names): the decoder linear layers' weights -- never embed_tokens (not
    a linear layer), lm_head (disabled by the recipe), norms or biases."""
    return t.dim() == 2 and t.is_floating_point() and bool(_MO_LINEAR.search(name))


def load_static(path: str) -> dict:
    """The step-0 static file of the emulated NVFP4 sampler (bench/mo_export.py amax). Raises on anything malformed."""
    import hashlib
    import json

    with open(path, "rb") as f:
        raw = f.read()
    rec = json.loads(raw)
    if rec.get("format") != STATIC_FORMAT:
        raise ValueError(f"{path}: format {rec.get('format')!r} != {STATIC_FORMAT}")
    if rec.get("grid") not in STATIC_GRIDS:
        raise ValueError(f"{path}: grid {rec.get('grid')!r} not in {STATIC_GRIDS}")
    act = rec.get("act_amax")
    if rec["grid"] in ("nvfp4", "int8sq") and (not isinstance(act, dict) or not act):
        raise ValueError(f"{path}: no act_amax")
    if rec["grid"] in ("int8sq", "int4awq"):
        t = rec.get("tensors") or {}
        tp = t.get("path", "")
        if tp and not os.path.isabs(tp):
            tp = os.path.join(os.path.dirname(os.path.abspath(path)), tp)
        if not tp or not os.path.exists(tp):
            raise ValueError(f"{path}: tensors file {t.get('path')!r} missing")
        with open(tp, "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if got != t.get("sha256"):
            raise ValueError(f"{path}: tensors file sha256 {got[:16]} != the record's {str(t.get('sha256'))[:16]}")
        nr = rec.get("narrow_range")
        if nr != {"input": False, "weight": False}:   # specopd_smooth implements Model Optimizer's default full range
            raise ValueError(f"{path}: narrow_range {nr!r} != {{'input': False, 'weight': False}} (the emulation's)")
        rec["_tensors_path"] = tp
    for k, v in (act or {}).items():
        if isinstance(v, bool) or not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
            raise ValueError(f"{path}: act_amax[{k}] = {v!r} (must be a finite positive float)")
    kv = rec.get("kv")
    if kv is not None:
        for k, v in kv.items():
            if not (isinstance(v, dict) and all(isinstance(v.get(x), (int, float)) and v[x] > 0 for x in ("k", "v"))):
                raise ValueError(f"{path}: kv[{k}] = {v!r}")
    rec["_sha256"] = hashlib.sha256(raw).hexdigest()
    rec["_path"] = path
    return rec


_STATIC_TENSORS: dict = {}


def static_tensors(rec: dict, device="cuda") -> dict:
    """{hf linear: (pqs bf16 [in], inv float32 [in])} of a smoothed static file, loaded once per process and device."""
    key = (rec["_tensors_path"], rec["tensors"]["sha256"], str(device))
    if key not in _STATIC_TENSORS:
        from safetensors.torch import load_file

        t = load_file(rec["_tensors_path"], device=str(device))
        names = sorted({k.rsplit(".", 1)[0] for k in t})
        bad = [n for n in names if f"{n}.pqs" not in t or f"{n}.inv" not in t]
        if bad:
            raise ValueError(f"{rec['_tensors_path']}: incomplete pqs / inv for {bad[:3]}")
        for n in names:
            if t[f"{n}.pqs"].dtype != torch.bfloat16 or t[f"{n}.inv"].dtype != torch.float32:
                raise ValueError(f"{n}: pqs {t[f'{n}.pqs'].dtype} / inv {t[f'{n}.inv'].dtype} (want bf16 / float32)")
        _STATIC_TENSORS[key] = {n: (t[f"{n}.pqs"], t[f"{n}.inv"]) for n in names}
    return _STATIC_TENSORS[key]


def is_embed(name: str) -> bool:
    return name.endswith("embed_tokens.weight")


def qdq_weight(name: str, w: torch.Tensor, g: Grid, w8a8: bool, chunk_elems: int = 1 << 25, stats=None):
    """A NEW tensor (the stream buffer is never written) with ``w`` quantized per output row, row-chunked so the fp32
    temporaries stay ~<= 0.5 GB even for the 152k x 3584 embedding / lm_head. ``stats`` (a 2-list) accumulates
    sum (wq - w)^2 and sum w^2 in fp64 for the relative RMS error audit."""
    if w.dim() != 2:
        raise NotImplementedError(f"{name}: {w.dim()}-D weight; only 2-D [out, in] (dense HF layout) is supported")
    fn = qdq_embed_rows if (w8a8 and is_embed(name)) else qdq_lastdim
    out = torch.empty_like(w)
    step = max(1, chunk_elems // max(1, w.shape[1]))
    for r0 in range(0, w.shape[0], step):
        src = w[r0 : r0 + step]
        dq = fn(src, g)
        out[r0 : r0 + step] = dq
        if stats is not None:
            sf = src.float()
            stats[0] += float(torch.sum((dq.float() - sf) ** 2, dtype=torch.float64))
            stats[1] += float(torch.sum(sf * sf, dtype=torch.float64))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Weights at every sync.
# ----------------------------------------------------------------------------------------------------------------------


def _same_model(engine_model: str, guard: str) -> bool:
    a, b = str(engine_model).rstrip("/"), str(guard).rstrip("/")
    if a == b:
        return True
    try:
        return os.path.realpath(a) == os.path.realpath(b)
    except Exception:  # noqa: BLE001
        return False


def spec_for_engine(vllm_config) -> Optional[QuantSpec]:
    """The spec this engine must run, or None (unset / empty / not the sampler model). Raises on a bad spec."""
    raw = os.environ.get(ENV, "")
    if not raw.strip():
        return None
    spec = QuantSpec.parse(raw)
    if spec.empty:
        return None
    guard = os.environ.get(ENV_MODEL, "")
    if not guard:
        raise RuntimeError(f"{ENV}={raw!r} is set but {ENV_MODEL} is not: refusing to guess which engine is the sampler")
    model = vllm_config.model_config.model
    if not _same_model(model, guard):
        if os.environ.get(ENV_ALLOW_SKIP, "") == "1":
            logger.warning(
                "[SPECOPD-QUANT-CFG] SKIP engine model=%s: not the sampler model %s=%s (%s=1) -> NOT patched",
                model, ENV_MODEL, guard, ENV_ALLOW_SKIP,
            )
            return None
        raise RuntimeError(
            f"[SPECOPD-QUANT] {ENV}={raw!r} is set in an engine whose model {model} is not {ENV_MODEL}={guard}: "
            f"refusing to run it unquantized (fail-closed; set {ENV_ALLOW_SKIP}=1 only for a deliberate skip)"
        )
    return spec


def _smooth_weight(name: str, w: torch.Tensor, grid: str, tens: dict, stats) -> torch.Tensor:
    """Model Optimizer's smoothed integer weight of one streamed HF weight (bf16): W' = (W * inv)[bf16] with the fixed
    step-0 inv, then INT8 per output channel (int8sq) or INT4 per block of 128 (int4awq), fresh scales. A NEW tensor;
    the relative error is measured against W' (what the quantizer sees)."""
    from verl.utils import specopd_smooth as sm

    mod = name[: -len(".weight")]
    if mod not in tens:
        raise RuntimeError(f"[SPECOPD-QUANT] w={grid}: no step-0 smoothing for {mod}")
    inv = tens[mod][1]
    if inv.numel() != w.shape[1]:
        raise RuntimeError(f"[SPECOPD-QUANT] {mod}: smoothing of width {inv.numel()} for a weight of {w.shape[1]} inputs")
    wp = sm.smooth_weight(w, inv.to(w.device))
    wq = sm.qdq_weight_int8_rows(wp) if grid == "int8sq" else sm.qdq_weight_int4_blocks(wp)
    stats[0] += float(torch.sum((wq.float() - wp.float()) ** 2, dtype=torch.float64))
    stats[1] += float(torch.sum(wp.float() * wp.float(), dtype=torch.float64))
    return wq


def _nvfp4_weight(name: str, w: torch.Tensor, stats) -> torch.Tensor:
    """Model Optimizer's NVFP4 weight fake quantization of one streamed HF weight (bf16, CUDA): a NEW tensor."""
    from verl.utils import specopd_nvfp4

    wq = specopd_nvfp4.nvfp4_qdq_weight(w)
    stats[0] += float(torch.sum((wq.float() - w.float()) ** 2, dtype=torch.float64))
    stats[1] += float(torch.sum(w.float() * w.float(), dtype=torch.float64))
    return wq


class SamplerQuant:
    """Per-worker weight fake-quant applied to every bucket of the actor stream (update_weights_from_ipc), after the
    optional SamplerNoise offset (Score Centering's prepare_sampler_weights: weights + delta, then quantize)."""

    def __init__(self, spec: QuantSpec):
        if spec.w is None:
            raise ValueError("SamplerQuant needs a w= grid")
        self.spec = spec
        self.sync = 0
        self._tens = None
        if spec.smooth:   # the fixed step-0 smoothing (tim_calibrate builds SamplerQuant without an engine config)
            path = os.environ.get(ENV_STATIC, "").strip()
            if not path:
                raise RuntimeError(f"[SPECOPD-QUANT] w={spec.smooth} needs {ENV_STATIC}")
            st = load_static(path)
            if st["grid"] != spec.smooth:
                raise RuntimeError(f"[SPECOPD-QUANT] {path} is grid {st['grid']}, the spec w={spec.smooth}")
            self._tens = static_tensors(st, "cuda" if torch.cuda.is_available() else "cpu")
        self._reset()

    @classmethod
    def from_env(cls, vllm_config) -> Optional["SamplerQuant"]:
        spec = spec_for_engine(vllm_config)
        return cls(spec) if spec is not None and spec.w is not None else None

    def _reset(self):
        self._n_tensors = 0
        self._n_elems = 0
        self._err2 = 0.0
        self._w2 = 0.0
        self._names: list[str] = []
        self._sums: list[float] = []  # float64 sum of every quantized tensor: the sync line's weight checksum
        self._t0 = None

    def apply(self, weights):
        """Quantize the bucket's target weights. The FSDP stream carries fp32 masters (bucketed_weight_transfer: "model
        parameters are in fp32 full precision"); Score Centering quantizes a bf16 COPY (train_rl.copy_sampler_weights casts to
        bf16, then quantize_tree), so the target is cast to bf16 first -- a no-op for a bf16 stream / checkpoint."""
        if self._t0 is None:
            self._t0 = time.perf_counter()
        out = []
        nv = self.spec.nvfp4
        sm = self.spec.smooth
        for name, w in weights:
            if not (is_mo_linear_weight(name, w) if (nv or sm) else is_weight_target(name, w)):
                out.append((name, w))
                continue
            if w.dtype != torch.bfloat16:
                w = w.to(torch.bfloat16)
            st = [0.0, 0.0]
            if nv:
                wq = _nvfp4_weight(name, w, st)
            elif sm:
                wq = _smooth_weight(name, w, sm, self._tens, st)
            else:
                wq = qdq_weight(name, w, self.spec.w, self.spec.w8a8, stats=st)
            self._err2 += st[0]
            self._w2 += st[1]
            self._sums.append(float(torch.sum(wq, dtype=torch.float64)))
            self._n_tensors += 1
            self._n_elems += w.numel()
            if len(self._names) < 4 or is_embed(name) or name.endswith("lm_head.weight"):
                self._names.append(name)
            out.append((name, wq))
        return out

    def finish_sync(self) -> dict:
        rec = dict(
            sync=self.sync, w=self.spec.w.name, spec=self.spec.canonical(),
            embed_dequant=("none (decoder linear layers only)" if self.spec.static_grid
                           else "bf16(q)*bf16(s)" if self.spec.w8a8 else "bf16(q*s)"),
            n_tensors=self._n_tensors, n_elems=self._n_elems,
            rel_rms_err=math.sqrt(self._err2 / self._w2) if self._w2 > 0 else float("nan"),
            ms=(time.perf_counter() - self._t0) * 1e3 if self._t0 is not None else 0.0,
            wsum=hashlib.sha256(repr(self._sums).encode()).hexdigest()[:16],
        )
        logger.warning(
            "[SPECOPD-QUANT] sync=%d spec=%s w=%s embed_dequant=%s n_tensors=%d n_elems=%d rel_rms_err=%.4e "
            "ms=%.0f wsum=%s names=%s",
            rec["sync"], rec["spec"], rec["w"], rec["embed_dequant"], rec["n_tensors"], rec["n_elems"],
            rec["rel_rms_err"], rec["ms"], rec["wsum"], ",".join(self._names[:8]),
        )
        self.sync += 1
        self._reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return rec


# ----------------------------------------------------------------------------------------------------------------------
# Activations / KV cache: engine patches (vLLM 0.19.1). Installed in the worker process before load_model; attached
# per MODULE of the target model after load_model (so the drafter and any other engine in the process stay clean).
# ----------------------------------------------------------------------------------------------------------------------

_ENGINE_SPECS: dict[int, QuantSpec] = {}  # id(vllm_config) -> spec, for the engines whose target model gets patched
_ENGINE_STATIC: dict[int, dict] = {}  # id(vllm_config) -> the w=nvfp4 static file (activation amax, KV scales)
_LOAD_HOOKED = False
_ACT_CLASSES: dict[type, type] = {}
_KV_CLASSES: dict[type, type] = {}


def _act_class(base: type) -> type:
    cls = _ACT_CLASSES.get(base)
    if cls is None:

        def apply(self, layer, x, bias=None):
            return base.apply(self, layer, qdq_lastdim(x, self._specopd_a), bias)

        cls = type(f"SpecopdActQuant{base.__name__}", (base,), {"apply": apply, "_specopd_patched": True})
        _ACT_CLASSES[base] = cls
    return cls


_NVFP4_CLASSES: dict[type, type] = {}


def _nvfp4_act_class(base: type) -> type:
    """The linear method with Model Optimizer's NVFP4 input quantizer in front of the GEMM (static amax on the method)."""
    cls = _NVFP4_CLASSES.get(base)
    if cls is None:
        from verl.utils.specopd_nvfp4 import define_op

        op = define_op()  # registered before torch.compile traces the model

        def apply(self, layer, x, bias=None):
            return base.apply(self, layer, op(x, self._specopd_nvfp4_amax), bias)

        cls = type(f"SpecopdNvfp4Act{base.__name__}", (base,), {"apply": apply, "_specopd_patched": True})
        _NVFP4_CLASSES[base] = cls
    return cls


def _patch_nvfp4_linears(model, static: dict, rec: dict) -> None:
    """w=nvfp4,a=nvfp4: every decoder LinearBase gets the NVFP4 input quantizer with its shards' static amax."""
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    act = static["act_amax"]
    packed = dict(getattr(model, "packed_modules_mapping", {}) or {})
    used, widths = set(), set()
    rec.update(nvfp4_missing=[], nvfp4_disagree=[], nvfp4_outside=[])
    for name, m in model.named_modules():
        if not isinstance(m, LinearBase):
            continue
        if _layer_idx(name + ".") is None:
            rec["nvfp4_outside"].append(name)
            continue
        qm = getattr(m, "quant_method", None)
        if type(qm) is not UnquantizedLinearMethod:
            if not getattr(qm, "_specopd_patched", False):
                rec["unexpected"].append(f"{name}:{type(qm).__name__}")
            continue
        prefix, last = name.rsplit(".", 1)
        shards = [f"{prefix}.{s}" for s in packed.get(last, [last])]
        vals = [act.get(s) for s in shards]
        if any(v is None for v in vals):
            rec["nvfp4_missing"].append(name)
            continue
        if len({float(v) for v in vals}) != 1:
            rec["nvfp4_disagree"].append((name, [float(v) for v in vals]))
            continue
        qm.__class__ = _nvfp4_act_class(UnquantizedLinearMethod)
        qm._specopd_nvfp4_amax = float(vals[0])
        used.update(shards)
        widths.add(int(getattr(m, "input_size_per_partition", getattr(m, "input_size", 0))))
        rec["n_linear"] += 1
        rec["linear_layers"].append(_layer_idx(name + "."))
    rec["nvfp4_unused"] = sorted(set(act) - used)
    rec["nvfp4_widths"] = sorted(widths)


_SMOOTH_CLASSES: dict = {}


def _smooth_act_class(base: type, grid: str) -> type:
    """The linear method with Model Optimizer's activation smoothing (and, for int8sq, its INT8 per-tensor quantizer)
    in front of the GEMM. One shard (or shards with bitwise-equal smoothing): one GEMM on the smoothed input. Shards
    with different smoothing: one GEMM per shard on its own smoothed input, outputs concatenated (the HF model's
    separate q / k / v, gate / up projections)."""
    key = (base, grid)
    cls = _SMOOTH_CLASSES.get(key)
    if cls is None:
        import torch.nn.functional as F

        from verl.utils.specopd_smooth import define_ops

        mul, q8 = define_ops()

        def act(layer, x, i):
            s = getattr(layer, f"_specopd_pqs{i}")
            return q8(x, s, layer._specopd_amax) if grid == "int8sq" else mul(x, s)

        def apply(self, layer, x, bias=None):
            if layer._specopd_nshard == 1:
                return base.apply(self, layer, act(layer, x, 0), bias)
            outs, r0 = [], 0
            for i, w in enumerate(layer._specopd_widths):
                b = bias[r0:r0 + w] if bias is not None else None
                outs.append(F.linear(act(layer, x, i), layer.weight[r0:r0 + w], b))
                r0 += w
            return torch.cat(outs, dim=-1)

        cls = type(f"SpecopdSmooth{grid}{base.__name__}", (base,), {"apply": apply, "_specopd_patched": True})
        _SMOOTH_CLASSES[key] = cls
    return cls


def _patch_smooth_linears(model, static: dict, grid: str, rec: dict) -> None:
    """int8sq / int4awq: every decoder LinearBase gets its shards' fixed smoothing (buffers allocated after the model
    load, outside vLLM's sleep pool) and, for int8sq, the per-tensor activation amax."""
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    act = static.get("act_amax") or {}
    tens = static_tensors(static, next(model.parameters()).device)
    packed = dict(getattr(model, "packed_modules_mapping", {}) or {})
    used = set()
    rec.update(smooth_missing=[], smooth_disagree=[], smooth_outside=[], smooth_split=[], smooth_fused=0)
    for name, m in model.named_modules():
        if not isinstance(m, LinearBase):
            continue
        if _layer_idx(name + ".") is None:
            rec["smooth_outside"].append(name)
            continue
        qm = getattr(m, "quant_method", None)
        if type(qm) is not UnquantizedLinearMethod:
            if not getattr(qm, "_specopd_patched", False):
                rec["unexpected"].append(f"{name}:{type(qm).__name__}")
            continue
        prefix, last = name.rsplit(".", 1)
        shards = [f"{prefix}.{s}" for s in packed.get(last, [last])]
        if any(s not in tens for s in shards) or (grid == "int8sq" and any(s not in act for s in shards)):
            rec["smooth_missing"].append(name)
            continue
        if grid == "int8sq" and len({float(act[s]) for s in shards}) != 1:
            rec["smooth_disagree"].append((name, [float(act[s]) for s in shards]))
            continue
        pqs = [tens[s][0] for s in shards]
        widths = list(getattr(m, "output_partition_sizes", None) or getattr(m, "output_sizes", None) or [m.weight.shape[0]])
        if len(widths) != len(shards) or sum(widths) != m.weight.shape[0]:
            rec["smooth_missing"].append(f"{name}: widths {widths} for {len(shards)} shards / {m.weight.shape[0]} rows")
            continue
        same = all(torch.equal(p.view(torch.int16), pqs[0].view(torch.int16)) for p in pqs[1:])
        n = 1 if same else len(shards)
        for i in range(n):
            m.register_buffer(f"_specopd_pqs{i}", pqs[i].clone(), persistent=False)
        if grid == "int8sq":
            a = float(act[shards[0]])
            if not a > 2.0 ** -24:
                rec["smooth_missing"].append(f"{name}: activation amax {a}")
                continue
            m.register_buffer("_specopd_amax", torch.tensor(a, dtype=torch.float32, device=pqs[0].device), persistent=False)
        m._specopd_nshard = n
        m._specopd_widths = widths
        qm.__class__ = _smooth_act_class(UnquantizedLinearMethod, grid)
        used.update(shards)
        rec["n_linear"] += 1
        rec["linear_layers"].append(_layer_idx(name + "."))
        if n > 1:
            rec["smooth_split"].append(name)
        else:
            rec["smooth_fused"] += 1
    rec["smooth_unused"] = sorted(set(tens) - used)
    rec["smooth_split"] = rec["smooth_split"][:4] + ([f"... {len(rec['smooth_split'])} split"] if len(rec["smooth_split"]) > 4 else [])


def _warm_nvfp4(widths, device) -> list:
    """Compile the kernel's row-count specializations (1 / multiple of 16 / other) for every input width BEFORE any
    CUDA-graph capture (a Triton compile cannot run inside a capture)."""
    from verl.utils.specopd_nvfp4 import nvfp4_qdq

    done = []
    for n in widths:
        for m in (1, 3, 16):
            nvfp4_qdq(torch.zeros(m, n, dtype=torch.bfloat16, device=device), 1.0)
            done.append((m, n))
    torch.cuda.synchronize(device)
    return done


def _attach_static_kv(runner, model, static: dict) -> dict:
    """The recipe's FP8 cache: the file's per-layer k / v scales into every FP8 attention layer (tensors + float
    mirrors), now, after every weight sync and after every KV wake-up. vLLM 0.19.1 resets them to 1.0 in BOTH places:
    Attention.process_weights_after_loading (run after every weight load -- verl's sync, tim_calibrate's re-stream)
    calls set_default_quant_scales when no quant method loads KV scales (without the re-application a re-streamed
    engine would decode at scale 1.0), and init_fp8_kv_scales at every KV wake-up. Both are wrapped per layer / per
    runner to re-apply."""
    from vllm.model_executor.layers.attention.attention import Attention

    kv = {int(k): (float(v["k"]), float(v["v"])) for k, v in static["kv"].items()}
    layers = {}
    for name, m in model.named_modules():
        if isinstance(m, Attention) and str(getattr(m, "kv_cache_dtype", "auto")).startswith("fp8"):
            layers[_layer_idx(name + ".")] = m
    missing = sorted(i for i in layers if i not in kv)
    if not layers or missing:
        raise RuntimeError(f"[SPECOPD-QUANT] FP8 attention layers {sorted(layers)} lack file k/v scales for {missing}")

    def apply():
        with torch.no_grad():
            for i, m in layers.items():
                k, v = kv[i]
                m._k_scale.fill_(k)
                m._v_scale.fill_(v)
                m._k_scale_float = float(torch.tensor(k, dtype=torch.float32))
                m._v_scale_float = float(torch.tensor(v, dtype=torch.float32))

    def ok():
        for i, m in layers.items():
            k, v = (float(torch.tensor(x, dtype=torch.float32)) for x in kv[i])
            if float(m._k_scale) != k or float(m._v_scale) != v or m._k_scale_float != k or m._v_scale_float != v:
                return False
        return True

    apply()
    st = dict(wakes=0, reloads=0, n_layers=len(layers), unused=sorted(set(kv) - set(layers)), ok_at_load=ok())
    for i, m in layers.items():
        pwal = getattr(m, "process_weights_after_loading", None)
        if pwal is None or getattr(pwal, "_specopd_kv", False):
            continue

        def pwal_then_restore(*a, _orig=pwal, _m=m, _i=i, **k):
            out = _orig(*a, **k)
            kk, vv = kv[_i]
            with torch.no_grad():
                _m._k_scale.fill_(kk)
                _m._v_scale.fill_(vv)
            _m._k_scale_float = float(torch.tensor(kk, dtype=torch.float32))
            _m._v_scale_float = float(torch.tensor(vv, dtype=torch.float32))
            st["reloads"] += 1
            return out

        pwal_then_restore._specopd_kv = True
        m.process_weights_after_loading = pwal_then_restore
    st["reload_hook"] = True
    orig = getattr(runner, "init_fp8_kv_scales", None)
    if orig is not None and not getattr(orig, "_specopd_kv", False):

        def init_fp8_kv_scales_then_restore(*a, **k):
            out = orig(*a, **k)
            st["wakes"] += 1
            reset = sum(1 for m in layers.values() if float(m._k_scale) == 1.0 and float(m._v_scale) == 1.0)
            apply()
            m0 = layers[min(layers)]
            logger.warning(
                "[SPECOPD-QUANT-KV] wake=%d reset=%d/%d kv_ok=%d k0=%.9g v0=%.9g q0=%.9g prob0=%.9g",
                st["wakes"], reset, len(layers), int(ok()), float(m0._k_scale), float(m0._v_scale),
                float(m0._q_scale), float(m0._prob_scale),
            )
            return out

        init_fp8_kv_scales_then_restore._specopd_kv = True
        runner.init_fp8_kv_scales = init_fp8_kv_scales_then_restore
        st["wake_hook"] = True
    else:
        st["wake_hook"] = orig is not None
    return st


def _kv_class(base: type) -> type:
    cls = _KV_CLASSES.get(base)
    if cls is None:

        def forward(self, query, key, value, output_shape=None):
            g, part = self._specopd_kv, self._specopd_kv_part
            heads = getattr(self, "_specopd_kv_heads", None)  # kvl0heads: layer 0 only; None = every head
            if heads is None:
                if key is not None and "k" in part:
                    key = kv_qdq(key.reshape(-1, self.num_kv_heads, self.head_size), g).reshape(key.shape)
                if value is not None and "v" in part:
                    value = kv_qdq(value.reshape(-1, self.num_kv_heads, self.head_size_v), g).reshape(value.shape)
            else:
                if key is not None and "k" in part:
                    key = kv_qdq_heads(key.reshape(-1, self.num_kv_heads, self.head_size), g, heads).reshape(key.shape)
                if value is not None and "v" in part:
                    value = kv_qdq_heads(
                        value.reshape(-1, self.num_kv_heads, self.head_size_v), g, heads
                    ).reshape(value.shape)
            return base.forward(self, query, key, value, output_shape)

        cls = type(f"SpecopdKVQuant{base.__name__}", (base,), {"forward": forward, "_specopd_patched": True})
        _KV_CLASSES[base] = cls
    return cls


def _layer_idx(name: str) -> Optional[int]:
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _idx_summary(ids) -> str:
    ids = sorted(set(i for i in ids if i is not None))
    if not ids:
        return "-"
    return f"{ids[0]}..{ids[-1]}({len(ids)})" if ids == list(range(ids[0], ids[-1] + 1)) else ",".join(map(str, ids))


def patch_target_model(model, spec: QuantSpec, static: Optional[dict] = None) -> dict:
    """Attach the activation / KV quantizers to the TARGET model's own modules (instance-level class swaps; nothing
    global). Returns the counts that go into the [SPECOPD-QUANT-CFG] line."""
    rec = dict(n_linear=0, linear_layers=[], lm_head=0, n_attn=0, attn_layers=[], unexpected=[])
    if spec.smooth:
        if static is None:
            raise RuntimeError(f"[SPECOPD-QUANT] w={spec.smooth} needs {ENV_STATIC}")
        _patch_smooth_linears(model, static, spec.smooth, rec)
        if rec["unexpected"]:
            raise RuntimeError(f"[SPECOPD-QUANT] w={spec.smooth} needs unquantized layers; found {rec['unexpected'][:4]}")
        return rec
    if spec.nvfp4:
        # Model Optimizer's NVFP4 recipe: decoder linear layers only (lm_head disabled), no fake-quantized cache
        if spec.a is not None:
            if static is None:
                raise RuntimeError(f"[SPECOPD-QUANT] a=nvfp4 needs {ENV_STATIC}")
            _patch_nvfp4_linears(model, static, rec)
        if rec["unexpected"]:
            raise RuntimeError(f"[SPECOPD-QUANT] a=nvfp4 needs unquantized layers; found {rec['unexpected'][:4]}")
        return rec
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, UnquantizedEmbeddingMethod

    for name, m in model.named_modules():
        if spec.a is not None and isinstance(m, LinearBase):
            qm = getattr(m, "quant_method", None)
            if type(qm) is UnquantizedLinearMethod:
                qm.__class__ = _act_class(UnquantizedLinearMethod)
                qm._specopd_a = spec.a
                rec["n_linear"] += 1
                rec["linear_layers"].append(_layer_idx(name + "."))
            elif not getattr(qm, "_specopd_patched", False):
                rec["unexpected"].append(f"{name}:{type(qm).__name__}")
        elif spec.a is not None and isinstance(m, ParallelLMHead):
            qm = getattr(m, "quant_method", None)
            if type(qm) is UnquantizedEmbeddingMethod:
                qm.__class__ = _act_class(UnquantizedEmbeddingMethod)
                qm._specopd_a = spec.a
                rec["lm_head"] += 1
            elif not getattr(qm, "_specopd_patched", False):
                rec["unexpected"].append(f"{name}:{type(qm).__name__}")
        if spec.kv is not None and isinstance(m, Attention):
            li = _layer_idx(name + ".")
            heads = None
            if li in spec.kv_skip:
                rec.setdefault("attn_skipped", []).append(li)
                continue
            if li == 0 and spec.kv_l0_heads:
                # kvl0heads: layer 0 is head-masked -- only the heads in S are snapped
                n_kv = int(m.num_kv_heads)
                if max(spec.kv_l0_heads) >= n_kv:
                    raise RuntimeError(
                        f"[SPECOPD-QUANT] kvl0heads={':'.join(map(str, spec.kv_l0_heads))} names a head outside "
                        f"layer 0's {n_kv} KV heads (0..{n_kv - 1})"
                    )
                heads = spec.l0_heads
                rec.setdefault("attn_l0_partial", []).append((li, heads))
            if not getattr(m, "_specopd_patched", False):
                m.__class__ = _kv_class(type(m))
            m._specopd_kv = spec.kv
            m._specopd_kv_part = spec.kv_part
            if heads is not None:
                m._specopd_kv_heads = heads
                continue  # counted in attn_l0_partial, not as a fully snapped layer
            rec["n_attn"] += 1
            rec["attn_layers"].append(li)
    if spec.a is not None:
        # A TIED head (tie_word_embeddings: vLLM's Qwen2 / Qwen3 set lm_head = model.embed_tokens, a
        # VocabParallelEmbedding, not a ParallelLMHead) is that ONE module, so the loop above never sees an LM head. Its
        # logits GEMM is quant_method.apply (LogitsProcessor._get_logits) and its row gather is quant_method.embedding
        # (VocabParallelEmbedding.forward); the activation class overrides apply only, so the head's input is quantized
        # and the embedding lookup is untouched -- the untied path's arithmetic. The weight is one tensor, synced as
        # embed_tokens (vLLM skips lm_head.* when tied), so the head shares the embedding's dequantization. A drafter
        # that shares the tied module then counts as patched and check_coverage refuses it.
        lm = getattr(model, "lm_head", None)
        emb = getattr(getattr(model, "model", None), "embed_tokens", None)
        if lm is not None and lm is emb and not isinstance(lm, ParallelLMHead):
            qm = getattr(lm, "quant_method", None)
            if type(qm) is UnquantizedEmbeddingMethod:
                qm.__class__ = _act_class(UnquantizedEmbeddingMethod)
                qm._specopd_a = spec.a
                rec["lm_head"] += 1
                rec["lm_head_tied"] = True
            elif not getattr(qm, "_specopd_patched", False):
                rec["unexpected"].append(f"lm_head(tied embed_tokens):{type(qm).__name__}")
    if rec["unexpected"]:
        raise RuntimeError(f"[SPECOPD-QUANT] a=... needs unquantized layers; found {rec['unexpected'][:4]}")
    return rec


def check_coverage(rec: dict, drafter: Optional[dict], spec: QuantSpec, num_layers: Optional[int]) -> None:
    """Fail closed on a partial patch: every decoder layer's attention must carry the KV quantizer, exactly one LM head
    must carry the activation quantizer (a ParallelLMHead, or a TIED head = the embed_tokens module, patched by
    patch_target_model's tied branch), at least one linear layer must, and the speculative drafter must carry none
    (only shared modules -- embed_tokens -- are the target's; a drafter sharing a TIED, hence patched, head fails)."""
    errs = []
    if spec.kv is not None and num_layers is not None:
        skipped = sorted(rec.get("attn_skipped", []))
        partial = list(rec.get("attn_l0_partial", []))
        want_skipped = [i for i in spec.kv_skip if i < num_layers]
        if list(skipped) != want_skipped or any(i >= num_layers for i in spec.kv_skip):
            errs.append(f"kvskip={spec.kv_skip} but skipped attention layers {skipped} (model has {num_layers})")
        if partial != ([(0, spec.l0_heads)] if spec.kv_l0_heads else []):
            errs.append(f"kvl0heads={spec.l0_heads} but head-masked layers {partial}")
        if rec["n_attn"] != num_layers - len(skipped) - len(partial):
            masked = f", -{len(partial)} head-masked" if partial or spec.kv_l0_heads else ""
            errs.append(
                f"KV quantizer on {rec['n_attn']} attention layers, model has {num_layers} (-{len(skipped)}{masked})"
            )
    if spec.smooth:
        if rec["lm_head"] != 0:
            errs.append(f"w={spec.smooth} on {rec['lm_head']} LM heads (the recipe leaves lm_head bf16)")
        for key, what in (("smooth_missing", "no step-0 smoothing for"), ("smooth_disagree", "fused shards disagree on amax"),
                          ("smooth_outside", "linear layers outside the decoder layers"),
                          ("smooth_unused", "step-0 smoothing never used")):
            if rec.get(key):
                errs.append(f"{what}: {rec[key][:4]}")
        if rec["n_linear"] == 0 or (num_layers and rec["n_linear"] % num_layers):
            errs.append(f"w={spec.smooth} on {rec['n_linear']} linear layers ({num_layers} decoder layers)")
    elif spec.nvfp4 and spec.a is not None:
        if rec["lm_head"] != 0:
            errs.append(f"a=nvfp4 on {rec['lm_head']} LM heads (Model Optimizer's recipe leaves lm_head bf16)")
        for key, what in (("nvfp4_missing", "no static amax for"), ("nvfp4_disagree", "fused shards disagree on amax"),
                          ("nvfp4_outside", "linear layers outside the decoder layers"),
                          ("nvfp4_unused", "static amax never used")):
            if rec.get(key):
                errs.append(f"{what}: {rec[key][:4]}")
        if num_layers and rec["n_linear"] % num_layers:
            errs.append(f"a=nvfp4 on {rec['n_linear']} linear layers, not a multiple of the {num_layers} decoder layers")
    elif spec.a is not None and rec["lm_head"] != 1:
        errs.append(f"activation quantizer on {rec['lm_head']} LM heads (expected 1: ParallelLMHead or tied embed_tokens)")
    if spec.a is not None and rec["n_linear"] == 0:
        errs.append("activation quantizer on 0 linear layers")
    if drafter is not None and (drafter.get("patched_linear") or drafter.get("patched_attn")):
        errs.append(f"the speculative drafter carries the sampler quantizer: {drafter}")
    if errs:
        raise RuntimeError("[SPECOPD-QUANT] incomplete / wrong patch: " + "; ".join(errs))


def count_patched(model) -> dict:
    """How many of a model's linear / lm_head / attention modules carry a SPECOPD quantizer (the drafter must be 0)."""
    n_lin = n_attn = n_tot = 0
    for _, m in model.named_modules():
        qm = getattr(m, "quant_method", None)
        if qm is not None:
            n_tot += 1
            n_lin += int(bool(getattr(qm, "_specopd_patched", False)))
        if hasattr(m, "num_kv_heads") and hasattr(m, "kv_cache_dtype"):
            n_tot += 1
            n_attn += int(bool(getattr(m, "_specopd_patched", False)))
    return dict(modules=n_tot, patched_linear=n_lin, patched_attn=n_attn)


_PATCHED: dict[int, dict] = {}  # id(vllm_config) -> the PATCHED record (proof that _after_load ran for that engine)


def _after_load(runner, vllm_config):
    spec = _ENGINE_SPECS.get(id(vllm_config))
    if spec is None:
        return
    model = runner.get_model() if hasattr(runner, "get_model") else runner.model
    static = _ENGINE_STATIC.get(id(vllm_config))
    rec = patch_target_model(model, spec, static)
    if static is not None and str(vllm_config.cache_config.cache_dtype).startswith("fp8"):
        rec["static_kv"] = _attach_static_kv(runner, model, static)
    if rec.get("nvfp4_widths"):
        rec["nvfp4_warm"] = len(_warm_nvfp4(rec["nvfp4_widths"], next(model.parameters()).device))
    drafter = getattr(getattr(runner, "drafter", None), "model", None)
    if drafter is not None:
        target_ids = {id(m) for m in model.modules()}
        shared = [n for n, m in drafter.named_modules() if id(m) in target_ids and n]
        dr = count_patched(drafter)
        dr["shared_modules_with_target"] = shared[:6]
    else:
        dr = None
    mc = vllm_config.model_config
    hf = getattr(mc, "hf_text_config", None) or getattr(mc, "hf_config", None)
    check_coverage(rec, dr, spec, getattr(hf, "num_hidden_layers", None))
    vc = vllm_config
    cc = vc.compilation_config
    import torch._inductor.config as inductor_config
    from vllm import envs as vllm_envs

    _PATCHED[id(vllm_config)] = dict(rec, drafter=dr)
    if static is not None:
        logger.warning(
            "[SPECOPD-QUANT-CFG] %s static=%s sha=%s act_amax=%d used_linear=%d widths=%s warm=%s split=%s fused=%s kv=%s",
            static["grid"].upper(), static.get("_path"), static.get("_sha256", "")[:16], len(static.get("act_amax") or {}),
            rec["n_linear"], rec.get("nvfp4_widths"), rec.get("nvfp4_warm"), rec.get("smooth_split"),
            rec.get("smooth_fused"), rec.get("static_kv"),
        )

    logger.warning(
        "[SPECOPD-QUANT-CFG] PATCHED model=%s spec=%s | W=%s (applied at every weight sync, embed_dequant=%s) | "
        "A=%s on %d linear layers (layers %s) + lm_head=%d%s | KV=%s part=%s on %d attention layers (layers %s), "
        "head-masked %s, kv_cache_dtype=%s | drafter=%s | compile_mode=%s cudagraph_mode=%s enforce_eager=%s "
        "VLLM_DISABLE_COMPILE_CACHE=%s inductor_div_rn=%s tp=%d",
        vc.model_config.model, spec.canonical(),
        spec.w.name if spec.w else "bf16",
        "n/a (decoder linear layers only)" if spec.nvfp4 else "bf16(q)*bf16(s)" if spec.w8a8 else "bf16(q*s)",
        spec.a.name if spec.a else "bf16", rec["n_linear"], _idx_summary(rec["linear_layers"]), rec["lm_head"],
        " (tied embed_tokens)" if rec.get("lm_head_tied") else "",
        spec.kv.name if spec.kv else "bf16", spec.kv_part, rec["n_attn"], _idx_summary(rec["attn_layers"]),
        rec.get("attn_l0_partial", "-"), vc.cache_config.cache_dtype, dr, getattr(cc, "mode", None),
        getattr(cc, "cudagraph_mode", None),
        vc.model_config.enforce_eager, vllm_envs.VLLM_DISABLE_COMPILE_CACHE,
        getattr(inductor_config, "emulate_divison_rounding", None),
        vc.parallel_config.tensor_parallel_size,
    )


def _hook_load_model():
    """Wrap vLLM's GPU Worker.load_model (covers the V1 and V2 model runners): attach right after the target and the
    drafter are built and BEFORE determine_available_memory / compile_or_warm_up_model run the first forward."""
    global _LOAD_HOOKED
    if _LOAD_HOOKED:
        return
    from vllm.v1.worker.gpu_worker import Worker

    orig = Worker.load_model

    def load_model(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        _after_load(self.model_runner, self.vllm_config)
        return out

    load_model._specopd_orig = orig
    Worker.load_model = load_model
    _LOAD_HOOKED = True


def assert_engine_patched(vllm_config) -> None:
    """Raise if this engine was registered for activation / KV quant but the post-load attach never ran (a vLLM path
    that bypassed Worker.load_model would otherwise run the sampler silently unpatched)."""
    if id(vllm_config) in _ENGINE_SPECS and id(vllm_config) not in _PATCHED:
        raise RuntimeError("[SPECOPD-QUANT] engine registered for A/KV quant but its model was never patched")


def install_engine_patches(vllm_config) -> Optional[QuantSpec]:
    """Called in the vLLM worker process BEFORE the model is built (vLLMColocateWorkerExtension.__new__, or offline
    tools before LLM()). No-op (returns None) unless SPECOPD_SAMPLER_QUANT is set AND this engine's model is
    SPECOPD_SAMPLER_QUANT_MODEL. Raises on a configuration that would silently run the wrong sampler."""
    spec = spec_for_engine(vllm_config)
    if spec is None:
        return None
    mc, cc = vllm_config.model_config, vllm_config.compilation_config
    if (spec.w is not None or spec.a is not None) and (
        getattr(vllm_config, "quant_config", None) is not None or getattr(mc, "quantization", None)
    ):
        raise RuntimeError(f"[SPECOPD-QUANT] {spec.canonical()} on top of vLLM quantization={mc.quantization}: pick one")
    if spec.kv is not None and vllm_config.cache_config.cache_dtype != "auto":
        raise RuntimeError(
            f"[SPECOPD-QUANT] kv={spec.kv.name} emulates Score Centering's bf16-container KV fake-quant; it needs "
            f"kv_cache_dtype=auto (got {vllm_config.cache_config.cache_dtype})"
        )
    cache_fp8 = str(vllm_config.cache_config.cache_dtype).startswith("fp8")
    static = None
    if os.environ.get(ENV_STATIC, "").strip() and not spec.static_grid:
        raise RuntimeError(f"[SPECOPD-QUANT] {ENV_STATIC} is set but the spec {spec.canonical()} is not w=nvfp4 / int8sq / int4awq")
    if (spec.nvfp4 and (spec.a is not None or cache_fp8)) or spec.smooth:
        path = os.environ.get(ENV_STATIC, "").strip()
        if not path:
            raise RuntimeError(
                f"[SPECOPD-QUANT] {spec.canonical()} needs {ENV_STATIC}=<static file> (activation amax"
                f"{' and the FP8 cache scales' if cache_fp8 else ''})"
            )
        static = load_static(path)
        if static["grid"] != spec.w.kind:
            raise RuntimeError(f"[SPECOPD-QUANT] {path} is grid {static['grid']}, the spec {spec.canonical()}")
        if static.get("model") and not _same_model(mc.model, static["model"]):
            raise RuntimeError(f"[SPECOPD-QUANT] {path} was made for {static['model']}, this engine serves {mc.model}")
        if cache_fp8 and not static.get("kv"):
            raise RuntimeError(f"[SPECOPD-QUANT] FP8 cache under {spec.canonical()} but {path} has no kv scales "
                               "(a unit-scale cache is not Model Optimizer's recipe)")
        if cache_fp8 and getattr(vllm_config.cache_config, "calculate_kv_scales", False):
            raise RuntimeError("[SPECOPD-QUANT] w=nvfp4 uses the file's KV scales: calculate_kv_scales must be off")
        _ENGINE_STATIC[id(vllm_config)] = static
    tp = vllm_config.parallel_config.tensor_parallel_size
    if spec.static_grid and tp != 1:
        raise RuntimeError(f"[SPECOPD-QUANT] {spec.canonical()} needs tensor_parallel_size=1 (per-tensor amax / smoothing of a whole layer; got {tp})")
    if spec.kv_l0_heads and tp != 1:
        # kvl0heads names GLOBAL KV heads; under TP each rank holds a slice and num_kv_heads is per rank
        raise RuntimeError(f"[SPECOPD-QUANT] kvl0heads needs tensor_parallel_size=1 (got {tp})")
    if spec.a is not None and tp != 1:
        logger.warning("[SPECOPD-QUANT-CFG] tp=%d: row-parallel inputs get a per-SHARD token absmax (Score Centering: full)", tp)
    # EVERY sampler-quant engine (W-only included, so every quantized setting shares the same compiled arithmetic):
    # (1) Triton lowers fp32 x / y to the APPROXIMATE div.full (Inductor's own note); the quantizers' x / scale must be
    #     IEEE-rounded like Score Centering's / eager's, or some elements land one grid step off. This makes every fp32
    #     division in this engine's compiled graphs div_rn (elsewhere: fp32-ulp-level effects, far below bf16).
    # (2) vLLM's AOT / Inductor caches are keyed without this module's code or that flag: a patched engine could load a
    #     CLEAN engine's compiled graph and silently run unpatched -> the compile cache must be off.
    import torch._inductor.config as inductor_config

    inductor_config.emulate_divison_rounding = True
    if not mc.enforce_eager:
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
        from vllm import envs as vllm_envs

        if not vllm_envs.VLLM_DISABLE_COMPILE_CACHE or vllm_envs.VLLM_USE_AOT_COMPILE:
            raise RuntimeError(
                "[SPECOPD-QUANT] a compiled sampler-quant engine needs VLLM_DISABLE_COMPILE_CACHE=1 set before vLLM "
                "starts (vllm.envs is already cached here)"
            )
    if isinstance(getattr(vllm_config, "additional_config", None), dict):
        vllm_config.additional_config["specopd_sampler_quant"] = spec.canonical()  # distinct compile hash
        if static is not None:
            vllm_config.additional_config["specopd_sampler_quant_static"] = static["_sha256"][:16]
    if spec.a is not None or spec.kv is not None or static is not None:
        _hook_load_model()
        _ENGINE_SPECS[id(vllm_config)] = spec
    logger.warning(
        "[SPECOPD-QUANT-CFG] INSTALL model=%s spec=%s (W at every sync: %s; A/KV attach after load_model: %s) pid=%d",
        mc.model, spec.canonical(), spec.w is not None, spec.a is not None or spec.kv is not None, os.getpid(),
    )
    return spec


def audit_weights(model, spec: QuantSpec, n_rows: int = 8) -> dict:
    """After a sync: read rows of the LIVE (fused) parameters back, undo the per-row scale and count the elements that
    are off the grid by more than bf16 storage explains (|q - grid(q)| > |q| * 2^-7). ~0 = quantized; an unquantized
    tensor shows most elements off-grid."""
    if spec.w is None:
        return {}
    g = spec.w
    res = {}
    picks = {}
    if spec.smooth:
        for n, p in model.named_parameters():
            if p.dim() != 2:
                continue
            for key in ("layers.0.self_attn.qkv_proj", "layers.0.self_attn.o_proj", "layers.0.mlp.gate_up_proj",
                        "mlp.down_proj", "embed_tokens", "lm_head"):
                if key in n and key not in picks:
                    picks[key] = (n, p)
        for key, (n, p) in picks.items():
            rows = p.detach()[:n_rows].float()
            if spec.smooth == "int4awq" and rows.shape[1] % 128 == 0:
                rows = rows.reshape(-1, 128)
            sc = rows.abs().amax(-1, keepdim=True).clamp_min(1e-30) / g.qmax
            q = rows / sc
            res[key] = round(float(((q - torch.round(q)).abs() > q.abs() * 2.0**-7 + 1e-6).float().mean()), 5)
        logger.warning("[SPECOPD-QUANT-AUDIT] w=%s off-grid fraction of live parameter rows (0 = quantized; "
                       "embed_tokens / lm_head are not targets): %s", g.name, res)
        return res
    if spec.nvfp4:
        from verl.utils.specopd_nvfp4 import on_grid_fraction_off

        for n, p in model.named_parameters():
            if p.dim() != 2:
                continue
            for key in ("layers.0.self_attn.qkv_proj", "layers.0.self_attn.o_proj", "layers.0.mlp.gate_up_proj",
                        "mlp.down_proj", "embed_tokens", "lm_head"):
                if key in n and key not in picks:
                    picks[key] = (n, p)
        for key, (n, p) in picks.items():
            res[key] = round(on_grid_fraction_off(p[:n_rows]), 5)
        logger.warning("[SPECOPD-QUANT-AUDIT] w=nvfp4 off-grid fraction of live parameter rows (0 = quantized; "
                       "embed_tokens / lm_head are not targets): %s", res)
        return res
    for n, p in model.named_parameters():
        if p.dim() != 2:
            continue
        for key in ("layers.0.self_attn.qkv_proj", "layers.0.self_attn.o_proj", "layers.0.mlp.gate_up_proj",
                    "mlp.down_proj", "embed_tokens", "lm_head"):
            if key in n and key not in picks:
                picks[key] = (n, p)
    for key, (n, p) in picks.items():
        rows = p.detach()[:n_rows].float()
        scale = rows.abs().amax(-1, keepdim=True).clamp_min(1e-30) / g.qmax
        q = rows / scale
        off = ((q - _to_grid(q, g)).abs() > q.abs() * 2.0**-7 + 1e-12).float().mean()
        res[key] = round(float(off), 5)
    logger.warning("[SPECOPD-QUANT-AUDIT] w=%s off-grid fraction of live parameter rows (0 = quantized): %s",
                   g.name, res)
    return res


def replica_env_overrides(is_sampler_engine: bool) -> dict:
    """Ray runtime_env vars for one vLLM server actor (vLLMReplica.launch_servers). Empty unless SPECOPD_SAMPLER_QUANT
    is set in this process: then teacher / reward engines get it blanked (never patched) and the sampler's engine gets
    VLLM_DISABLE_COMPILE_CACHE=1 (see install_engine_patches)."""
    if not os.environ.get(ENV, "").strip():
        return {}
    if not is_sampler_engine:
        return {ENV: "", ENV_MODEL: "", ENV_STATIC: ""}
    return {"VLLM_DISABLE_COMPILE_CACHE": "1"}
