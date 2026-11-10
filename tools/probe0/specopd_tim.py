# Copyright 2026 The authors (see README.md)
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
"""Training-inference mismatch (TIM) injection on the rollout SAMPLER only (opt-in: SPECOPD_SAMPLER_NOISE=<sigma>).

Question: does a sampler q != trainer p destabilise reverse-KL OPD the way it destabilises RL (Score Centering,
arXiv 2609.20807)? The trainer is never touched: the noise is added to the weight STREAM the rollout engine loads
after every sync (vllm_rollout/utils.py update_weights_from_ipc), so the FSDP actor, its log-probs, the advantages
and the checkpoints are all clean. Level-2 sleep discards the engine's weights every step and every sync streams
them again, so the offset is re-applied on every sync and never accumulates.

Noise spec (one per sync, identical on every replica):
  * TARGETS: every 2-D ``*.weight`` inside a transformer block (``.layers.``) = the q/k/v/o and gate/up/down
    projections. NOT the embedding, NOT lm_head, NOT norms or biases (1-D). This is exactly the set vLLM's fp8
    rollout quantizes (Linear layers in the blocks; lm_head/embeddings stay bf16), so the noise perturbs the same
    weights as the fp8 rollout. Names are the actor stream's HF names (model.layers.N.self_attn.q_proj.weight, ...),
    i.e. BEFORE vLLM fuses them into qkv_proj/gate_up_proj -- so an offline loader that feeds the HF checkpoint to
    ``model.load_weights`` gets the identical offset (bench/tim_calibrate.py).
  * OFFSET: W_noisy = W + sigma * s_name * eps_name, computed in fp32 and cast back to the stream dtype.
    eps_name ~ N(0, I) is regenerated each sync from a fixed seed + the parameter name (torch CUDA generator, same
    shape): a SYSTEMATIC offset, like quantization error, not fresh randomness. s_name = std(W_name) of the FIRST
    sync (the initial student), frozen, so the sampler's bias is one fixed vector n = sigma * s * eps for the whole
    run. (CUDA randn depends on the device's SM count: calibrate on the GPU type that trains.)
  * FEEDBACK-LOOP DIAGNOSTIC, computed on the CLEAN streamed weights W_t (after t optimizer steps):
        d_eps = sum_name <W_t, eps_name> / ||eps||            (projection on the unit eps direction)
        d_n   = sum_name s_name <W_t, eps_name> / ||n/sigma|| (projection on the unit direction of the ACTUAL offset)
    d_n(t) - d_n(0) is how far the trainer itself moved along the sampler's bias; divided by ||n|| (= sigma * nnorm1,
    logged) it is the fraction of the offset the trainer has absorbed. Positive = toward the sampler. With sigma=0
    nothing is added and the same projection is logged for the seed's eps as a scale reference (for runs without
    noise, e.g. a bf16 or an fp8 rollout).

One line per sync per replica:  [SPECOPD-TIM] step=<optimizer steps applied> sigma=.. seed=.. d_eps=.. d_n=..
nnorm1=.. n_tensors=.. n_elems=.. dtype=.. ms=..   (step 0 = the initial sync, before the first rollout).

Unset SPECOPD_SAMPLER_NOISE = the hook is never constructed: byte-identical behaviour.
"""

import hashlib
import logging
import math
import os
import time

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

DEFAULT_SEED = 20261001


def is_target(name: str, tensor: torch.Tensor) -> bool:
    """A transformer-block projection weight: 2-D, ``.layers.`` in the name, ends in ``.weight``."""
    return tensor.dim() == 2 and ".layers." in name and name.endswith(".weight") and tensor.is_floating_point()


def name_seed(seed: int, name: str) -> int:
    """Stable 63-bit seed for (run seed, parameter name). Python's hash() is salted per process -- never use it."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little") & ((1 << 63) - 1)


def make_eps(seed: int, name: str, shape, device) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(name_seed(seed, name))
    return torch.randn(shape, generator=gen, device=device, dtype=torch.float32)


class SamplerNoise:
    """Per-worker state: the frozen per-tensor scales, the sync counter and the running projection sums."""

    def __init__(self, sigma: float, seed: int = DEFAULT_SEED):
        if not (sigma >= 0.0 and math.isfinite(sigma)):
            raise ValueError(f"SPECOPD_SAMPLER_NOISE must be a finite sigma >= 0, got {sigma}")
        self.sigma = float(sigma)
        self.seed = int(seed)
        self.scales: dict[str, float] = {}  # s_name = std(W_name) at the first sync, frozen
        self.sync = 0
        self._reset()

    @classmethod
    def from_env(cls):
        raw = os.environ.get("SPECOPD_SAMPLER_NOISE", "")
        if raw == "":
            return None
        return cls(float(raw), int(os.environ.get("SPECOPD_SAMPLER_NOISE_SEED", DEFAULT_SEED)))

    def _reset(self):
        self._dot = 0.0  # sum <W, eps>
        self._sdot = 0.0  # sum s <W, eps>
        self._eps2 = 0.0  # sum ||eps||^2
        self._seps2 = 0.0  # sum s^2 ||eps||^2
        self._n_tensors = 0
        self._n_elems = 0
        self._dtype = None
        self._t0 = None

    def apply(self, weights):
        """One bucket of (name, tensor): returns it with every target replaced by a NOISY COPY (the bucket's
        tensors are views into the shared IPC buffer and are never written), accumulating the projection."""
        if self._t0 is None:
            self._t0 = time.perf_counter()
        out = []
        for name, w in weights:
            if not is_target(name, w):
                out.append((name, w))
                continue
            wf = w.to(torch.float32, copy=True)  # a COPY even for an fp32 stream: add_ below must not hit the buffer
            if name not in self.scales:
                self.scales[name] = float(wf.std())
            s = self.scales[name]
            eps = make_eps(self.seed, name, w.shape, w.device)
            dot = float(torch.dot(wf.reshape(-1).double(), eps.reshape(-1).double()))
            e2 = float(torch.dot(eps.reshape(-1), eps.reshape(-1)).double())
            self._dot += dot
            self._sdot += s * dot
            self._eps2 += e2
            self._seps2 += s * s * e2
            self._n_tensors += 1
            self._n_elems += w.numel()
            self._dtype = w.dtype
            if self.sigma > 0.0:
                out.append((name, wf.add_(eps, alpha=self.sigma * s).to(w.dtype)))
            else:
                out.append((name, w))
            del wf, eps
        return out

    def finish_sync(self) -> dict:
        """Log the sync's projection line, advance the counter, reset the sums. Returns the record."""
        rec = dict(
            step=self.sync,
            sigma=self.sigma,
            seed=self.seed,
            d_eps=self._dot / math.sqrt(self._eps2) if self._eps2 > 0 else float("nan"),
            d_n=self._sdot / math.sqrt(self._seps2) if self._seps2 > 0 else float("nan"),
            nnorm1=math.sqrt(self._seps2),
            n_tensors=self._n_tensors,
            n_elems=self._n_elems,
            dtype=str(self._dtype).replace("torch.", ""),
            ms=(time.perf_counter() - self._t0) * 1e3 if self._t0 is not None else 0.0,
        )
        logger.warning(
            "[SPECOPD-TIM] step=%d sigma=%g seed=%d d_eps=%.10e d_n=%.10e nnorm1=%.6e n_tensors=%d n_elems=%d "
            "dtype=%s ms=%.0f",
            rec["step"], rec["sigma"], rec["seed"], rec["d_eps"], rec["d_n"], rec["nnorm1"], rec["n_tensors"],
            rec["n_elems"], rec["dtype"], rec["ms"],
        )
        self.sync += 1
        self._reset()
        # The fp32 temporaries (<= ~1.6 GB for a 7B MLP matrix) live in torch's caching allocator, outside vLLM's
        # sleep-mode pool: hand them back before the KV cache wakes up.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return rec


def audit_engine(model_runner) -> None:
    """One [SPECOPD-TIM-CFG] line per worker after its FIRST sync: what the engine ACTUALLY runs, so a run's log shows
    its sampler (verl runs vLLM at WARN, and verl's own 'FP8 model detected' message is INFO-level -- invisible).
    Prints the quantization config vLLM resolved, the KV-cache dtype, and the live dtype of layer 0's parameters
    (an fp8 rollout shows float8_e4m3fn there; a bf16 one bfloat16). Never raises: it is an audit, not a gate."""
    try:
        vc = model_runner.vllm_config
        qc = vc.quant_config
        layer0 = {}
        for n, p in model_runner.model.named_parameters():
            if ".layers.0." in n:
                layer0[n.split(".layers.0.", 1)[1]] = str(p.dtype).replace("torch.", "")
        kv = getattr(model_runner, "kv_caches", None)
        kv_dtype = None
        if kv:
            first = kv[0]
            first = first[0] if isinstance(first, (list, tuple)) else first
            kv_dtype = str(getattr(first, "dtype", None)).replace("torch.", "")
        logger.warning(
            "[SPECOPD-TIM-CFG] quantization=%s quant_config=%s cache_dtype=%s kv_tensor_dtype=%s layer0=%s",
            getattr(vc.model_config, "quantization", None), type(qc).__name__ if qc is not None else None,
            vc.cache_config.cache_dtype, kv_dtype, ",".join(f"{k}:{v}" for k, v in sorted(layer0.items())),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[SPECOPD-TIM-CFG] audit failed: %r", e)
