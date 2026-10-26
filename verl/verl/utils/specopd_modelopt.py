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
"""NVIDIA Model Optimizer quantization of the SAMPLER (the student's rollout engine) during training, with vLLM's real
kernels (launcher knob SAMPLER_MODELOPT). Opt-in: nothing here is imported unless
SPECOPD_SAMPLER_MODELOPT is set, so every other run is unchanged.

Recipe fp8 = Model Optimizer's FP8 per-tensor W8A8 recipe (static per-tensor activation scales) with its calibrated FP8
KV cache (`general/ptq/fp8_default-kv_fp8`), served by vLLM 0.19.1's ModelOptFp8LinearMethod and FP8 attention cache.

Step 0 (offline, bench/mo_export.py + bench/mo_scales.py): Model Optimizer calibrates and exports the student; the
scale file (SPECOPD_SAMPLER_MODELOPT_SCALES, JSON, format "specopd_modelopt/v1") pins the exported quantization config,
every fused linear layer's static input scale (vLLM's max over its shards, as ModelOptFp8LinearMethod takes it) and
every layer's k / v cache scale.

The engine (vllm_async_server._apply_quantization): the bf16 student, quantization="modelopt" with the exported config
as hf_overrides.quantization_config, kv_cache_dtype=fp8 (launcher). Its first weight sync, before any rollout, writes
everything below.

At EVERY weight sync (vLLMColocateWorkerExtension.update_weights_from_ipc):
  * each streamed bf16 weight of a quantized linear shard is quantized with Model Optimizer's own FP8 numerics
    (mo_fp8_weight: amax = max|W| of the current weights, scale = amax / 448 as Model Optimizer computes it on the same
    device, q = (W / scale) to float8_e4m3fn -- export_hf_checkpoint's `_export_quantized_weight` +
    `to_quantized_weight`), the shards of a fused layer are concatenated and, if their scales differ, re-rounded
    to the largest with vLLM's own requantize_with_max_scale (what ModelOptFp8LinearMethod.process_weights_after_loading
    does to an export), and the result is written INTO the live parameters (same storage, so CUDA graphs stay valid);
  * every other tensor (biases, norms, embeddings, the output layer) goes through vLLM's normal load;
  * vLLM's own post-processing of these linear layers is suppressed for the sync (it would transpose them again);
  * the static input scales are rewritten from the file (a level-2 sleep discards parameters that the
    stream does not carry), and the k / v scales too.
After EVERY KV-cache wake-up (model_runner.init_fp8_kv_scales, wrapped): vLLM resets every attention
layer's _k_scale / _v_scale tensors to 1.0 there; the file's values are written back (tensors and float mirrors).
The query and probability scales stay vLLM's 1.0 (as when vLLM serves this checkpoint).

Fresh at every sync: the FP8 weights and their weight scales. Fixed from step 0: the activation scales and the KV
scales (stale by design, as when serving a fixed checkpoint).

Audit lines (never raise):
  [SPECOPD-MO] rid=<host:pid> sync=<n> step=<s> recipe=fp8 layers=<written>/<all> wsum=<hash of per-layer byte sums> changed=<layers
      whose bytes changed since the previous sync> requant=<fused layers re-rounded to the max shard scale>
      wscale=<min>:<max> in_ok=<static scales == file> kv_ok=<k/v tensors+floats == file> relerr=<relative
      dequantization error of the first layer-0 linear layer written> ms=<t>
  [SPECOPD-MO-KV] rid=<host:pid> wake=<n> sync=<n> reset=<layers vLLM had set to 1.0> kv_ok=<after re-applying> k0=..,v0=..,q0=..,prob0=..
Errors found during the stream are raised at the end of the sync (a raise mid-stream would block the sender).
"""
import contextlib
import hashlib
import json
import logging
import os
import re
import socket
import time

logger = logging.getLogger(__name__)

ENV = "SPECOPD_SAMPLER_MODELOPT"
ENV_SCALES = "SPECOPD_SAMPLER_MODELOPT_SCALES"
ENV_MODEL = "SPECOPD_SAMPLER_MODELOPT_MODEL"
FORMAT = "specopd_modelopt/v1"
RECIPES = ("fp8", "fp8fl")   # fp8fl: the export with the first and last decoder layers unquantized (cache bf16)
FP8_MAX = 448.0
TAG, TAG_KV = "[SPECOPD-MO]", "[SPECOPD-MO-KV]"
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def enabled():
    return bool(os.environ.get(ENV, "").strip())


def load_scales(path):
    with open(path) as f:
        rec = json.load(f)
    if rec.get("format") != FORMAT:
        raise ValueError(f"{path}: format {rec.get('format')!r} != {FORMAT}")
    if rec.get("recipe") not in RECIPES:
        raise ValueError(f"{path}: recipe {rec.get('recipe')!r} not in {RECIPES}")
    for k in ("quantization_config", "input_scale", "kv"):
        if not rec.get(k):
            raise ValueError(f"{path}: missing {k}")
    return rec


def server_quantization(model_path):
    """(quantization, quantization_config) for the policy's rollout server, or None when the knob is off. Fail-closed:
    the recipe must match the file and the engine must be the model the file was made for."""
    if not enabled():
        return None
    recipe = os.environ[ENV].strip()
    rec = load_scales(os.environ[ENV_SCALES])
    if rec["recipe"] != recipe:
        raise ValueError(f"{ENV}={recipe} but the scale file is recipe {rec['recipe']}")
    want = os.environ.get(ENV_MODEL, "")
    if os.path.realpath(model_path.rstrip("/")) != os.path.realpath(want.rstrip("/")):
        raise RuntimeError(f"{ENV} is set for {want!r} but this engine serves {model_path!r} (refusing)")
    return "modelopt", dict(rec["quantization_config"])


def replica_env_overrides(is_sampler_engine):
    """Ray runtime_env vars of one vLLM server actor: teacher / reward engines get the knob blanked."""
    if not enabled() or is_sampler_engine:
        return {}
    return {ENV: "", ENV_SCALES: "", ENV_MODEL: ""}


# --------------------------------------------------------------------------------------------- Model Optimizer numerics
def mo_fp8_weight(w):
    """Model Optimizer's per-tensor FP8 export of one bf16 weight (modelopt 0.47.0 unified_export_hf
    `_export_quantized_weight` + quant_utils.to_quantized_weight), the SAME expressions on the SAME device: amax = max|W|
    (the max calibrator's per-tensor amax, in W's dtype, not 1-dim) cast to float32; scale = amax / 448 as a tensor op
    (`torch.tensor(weight_quantizer.amax / weight_quantizer.maxbound)`: on CUDA a division by a Python scalar such as
    `amax.item() / 448` is a multiplication by fp32(1/448), which can move the scale by one ulp against an exact
    division; on CPU both are exact); q = (W.to(bf16) / scale).to(float8_e4m3fn) with the scale a 0-dim tensor on
    W's device (an element-wise division). Returns (q, scale)."""
    import torch

    amax = w.abs().amax().to(torch.float32)
    scale = amax / FP8_MAX
    q = (w.to(torch.bfloat16) / scale).to(torch.float8_e4m3fn)
    return q, scale


def fuse_shards(qs, scales, widths):
    """Concatenate the shards and do what ModelOptFp8LinearMethod.process_weights_after_loading does to them: if the
    shard scales differ, re-round every shard to the largest (vLLM's requantize_with_max_scale, imported). Returns
    (weight [out, in] float8, weight_scale 0-dim float32, requantized?)."""
    import torch

    w = torch.cat(qs, 0)
    s = torch.stack([x.reshape(()) for x in scales]).to(torch.float32)
    if not bool((s == s[0]).all()):
        from vllm.model_executor.layers.quantization.utils.w8a8_utils import requantize_with_max_scale

        mx, w = requantize_with_max_scale(w, s, list(widths))
        return w, mx.reshape(()), True
    return w, s.max().reshape(()), False


def _f(x):
    return "%.9g" % x


class SamplerModelOpt:
    """One per rollout worker (TP1). See the module docstring."""

    def __init__(self, model_runner, scales, recipe="fp8"):
        import torch

        self.recipe, self.scales = recipe, scales
        self.model_runner = model_runner
        vc = model_runner.vllm_config
        tp = vc.parallel_config.tensor_parallel_size
        if tp != 1:
            raise RuntimeError(f"SAMPLER_MODELOPT is built for rollout TP=1 (got {tp})")
        qc = type(getattr(vc, "quant_config", None)).__name__
        if qc != "ModelOptFp8Config":
            raise RuntimeError(f"SAMPLER_MODELOPT={recipe} needs a ModelOptFp8Config engine (got {qc})")
        want_skip = [str(i) for i in scales.get("kv_bf16_layers", [])]
        got_skip = [str(x) for x in (getattr(vc.cache_config, "kv_cache_dtype_skip_layers", None) or [])]
        if sorted(got_skip) != sorted(want_skip):
            raise RuntimeError(f"SAMPLER_MODELOPT={recipe}: the engine keeps layers {got_skip} in bf16, the scale file "
                               f"{want_skip} (kv_cache_dtype_skip_layers must match the export)")
        if not str(vc.cache_config.cache_dtype).startswith("fp8"):
            raise RuntimeError(f"SAMPLER_MODELOPT=fp8 needs the FP8 KV cache (got {vc.cache_config.cache_dtype})")
        model = model_runner.get_model()
        packed = dict(getattr(model, "packed_modules_mapping", {}) or {})
        self.layers = {}
        for name, m in model.named_modules():
            if type(getattr(m, "quant_method", None)).__name__ != "ModelOptFp8LinearMethod":
                continue
            prefix, last = name.rsplit(".", 1)
            shards = [f"{prefix}.{s}.weight" for s in packed.get(last, [last])]
            widths = list(getattr(m, "logical_widths", []))
            if len(shards) != len(widths):
                raise RuntimeError(f"{name}: {len(shards)} shard names for logical widths {widths}")
            self.layers[name] = dict(m=m, shards=shards, widths=widths)
        missing = sorted(set(self.layers) ^ set(scales["input_scale"]))
        if missing:
            raise RuntimeError(f"quantized linear layers differ from the scale file: {missing[:4]} ...")
        self.shard_to_layer = {s: n for n, L in self.layers.items() for s in L["shards"]}
        try:
            from vllm.model_executor.layers.attention.attention import Attention
        except Exception:  # noqa: BLE001
            Attention = ()

        self.attn = {}
        for name, m in model.named_modules():
            is_attn = (Attention and isinstance(m, Attention)) or type(m).__name__ == "Attention"
            if is_attn and str(getattr(m, "kv_cache_dtype", "auto")).startswith("fp8"):
                mm = _LAYER.search(name)
                self.attn[int(mm.group(1))] = m
        if sorted(self.attn) != sorted(int(i) for i in scales["kv"]):
            raise RuntimeError(f"FP8 attention layers {sorted(self.attn)} != the file's {sorted(scales['kv'])[:4]}...")
        self.kv = {i: (float(v["k"]), float(v["v"])) for i, v in ((int(k), v) for k, v in scales["kv"].items())}
        self.device = next(iter(self.layers.values()))["m"].weight.device
        self.sync = 0
        self.wakes = 0
        self.rid = "%s:%d" % (socket.gethostname(), os.getpid())
        self.prev_sums = {}
        self._begin()
        self._torch = torch

    @classmethod
    def from_env(cls, model_runner):
        if not enabled():
            return None
        recipe = os.environ[ENV].strip()
        rec = load_scales(os.environ[ENV_SCALES])
        if rec["recipe"] != recipe:
            raise ValueError(f"{ENV}={recipe} but the scale file is recipe {rec['recipe']}")
        self = cls(model_runner, rec, recipe)
        self.install_wake_hook()
        return self

    # ------------------------------------------------------------------------------------------------ per sync
    def _begin(self):
        self.pending, self.written, self.errors = {}, set(), []
        self.n_requant, self.wscales, self.t0 = 0, [], time.time()
        self.sentinel = None

    def apply(self, bucket):
        """Takes the bucket's quantized-layer shards (writing a fused layer once all its shards have arrived) and
        returns the other tensors for vLLM's normal load. Never raises (errors are raised by finish_sync)."""
        out = []
        for name, t in bucket:
            ln = self.shard_to_layer.get(name)
            if ln is None:
                out.append((name, t))
                continue
            try:
                p = self.pending.setdefault(ln, {})
                p[name] = t.detach().to(self.device).clone()    # the IPC bucket buffer is reused
                if len(p) == len(self.layers[ln]["shards"]):
                    self._write(ln, self.pending.pop(ln))
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"{name}: {type(e).__name__}: {str(e)[:240]}")
        return out

    def _write(self, ln, shards):
        import torch

        L = self.layers[ln]
        m = L["m"]
        qs, ss = [], []
        for s in L["shards"]:
            q, sc = mo_fp8_weight(shards[s])
            qs.append(q)
            ss.append(sc)
        w, ws, req = fuse_shards(qs, ss, L["widths"])
        if tuple(m.weight.shape) != (w.shape[1], w.shape[0]) or m.weight.dtype != torch.float8_e4m3fn:
            raise RuntimeError(f"{ln}: live weight {tuple(m.weight.shape)} {m.weight.dtype} != transposed {tuple(w.shape)}")
        with torch.no_grad():
            m.weight.data.copy_(w.t())
            m.weight_scale.data.copy_(ws.to(m.weight_scale.dtype).reshape(m.weight_scale.shape))
        self.written.add(ln)
        self.n_requant += int(req)
        self.wscales.append(float(ws))
        if self.sentinel is None and ".layers.0." in ln:
            ref = torch.cat([shards[s].to(torch.float32) for s in L["shards"]], 0)
            dq = w.to(torch.float32) * ws
            self.sentinel = float(((dq - ref).norm() / ref.norm()).item())

    def write_static(self):
        """Static input scales and k / v scales from the file (a level-2 sleep discards the former; vLLM resets the
        latter)."""
        import torch

        with torch.no_grad():
            for ln, L in self.layers.items():
                m = L["m"]
                m.input_scale.data.copy_(torch.tensor(float(self.scales["input_scale"][ln]), dtype=m.input_scale.dtype,
                                                      device=m.input_scale.device).reshape(m.input_scale.shape))
        self.apply_kv()

    def apply_kv(self):
        import torch

        with torch.no_grad():
            for i, m in self.attn.items():
                k, v = self.kv[i]
                m._k_scale.fill_(k)
                m._v_scale.fill_(v)
                m._k_scale_float = float(torch.tensor(k, dtype=torch.float32))
                m._v_scale_float = float(torch.tensor(v, dtype=torch.float32))

    def kv_ok(self):
        import torch

        for i, m in self.attn.items():
            k, v = (float(torch.tensor(x, dtype=torch.float32)) for x in self.kv[i])
            if float(m._k_scale) != k or float(m._v_scale) != v or m._k_scale_float != k or m._v_scale_float != v:
                return False
        return True

    def in_ok(self):
        for ln, L in self.layers.items():
            want = float(self._torch.tensor(float(self.scales["input_scale"][ln]), dtype=self._torch.float32))
            if float(L["m"].input_scale) != want:
                return False
        return True

    def finish_sync(self, step=None):
        import torch

        self.sync += 1
        if self.pending:
            self.errors.append(f"incomplete fused layers: {sorted(self.pending)[:3]}")
        if self.written != set(self.layers):
            self.errors.append(f"{len(self.layers) - len(self.written)} quantized layers not written this sync")
        self.write_static()
        sums, changed = {}, 0
        for ln, L in self.layers.items():
            x = L["m"].weight.data.reshape(-1).view(torch.uint8)
            sums[ln] = int(x.to(torch.int64).sum().item())
            changed += int(self.prev_sums.get(ln) is not None and self.prev_sums[ln] != sums[ln])
        wsum = hashlib.sha256(json.dumps(sorted(sums.items())).encode()).hexdigest()[:16]
        self.prev_sums = sums
        rec = dict(rid=self.rid, sync=self.sync, step=step if step is not None else "na", recipe=self.recipe,
                   layers="%d/%d" % (len(self.written), len(self.layers)), wsum=wsum,
                   changed=changed if self.sync > 1 else "na", requant=self.n_requant,
                   wscale=("%s:%s" % (_f(min(self.wscales)), _f(max(self.wscales)))) if self.wscales else "na",
                   in_ok=int(self.in_ok()), kv_ok=int(self.kv_ok()),
                   relerr=_f(self.sentinel) if self.sentinel is not None else "na",
                   ms=int(1000 * (time.time() - self.t0)))
        line = TAG + " " + " ".join("%s=%s" % kv for kv in rec.items())
        try:
            logger.warning("%s", line)
        except Exception:  # noqa: BLE001
            pass
        errors = list(self.errors)
        self._begin()
        if errors:
            raise RuntimeError("SAMPLER_MODELOPT sync failed: " + "; ".join(errors[:5]))
        return rec

    @contextlib.contextmanager
    def no_linear_reprocess(self):
        """vLLM's post-load processing would transpose / re-scale the live linear layers again: off for the sync."""
        from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8LinearMethod

        orig = ModelOptFp8LinearMethod.process_weights_after_loading
        ModelOptFp8LinearMethod.process_weights_after_loading = lambda self_, layer: None
        try:
            yield
        finally:
            ModelOptFp8LinearMethod.process_weights_after_loading = orig

    # ------------------------------------------------------------------------------------------------ per KV wake-up
    def install_wake_hook(self):
        mr = self.model_runner
        orig = mr.init_fp8_kv_scales

        def init_fp8_kv_scales_then_restore():
            orig()
            self.wakes += 1
            try:
                reset = sum(1 for m in self.attn.values() if float(m._k_scale) == 1.0 and float(m._v_scale) == 1.0)
                self.apply_kv()
                m0 = self.attn.get(min(self.attn)) if self.attn else None
                line = "%s rid=%s wake=%d sync=%d reset=%d/%d kv_ok=%d k0=%s v0=%s q0=%s prob0=%s kf0=%s" % (
                    TAG_KV, self.rid, self.wakes, self.sync, reset, len(self.attn), int(self.kv_ok()),
                    _f(float(m0._k_scale)),
                    _f(float(m0._v_scale)), _f(float(m0._q_scale)), _f(float(m0._prob_scale)), _f(m0._k_scale_float))
            except Exception as e:  # noqa: BLE001
                line = "%s rid=%s wake=%d audit failed: %r" % (TAG_KV, self.rid, self.wakes, e)
            try:
                logger.warning("%s", line)
            except Exception:  # noqa: BLE001
                pass

        mr.init_fp8_kv_scales = init_fp8_kv_scales_then_restore


# ------------------------------------------------------------------------------------------------ reading the lines
_KV = re.compile(r"(\w+)=(\S+)")


def parse_line(line):
    """A [SPECOPD-MO] or [SPECOPD-MO-KV] line -> dict with 'kind' ('sync' | 'wake'); None otherwise."""
    for tag, kind in ((TAG_KV, "wake"), (TAG, "sync")):
        i = line.find(tag + " ")
        if i >= 0:
            body = line[i + len(tag) + 1:].strip()
            d = dict(_KV.findall(body.split(" audit failed:")[0]))
            d["kind"] = kind
            d["failed"] = " audit failed:" in body
            return d
    return None
