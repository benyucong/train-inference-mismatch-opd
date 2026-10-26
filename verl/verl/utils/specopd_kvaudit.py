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
"""[SPECOPD-KV-CFG]: what the rollout engine's KV cache really is, at EVERY weight sync.

Called from vLLMColocateWorkerExtension.update_weights_from_ipc next to audit_engine, only when the engine's
cache_config.cache_dtype is not 'auto' (a bf16-KV engine prints nothing, so its log is unchanged). One line
per replica (= per rollout worker, TP1) per sync:

  [SPECOPD-KV-CFG] rid=<host>:<pid> sync=<n> step=<s> cache_dtype=<x> skip=<i,j|-> target=<runs> drafter=<runs|none>
      backend=<runs> dbackend=<runs|none> groups=<g:backend:n|...> scales=<k:min:max,v:..,q:..,prob:..>
      fscales=<k:min:max,...> geom=<kv_heads>x<head_size> blocks=<num_gpu_blocks> block_size=<b> cap_tokens=<t>
      cap_gib=<g> kvskip=<FA3 kv-skip patch stats|off>

  <runs> = run-length compressed per-layer labels "<first>[-<last>]:<kv_cache_dtype>,<kv_cache_torch_dtype>,<storage>"
  joined by '|' (the layer index is the module name's `layers.<i>.`); kv_cache_dtype and kv_cache_torch_dtype are the
  Attention layer's attributes; storage is the dtype of the layer's bound KV tensor
  (torch. prefix dropped). scales: min:max over the TARGET attention layers of the _k/_v/_q/_prob_scale buffers
  (fscales: of their float mirrors _k_scale_float, ...); 'na' = no layer has it, 'partial' = some layers lack it.
  blocks x block_size = KV capacity in tokens; cap_gib = the KV tensors' bytes.

It never raises: an exception prints `[SPECOPD-KV-CFG] rid=... sync=... audit failed: <repr>` in place of the record.
"""
import logging
import os
import re
import socket

logger = logging.getLogger(__name__)

TAG = "[SPECOPD-KV-CFG]"
SCALES = ("k", "v", "q", "prob")
_LAYER_IDX = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _dt(x):
    return str(x).replace("torch.", "") if x is not None else "none"


def _f(x):
    return "%.9g" % x


def _rle(items):
    """[(idx, label)] sorted by idx -> 'a-b:label|c:label' (consecutive equal labels merged)."""
    runs = []
    for i, lab in sorted(items):
        if runs and runs[-1][2] == lab and runs[-1][1] == i - 1:
            runs[-1][1] = i
        else:
            runs.append([i, i, lab])
    return "|".join(("%d" % a if a == b else "%d-%d" % (a, b)) + ":" + lab for a, b, lab in runs) or "none"


def _attention_cls():
    try:
        from vllm.model_executor.layers.attention.attention import Attention
        return Attention
    except Exception:  # noqa: BLE001
        try:
            from vllm.attention.layer import Attention
            return Attention
        except Exception:  # noqa: BLE001
            return None


def _attn_modules(model, attn_cls):
    out = []
    if model is None:
        return out
    for n, (name, mod) in enumerate((nm, m) for nm, m in model.named_modules()
                                    if (attn_cls is not None and isinstance(m, attn_cls))
                                    or type(m).__name__ == "Attention"):
        mm = _LAYER_IDX.search(name)
        out.append((int(mm.group(1)) if mm else n, name, mod))
    return out


def _storage(mod):
    kv = getattr(mod, "kv_cache", None)
    t = kv[0] if isinstance(kv, (list, tuple)) and kv else kv
    dt = getattr(t, "dtype", None)
    return _dt(dt) if dt is not None else "none"


def _minmax(vals):
    if not vals:
        return None
    return (min(vals), max(vals))


def _tval(t):
    """All elements of a scale buffer as floats (a 0-d / 1-element tensor in vLLM; a float in stubs)."""
    try:
        return [float(x) for x in t.detach().float().reshape(-1).cpu().tolist()]
    except AttributeError:
        return [float(t)]


def _scale_field(layers, attr):
    vals, have = [], 0
    for _, _, mod in layers:
        if hasattr(mod, attr):
            have += 1
            vals += _tval(getattr(mod, attr))
    if have == 0:
        return "na"
    if have < len(layers):
        return "partial"
    lo, hi = _minmax(vals)
    return "%s:%s" % (_f(lo), _f(hi))


def _backend(mod):
    be = getattr(mod, "attn_backend", None)
    try:
        return be.get_name() if hasattr(be, "get_name") else str(be)
    except Exception:  # noqa: BLE001
        return "?"


def collect(model_runner):
    """The audit record of one engine (no rid / sync / step / kvskip; kv_audit adds them)."""
    attn_cls = _attention_cls()
    vc = model_runner.vllm_config
    cc = vc.cache_config
    get_model = getattr(model_runner, "get_model", None)
    target_model = get_model() if callable(get_model) else model_runner.model
    tl = _attn_modules(target_model, attn_cls)
    dmodel = getattr(getattr(model_runner, "drafter", None), "model", None)
    dl = _attn_modules(dmodel, attn_cls)

    def labels(layers):
        return [(i, "%s,%s,%s" % (getattr(m, "kv_cache_dtype", "?"), _dt(getattr(m, "kv_cache_torch_dtype", None)),
                                  _storage(m))) for i, _, m in layers]

    rec = dict(cache_dtype=str(cc.cache_dtype))
    skip = getattr(cc, "kv_cache_dtype_skip_layers", None) or []
    rec["skip"] = ",".join(str(x) for x in skip) or "-"
    rec["target"] = _rle(labels(tl))
    rec["drafter"] = _rle(labels(dl)) if dl else "none"
    rec["backend"] = _rle([(i, _backend(m)) for i, _, m in tl])
    rec["dbackend"] = _rle([(i, _backend(m)) for i, _, m in dl]) if dl else "none"
    # the backend per KV cache group (layer_name -> backend through the modules' own layer_name)
    byname = {}
    for _, name, m in tl + dl:
        byname[getattr(m, "layer_name", name)] = _backend(m)
    kcc = getattr(model_runner, "kv_cache_config", None)
    groups = []
    for g, grp in enumerate(getattr(kcc, "kv_cache_groups", None) or []):
        names = list(getattr(grp, "layer_names", []) or [])
        bes = sorted({byname.get(n, "?") for n in names}) or ["?"]
        groups.append("%d:%s:%d" % (g, "+".join(bes), len(names)))
    rec["groups"] = "|".join(groups) or "na"
    rec["scales"] = ",".join("%s:%s" % (s, _scale_field(tl, "_%s_scale" % s)) for s in SCALES)
    rec["fscales"] = ",".join("%s:%s" % (s, _scale_field(tl, "_%s_scale_float" % s)) for s in SCALES)
    m0 = tl[0][2] if tl else None
    rec["geom"] = "%sx%s" % (getattr(m0, "num_kv_heads", "?"), getattr(m0, "head_size", "?")) if m0 is not None else "?"
    blocks = getattr(cc, "num_gpu_blocks", None) or getattr(kcc, "num_blocks", None)
    bs = getattr(cc, "block_size", None)
    rec["blocks"] = int(blocks) if blocks else 0
    rec["block_size"] = int(bs) if bs else 0
    rec["cap_tokens"] = rec["blocks"] * rec["block_size"]
    tens = getattr(kcc, "kv_cache_tensors", None)
    rec["cap_gib"] = ("%.4f" % (sum(int(getattr(t, "size", 0)) for t in tens) / 2 ** 30)) if tens else "na"
    return rec


FIELDS = ("rid", "sync", "step", "cache_dtype", "skip", "target", "drafter", "backend", "dbackend", "groups", "scales",
          "fscales", "geom", "blocks", "block_size", "cap_tokens", "cap_gib", "kvskip")


def format_line(rec):
    return TAG + " " + " ".join("%s=%s" % (k, rec[k]) for k in FIELDS)


def kv_audit(worker, step=None):
    """One [SPECOPD-KV-CFG] line for this worker's engine, at every sync; nothing when cache_dtype is 'auto'. Returns the
    line (None when nothing was printed). Never raises."""
    rid = "%s:%d" % (socket.gethostname(), os.getpid())
    sync = "?"
    try:
        cd = str(getattr(getattr(getattr(worker.model_runner, "vllm_config", None), "cache_config", None),
                         "cache_dtype", "auto"))
        if cd == "auto":
            return None
        sync = getattr(worker, "_specopd_kv_sync", -1) + 1
        worker._specopd_kv_sync = sync
        rec = collect(worker.model_runner)
        from verl.utils.specopd_kvskip import stats_str  # stdlib-only module
        rec.update(rid=rid, sync=sync, step=step if step is not None else "na",
                   kvskip=stats_str(getattr(worker, "_specopd_kvskip_stats", None)))
        line = format_line(rec)
    except Exception as e:  # noqa: BLE001
        line = "%s rid=%s sync=%s audit failed: %r" % (TAG, rid, sync, e)
    try:
        logger.warning("%s", line)
    except Exception:  # noqa: BLE001
        pass
    return line
