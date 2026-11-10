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
"""The FA3 kv-skip patch in the ROLLOUT worker.

`install_fa3_kvskip_patch` is the one in bench/g0_kernels.py: its logic is identical; only the stats attribute on the
flash_attn module (`_g0_fa3_kvskip_stats` -> `_specopd_fa3_kvskip_stats`) and the docstring differ.

What it does: vLLM 0.19.1 + kv_cache_dtype_skip_layers + FA3 computes FA3's AOT scheduler metadata with the GLOBAL
cache dtype also for the attention group of the layers kept in bf16; where the metadata layout depends on the dtype the
bf16 kernel raises "scheduler_metadata must have shape". The patch acts only inside build() of a builder whose own
kv_cache_dtype is bf16/fp16 while the passed qkv_dtype differs, only on the non-CUDA-graph path (num_splits == 0), and
there returns the own-dtype metadata ONLY if its shape differs from vLLM's (exactly where the stock kernel raises).
Wherever stock vLLM runs, it runs bit-for-bit as stock. It does NOT keep layer 0 in bf16 -- vLLM's own
kv_cache_dtype_skip_layers does that.

Installed by `install_in_worker()` from vLLMColocateWorkerExtension.update_weights_from_ipc on the worker's FIRST weight
sync (before the first rollout), only when SPECOPD_FA3_KVSKIP_PATCH=1 (launcher knob KV_SKIP_LAYERS). Fail-closed: an
import or install failure raises (the sync fails and the job stops) instead of running an unpatched engine. Without the
env the worker never imports this module. The stats (installed, checked, fallback, first_fallback) are printed on every
[SPECOPD-KV-CFG] line (verl/utils/specopd_kvaudit.py).

Stdlib only at import time (torch / vllm are imported inside install_in_worker), so it loads by path without them.
"""

ENV = "SPECOPD_FA3_KVSKIP_PATCH"
FA3_KVSKIP_PATCH = "fa3_kvskip"  # the patch's label in its stats: see install_fa3_kvskip_patch


def install_fa3_kvskip_patch(fa, torch):
    """bench/g0_kernels.py's install_fa3_kvskip_patch (logic identical, stats attribute renamed to
    _specopd_fa3_kvskip_stats). `fa` is vllm.v1.attention.backends.flash_attn (or a stand-in module). Idempotent.
    Returns the live stats dict."""
    import threading
    st = getattr(fa, "_specopd_fa3_kvskip_stats", None)
    if st is not None:
        return st
    st = dict(patch=FA3_KVSKIP_PATCH, installed=True, checked=0, fallback=0, first_fallback=None)
    tl = threading.local()
    orig_gsm, cls = fa.get_scheduler_metadata, fa.FlashAttentionMetadataBuilder
    orig_build = cls.build
    own16 = (torch.bfloat16, torch.float16)

    def build(self, *args, **kwargs):
        prev, tl.builder = getattr(tl, "builder", None), self
        try:
            return orig_build(self, *args, **kwargs)
        finally:
            tl.builder = prev

    def get_scheduler_metadata(*args, **kwargs):
        md = orig_gsm(*args, **kwargs)
        b = getattr(tl, "builder", None)
        own, q = getattr(b, "kv_cache_dtype", None), kwargs.get("qkv_dtype")
        if b is None or args or own not in own16 or q is None or q == own or kwargs.get("num_splits", 0) != 0:
            return md
        st["checked"] += 1
        md_own = orig_gsm(**dict(kwargs, qkv_dtype=own))
        if tuple(md_own.shape) == tuple(md.shape):
            return md
        st["fallback"] += 1
        if st["first_fallback"] is None:
            st["first_fallback"] = dict(stock_shape=list(md.shape), own_shape=list(md_own.shape), qkv_dtype=str(q),
                                        own_dtype=str(own), batch_size=kwargs.get("batch_size"),
                                        max_seqlen_q=kwargs.get("max_seqlen_q"), causal=kwargs.get("causal"))
        return md_own

    cls.build = build
    fa.get_scheduler_metadata = get_scheduler_metadata
    fa._specopd_fa3_kvskip_stats = st
    return st


def install_in_worker():
    """Worker side (the EngineCore process that runs the model runner): install the patch into vLLM's flash_attn backend
    module. FAIL-CLOSED: any import or install error propagates. Returns the live stats dict."""
    import torch
    import vllm.v1.attention.backends.flash_attn as fa

    st = install_fa3_kvskip_patch(fa, torch)
    if st.get("installed") is not True or getattr(fa, "_specopd_fa3_kvskip_stats", None) is not st:
        raise RuntimeError(f"[SPECOPD-KVSKIP] the FA3 kv-skip patch did not install: {st!r}")
    return st


def stats_str(st):
    """The one-token form printed on the [SPECOPD-KV-CFG] line ('off' when not installed)."""
    if not st:
        return "off"
    ff = st.get("first_fallback")
    ffs = "none" if ff is None else ("bs%s_q%s_%s->%s" % (ff.get("batch_size"), ff.get("max_seqlen_q"),
                                                          "x".join(map(str, ff.get("stock_shape") or [])),
                                                          "x".join(map(str, ff.get("own_shape") or []))))
    return "patch:%s,installed:%s,checked:%d,fallback:%d,first_fallback:%s" % (
        st.get("patch"), st.get("installed"), int(st.get("checked", 0)), int(st.get("fallback", 0)), ffs)
