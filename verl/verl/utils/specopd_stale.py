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
"""Stale SAMPLER for synchronous OPD (opt-in: SPECOPD_STALE_PERIOD=K or SPECOPD_STALE_LAG=k).

The rollout engine generates with the student's weights from EARLIER weight syncs while the FSDP trainer stays
current: async / stale rollouts, emulated under full control inside the synchronous trainer. The trainer, its
log-probs, the OPD advantages (recomputed from the CURRENT weights) and the checkpoints are untouched; the only
effect is which tokens the sampler emits. verl's rollout_corr/kl (rollout log-prob vs trainer old_log_prob) measures
the resulting mismatch per step.

Where: receiver side, in each vLLM worker process (vllm_rollout/utils.py update_weights_from_ipc). Level-2 sleep
wipes the GPU weights every step and every wake streams them again, bucket by bucket, as HF-named tensors; this hook
sits FIRST in the per-bucket callback. It copies each CLEAN incoming tensor into a CPU cache (CPU memory survives the
sleep) and hands the engine the cached tensor of the version it must serve instead. Every later hook (SamplerNoise,
SamplerQuant) then perturbs / quantizes the STALE weights -- a stale quantized sampler -- and the cache never sees
their output. Only the actor stream is touched: the EAGLE-3 drafter is loaded from its own file afterwards, and the
teacher engine never receives an IPC sync.

Versions. Sync t (t = 0 is the initial sync before the first rollout; with verl's sync trainer one sync per global
step, i.e. version t = the trainer after t train batches) serves version v(t):
  * period K  (SPECOPD_STALE_PERIOD, DEFAULT semantics = Score Centering's sampler.staleness = K - 1):
        v = t - t % K. The sampler is refreshed every K syncs; lag cycles 0, 1, ..., K-1. Caches ONE version.
        Paper (github.com/martin-marek/score-centering @7c56e9ee, train_rl.py L364-369): refresh_period =
        staleness + 1; `if step % refresh_period == 0: sampler_weights = sampler_snapshot(model.weights)`.
  * lag k     (SPECOPD_STALE_LAG): v = max(t - k, 0). Fixed lag: once t >= k the served weights are EXACTLY the
        trainer's weights from k syncs ago. The first k syncs (t < k) serve the INITIAL weights (lag ramps 0..k-1),
        as AsyncOPD's controlled cache protocol (arXiv 2606.24143, App. B: staleness ramps then plateaus at k).
        Caches k versions in a ring: slot t % k holds version t; at t >= k it is read (version t - k) and then
        overwritten in place, tensor by tensor.
  K = 1 / k = 0 is a pass-through (nothing cached, the line is still logged).
Served weights are BIT-FOR-BIT the bytes streamed at sync v (same dtype). With SPECOPD_STALE_VERIFY=1 (default) every
served tensor is checked against the exact checksum recorded when version v arrived; a mismatch fails the sync.

verl's own staleness metric. update_weights_from_ipc returns the trainer step of the SERVED weights (served_step);
ServerAdapter.update_weights tags the replica's samples with it (served_global_steps -> set_global_steps) instead of
the trainer step, so training/off_policy/trajectory_staleness = (global_steps - 1) - max_global_steps reads the lag of
the sync that generated the step's rollouts (k in lag mode once t >= k, t % K in period mode) instead of 0, and
trajectory_spans stays 1. The tag feeds only those trajectory_* metrics (the replay buffer orders prompts by its own
trainer-side tags). If the engine returns no step (an unpatched vllm_async_server.collective_rpc), the tag falls back
to the trainer step with one WARNING: the metric then reads 0 and only [SPECOPD-STALE] lag= holds the truth.

With SAMPLER_NOISE: [SPECOPD-TIM] step=t is TIM's own sync counter, but its d_eps / d_n projection is taken on the
SERVED weights (this hook runs first), i.e. on version served=v of the [SPECOPD-STALE] line of the same sync (the n-th
TIM line of a worker pairs with its n-th STALE line). Join the two on sync before any d_n analysis: with STALENESS=64
d_n stays flat for 65 syncs.

Re-syncs. update_weights_from_ipc receives the trainer's global step (specopd_step, see rpc_kwargs); a sync whose step
equals the previous one, or None after numbered syncs (verl's _validate re-syncs with a colocated reward model), is a
RE-SYNC of the same trainer weights: it does not advance t and serves the version served last time -- if that version
is still cached (always in period mode), else the sync fails. Without the step (None throughout) every sync is new.

Errors never HANG the job. verl's BucketedWeightSender blocks without a timeout on the ack of every bucket, and the
receiver acks a bucket only after the callback returns; an exception raised before or during the stream (in
begin_sync / apply, or by the constructor) would leave the trainer waiting to its time limit. So inside a sync every
error is logged at ERROR, the rest of the sync passes the fresh incoming tensors through (nothing more is cached), and
finish_sync -- called after the receiver's last ack -- raises it: the update_weights_from_ipc RPC fails and the step
crashes. A failed sampler stays failed: every later sync passes through and raises again. for_worker() turns a
configuration error into the same deferred raise.

Memory (per worker = per rollout GPU; every worker receives the FULL stream, also under TP > 1). The verl FSDP stream
is the actor's fp32 master weights (the [SPECOPD-TIM] line logs the stream dtype), so one version of
R1-Distill-Qwen-7B is 7.62e9 x 4 B = 30.5 GB per worker, x 4 workers = 122 GB per 4-GPU node:
    SPECOPD_STALE_CACHE_DTYPE=exact (default): period any K -> 122 GB/node; lag k -> 122 k GB/node (k <= 2 at 300 GB)
    SPECOPD_STALE_CACHE_DTYPE=bf16: fp32 tensors cached as bf16 (= vLLM's own round-to-nearest-even cast on load):
        61 GB per version per node (lag k <= 4 at 300 GB). EVERY served fp32 tensor is then bf16, also at fresh syncs,
        so the engine's parameters equal the exact mode's; a noise / quant hook then perturbs the bf16-rounded weights
        (Score Centering's order: cast to the sampler dtype first, its train_rl.py L172-179, L189-213), not the fp32 ones.
The worker guard fails the first sync when n_versions x version_bytes x SPECOPD_STALE_WORKERS_PER_NODE (default 4) >
SPECOPD_STALE_MAX_GB_PER_NODE (default 300). preflight() applies the same rule in the launcher before Ray starts, and
also checks the cache plus the trainer's own CPU state (SPECOPD_STALE_RESERVE_B_PER_PARAM x params; the launcher sets
16 B/param for an offloaded FSDP actor, +2 with the REF_KL reference) against this node's memory limit: the tightest
cgroup limit of the job (slurm --mem) or MemTotal. Process RSS and Ray's object store are not in that sum: the
[SPECOPD-STALE-PLAN] line prints the limit, the headroom and MemAvailable for the first run to check.
SPECOPD_STALE_SPILL_DIR=<node-local disk>: the cache is file-backed (mmap, unlinked at once -> freed when the worker
exits) instead of RAM; the guard then checks free disk. A tmpfs/ramfs dir is RAM and keeps the RAM guard.
Pinning (SPECOPD_STALE_PIN=1, default): cudaHostRegister on each exact-size CPU tensor -- NOT tensor.pin_memory(),
whose caching host allocator rounds every block up to a power of two (a 272 MB fp32 MLP matrix would take 512 MB).

One line per sync per worker (worker = host/pid/CUDA_VISIBLE_DEVICES, LAST so anchored parsers keep working):
  [SPECOPD-STALE] sync=t served=v lag=t-v mode=period|lag k=.. cache_gb=.. node_gb=.. step=.. served_step=.. resync=0|1
  cache_dtype=.. n_tensors=.. stream_gb=.. digest_in=.. digest_served=.. verified=.. ms=.. worker=..
digest_served at sync t equals digest_in of sync v (same worker): the audit trail of the bit-exact claim across lines.

Unset SPECOPD_STALE_PERIOD and SPECOPD_STALE_LAG = for_worker() returns None, rpc_kwargs() is {} and
served_global_steps() returns the trainer step: byte-identical.
"""

import hashlib
import logging
import os
import shutil
import socket
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

ENV_PERIOD = "SPECOPD_STALE_PERIOD"
ENV_LAG = "SPECOPD_STALE_LAG"
ENV_MAX_GB = "SPECOPD_STALE_MAX_GB_PER_NODE"
ENV_WORKERS = "SPECOPD_STALE_WORKERS_PER_NODE"
ENV_CACHE_DTYPE = "SPECOPD_STALE_CACHE_DTYPE"
ENV_SPILL = "SPECOPD_STALE_SPILL_DIR"
ENV_PIN = "SPECOPD_STALE_PIN"
ENV_VERIFY = "SPECOPD_STALE_VERIFY"
ENV_RESERVE = "SPECOPD_STALE_RESERVE_B_PER_PARAM"
DEFAULT_MAX_GB_PER_NODE = 300.0
DEFAULT_WORKERS_PER_NODE = 4  # 4 GPUs per node, one rollout worker each
_CHUNK = 1 << 26  # words per checksum chunk: 512 MB of int64 temporaries at most


def _error(msg: str, cls: type = RuntimeError) -> Exception:
    """Log at ERROR, then return the exception to raise. Raised directly only where it reaches the driver: the launcher
    (preflight) and finish_sync, after the receiver's last ack. Inside a sync errors are deferred (see module doc)."""
    logger.error(msg)
    return cls(msg)


def enabled() -> bool:
    return bool(os.environ.get(ENV_PERIOD, "").strip() or os.environ.get(ENV_LAG, "").strip())


def rpc_kwargs(global_steps) -> dict:
    """Extra kwargs for the update_weights_from_ipc RPC (vllm_rollout.py): the trainer step behind the streamed weights,
    only when the knob is on -- otherwise {} and the RPC is unchanged."""
    return {"specopd_step": global_steps} if enabled() else {}


_warned_no_served_step = False


def served_global_steps(results, global_steps):
    """ServerAdapter.update_weights side: the step to tag this replica's samples with (set_global_steps) = the trainer
    step of the weights its engine SERVES, as update_weights_from_ipc returned it (one value per worker, through
    collective_rpc). Knob off, or no trainer step: ``global_steps`` unchanged."""
    global _warned_no_served_step
    if not enabled() or global_steps is None:
        return global_steps
    steps = {r for r in results if r is not None} if isinstance(results, (list, tuple)) else set()
    if not steps:
        if not _warned_no_served_step:
            _warned_no_served_step = True
            logger.warning(
                "[SPECOPD-STALE] the engine returned no served step (collective_rpc gave %r): samples keep the trainer "
                "step, so off_policy/trajectory_staleness reads 0 -- read the lag from the [SPECOPD-STALE] lines",
                results,
            )
        return global_steps
    if len(steps) > 1:
        logger.error("[SPECOPD-STALE] the engine's workers serve different steps %s: tagging the oldest", sorted(steps))
    return min(steps)


def tensor_digest(t: torch.Tensor) -> tuple[int, int]:
    """Exact checksum of a tensor's BYTES: (sum x, sum x^2) over its int16 words (uint8 for 1-byte dtypes), in int64 on
    the tensor's own device, chunked. Integer arithmetic, so device- and order-independent. An audit of the bit-exact
    claim (two trained versions agree on both sums only by accident), not a cryptographic hash."""
    flat = t.detach().contiguous().reshape(-1)
    words = flat.view(torch.int16) if flat.element_size() >= 2 else flat.view(torch.uint8)
    s1 = s2 = 0
    for i in range(0, words.numel(), _CHUNK):
        c = words[i : i + _CHUNK].to(torch.int64)
        s1 += int(c.sum())
        s2 += int(c.mul_(c).sum())
    return s1, s2


def version_digest(digests: dict) -> str:
    """One short id per version from the per-tensor checksums (name-ordered, so bucket order does not matter)."""
    h = hashlib.sha256()
    for name in sorted(digests):
        h.update(f"{name}:{digests[name][0]}:{digests[name][1]};".encode())
    return h.hexdigest()[:16]


def _is_ram_backed_fs(path: str) -> bool:
    """True if ``path`` is on tmpfs/ramfs (spilling there still costs RAM): longest /proc/mounts prefix."""
    try:
        real, best, fstype = os.path.realpath(path), "", ""
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mnt = parts[1].replace("\\040", " ")
                if (real == mnt or real.startswith(mnt.rstrip("/") + "/")) and len(mnt) > len(best):
                    best, fstype = mnt, parts[2]
        return fstype in ("tmpfs", "ramfs")
    except OSError:
        return False


def _pin(t: torch.Tensor) -> bool:
    """Page-lock a non-empty exact-size CPU tensor in place (cudaHostRegister); False if CUDA is absent or it fails."""
    if not torch.cuda.is_available():
        return False
    try:
        cudart = torch.cuda.cudart()
        return int(cudart.cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)) == 0
    except Exception:  # noqa: BLE001
        return False


class StaleSampler:
    """Per-worker state: the version counter, the CPU ring of cached versions and the per-version checksums."""

    def __init__(
        self,
        mode: str,
        k: int,
        *,
        max_gb_per_node: float = DEFAULT_MAX_GB_PER_NODE,
        workers_per_node: int = DEFAULT_WORKERS_PER_NODE,
        cache_dtype: str = "exact",
        spill_dir: Optional[str] = None,
        pin: bool = True,
        verify: bool = True,
    ):
        if mode not in ("period", "lag"):
            raise _error(f"[SPECOPD-STALE] mode must be period or lag, got {mode!r}", ValueError)
        if k < (1 if mode == "period" else 0):
            raise _error(f"[SPECOPD-STALE] {mode} must be >= {1 if mode == 'period' else 0}, got {k}", ValueError)
        if cache_dtype not in ("exact", "bf16"):
            raise _error(f"[SPECOPD-STALE] {ENV_CACHE_DTYPE} must be exact or bf16, got {cache_dtype!r}", ValueError)
        if workers_per_node < 1 or not max_gb_per_node > 0:
            raise _error("[SPECOPD-STALE] workers_per_node must be >= 1 and max_gb_per_node > 0", ValueError)
        if spill_dir is not None and not os.path.isdir(spill_dir):
            raise _error(f"[SPECOPD-STALE] {ENV_SPILL}={spill_dir} is not a directory", ValueError)
        self.mode, self.k = mode, int(k)
        self.identity = (mode == "lag" and k == 0) or (mode == "period" and k == 1)
        self.n_slots = 0 if self.identity else (self.k if mode == "lag" else 1)
        self.max_bytes_per_node = float(max_gb_per_node) * 1e9
        self.workers_per_node = int(workers_per_node)
        self.cache_dtype = cache_dtype
        self.spill_dir = spill_dir
        self.ram_backed = spill_dir is None or _is_ram_backed_fs(spill_dir)
        self.pin = pin and spill_dir is None
        self.verify = verify
        host, cvd = socket.gethostname().split(".")[0], os.environ.get("CUDA_VISIBLE_DEVICES", "-")
        self.worker = f"{host}/{os.getpid()}/{cvd}"  # which rollout worker wrote a [SPECOPD-STALE] line
        self.t = -1  # index of the newest version received (sync counter, re-syncs excluded)
        self.served_step = None  # trainer step of the weights the last sync served: update_weights_from_ipc returns it
        self._last_step = None
        self._step_of: dict[int, Optional[int]] = {}
        self._layout: Optional[dict] = None  # name -> (shape, dtype) of the stream, frozen after the first sync
        self._slots: list[dict[str, torch.Tensor]] = [{} for _ in range(self.n_slots)]
        self._slot_ver: list[Optional[int]] = [None] * self.n_slots
        self._digests: dict[int, dict] = {}  # version -> name -> checksum, for cached versions + the newest
        self._cache_bytes = 0
        self._n_files = 0
        self._pin_failed = False
        self._open = False
        self._failed: Optional[Exception] = None  # sticky: the first error of any sync
        self._err: Optional[Exception] = None  # the first error of the current sync, raised by finish_sync

    @classmethod
    def from_env(cls) -> Optional["StaleSampler"]:
        period, lag = os.environ.get(ENV_PERIOD, "").strip(), os.environ.get(ENV_LAG, "").strip()
        if not period and not lag:
            return None
        if period and lag:
            raise _error(f"[SPECOPD-STALE] set {ENV_PERIOD} or {ENV_LAG}, not both", ValueError)
        mode, raw = ("period", period) if period else ("lag", lag)
        try:
            k = int(raw)
        except ValueError as e:
            raise _error(
                f"[SPECOPD-STALE] {ENV_PERIOD if period else ENV_LAG}={raw!r} is not an integer", ValueError
            ) from e
        return cls(
            mode,
            k,
            max_gb_per_node=float(os.environ.get(ENV_MAX_GB, DEFAULT_MAX_GB_PER_NODE)),
            workers_per_node=int(os.environ.get(ENV_WORKERS, DEFAULT_WORKERS_PER_NODE)),
            cache_dtype=os.environ.get(ENV_CACHE_DTYPE, "exact").strip() or "exact",
            spill_dir=os.environ.get(ENV_SPILL, "").strip() or None,
            pin=os.environ.get(ENV_PIN, "1") != "0",
            verify=os.environ.get(ENV_VERIFY, "1") != "0",
        )

    def served_version(self, t: int) -> int:
        return max(t - self.k, 0) if self.mode == "lag" else t - t % self.k

    # ------------------------------------------------------------------------------------------------------------
    def begin_sync(self, step: Optional[int] = None, lora: bool = False) -> None:
        """Before the first bucket: classify the sync (new version or re-sync), pick the slot to read and to write.
        Never raises (see the module doc): a refusal turns the sync into a pass-through that finish_sync fails."""
        self._err = None
        self._seen: dict[str, None] = {}
        self._layout_new: dict = {}
        self._digest_in: dict = {}
        self._stream_bytes = 0
        self._read = self._write = None
        self._t0 = time.perf_counter()
        if self._open:  # the previous sync failed inside receive_weights: the slot it was writing is half written
            self._defer(RuntimeError(f"[SPECOPD-STALE] begin_sync() before sync {self.t} finished"))
        self._open = True
        resync = self.t >= 0 and (step == self._last_step if step is not None else self._last_step is not None)
        if not resync:
            self.t += 1
            self._step_of[self.t] = step
            if step is not None:
                self._last_step = step
        t, v = self.t, self.served_version(self.t)
        self._resync, self._v, self._fresh = resync, v, v == t
        if self._failed is not None:
            self._defer(self._failed)
        elif lora:
            self._defer(RuntimeError("[SPECOPD-STALE] LoRA adapter syncs cannot be served stale: unset the knob"))
        elif not self.identity:
            if not self._fresh:
                if v not in self._slot_ver:
                    self._defer(
                        RuntimeError(
                            f"[SPECOPD-STALE] sync {t} (re-sync={resync}) must serve version {v}, no longer cached "
                            f"(mode={self.mode} k={self.k}): a re-sync after the ring advanced cannot be served in "
                            f"lag mode -- use {ENV_PERIOD} or a reward model with its own pool"
                        )
                    )
                    return
                self._read = self._slot_ver.index(v)
            if not resync and (self.mode == "lag" or t % self.k == 0):
                self._write = t % self.n_slots
                self._slot_ver[self._write] = None  # overwritten below, each tensor right AFTER its old bytes are read

    def apply(self, weights):
        """One bucket of (name, tensor) -> the same names with the tensors of the served version. The incoming tensors
        are views into the shared IPC buffer: they are copied out (never kept) and never written. Never raises: after
        an error the rest of the sync gets the incoming tensors unchanged and finish_sync raises it."""
        if not self._open:
            self._defer(RuntimeError("[SPECOPD-STALE] apply() outside begin_sync()/finish_sync()"))
        out = []
        for name, w in weights:
            if self._err is None:
                try:
                    w = self._apply_one(name, w)
                except Exception as e:  # noqa: BLE001 -- deferred to finish_sync, see the module doc
                    self._defer(e)
            out.append((name, w))
        return out

    def _apply_one(self, name: str, w: torch.Tensor) -> torch.Tensor:
        if name in self._seen:
            raise RuntimeError(f"[SPECOPD-STALE] {name} streamed twice in sync {self.t}")
        self._seen[name] = None
        self._stream_bytes += w.numel() * w.element_size()
        sig = (tuple(w.shape), w.dtype)
        if self._layout is None:
            self._layout_new[name] = sig
        elif self._layout.get(name) != sig:
            raise RuntimeError(f"[SPECOPD-STALE] {name} {sig} does not match the first sync: {self._layout.get(name)}")
        if self.identity:
            return w
        src = w.to(torch.bfloat16) if self.cache_dtype == "bf16" and w.dtype == torch.float32 else w
        if self.verify:
            self._digest_in[name] = tensor_digest(src)
        if self._fresh:
            served = src
        else:
            # H2D copy (copy=True: on CPU a plain .to() would alias the slot that the write below overwrites).
            served = self._slots[self._read][name].to(w.device, copy=True)
            if self.verify and tensor_digest(served) != self._digests[self._v][name]:
                raise RuntimeError(f"[SPECOPD-STALE] served {name} is not version {self._v} bit-for-bit")
        if self._write is not None:
            self._buffer(self._write, name, src).copy_(src)  # D2H of the CLEAN tensor, after the read above
        return served

    def _defer(self, e: Exception) -> None:
        """Record the first error of this sync (and of the sampler) and log it now; finish_sync raises it."""
        if self._err is None:
            self._err = e
            logger.error(
                "[SPECOPD-STALE] sync %d failed: %s -- the rest of the stream passes through, finish_sync raises "
                "(worker=%s)",
                self.t,
                e,
                self.worker,
            )
        if self._failed is None:
            self._failed = e

    def _fail(self, msg: str) -> Exception:
        e = _error(msg)
        if self._failed is None:
            self._failed = e
        return e

    def finish_sync(self) -> dict:
        """After the last bucket, i.e. after the receiver's last ack: raising here fails the update_weights_from_ipc
        RPC and so crashes the step. Deferred errors, completeness, ring bookkeeping, the memory guard, one log line."""
        if not self._open:
            raise self._fail("[SPECOPD-STALE] finish_sync() without begin_sync()")
        self._open = False
        if self._err is not None:
            raise RuntimeError(
                f"[SPECOPD-STALE] sync {self.t} failed (the engine may hold a MIX of versions; every later sync "
                f"fails too): {self._err}"
            ) from self._err
        if self._layout is None:
            self._layout = self._layout_new
            msg = self._guard_msg(self._cache_bytes, final=True) if self.n_slots else None
            if msg:
                raise self._fail(msg)
        else:
            missing = self._layout.keys() - self._seen.keys()
            if missing:
                missing = sorted(missing)
                raise self._fail(f"[SPECOPD-STALE] sync {self.t} never streamed {len(missing)} tensors: {missing[:4]}")
        if self._write is not None:
            self._slot_ver[self._write] = self.t
        verified = "off" if not self.verify or self.identity else ("fresh" if self._fresh else "ok")
        digest_served = "-"
        if self.verify and not self.identity:
            if not self._resync:
                self._digests[self.t] = self._digest_in
            elif self._digests.get(self.t, self._digest_in) != self._digest_in:
                raise self._fail(f"[SPECOPD-STALE] re-sync of version {self.t} streamed DIFFERENT weights")
            digest_served = version_digest(self._digests[self._v])
            keep = set(self._slot_ver) | {self.t}  # cached versions + the newest (a re-sync is checked against it)
            self._digests = {v: d for v, d in self._digests.items() if v in keep}
        self.served_step = self._step_of.get(self._v)
        rec = dict(
            sync=self.t,
            served=self._v,
            lag=self.t - self._v,
            mode=self.mode,
            k=self.k,
            cache_gb=self._cache_bytes / 1e9,
            node_gb=self._cache_bytes * self.workers_per_node / 1e9,
            step=self._step_of.get(self.t),
            served_step=self.served_step,
            resync=int(self._resync),
            cache_dtype=self.cache_dtype + ("+spill" if self.spill_dir else ""),
            n_tensors=len(self._seen),
            stream_gb=self._stream_bytes / 1e9,
            digest_in=version_digest(self._digest_in) if self._digest_in else "-",
            digest_served=digest_served,
            verified=verified,
            ms=(time.perf_counter() - self._t0) * 1e3,
            worker=self.worker,
        )
        logger.warning(  # one mapping argument: logging fills the %(name) fields from rec
            "[SPECOPD-STALE] sync=%(sync)d served=%(served)d lag=%(lag)d mode=%(mode)s k=%(k)d cache_gb=%(cache_gb).2f "
            "node_gb=%(node_gb).1f step=%(step)s served_step=%(served_step)s resync=%(resync)d "
            "cache_dtype=%(cache_dtype)s n_tensors=%(n_tensors)d stream_gb=%(stream_gb).2f digest_in=%(digest_in)s "
            "digest_served=%(digest_served)s verified=%(verified)s ms=%(ms).0f worker=%(worker)s",
            rec,
        )
        # The served copies and checksum temporaries live in torch's caching allocator, outside vLLM's sleep-mode pool:
        # hand them back before the KV cache wakes up.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return rec

    # ------------------------------------------------------------------------------------------------------------
    def _guard_msg(self, version_bytes: int, final: bool, allocated: int = 1) -> Optional[str]:
        """The refusal, if any: n_slots versions x version_bytes x workers per node over the cap (``allocated``
        versions already sit on the spill disk: 1 after the first sync, 0 in preflight)."""
        need = self.n_slots * version_bytes * self.workers_per_node
        if self.ram_backed and need > self.max_bytes_per_node:
            return (
                f"[SPECOPD-STALE] refusing: mode={self.mode} k={self.k} caches {self.n_slots} version(s) x "
                f"{'' if final else '>= '}{version_bytes / 1e9:.2f} GB ({self.cache_dtype}) x {self.workers_per_node} "
                f"workers/node = {need / 1e9:.1f} GB > {ENV_MAX_GB}={self.max_bytes_per_node / 1e9:g}. Options: "
                + (f"{ENV_CACHE_DTYPE}=bf16 (halves an fp32 stream), " if self.cache_dtype == "exact" else "")
                + f"{ENV_SPILL}=<node-local disk>, a smaller k, or a higher cap."
            )
        if final and not self.ram_backed:
            more = (self.n_slots - allocated) * version_bytes * self.workers_per_node
            free = shutil.disk_usage(self.spill_dir).free
            if more > free:
                return (
                    f"[SPECOPD-STALE] refusing: {self.n_slots - allocated} more version(s) x {version_bytes / 1e9:.2f} "
                    f"GB x {self.workers_per_node} workers = {more / 1e9:.1f} GB on {self.spill_dir}, "
                    f"{free / 1e9:.1f} GB free"
                )
        return None

    def _buffer(self, slot: int, name: str, like: torch.Tensor) -> torch.Tensor:
        buf = self._slots[slot].get(name)
        if buf is None:
            buf = self._slots[slot][name] = self._alloc(like)
        return buf

    def _alloc(self, like: torch.Tensor) -> torch.Tensor:
        """One exact-size CPU tensor: file-backed under a spill dir, else (pinned) RAM. Budget-checked BEFORE the
        allocation while the first version is still streaming (the layout is not known yet: a running lower bound)."""
        nbytes = like.numel() * like.element_size()
        if self._layout is None:
            msg = self._guard_msg(self._cache_bytes + nbytes, final=False)
            if msg:
                raise RuntimeError(msg)
        self._cache_bytes += nbytes
        if self.spill_dir is not None and like.numel() > 0:
            path = os.path.join(self.spill_dir, f".specopd_stale_{os.getpid()}_{self._n_files}.bin")
            self._n_files += 1
            buf = torch.from_file(path, shared=True, size=like.numel(), dtype=like.dtype).view(like.shape)
            os.unlink(path)  # the mapping keeps the data; the disk space is freed when this process exits
            return buf
        buf = torch.empty(like.shape, dtype=like.dtype, device="cpu")
        # Empty tensors have nothing to page-lock: skipping them must not read as a failure.
        if self.pin and buf.numel() and not self._pin_failed and not _pin(buf) and torch.cuda.is_available():
            self._pin_failed = True
            logger.warning("[SPECOPD-STALE] cudaHostRegister failed: caching in pageable memory (slower copies)")
        return buf


class _Refused:
    """for_worker()'s stand-in when the configuration itself is invalid: passes every sync through and raises at
    finish_sync, so the error crashes the step instead of hanging the sender (see the module doc)."""

    served_step = None

    def __init__(self, err: Exception):
        self.err = err

    def begin_sync(self, step: Optional[int] = None, lora: bool = False) -> None:
        pass

    def apply(self, weights):
        return weights

    def finish_sync(self) -> dict:
        raise RuntimeError(f"[SPECOPD-STALE] invalid configuration: {self.err}") from self.err


def for_worker():
    """update_weights_from_ipc's constructor: StaleSampler.from_env(), except that a configuration error comes back as
    a _Refused stand-in -- raised there, before receive_weights, it would hang the sender. None when the knob is off."""
    try:
        return StaleSampler.from_env()
    except Exception as e:  # noqa: BLE001
        logger.error("[SPECOPD-STALE] invalid configuration, every weight sync will fail: %r", e)
        return _Refused(e)


# ----------------------------------------------------------------------------------------------------------------
_ST_INT_BYTES = {"I64": 8, "U64": 8, "I32": 4, "U32": 4, "I16": 2, "U16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def _node_memory(proc: str = "/proc", cgroot: str = "/sys/fs/cgroup") -> dict:
    """What this job may use on this node, in bytes (None where unreadable, e.g. off Linux): ``cgroup`` = the tightest
    memory limit on the path from this process's cgroup up to the root (v2 memory.max, v1 memory.limit_in_bytes; slurm
    puts --mem on the job's cgroup, an ANCESTOR of the batch step's), plus /proc/meminfo's MemTotal and MemAvailable."""
    out = {"cgroup": None, "total": None, "avail": None}
    try:
        with open(os.path.join(proc, "meminfo")) as f:
            info = {p[0].rstrip(":"): int(p[1]) * 1024 for p in (line.split() for line in f) if len(p) >= 2}
        out["total"], out["avail"] = info.get("MemTotal"), info.get("MemAvailable")
    except (OSError, ValueError):
        pass
    limits = []
    try:
        with open(os.path.join(proc, "self", "cgroup")) as f:
            entries = [line.rstrip("\n").split(":", 2) for line in f if line.count(":") >= 2]
    except OSError:
        entries = []
    for hid, ctrls, path in entries:
        if hid == "0" and ctrls == "":
            root, fname = cgroot, "memory.max"
        elif "memory" in ctrls.split(","):
            root, fname = os.path.join(cgroot, "memory"), "memory.limit_in_bytes"
        else:
            continue
        p = path.strip() or "/"
        while True:
            try:
                with open(os.path.join(root, p.lstrip("/"), fname)) as f:
                    raw = f.read().strip()
                if raw != "max" and int(raw) < 1 << 60:  # v1 "unlimited" is ~2^63
                    limits.append(int(raw))
            except (OSError, ValueError):
                pass
            if p == "/":
                break
            p = os.path.dirname(p.rstrip("/")) or "/"
    out["cgroup"] = min(limits) if limits else None
    return out


def preflight(model_dir: str) -> str:
    """Launcher-side twin of the in-worker guard, run BEFORE Ray starts. Sizes one cached version from the checkpoint's
    safetensors headers, assuming verl's fp32 FSDP stream (every float tensor 4 B, 2 B under cache_dtype=bf16), applies
    the same cap, then checks cache + the trainer's reserve (SPECOPD_STALE_RESERVE_B_PER_PARAM x params) against this
    node's memory limit. Raises on refusal, returns the [SPECOPD-STALE-PLAN] line."""
    import glob
    import json
    import math
    import struct

    s = StaleSampler.from_env()
    if s is None:
        return "[SPECOPD-STALE-PLAN] off"
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise _error(f"[SPECOPD-STALE-PLAN] no *.safetensors under {model_dir}", ValueError)
    version_bytes = n_params = 0
    for f in files:
        with open(f, "rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for name, meta in header.items():
            if name != "__metadata__":
                n = math.prod(meta["shape"])
                n_params += n
                version_bytes += n * _ST_INT_BYTES.get(meta["dtype"], 2 if s.cache_dtype == "bf16" else 4)
    if s.n_slots:
        msg = s._guard_msg(version_bytes, final=True, allocated=0)
        if msg:
            raise _error(msg)
    cache = s.n_slots * version_bytes * s.workers_per_node
    cache_ram = cache if s.ram_backed else 0  # a disk spill is page cache: reclaimable, not counted
    b_per_param = float(os.environ.get(ENV_RESERVE, "0") or 0)
    reserve = b_per_param * n_params
    need = cache_ram + reserve
    mem = _node_memory()
    known = [(x, src) for x, src in ((mem["cgroup"], "cgroup"), (mem["total"], "MemTotal")) if x]
    limit, src = min(known) if known else (None, "unknown")
    budget = (
        f"RAM need {need / 1e9:.1f} GB = cache {cache_ram / 1e9:.1f} + trainer reserve "
        f"{reserve / 1e9:.1f} ({b_per_param:g} B/param x {n_params / 1e9:.2f}e9 params); node limit "
        + (f"{limit / 1e9:.1f} GB ({src}), headroom {(limit - need) / 1e9:.1f} GB" if limit else "unknown")
        + (f", MemAvailable {mem['avail'] / 1e9:.1f} GB" if mem["avail"] else "")
        + " (process RSS and Ray's object store not counted)"
    )
    if limit is not None and need > limit:
        raise _error(
            f"[SPECOPD-STALE-PLAN] refusing: {budget}. Options: {ENV_CACHE_DTYPE}=bf16, {ENV_SPILL}=<node-local "
            f"disk>, a smaller k, or a larger --mem."
        )
    return (
        f"[SPECOPD-STALE-PLAN] mode={s.mode} k={s.k} versions={s.n_slots} x {version_bytes / 1e9:.2f} GB "
        f"({s.cache_dtype}, fp32 stream assumed) x {s.workers_per_node} workers = {cache / 1e9:.1f} GB/node "
        f"{'on ' + s.spill_dir if s.spill_dir else 'RAM'}; cap {s.max_bytes_per_node / 1e9:g} GB; {budget}"
    )
