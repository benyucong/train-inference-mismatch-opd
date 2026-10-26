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
"""EngineCore scheduler plugin: per-request and per-step speculative-decoding telemetry (identity depth policy).

Loaded with ``--scheduler-cls specopd_vllm.depth_sched.DepthAsyncScheduler`` (``DepthScheduler`` when async
scheduling is off). vLLM resolves the qualname inside EngineCore (vllm/config/scheduler.py:185-187, called from
vllm/v1/engine/core.py:128), which verl's monkeypatches cannot reach. TELEMETRY ONLY: every override hands the
base exactly the arguments it received, so scheduling, the SchedulerOutput sent to the workers and vLLM's own
spec-decode counters are unchanged. Nothing here may raise during scheduling: telemetry runs in try/except,
errors are counted and reported by ``pull``.

Hooks (vLLM 0.19.1, paths under vllm/):
  Scheduler.__init__(vllm_config, kv_cache_config, structured_output_manager, block_size, mm_registry,
                     include_finished_set, log_stats)                  v1/core/sched/scheduler.py:68
  _update_after_schedule(scheduler_output)                              scheduler.py:978 (+ async_scheduler.py:18),
                                                                        called at scheduler.py:948
  update_from_output(scheduler_output, model_runner_output)             scheduler.py:1302
  make_spec_decoding_stats(spec_decoding_stats, num_draft_tokens, num_accepted_tokens,
                           num_invalid_spec_tokens, request_id)         scheduler.py:1977, called once per request
                                                                        per step at scheduler.py:1384-1390
  _free_request(request, delay_free_blocks=False)                       scheduler.py:1818; its return value becomes
                                                                        EngineCoreOutput.kv_transfer_params (:1419)
  finish_requests(request_ids, finished_status)                         scheduler.py:1755; frees through
                                                                        _free_request but DISCARDS its return (:1814)
  EngineCore.specopd_rpc(op, payload)                                   injected below; reached as a UTILITY request
                                                                        (getattr dispatch, v1/engine/core.py:1252-1264)
Version / hook-signature drift is checked in ``__new__`` BEFORE any override can run: on failure the engine gets
vLLM's own AsyncScheduler / Scheduler instance (a true pass-through; ``specopd_rpc`` then answers ok=False).

Per-request record, exported once at finish as ``kv_transfer_params["specopd"]`` (-> RequestOutput.kv_transfer_params,
v1/engine/output_processor.py:618,644 -> verl extra_fields["kv_transfer_params"]):
  {v, K, steps, drafted, accepted, tried[K], acc[K], H[(K+1)^2], t_arrival, t_sched0, t_fin, gen_tokens,
   prompt_tokens, preemptions}
  steps = verify events (vLLM's num_drafts); drafted/accepted = exactly what vLLM's counters receive;
  tried[j] += 1 iff draft position j was verified (kd > j and a >= j), acc[j] += 1 iff a > j, so the per-request
  conditional acceptance is alpha_j = acc[j] / tried[j]; H[kd*(K+1)+a] counts (drafts verified, accepted) pairs,
  a sufficient statistic under any depth policy. Times are time.time() so they join with verl's [OPD-CP] clocks.
  ``accepted`` counts drafts accepted past a stop token on the final step (num_accepted is computed before stop
  truncation, scheduler.py:1371): clip with gen_tokens if that matters.
  NO record exists for a request the FRONTEND finishes -- aborts, including the detokenizer's stop strings
  (v1/engine/output_processor.py:630,670 -> EngineCore.abort_requests, v1/engine/core.py:332) and failed KV loads
  (scheduler.py:1489): finish_requests discards _free_request's return (scheduler.py:1814). Their counts go to the
  per-pull ``orphans`` tally instead, so sum(records) + orphans == totals == vLLM's counters still holds.

Per-engine state, returned (and reset) by ``specopd_rpc("pull")``:
  cells  "n,kv,kd" -> [steps, sum dt, sum dt^2, drafted, accepted, emitted] over PURE-DECODE steps, n = requests in
         the step, kv = longest draft list verified, kd = drafter depth run in the same step (identity policy:
         kd = K); emitted = tokens vLLM actually appended (EngineCoreOutput.new_token_ids: after stop truncation,
         without the already-finished rows the base skips, scheduler.py:1351). dt = this step's update_from_output
         completion - max(previous completion, this step's schedule time): the pipelined GPU step under the async
         batch queue (step t+1 is scheduled before step t completes), schedule->completion in sync mode and for the
         first step after idle.
  hist_n "n" -> H over all steps with n scheduled requests.
  totals {drafts, drafted, accepted, acc_pos[K]} == the deltas of vLLM's spec_decode counters since the last pull.
  orphans {requests, steps, drafted, accepted, acc[K], gen_tokens}: requests freed without a record (see above).
"""

from __future__ import annotations

import copy
import inspect
import os
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import vllm
from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore

if TYPE_CHECKING:  # annotations only (``from __future__ import annotations``)
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.spec_decode.metrics import SpecDecodingStats

# Inside vLLM's "vllm" logger tree (vllm/logger.py:63-69): the only namespace with a handler in EngineCore.
logger = init_logger("vllm.specopd.depth_sched")

SCHEMA = 1
_VLLM_VERSION = "0.19.1"
_MAX_ERR_LOGS = 5
_MAX_ERRORS = 1000  # after this many telemetry errors, fall back to pure pass-through
# Parameter names of the overridden base methods in 0.19.1; any drift -> pass-through.
_SIGS = {
    "_update_after_schedule": ("self", "scheduler_output"),
    "update_from_output": ("self", "scheduler_output", "model_runner_output"),
    "make_spec_decoding_stats": (
        "self",
        "spec_decoding_stats",
        "num_draft_tokens",
        "num_accepted_tokens",
        "num_invalid_spec_tokens",
        "request_id",
    ),
    "_free_request": ("self", "request", "delay_free_blocks"),
    "finish_requests": ("self", "request_ids", "finished_status"),
}
# Per-request accumulator (a list on the Request object; Request has no __slots__, v1/request.py:58).
_STEPS, _DRAFTED, _ACCEPTED, _T0, _TRIED, _ACC, _H = range(7)
_STATIC: list = []  # memo of _static_check()


def _static_check() -> str | None:
    """None if this vLLM is the one the overrides were written against; otherwise the reason (-> pass-through)."""
    if not _STATIC:
        try:
            reason = None
            if vllm.__version__ != _VLLM_VERSION:
                reason = f"vllm {vllm.__version__} != {_VLLM_VERSION}"
            else:
                for base in (Scheduler, AsyncScheduler):
                    for name, params in _SIGS.items():
                        if tuple(inspect.signature(getattr(base, name)).parameters) != params:
                            reason = f"{base.__name__}.{name} signature differs from vllm {_VLLM_VERSION}"
                            break
                    if reason:
                        break
        except Exception as e:
            reason = f"startup check failed: {e!r}"
        _STATIC.append(reason)
    return _STATIC[0]


class _DepthMixin:
    _sp_on = False  # class default: pass-through until __init__ has run its checks
    _sp_export = False
    _sp_orphaning = 0  # > 0 while finish_requests runs (its _free_request return value is discarded)

    def __new__(cls, *args, **kwargs):
        # EngineCore builds the scheduler with keywords (v1/engine/core.py:143-150).
        vcfg = kwargs.get("vllm_config", args[0] if args else None)
        want = getattr(getattr(vcfg, "scheduler_config", None), "async_scheduling", None)
        reason = _static_check()
        if reason is not None:
            # vLLM calls the hooks with keywords (e.g. scheduler.py:1384-1390): overrides written against another
            # signature would raise inside EngineCore. Hand back vLLM's own class instead -- nothing of ours runs.
            base = AsyncScheduler if (want if want is not None else issubclass(cls, AsyncScheduler)) else Scheduler
            logger.warning("[specopd] telemetry DISABLED, using vLLM's own %s: %s", base.__name__, reason)
            obj = object.__new__(base)
            obj.__init__(*args, **kwargs)  # not an instance of cls, so Python will not call __init__ itself
            return obj
        # scheduler_cls bypasses vLLM's Async/Sync choice (config/scheduler.py:167-175), and a base that disagrees
        # with async_scheduling corrupts the output placeholders. Log and build the sibling that matches.
        if want is not None and cls in (DepthAsyncScheduler, DepthScheduler):
            target = DepthAsyncScheduler if want else DepthScheduler
            if cls is not target:
                logger.warning(
                    "[specopd] %s requested but async_scheduling=%s: using %s", cls.__name__, want, target.__name__
                )
                obj = object.__new__(target)
                obj.__init__(*args, **kwargs)  # not an instance of cls, so Python will not call __init__ itself
                return obj
        return object.__new__(cls)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        K = self.num_spec_tokens  # engine K_max, fixed at engine start (scheduler.py:213-222)
        self._sp_K = K
        self._sp_kd = K  # identity policy: the drafter runs K deep on every step (no depth control in this build)
        self._sp_version = 0
        self._sp_err = 0
        self._sp_err_last = ""
        self._sp_tsched: dict[int, float] = {}  # id(scheduler_output) -> perf_counter at schedule time
        self._sp_t_out = 0.0  # perf_counter at the last non-empty update_from_output
        self._sp_n = 0  # requests in the step being processed (for hist_n)
        self._sp_st = [0, 0]  # this step's drafted, accepted
        self._sp_reset()
        self._sp_reason = self._sp_check()
        self._sp_on = self._sp_reason is None
        # The per-request record rides kv_transfer_params, which a KV connector owns (verl's PD path reads it,
        # vllm_async_server.py:576,829): keep the per-step telemetry but never touch kv_transfer_params then.
        self._sp_export = self._sp_on and self.connector is None
        if self._sp_on:
            # WARNING, not INFO: verl runs vLLM at VLLM_LOGGING_LEVEL=WARN (constants_ppo.py:89), and a loaded
            # plugin must leave a trace in the log.
            logger.warning(
                "[specopd] telemetry scheduler %s active: K=%d async=%s log_stats=%s per-request export=%s",
                type(self).__name__, K, self.scheduler_config.async_scheduling, self.log_stats, self._sp_export,
            )
            if not self._sp_export:
                logger.warning("[specopd] KV connector configured: per-request records DISABLED (cells only)")
        else:
            logger.warning("[specopd] telemetry DISABLED, pure pass-through: %s", self._sp_reason)

    # ------------------------------------------------------------------ checks / state

    def _sp_check(self) -> str | None:
        """None if telemetry may run; otherwise the reason for pure pass-through (hooks still forward unchanged).
        Version and signatures were already checked in __new__."""
        try:
            spec = self.vllm_config.speculative_config
            if spec is not None and not self.use_eagle:
                return f"speculative method {spec.method!r} is not eagle-family"
            if isinstance(self, AsyncScheduler) != bool(self.scheduler_config.async_scheduling):
                return "scheduler base class does not match async_scheduling"
            return None
        except Exception as e:
            return f"startup check failed: {e!r}"

    def _sp_reset(self) -> None:
        # Rebinds fresh containers: whatever a pull returned is never mutated again.
        self._sp_cells: dict[tuple[int, int, int], list] = {}
        self._sp_hist: dict[int, list[int]] = {}
        self._sp_tot = [0, 0, 0]  # drafts, drafted, accepted
        self._sp_tot_pos = [0] * self._sp_K
        self._sp_orph = [0, 0, 0, 0, 0]  # requests, steps, drafted, accepted, gen_tokens
        self._sp_orph_acc = [0] * self._sp_K
        self._sp_steps = 0
        self._sp_empty = 0
        self._sp_t_reset = time.time()

    def _sp_new_acc(self, t0: float | None) -> list:
        K = self._sp_K
        return [0, 0, 0, t0, [0] * K, [0] * K, [0] * (K + 1) ** 2]

    def _sp_fail(self, e: Exception) -> None:
        self._sp_err += 1
        self._sp_err_last = repr(e)[:300]
        if self._sp_err <= _MAX_ERR_LOGS:
            logger.exception("[specopd] telemetry error #%d (scheduling unaffected)", self._sp_err)
        if self._sp_err >= _MAX_ERRORS and self._sp_on:
            self._sp_on = self._sp_export = False
            self._sp_reason = f"{self._sp_err} telemetry errors, last {self._sp_err_last}"
            logger.error("[specopd] telemetry DISABLED after %d errors", self._sp_err)

    # ------------------------------------------------------------------ vLLM hooks

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        if self._sp_on and scheduler_output.num_scheduled_tokens:  # empty schedules advance no state
            try:
                t = time.perf_counter()
                if len(self._sp_tsched) > 64:  # every output is normally popped in update_from_output
                    self._sp_tsched.clear()
                self._sp_tsched[id(scheduler_output)] = t
                if scheduler_output.scheduled_new_reqs:
                    now = time.time()
                    for nr in scheduler_output.scheduled_new_reqs:
                        req = self.requests.get(nr.req_id)
                        if req is not None and getattr(req, "_specopd", None) is None:
                            req._specopd = self._sp_new_acc(now)
            except Exception as e:
                self._sp_fail(e)
        super()._update_after_schedule(scheduler_output)

    def update_from_output(
        self, scheduler_output: SchedulerOutput, model_runner_output: ModelRunnerOutput
    ) -> dict[int, EngineCoreOutputs]:
        pre = None
        if self._sp_on:
            try:
                pre = self._sp_pre(scheduler_output)
            except Exception as e:
                self._sp_fail(e)
        outputs = super().update_from_output(
            scheduler_output=scheduler_output, model_runner_output=model_runner_output
        )
        if pre is not None:
            try:
                self._sp_post(pre, outputs)
            except Exception as e:
                self._sp_fail(e)
        return outputs

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        if self._sp_on:
            try:
                self._sp_observe(num_draft_tokens, num_accepted_tokens, num_invalid_spec_tokens, request_id)
            except Exception as e:
                self._sp_fail(e)
        # Identity policy: the base (and so vLLM's Prometheus counters) sees the untouched values.
        return super().make_spec_decoding_stats(
            spec_decoding_stats,
            num_draft_tokens=num_draft_tokens,
            num_accepted_tokens=num_accepted_tokens,
            num_invalid_spec_tokens=num_invalid_spec_tokens,
            request_id=request_id,
        )

    def _free_request(self, request: Request, delay_free_blocks: bool = False) -> dict[str, Any] | None:
        params = super()._free_request(request, delay_free_blocks=delay_free_blocks)
        if not self._sp_on:
            return params
        try:
            if self._sp_orphaning:  # via finish_requests: the return value is dropped (scheduler.py:1814)
                self._sp_orphan(request)
                return params
            if not self._sp_export:
                return params
            rec = self._sp_record(request)
        except Exception as e:
            self._sp_fail(e)
            return params
        return {**(params or {}), "specopd": rec}

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[tuple[str, int]]:
        # Frontend aborts (incl. stop strings, output_processor.py:630,670 -> core.py:332), failed KV loads
        # (scheduler.py:1489) and ended streaming sessions (:1746) free here; mark them as orphans.
        self._sp_orphaning += 1
        try:
            return super().finish_requests(request_ids=request_ids, finished_status=finished_status)
        finally:
            self._sp_orphaning -= 1

    # ------------------------------------------------------------------ telemetry bodies

    def _sp_pre(self, so) -> tuple | None:
        t_sched = self._sp_tsched.pop(id(so), None)
        n = len(so.num_scheduled_tokens)
        if not n:
            self._sp_empty += 1
            return None
        kv = nspec = 0
        for ids in so.scheduled_spec_decode_tokens.values():
            lv = len(ids)
            nspec += lv
            if lv > kv:
                kv = lv
        self._sp_n = n
        self._sp_st = [0, 0]
        # Pure decode: every scheduled request runs exactly 1 + its drafts (no prefill chunk in the step).
        return n, kv, so.total_num_scheduled_tokens == n + nspec, t_sched

    def _sp_post(self, pre: tuple, outputs) -> None:
        n, kv, pure, t_sched = pre
        t = time.perf_counter()
        t0 = self._sp_t_out if t_sched is None else max(self._sp_t_out, t_sched)
        self._sp_t_out = t
        self._sp_steps += 1
        if not pure or t0 <= 0.0:
            return
        dt = t - t0
        # What vLLM appended (stop-truncated, discarded async tokens and finished rows excluded), not sampled.
        emitted = sum(len(o.new_token_ids) for eco in outputs.values() for o in eco.outputs)
        key = (n, kv, self._sp_kd)
        c = self._sp_cells.get(key)
        if c is None:
            c = self._sp_cells[key] = [0, 0.0, 0.0, 0, 0, 0]
        c[0] += 1
        c[1] += dt
        c[2] += dt * dt
        c[3] += self._sp_st[0]
        c[4] += self._sp_st[1]
        c[5] += emitted

    def _sp_observe(self, kd: int, a: int, invalid: dict[str, int] | None, request_id: str) -> None:
        K = self._sp_K
        # What vLLM's counter receives (scheduler.py:1987-1988); invalid is only non-empty with structured output.
        drafted = kd - invalid.get(request_id, 0) if invalid else kd
        hidx = kd * (K + 1) + a
        tot = self._sp_tot  # mirrors SpecDecodingStats.observe_draft (v1/spec_decode/metrics.py:39-45)
        tot[0] += 1
        tot[1] += drafted
        tot[2] += a
        pos = self._sp_tot_pos
        for j in range(a):
            pos[j] += 1
        st = self._sp_st
        st[0] += drafted
        st[1] += a
        h = self._sp_hist.get(self._sp_n)
        if h is None:
            h = self._sp_hist[self._sp_n] = [0] * (K + 1) ** 2
        h[hidx] += 1
        req = self.requests.get(request_id)
        if req is None:
            return
        r = getattr(req, "_specopd", None)
        if r is None:
            r = req._specopd = self._sp_new_acc(None)
        r[_STEPS] += 1
        r[_DRAFTED] += drafted
        r[_ACCEPTED] += a
        tried, acc = r[_TRIED], r[_ACC]
        for j in range(min(kd, a + 1)):  # position j verified iff j < kd and all of 0..j-1 accepted
            tried[j] += 1
        for j in range(a):
            acc[j] += 1
        r[_H][hidx] += 1

    def _sp_orphan(self, req) -> None:
        r = getattr(req, "_specopd", None)
        o = self._sp_orph
        o[0] += 1
        o[4] += req.num_output_tokens
        if r is not None:
            o[1] += r[_STEPS]
            o[2] += r[_DRAFTED]
            o[3] += r[_ACCEPTED]
            oa = self._sp_orph_acc
            for j, v in enumerate(r[_ACC]):
                oa[j] += v

    def _sp_record(self, req) -> dict:
        r = getattr(req, "_specopd", None) or self._sp_new_acc(None)
        return {
            "v": SCHEMA,
            "K": self._sp_K,
            "steps": r[_STEPS],
            "drafted": r[_DRAFTED],
            "accepted": r[_ACCEPTED],
            "tried": list(r[_TRIED]),
            "acc": list(r[_ACC]),
            "H": list(r[_H]),
            "t_arrival": req.arrival_time,
            "t_sched0": r[_T0],
            "t_fin": time.time(),
            "gen_tokens": req.num_output_tokens,
            "prompt_tokens": req.num_prompt_tokens,
            "preemptions": req.num_preemptions,
        }

    # ------------------------------------------------------------------ control channel

    def specopd_handle(self, op: str, payload: dict | None = None) -> dict:
        """ops: pull {reset=True} -> header + cells/hist_n/totals (then reset); get -> header;
        set_policy {mode='identity', K, version} -> {ok, K, version} (only the identity policy exists)."""
        try:
            payload = payload or {}
            if op == "pull":
                return self._sp_pull(bool(payload.get("reset", True)))
            if op == "get":
                return self._sp_header()
            if op == "set_policy":
                return self._sp_set_policy(payload)
            return {"ok": False, "error": f"unknown op {op!r}"}
        except Exception as e:
            self._sp_fail(e)
            return {"ok": False, "error": repr(e)}

    def _sp_header(self) -> dict:
        return {
            "v": SCHEMA,
            "ok": True,
            "K": self._sp_K,
            "version": self._sp_version,
            "policy": "identity",
            "passthrough": not self._sp_on,
            "reason": self._sp_reason,
            "export": self._sp_export,
            "log_stats": bool(self.log_stats),
            "async": isinstance(self, AsyncScheduler),
            "pid": os.getpid(),
            "t": time.time(),
            "n_requests": len(self.requests),
            "errors": self._sp_err,
            "err_last": self._sp_err_last,
        }

    def _sp_pull(self, reset: bool) -> dict:
        out = self._sp_header()
        out.update(engine_steps=self._sp_steps, empty_schedules=self._sp_empty, t_since=self._sp_t_reset)
        cells, hist, tot, pos = self._sp_cells, self._sp_hist, self._sp_tot, self._sp_tot_pos
        orph, orph_acc = self._sp_orph, self._sp_orph_acc
        if reset:
            # Swap-and-reset IS the copy: the old containers are never mutated again, so the output thread can
            # serialize them after we return while the busy loop keeps stepping.
            self._sp_reset()
        else:
            cells, hist, tot, pos, orph, orph_acc = copy.deepcopy((cells, hist, tot, pos, orph, orph_acc))
        out["cells"] = {f"{n},{kv},{kd}": v for (n, kv, kd), v in cells.items()}
        out["hist_n"] = {str(n): h for n, h in hist.items()}
        out["totals"] = {"drafts": tot[0], "drafted": tot[1], "accepted": tot[2], "acc_pos": pos}
        out["orphans"] = dict(zip(("requests", "steps", "drafted", "accepted", "gen_tokens"), orph), acc=orph_acc)
        return out

    def _sp_set_policy(self, p: dict) -> dict:
        if p.get("mode", "identity") != "identity" or int(p.get("K", self._sp_K)) != self._sp_K:
            return {
                "ok": False,
                "K": self._sp_K,
                "version": self._sp_version,
                "error": "telemetry build: only {mode: 'identity', K: engine K} is accepted",
            }
        self._sp_version = int(p.get("version", self._sp_version + 1))
        return {"ok": True, "K": self._sp_K, "version": self._sp_version}


class DepthAsyncScheduler(_DepthMixin, AsyncScheduler):
    """Async scheduling on (vLLM's default with eagle-family SD or no SD)."""


class DepthScheduler(_DepthMixin, Scheduler):
    """Async scheduling off (verl ASYNC_SCHED=off, or a CPU ngram drafter)."""


def _engine_specopd_rpc(self, op: str, payload: dict | None = None) -> dict:
    """EngineCore UTILITY method: ``engine_core.call_utility[_async]("specopd_rpc", op, payload)``.

    Utility requests are handled on the EngineCore busy-loop thread between engine steps
    (v1/engine/core.py:1136-1175, dispatch at :1252-1264), so this never races schedule()/update_from_output().
    Returns plain dict/list/int/float/str, which msgpack round-trips without VLLM_ALLOW_INSECURE_SERIALIZATION.
    """
    handle = getattr(self.scheduler, "specopd_handle", None)
    if handle is None:
        return {"ok": False, "error": f"scheduler {type(self.scheduler).__name__} is not a specopd scheduler"}
    return handle(op, payload or {})


if not hasattr(EngineCore, "specopd_rpc"):
    EngineCore.specopd_rpc = _engine_specopd_rpc
