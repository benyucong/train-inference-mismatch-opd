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
"""Trainer side of the specopd speculative-decoding telemetry (opt-in: SPECOPD_TELEMETRY=1).

The rollout engines run the ``specopd_vllm.depth_sched`` scheduler plugin, which keeps per-step cost cells inside
EngineCore and exports one record per finished request through ``kv_transfer_params["specopd"]``. Per training step:

  * ``pull_engines`` (sync trainer, on_sample_end, BEFORE the replicas sleep): every rollout server's
    ``specopd_rpc("pull")`` -> per-engine (replica_rank, dp_rank) cells / hist_n / totals / orphans, reset;
  * ``step_report`` (trainer_base._step_once, after the /metrics scrape): per-request records from TransferQueue
    ``extra_fields`` joined with uid / data_source / replica_rank, one JSON line to
    ``$SPECOPD_TELEMETRY_DIR/step_NNNNN.jsonl`` (temp + os.replace) and a few ``specopd_tel/*`` scalars;
  * ``discard_validation`` (on_validate_end): pull-and-drop the validation traffic and re-baseline the /metrics
    scrape, so neither leaks into the next training step.

Estimators: acceptance is aggregated PER REQUEST first and then weighted by generated tokens
(pooled per-position counts are depth-policy-weighted, the per-request alpha_j = acc[j]/tried[j] is not).
Cross-check: sum of per-request ``drafted`` + the engines' orphans (requests the frontend finished -- aborts, stop
strings -- which carry no record) == sum of the engines' totals == the Prometheus delta. Nothing here may fail a
training step: every entry point catches, logs and returns.

SYNC TRAINER ONLY (trainer.trainer_mode=sync, the launcher's mode): only trainer_sync pulls the engines, and the
async trainers' FullyAsyncLLMServerClient rebuilds each TokenOutput copying just four extra_fields keys
(workers/rollout/llm_server.py:276-351), dropping kv_transfer_params and replica_rank. There, every sample would be
reported missing; step_report warns once instead of pretending.

Timing: the engine pull runs inside the "gen" timer (on_sample_end, before sleep) under its own ``specopd_pull``
timer; compare generation across runs as ``timing_s/gen - timing_s/specopd_pull``.
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_PREFIX = "specopd_tel"
_WARNED: set[str] = set()


def enabled() -> bool:
    return os.environ.get("SPECOPD_TELEMETRY", "0") not in ("0", "", "false", "False")


def pull_engines(server_handles: list, timeout: float = 120.0) -> list[dict]:
    """``specopd_rpc("pull")`` on every rollout server (resets the engines' per-step state) -> one dict per
    EngineCore (a replica with DP>1 has several; headless node_rank>0 servers return None).

    The sync trainer calls this inside its "gen" timer, wrapped in its own ``specopd_pull`` timer: the comparable
    generation time of a run with telemetry is ``timing_s/gen - timing_s/specopd_pull``."""
    t0 = time.perf_counter()
    try:
        import ray

        refs = [h.specopd_rpc.remote("pull", {"reset": True}) for h in server_handles]
        out = [e for r in ray.get(refs, timeout=timeout) if r is not None for e in (r if isinstance(r, list) else [r])]
    except Exception as e:
        logger.warning(f"[SPECOPD-TEL] engine pull failed: {e!r}")
        out = [{"ok": False, "error": repr(e)}]
    for r in out:
        r["pull_s"] = time.perf_counter() - t0
    return out


def discard_validation(llm_server_manager) -> None:
    """Drop what validation added to the engines' telemetry and to the /metrics delta baseline."""
    try:
        from verl.utils.specopd_metrics import rebaseline

        pull_engines(llm_server_manager.server_handles)
        rebaseline(llm_server_manager.get_addresses())
    except Exception as e:
        logger.warning(f"[SPECOPD-TEL] validation discard failed: {e!r}")


def _accept_len(r: dict) -> float | None:
    return 1.0 + r["accepted"] / r["steps"] if r.get("steps") else None


def _weighted(pairs: list[tuple[float, float]]) -> float | None:
    w = sum(p[1] for p in pairs)
    return sum(v * wt for v, wt in pairs) / w if w > 0 else None


def _field_list(data, key) -> list:
    """A non-tensor TransferQueue field as a plain list. ``data.pop(key)`` returns the NonTensorStack (``.tolist()``,
    upstream's pattern: trainer_base.py extra_fields readers); ``data[key]`` returns a list-like LinkedList without
    ``.tolist()``."""
    v = data.pop(key)
    return v.tolist() if hasattr(v, "tolist") else list(v)


def _fetch_samples(batch) -> list[dict]:
    import transfer_queue as tq

    keep = [(k, t) for k, t in zip(batch.keys, batch.tags, strict=True) if not t.get("is_padding", False)]
    keys = [k for k, _ in keep]
    extra = tq.kv_batch_get(keys=keys, partition_id=batch.partition_id, select_fields=["extra_fields"])
    extra = _field_list(extra, "extra_fields")
    try:
        sources = tq.kv_batch_get(keys=keys, partition_id=batch.partition_id, select_fields=["data_source"])
        sources = _field_list(sources, "data_source")
    except Exception:  # a missing label must not cost the telemetry
        sources = [None] * len(keys)
    samples = []
    for (key, tag), ef, src in zip(keep, extra, sources, strict=True):
        ef = getattr(ef, "data", ef)
        ef = ef if isinstance(ef, dict) else {}
        kv = ef.get("kv_transfer_params")
        rec = kv.get("specopd") if isinstance(kv, dict) else None
        samples.append(
            {
                "key": key,
                "uid": key.rsplit("_", 2)[0],  # agent_loop_tq key = {uid}_{session_id}_{index}
                "data_source": getattr(src, "data", src),
                "replica_rank": ef.get("replica_rank"),
                "response_len": tag.get("response_len"),
                "spec": rec,
            }
        )
    return samples


def _scalars(samples: list[dict], engines: list[dict] | None, prom: dict) -> dict[str, float]:
    recs = [s for s in samples if isinstance(s["spec"], dict)]
    out = {f"{_PREFIX}/samples": float(len(samples)), f"{_PREFIX}/missing": float(len(samples) - len(recs))}
    # Acceptance length per request, then weighted by the request's generated tokens.
    al = [(_accept_len(s["spec"]), s["spec"].get("gen_tokens") or 0) for s in recs]
    v = _weighted([(a, w) for a, w in al if a is not None])
    if v is not None:
        out[f"{_PREFIX}/accept_len"] = v
    # Conditional acceptance at draft position j (1-indexed): alpha_j = acc[j]/tried[j] per request, then weighted.
    K = max((len(s["spec"].get("tried") or []) for s in recs), default=0)
    for j in range(K):
        pairs = [
            (s["spec"]["acc"][j] / s["spec"]["tried"][j], s["spec"].get("gen_tokens") or 0)
            for s in recs
            if len(s["spec"].get("tried") or []) > j and s["spec"]["tried"][j] > 0
        ]
        v = _weighted(pairs)
        if v is not None:
            out[f"{_PREFIX}/alpha_{j + 1}"] = v
    by_src: dict[str, list] = {}
    for s, (a, w) in zip(recs, al, strict=True):
        if a is not None:
            by_src.setdefault(str(s["data_source"]), []).append((a, w))
    for src, pairs in by_src.items():
        v = _weighted(pairs)
        if v is not None:
            out[f"{_PREFIX}/accept_len/{src}"] = v
    # Critical path: the sample whose request finished LAST in its engine (wall clock).
    fin = [s for s in recs if s["spec"].get("t_fin") is not None]
    if fin:
        crit = max(fin, key=lambda s: s["spec"]["t_fin"])
        a = _accept_len(crit["spec"])
        if a is not None:
            out[f"{_PREFIX}/crit_accept_len"] = a
        out[f"{_PREFIX}/crit_gen_tokens"] = float(crit["spec"].get("gen_tokens") or 0)
        if crit["replica_rank"] is not None:
            out[f"{_PREFIX}/crit_replica"] = float(crit["replica_rank"])
    # Cross-check: sum of per-request drafts (+ record-less orphans) == engines' totals == Prometheus delta
    # (specopd_metrics.py).
    out[f"{_PREFIX}/drafted_req"] = float(sum(s["spec"].get("drafted", 0) for s in recs))
    out[f"{_PREFIX}/accepted_req"] = float(sum(s["spec"].get("accepted", 0) for s in recs))
    checks = [out[f"{_PREFIX}/drafted_req"]]
    if engines:
        ok = [e for e in engines if e.get("ok")]
        orph = [e["orphans"] for e in ok if isinstance(e.get("orphans"), dict)]
        out[f"{_PREFIX}/pull_s"] = float(engines[0].get("pull_s", 0.0))
        out[f"{_PREFIX}/engines_ok"] = float(len(ok))
        out[f"{_PREFIX}/engines_failed"] = float(len(engines) - len(ok))
        out[f"{_PREFIX}/engine_errors"] = float(sum(e.get("errors", 0) for e in ok))
        out[f"{_PREFIX}/passthrough"] = float(sum(bool(e.get("passthrough")) for e in ok))
        out[f"{_PREFIX}/orphans"] = float(sum(o.get("requests", 0) for o in orph))
        out[f"{_PREFIX}/drafted_orphan"] = float(sum(o.get("drafted", 0) for o in orph))
        checks[0] += out[f"{_PREFIX}/drafted_orphan"]
        out[f"{_PREFIX}/drafted_engine"] = float(sum(e["totals"]["drafted"] for e in ok if "totals" in e))
        checks.append(out[f"{_PREFIX}/drafted_engine"])
        # Pooled pure-decode step time (a diagnostic of the per-replica cost surface, not a policy input).
        cells = [c for e in ok for c in (e.get("cells") or {}).values()]
        n_steps = sum(c[0] for c in cells)
        if n_steps:
            out[f"{_PREFIX}/decode_step_ms"] = 1e3 * sum(c[1] for c in cells) / n_steps
    if "specdec/draft_tokens" in prom:
        out[f"{_PREFIX}/drafted_prom"] = float(prom["specdec/draft_tokens"])
        checks.append(out[f"{_PREFIX}/drafted_prom"])
    out[f"{_PREFIX}/xcheck_ok"] = float(len(checks) > 1 and all(round(c) == round(checks[0]) for c in checks))
    return out


def _write(out_dir: str, name: str, line: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    final, tmp = os.path.join(out_dir, name), os.path.join(out_dir, f".{name}.partial")
    with open(tmp, "w") as fh:
        fh.write(json.dumps(line, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)


def step_report(batch, engines: list[dict] | None, prom: dict, global_steps: int, trigger: int = 0) -> dict:
    """Join this step's per-request records with the pulled engine cells; write the JSON line; return scalars."""
    try:
        if batch.partition_id != "train":  # validation records are dropped
            return {}
        if engines is None and "no_pull" not in _WARNED:
            _WARNED.add("no_pull")
            logger.warning(
                "[SPECOPD-TEL] no engine pull this step: SPECOPD_TELEMETRY supports the SYNC trainer only "
                "(async trainers neither pull nor keep kv_transfer_params); records will read as missing"
            )
        samples = _fetch_samples(batch)
        scalars = _scalars(samples, engines, prom or {})
        out_dir = os.environ.get("SPECOPD_TELEMETRY_DIR")
        if out_dir:
            name = f"step_{global_steps:05d}" + (f"_{trigger}" if trigger else "") + ".jsonl"
            line = {
                "v": 1,
                "step": global_steps,
                "trigger": trigger,
                "t": time.time(),
                "engines": engines,
                "samples": samples,
                "prom": {k: v for k, v in (prom or {}).items() if k.startswith("specdec/")},
                "scalars": scalars,
            }
            _write(out_dir, name, line)
        n, miss = int(scalars[f"{_PREFIX}/samples"]), int(scalars[f"{_PREFIX}/missing"])
        g = lambda k: scalars.get(f"{_PREFIX}/{k}")  # noqa: E731
        print(
            f"[SPECOPD-TEL] step={global_steps} records={n - miss}/{n} accept_len={g('accept_len')} "
            f"drafted req={g('drafted_req')} orphan={g('drafted_orphan')} engine={g('drafted_engine')} "
            f"prom={g('drafted_prom')} xcheck_ok={g('xcheck_ok')} pull_s={g('pull_s')}",
            flush=True,
        )
        scalars[f"{_PREFIX}/ok"] = 1.0
        return scalars
    except Exception as e:
        logger.warning(f"[SPECOPD-TEL] step {global_steps} telemetry failed: {e!r}")
        return {f"{_PREFIX}/ok": 0.0}
