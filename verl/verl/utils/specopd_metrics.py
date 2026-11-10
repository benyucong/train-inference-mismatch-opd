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
"""Per-step speculative-decoding counters, scraped from the rollout servers' Prometheus /metrics.

vLLM's periodic stat LOG line never prints inside verl's server, but its Prometheus counters are
recorded whenever the engine runs with disable_log_stats=False. Differencing the cumulative
counters between steps gives, per training step: accepted and drafted tokens, draft count and
generated tokens -> acceptance rate and mean acceptance length, which a speculation controller needs
and which a run without speculation reports as zeros (so runs with and without it log the same keys).

Opt-in with SPECOPD_SPEC_METRICS=1. A failed scrape reports specdec/scrape_ok=0 and never raises:
telemetry must not kill a training step.

KV memory:
  (a) the same scrape also reads the counter vllm:num_preemptions -> kvmem/preemptions, the per-step delta summed over
      the replicas (NaN when a server does not expose the counter: never a silent zero). Each server keeps its own
      baseline: a scrape where a server answers without the counter makes that step NaN and drops the server's
      baseline, and its next scrape with the counter only re-baselines it (NaN again), so a delta is never taken from
      a stale or zero base (a server that lost the counter may have restarted and reset it);
  (b) KvMemPoller, a daemon thread the trainer starts at generation start and stops at its end, reads the gauges
      vllm:kv_cache_usage_perc and vllm:num_requests_running from every rollout server every 2 s -> kvmem/usage_max
      (max over replicas and samples), kvmem/running_max, kvmem/poll_ok (1 iff every poll of every server answered
      with both gauges; a server that never answered makes the two maxima NaN). It never raises.
With SPECOPD_KV_METRICS unset the scraper's names, calls and output dict are exactly the ones above.
Memory only: none of these is a latency or throughput measurement.
"""

import math
import os
import re
import threading
import urllib.request

_NAMES = {
    "vllm:spec_decode_num_accepted_tokens": "accepted_tokens",
    "vllm:spec_decode_num_draft_tokens": "draft_tokens",
    "vllm:spec_decode_num_drafts": "num_drafts",
    "vllm:generation_tokens": "generation_tokens",
}
# name[_total]{labels} value -- exact-name match, so *_per_pos and *_created series are skipped.
_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*?)(?:_total)?(\{[^}]*\})?\s+([-+0-9.eE]+|NaN)$")
_PREV: dict[str, float] = {}
_PREV_KV: dict[str, float] = {}   # KV memory (a): each server's last vllm:num_preemptions (its own delta baseline)


# KV memory (a): read only with SPECOPD_KV_METRICS=1
_KV_NAMES = {"vllm:num_preemptions": "preemptions"}
# KV memory (b): the poller's gauges (usage: max over the server's series; running: their sum)
_GAUGES = {"vllm:kv_cache_usage_perc": "usage", "vllm:num_requests_running": "running"}


def kv_metrics_on() -> bool:
    return os.environ.get("SPECOPD_KV_METRICS", "0") not in ("0", "", "false", "False")


def _names() -> dict:
    return dict(_NAMES, **_KV_NAMES) if kv_metrics_on() else _NAMES


def _metrics_text(addr: str, timeout: float) -> str:
    url = addr if addr.startswith("http") else f"http://{addr}"
    with urllib.request.urlopen(f"{url.rstrip('/')}/metrics", timeout=timeout) as r:
        return r.read().decode()


def _scrape(addr: str, timeout: float = 5.0, names: dict = None) -> dict[str, float]:
    names = _NAMES if names is None else names
    text = _metrics_text(addr, timeout)
    tot: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if m and m.group(1) in names and m.group(3) != "NaN":
            key = names[m.group(1)]
            tot[key] = tot.get(key, 0.0) + float(m.group(3))
    return tot


def rebaseline(addresses: list[str]) -> None:
    """Move the delta baseline to the servers' current counters, dropping the traffic since the last scrape
    (e.g. validation) from the next step's delta. Used by SPECOPD_TELEMETRY's validation discard only."""
    if os.environ.get("SPECOPD_SPEC_METRICS", "0") in ("0", "", "false", "False"):
        return
    names = _names()
    cur: dict[str, float] = {}
    kv_now: dict[str, float] = {}
    try:
        for a in addresses:
            got = _scrape(a, names=names)
            if "preemptions" in got:
                kv_now[a] = got["preemptions"]
            for k, v in got.items():
                cur[k] = cur.get(k, 0.0) + v
    except Exception:
        return
    _PREV.clear()
    _PREV.update(cur)
    if kv_metrics_on():
        _PREV_KV.clear()
        _PREV_KV.update(kv_now)


def spec_decode_step_metrics(addresses: list[str]) -> dict[str, float]:
    if os.environ.get("SPECOPD_SPEC_METRICS", "0") in ("0", "", "false", "False"):
        return {}
    kv = kv_metrics_on()
    names = _names()
    cur: dict[str, float] = {}
    kv_now: dict[str, float] = {}
    try:
        for a in addresses:
            got = _scrape(a, names=names)
            if "preemptions" in got:
                kv_now[a] = got["preemptions"]
            for k, v in got.items():
                cur[k] = cur.get(k, 0.0) + v
    except Exception:
        out = {"specdec/scrape_ok": 0.0}
        if kv:
            out["kvmem/preemptions"] = float("nan")
        return out
    delta = {k: cur.get(k, 0.0) - _PREV.get(k, 0.0) for k in _NAMES.values()}
    if kv:
        # per server: NaN unless it exposes the counter now AND has its own baseline (or this is the run's first scrape,
        # whose base is 0, as for the specdec/* deltas); a server without the counter loses its baseline, so its next
        # good scrape only re-baselines it. The step's sum is NaN if any server's part is (no silent zero, no invented
        # delta from another server's partial sum).
        first = not _PREV
        parts = [float("nan") if a not in kv_now else kv_now[a] if first else
                 (kv_now[a] - _PREV_KV[a]) if a in _PREV_KV else float("nan") for a in addresses]
        kv_delta = sum(parts) if parts and all(p == p for p in parts) else float("nan")
        _PREV_KV.clear()
        _PREV_KV.update(kv_now)
    _PREV.clear()
    _PREV.update(cur)
    out = {f"specdec/{k}": v for k, v in delta.items()}
    out["specdec/scrape_ok"] = 1.0
    if delta.get("draft_tokens", 0.0) > 0:
        out["specdec/accept_rate"] = delta["accepted_tokens"] / delta["draft_tokens"]
    if delta.get("num_drafts", 0.0) > 0:
        out["specdec/mean_accept_len"] = 1.0 + delta["accepted_tokens"] / delta["num_drafts"]
    if kv:
        out["kvmem/preemptions"] = kv_delta
    return out


def _scrape_gauges(addr: str, timeout: float) -> dict[str, float]:
    """{'usage': max over series, 'running': sum over series} of one server; a gauge it does not expose is absent."""
    text = _metrics_text(addr, timeout)
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if m and m.group(1) in _GAUGES and m.group(3) != "NaN":
            k, v = _GAUGES[m.group(1)], float(m.group(3))
            if k == "usage":
                out[k] = max(out.get(k, v), v)
            else:
                out[k] = out.get(k, 0.0) + v
    return out


class KvMemPoller:
    """KV memory (b): the KV-use high-water mark of one generation phase. start() polls every server at once and then
    every `period` s in a daemon thread; stop() ends it and returns {kvmem/usage_max, kvmem/running_max, kvmem/poll_ok}.
    Neither ever raises."""

    def __init__(self, addresses, period: float = 2.0, timeout: float = 1.5):
        self.addresses = list(addresses or [])
        self.period, self.timeout = period, timeout
        self._stop = threading.Event()
        self._thread = None
        self._usage: dict = {}
        self._running: dict = {}
        self._fails = 0
        self._polls = 0

    def _poll_once(self):
        for a in self.addresses:
            self._polls += 1
            try:
                g = _scrape_gauges(a, self.timeout)
            except Exception:
                self._fails += 1
                continue
            if "usage" not in g or "running" not in g:
                self._fails += 1
            if "usage" in g:
                self._usage[a] = max(self._usage.get(a, -math.inf), g["usage"])
            if "running" in g:
                self._running[a] = max(self._running.get(a, -math.inf), g["running"])

    def _loop(self):
        try:
            while True:
                self._poll_once()
                if self._stop.wait(self.period):
                    return
        except Exception:
            self._fails += 1

    def start(self):
        try:
            self._thread = threading.Thread(target=self._loop, name="specopd-kvmem-poller", daemon=True)
            self._thread.start()
        except Exception:
            self._thread = None
            self._fails += 1
        return self

    @classmethod
    def start_for(cls, addresses, period: float = 2.0, timeout: float = 1.5):
        try:
            return cls(addresses, period=period, timeout=timeout).start()
        except Exception:
            return None

    def stop(self) -> dict[str, float]:
        try:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=self.period + self.timeout * max(1, len(self.addresses)) + 1.0)
            alive = self._thread is not None and self._thread.is_alive()
            every = bool(self.addresses) and all(a in self._usage and a in self._running for a in self.addresses)
            ok = every and self._fails == 0 and self._polls > 0 and not alive and self._thread is not None
            usage = max(self._usage.values()) if every else float("nan")
            running = max(self._running.values()) if every else float("nan")
            return {"kvmem/usage_max": float(usage), "kvmem/running_max": float(running),
                    "kvmem/poll_ok": 1.0 if ok else 0.0}
        except Exception:
            return {"kvmem/usage_max": float("nan"), "kvmem/running_max": float("nan"), "kvmem/poll_ok": 0.0}
