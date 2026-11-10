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
"""Measurement helpers for verl's v1 ASYNC trainers.
Stdlib only, never imported by the sync trainer path, never raises into a training step.

1. ``[STEP-WALL] step=<n> t=<unix float>``: one line per COMPLETED training step, printed by the trainer (driver side,
   the TaskRunner actor that also prints the console metrics) after the step's metrics are logged, plus one clock-start
   line before the training loop stamped with the last completed step (``step=0`` on a fresh run), so
   wall(1) = t(1) - t(0) holds the warmup submission and the pipeline fill. The console logger
   prints no time, and in async modes ``timing_s/gen`` is the trainer waiting for finished groups (plus abort / sleep),
   not generation, so per-step wall time comes from consecutive stamps. ``STEP_WALL_RE`` is the line format that
   ``parse_step_wall`` reads back from a log (``step_wall_line`` writes it); the line is unique per step, so Ray log
   dedup (normalizes numbers across PROCESSES within a few seconds) cannot fold it, and the launcher sets
   RAY_DEDUP_LOGS=0 besides.

2. ``SampleWaitClock``: what ``ReplayBufferAsync.wait_for_sampleable`` spent per sample() call -- the seconds and polls
   until enough finished groups existed, and the polls / seconds during which a FULL batch was already finished but the
   dropless ``wait`` rule (an in-flight prompt at span >= max_off_policy_threshold) held it back. The second is the
   throughput price of the lag budget itself; nothing upstream logs it. Logged as training/async/* (plain Python
   numbers, so the console line reads ``key:1.5``, never a numpy repr).
"""

from __future__ import annotations

import re
import sys
import time

STEP_WALL_TAG = "[STEP-WALL]"
# The [STEP-WALL] line that step_wall_line writes and parse_step_wall reads. Ray prefixes forwarded actor lines with
# "(TaskRunner pid=...) ", so parse_step_wall searches for the tag anywhere in the text instead of anchoring at ^.
STEP_WALL_RE = re.compile(r"\[STEP-WALL\] step=(\d+) t=(\d+\.\d+)(?:\s|$)")

ASYNC_METRIC_PREFIX = "training/async"


def step_wall_line(step: int, t: float) -> str:
    """The exact line printed for a completed step (microsecond resolution)."""
    return f"{STEP_WALL_TAG} step={int(step)} t={float(t):.6f}"


def emit_step_wall(step: int, t: float | None = None, file=None) -> str:
    """Print (flushed) and return the [STEP-WALL] line for ``step``; ``t`` defaults to time.time() (unix seconds)."""
    line = step_wall_line(step, time.time() if t is None else t)
    try:
        print(line, file=file if file is not None else sys.stdout, flush=True)
    except Exception:  # a closed stdout must not cost a training step
        pass
    return line


def parse_step_wall(text: str) -> list[tuple[int, float]]:
    """All (step, t) stamps in ``text``, in order of appearance."""
    return [(int(m.group(1)), float(m.group(2))) for m in STEP_WALL_RE.finditer(text)]


class SampleWaitClock:
    """Accounting for one ``wait_for_sampleable`` call. Call ``poll(enough, sampleable, now)`` once per poll that does
    NOT return (``enough`` = the call's own exit test, ``sampleable`` = len(sampleable) >= target), then ``done(now)``.

    A poll with ``sampleable and not enough`` is one where only the wait rule held a full batch back (the only other
    exit condition of ReplayBufferAsync._has_enough_samples). Held seconds run from the first such poll to the next poll
    that is not held, or to ``done``.
    """

    def __init__(self, now: float | None = None):
        self.t_start = time.monotonic() if now is None else now
        self.polls = 0
        self.held_polls = 0
        self.held_s = 0.0
        self._held_since: float | None = None

    def poll(self, enough: bool, sampleable: bool, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        held = bool(sampleable) and not bool(enough)
        if held:
            self.held_polls += 1
            if self._held_since is None:
                self._held_since = now
        elif self._held_since is not None:
            self.held_s += now - self._held_since
            self._held_since = None
        self.polls += 1

    def done(self, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now
        held_s = self.held_s + (now - self._held_since if self._held_since is not None else 0.0)
        return {
            f"{ASYNC_METRIC_PREFIX}/sample_wait_s": float(now - self.t_start),
            f"{ASYNC_METRIC_PREFIX}/sample_polls": int(self.polls),
            f"{ASYNC_METRIC_PREFIX}/wait_rule_held_polls": int(self.held_polls),
            f"{ASYNC_METRIC_PREFIX}/wait_rule_held_s": float(held_s),
        }
