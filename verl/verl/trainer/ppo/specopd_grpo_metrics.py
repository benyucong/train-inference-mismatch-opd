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
"""Driver-side GRPO group metrics.

``grpo/nondegenerate_frac``: the fraction of prompt groups (uid) with 0 < #correct < group size, i.e. the groups whose
GRPO advantage is not identically zero (the groups that carry any GRPO gradient).
A response is CORRECT when its verifier score (the sum of its token-level scores = the math verifier's 0/1 reward) is
above ``CORRECT_THRESHOLD``. Synthetic padding rows (balance padding) are excluded; multi-turn sessions count once, by
their final output, exactly as GRPO's own grouping in compute_advantage_for_multi_trajectories.

Also logged: ``grpo/n_groups``, ``grpo/group_size_min`` / ``_max`` (a short group would make "< 8" read differently;
here "< group size"), ``grpo/all_correct_frac`` and ``grpo/all_wrong_frac`` (the two degenerate kinds; they sum with
nondegenerate_frac to 1).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence

import numpy as np

CORRECT_THRESHOLD = 0.5


def grpo_group_metrics(
    seq_scores: Sequence[float],
    uids: Sequence,
    batch_keys: Optional[Sequence[str]] = None,
    is_padding: Optional[Sequence[bool]] = None,
) -> dict:
    """seq_scores[i] = response i's verifier score, uids[i] its prompt-group id, batch_keys[i] its TransferQueue key
    ``{uid}_{session}_{index}`` (optional: only the final output of each session counts), is_padding[i] True for a
    synthetic padding row (excluded). Returns {} when no real row is left."""
    n = len(seq_scores)
    if (
        len(uids) != n
        or (batch_keys is not None and len(batch_keys) != n)
        or (is_padding is not None and len(is_padding) != n)
    ):
        raise ValueError("grpo_group_metrics: seq_scores, uids, batch_keys and is_padding must have equal length")
    rows = [i for i in range(n) if not (is_padding is not None and bool(is_padding[i]))]
    if batch_keys is not None:
        final: dict[str, tuple[int, int]] = {}
        for i in rows:
            uid, session, index = str(batch_keys[i]).rsplit("_", 2)
            sk = f"{uid}_{session}"
            if sk not in final or final[sk][0] < int(index):
                final[sk] = (int(index), i)
        rows = sorted(i for _, i in final.values())
    groups: dict = defaultdict(list)
    for i in rows:
        groups[str(uids[i])].append(float(seq_scores[i]) > CORRECT_THRESHOLD)
    if not groups:
        return {}
    sizes = np.array([len(v) for v in groups.values()])
    n_correct = np.array([sum(v) for v in groups.values()])
    return {
        "grpo/nondegenerate_frac": float(np.mean((n_correct > 0) & (n_correct < sizes))),
        "grpo/all_correct_frac": float(np.mean(n_correct == sizes)),
        "grpo/all_wrong_frac": float(np.mean(n_correct == 0)),
        "grpo/n_groups": int(len(sizes)),
        "grpo/group_size_min": int(sizes.min()),
        "grpo/group_size_max": int(sizes.max()),
    }
