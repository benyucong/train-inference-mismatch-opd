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
"""Plain REINFORCE for the specopd RL contrast (launcher ALGO=reinforce; ``algorithm.adv_estimator=specopd_reinforce``).

The advantage of every response token is the response's verifier score, A_t = R(y) = sum_t token_level_rewards[t]
(the math verifier's 0/1 reward sits on the last response token), broadcast over the response mask. No baseline, no
group mean, no std normalization, no whitening, no discount: the policy gradient is E_q[R(y) grad log p(y)]. With a 0/1
verifier every advantage is >= 0, so E_q[A] >= 0 by construction -- the contrast with GRPO (whose group-mean
baseline makes E_q[A] ~ 0 per group).

Registered in core_algos' ADV_ESTIMATOR_REGISTRY under ``ESTIMATOR_NAME`` (core_algos imports this module at its end),
so it goes through verl's generic ``compute_advantage`` branch like any registered estimator. Inert unless selected.

``reinforce_advantage_metrics`` is the driver-side log: ``reinforce/adv_mean`` (token-weighted mean of the advantage
over real response tokens = the token-weighted E_q[A] the step trained on), ``reinforce/adv_seq_mean`` (per-response
mean), ``reinforce/correct_frac`` (responses with score > 0.5), ``reinforce/nonzero_adv_frac``,
``reinforce/n_responses`` and ``reinforce/n_tokens``. Synthetic balance-padding rows are excluded.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

ESTIMATOR_NAME = "specopd_reinforce"
CORRECT_THRESHOLD = 0.5


def compute_specopd_reinforce_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config=None,
    index=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A[i, t] = (sum_t' token_level_rewards[i, t']) * response_mask[i, t]; returns (advantages, returns), equal.

    ``config`` / ``index`` (the uid groups) / any other kwarg verl's generic branch passes are accepted and ignored:
    there is no baseline, so the groups play no role."""
    if token_level_rewards.shape != response_mask.shape:
        raise ValueError(
            f"specopd_reinforce: token_level_rewards {tuple(token_level_rewards.shape)} != response_mask "
            f"{tuple(response_mask.shape)}"
        )
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        advantages = scores.unsqueeze(-1) * response_mask.to(scores.dtype)
    return advantages, advantages


def reinforce_advantage_metrics(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    seq_scores: Sequence[float],
    is_padding: Optional[Sequence[bool]] = None,
) -> dict:
    """Advantage statistics over the real (non-padding) rows; {} when no real row is left."""
    n = advantages.shape[0]
    if (
        response_mask.shape != advantages.shape
        or len(seq_scores) != n
        or (is_padding is not None and len(is_padding) != n)
    ):
        raise ValueError("reinforce_advantage_metrics: advantages, response_mask, seq_scores, is_padding must align")
    keep = [i for i in range(n) if not (is_padding is not None and bool(is_padding[i]))]
    if not keep:
        return {}
    idx = torch.tensor(keep, dtype=torch.long, device=advantages.device)
    adv = advantages.index_select(0, idx).float()
    mask = response_mask.index_select(0, idx).float()
    n_tok = mask.sum()
    tok_len = mask.sum(-1).clamp(min=1.0)
    seq_adv = (adv * mask).sum(-1) / tok_len
    scores = torch.tensor([float(seq_scores[i]) for i in keep], dtype=torch.float64)
    return {
        "reinforce/adv_mean": float((adv * mask).sum() / n_tok) if n_tok > 0 else float("nan"),
        "reinforce/adv_seq_mean": float(seq_adv.mean()),
        "reinforce/correct_frac": float((scores > CORRECT_THRESHOLD).double().mean()),
        "reinforce/nonzero_adv_frac": float(((adv * mask).abs().sum(-1) > 0).float().mean()),
        "reinforce/n_responses": int(len(keep)),
        "reinforce/n_tokens": int(n_tok.item()),
    }
