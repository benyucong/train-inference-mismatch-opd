# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.trainer.ppo.score_centering import score_centering_logits_processor
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

# Per-token outputs of the score-centering logits processor (verl/trainer/ppo/score_centering.py).
SCORE_CENTERING_KEYS = ("sc_correction", "sc_sampler_head_mass", "sc_train_head_mass")

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


def is_score_centering_enabled(config: ActorConfig) -> bool:
    """Whether the actor's rollout-correction mirror turns score centering on (the same flag the v1 trainer
    reads for the ``score_centering`` micro-batch flag)."""
    rollout_correction = config.policy_loss.get("rollout_correction", None) or {}
    return bool(rollout_correction.get("score_centering", False))


def _distillation_metric_name(key: str) -> str:
    # The policy loss names its own metrics actor/*; bypass mode adds rollout_corr/* mismatch metrics, which keep
    # their name (slicing len("actor/") off every key would turn rollout_corr/kl into t_corr/kl).
    return f"distillation/{key[len('actor/') :]}" if key.startswith("actor/") else key


def _global_token_mean_metric(values: torch.Tensor, response_mask: torch.Tensor, config: ActorConfig) -> Metric:
    """Token mean of ``values`` over the valid response tokens of the whole global batch.

    Uses agg_loss's token-mean with the global token count (``config.global_batch_info``), SUM-aggregated over
    micro-batches and averaged over DP ranks exactly like ``distillation/loss``. It is therefore token-weighted
    however dynamic bsz splits the batch, unlike a per-micro-batch mean (AggregationType.MEAN averages the
    micro-batch means unweighted).
    """
    dp_size = config.global_batch_info.get("dp_size", 1) or 1
    batch_num_tokens = config.global_batch_info.get("batch_num_tokens", None)
    with torch.no_grad():
        value = agg_loss(
            loss_mat=values.detach().float(),
            loss_mask=response_mask,
            loss_agg_mode="token-mean",
            dp_size=dp_size,
            batch_num_tokens=batch_num_tokens,
        )
    global_normalized = dp_size > 1 or batch_num_tokens is not None
    return Metric(value=value, aggregation=AggregationType.SUM if global_normalized else AggregationType.MEAN)


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    if response_mask.is_nested:
        distillation_losses_response = distillation_losses[response_mask.bool().to_padded_tensor(False)]
    else:
        distillation_losses_response = distillation_losses[response_mask.bool()]
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    match config.strategy:
        # VeOmni uses FSDP2 internally, so its loss computation is identical to FSDP.
        case "fsdp" | "veomni" | "fsdp2":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
        case "megatron":
            import verl.trainer.distillation.megatron.losses as megatron_losses

            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
    )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    distillation_loss_config = distillation_config.distillation_loss
    score_centering = is_score_centering_enabled(config)

    # Called as logits processor
    if student_logits is not None:
        if score_centering:
            # Vanilla (k1) OPD leaves the slot free -- it has no top-k loss -- so it carries the sampler-head
            # centering term instead. Top-k distillation + score centering is rejected by validate_config.
            if distillation_loss_config.loss_settings.use_topk:
                raise NotImplementedError("score centering is not supported with top-k distillation losses.")
            return score_centering_logits_processor(student_logits, data, config, data_format)
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # Called as final policy loss
    if score_centering and not distillation_loss_config.use_policy_gradient:
        raise ValueError("score centering with distillation needs distillation_loss.use_policy_gradient=True.")
    if score_centering and "sc_correction" not in model_output:
        # same guard as score_centering_ppo_loss: without the micro-batch flag the run would silently train plain PG
        raise RuntimeError(
            "score centering is enabled but the logits processor did not run: the trainer must set the "
            "score_centering micro-batch flag (supported by the v1 trainer only)."
        )
    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
    if not distillation_loss_config.use_task_rewards and not distillation_loss_config.use_policy_gradient:
        # no need to compute policy loss
        policy_loss = 0.0
        policy_metrics = {}
    else:
        task_model_output = model_output
        if score_centering and distillation_loss_config.score_centering_probe:
            # probe: the centering term enters no loss, the task-reward policy-gradient term included
            task_model_output = {k: v for k, v in model_output.items() if k not in SCORE_CENTERING_KEYS}
        policy_loss, policy_metrics = ppo_loss(config, task_model_output, data, dp_group)
        if not distillation_loss_config.use_task_rewards:
            policy_loss = 0.0

    # Combine distillation with policy loss
    policy_metrics.update(distill_metrics)
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
    )
    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)

    return policy_loss, policy_metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode

    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    unclamped_distillation_losses = distillation_losses.detach()
    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # The per-token estimator itself (k1 = log p_theta - log p_T for vanilla OPD), token-mean over the global batch,
    # before any clamp, in every mode. distillation/loss is the value of whatever loss is optimized: PPO clip gives
    # ~token-mean(k1) (ratio ~1), but bypass REINFORCE gives token-mean(-A log p_theta) (+ the SC term), which is not
    # a divergence. Compare reverse-KL / convergence across runs on this metric, never on distillation/loss.
    dense_response_mask = response_mask.to_padded_tensor(False) if response_mask.is_nested else response_mask
    distillation_metrics[f"distillation/{loss_config.loss_mode}_mean"] = _global_token_mean_metric(
        unclamped_distillation_losses, dense_response_mask.bool(), config
    )

    if loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = data["old_log_probs"].to_padded_tensor(0.0)
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        rollout_is_weights = data.get("rollout_is_weights", None)
        if rollout_is_weights is not None and rollout_is_weights.is_nested:
            # Decoupled-mode IS weights (algorithm.rollout_correction.rollout_is with bypass_mode=False) are computed
            # on the driver and stored NESTED over the response like old_log_probs; ppo_loss pads them via
            # select().to_padded_tensor(), this path did not (the padded loss times a jagged tensor raises).
            rollout_is_weights = rollout_is_weights.to_padded_tensor(0.0)
        pg_config, pg_kwargs = loss_config, {}
        sc_outputs = {
            key: no_padding_2_padding(model_output[key], data) for key in SCORE_CENTERING_KEYS if key in model_output
        }
        if loss_config.policy_loss_mode == "bypass_mode":
            # old_log_probs are the sampler's rollout_log_probs (algorithm.rollout_correction.bypass_mode) and the
            # rollout correction -- loss_type, IS rule, score_centering -- is the actor's policy_loss mirror, the
            # same one the score-centering logits processor reads; global_batch_info was set on it above.
            pg_config = config
            if sc_outputs and not loss_config.score_centering_probe:
                # L_t += A_t * sc_correction_t after the IS weighting, A_t = log p_T - log p_theta (detached)
                pg_kwargs["sc_correction"] = sc_outputs["sc_correction"]
        elif sc_outputs:
            raise ValueError("score centering with distillation needs distillation_loss.policy_loss_mode=bypass_mode.")
        advantages = -distillation_losses.detach()
        if loss_config.adv_offset is not None:
            # Advantage-offset causal test: A_t + c on every response token, in either policy-gradient path (and in
            # the score-centering term, which bypass REINFORCE scales by the same advantages). Off (None) = the
            # advantages above, untouched.
            adv_offset = float(loss_config.adv_offset)
            advantages = advantages + adv_offset
            distillation_metrics["distillation/adv_offset"] = adv_offset
            distillation_metrics["distillation/adv_mean_with_offset"] = _global_token_mean_metric(
                advantages, response_mask.bool(), config
            )
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=pg_config,
            rollout_is_weights=rollout_is_weights,
            **pg_kwargs,
        )
        pg_metrics = {_distillation_metric_name(k): v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
        response_mask_bool = response_mask.bool()
        if loss_config.policy_loss_mode == "bypass_mode":
            # Sampler-trainer mismatch on the realized tokens, token-weighted over the global batch: the same
            # definitions as the driver-side rollout_corr/kl and rollout_corr/k3_kl of the recompute (non-bypass)
            # runs, with log p_theta from this training forward instead of the recompute (same weights). The
            # loss-side rollout_corr/* that bypass mode logs are per-micro-batch means averaged unweighted.
            with torch.no_grad():
                log_ratio = (log_prob.detach() - old_log_prob).float()  # log p_theta - log q
                distillation_metrics["distillation/mismatch_kl"] = _global_token_mean_metric(
                    -log_ratio, response_mask_bool, config
                )
                distillation_metrics["distillation/mismatch_k3_kl"] = _global_token_mean_metric(
                    torch.expm1(log_ratio) - log_ratio, response_mask_bool, config
                )
        # Logged identically whether the term is applied or probed, so the two arms compare metric for metric.
        for key, value in sc_outputs.items():
            distillation_metrics[f"distillation/{key}"] = _global_token_mean_metric(value, response_mask_bool, config)
        if sc_outputs:
            # Entropy of the sampler's top-k head, -sum_H q log q (a lower bound on H(q): the tail is not seen). Bypass
            # skips the old_log_prob forward and with it actor/entropy; this is the entropy-collapse diagnostic of
            # the two configurations that fetch the head (score centering applied vs its probe). Response row r of the
            # padded head is the sampler's prediction of response token r, the same alignment as log_probs.
            with torch.no_grad():
                head_probs = no_padding_2_padding(data["rollout_topk_log_probs"], data).float().exp()
                head_entropy = -torch.special.xlogy(head_probs, head_probs).sum(-1)
            distillation_metrics["distillation/sc_sampler_head_entropy"] = _global_token_mean_metric(
                head_entropy, response_mask_bool, config
            )
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )

    return distillation_loss, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    overlap_count = model_output.get("overlap_count")
    overlap_token_advantage = model_output.get("overlap_token_advantage")
    if overlap_count is not None and overlap_token_advantage is not None:
        overlap_count = no_padding_2_padding(overlap_count, data)
        overlap_token_advantage = no_padding_2_padding(overlap_token_advantage, data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    overlap_metrics = {}
    if overlap_count is not None and overlap_token_advantage is not None:
        assert overlap_count.shape == overlap_token_advantage.shape == response_mask_bool.shape
        valid_overlap_count = overlap_count[response_mask_bool]
        k = distillation_config.distillation_loss.topk
        assert k is not None
        # Diagnostics for tracking teacher/student top-k overlap in OPD, following
        # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016):
        # overlap ratio and average teacher-token KL contribution on overlapped tokens.
        overlap_metrics["distillation/overlap_ratio"] = (valid_overlap_count.float().mean() / k).item()
        overlap_position_mask = response_mask_bool & (overlap_count > 0)
        if overlap_position_mask.any():
            overlap_metrics["distillation/overlap_token_advantage"] = (
                overlap_token_advantage[overlap_position_mask].mean().item()
            )
        else:
            overlap_metrics["distillation/overlap_token_advantage"] = 0.0

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
        **overlap_metrics,
    }

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, distillation_metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
    )
    # Since k1 can be negative, log the mean absolute loss.
    metrics = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),
    }
    return distillation_losses, metrics
