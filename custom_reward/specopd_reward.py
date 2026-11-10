"""Task reward for the training launcher: the math verifier, with a guard for prompts it cannot grade.

OPD here trains on the teacher's token-level signal only (distillation_loss.use_task_rewards=False), so the
task reward is LOGGED, never optimized (the GRPO and REINFORCE runs optimize it). Math prompts use the math
verifier (custom_reward.ttrl_math). A prompt whose data_source starts with "deepcoder" (a code problem, which
the math verifier cannot grade) scores 0.0: the math verifier re-raises on inputs it cannot parse, and such an
error would otherwise stop a training step.
"""
from custom_reward.ttrl_math import reward_func as _math_reward


def reward_func(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if isinstance(data_source, str) and data_source.startswith("deepcoder"):
        return 0.0
    return _math_reward(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)
