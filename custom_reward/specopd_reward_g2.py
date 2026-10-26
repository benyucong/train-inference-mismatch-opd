"""Reward that grades either answer format: a \\boxed{} answer if present, else the last 'Answer:' line, else 0.

The DAPO-17k prompts ask for the answer on an 'Answer:' line, but custom_reward.ttrl_math credits only a \\boxed{}
answer, so a response that follows the prompt's format earns nothing even when its answer is correct. This reward:
  - a response with a \\boxed{} answer is graded exactly as before (ttrl_math.compute_score, byte-identical result);
  - otherwise the last 'Answer:' line after </think> (else anywhere) is extracted (answer_line below) and graded by the
    same ttrl_math grader (fast=False);
  - otherwise the score is 0.
A prompt whose data_source starts with "deepcoder" (a code problem, which the math grader cannot grade) scores 0.0,
as in custom_reward.specopd_reward (the \\boxed-only reward, unchanged), so a grader error on it cannot stop a step.
bench/ho_eval.py grades held-out samples with both.
"""
import re

from custom_reward.ttrl_math import compute_score, extract_answer, grade

ANS_RE = re.compile(r"(?im)^\s*\**\s*answer\s*\**\s*:\s*(.+?)\s*$")   # = bench/ho_eval.py G2_ANS_RE (asserted there)


def answer_line(text):
    """DAPO's requested format, 'Answer: $Answer' on the last line: the last such line after </think> (else anywhere)."""
    tail = text.split("</think>")[-1]
    m = ANS_RE.findall(tail) or ANS_RE.findall(text)
    if not m:
        return None
    x = m[-1].strip().strip("*").strip()
    x = re.sub(r"^\$+|\$+$", "", x).strip().rstrip(".").strip()
    return x or None


def compute_score_g2(model_response, gt_answer):
    if extract_answer(model_response) is not None:          # \boxed present: ttrl_math's grading, unchanged
        res = compute_score(model_response, gt_answer)
        res["answer_source"] = "boxed"
        return res
    pred = answer_line(model_response)
    if pred is None:
        return {"score": 0.0, "format_score": 0.0, "acc": False, "extracted_gt": gt_answer, "pred": "",
                "answer_source": "none"}
    gts = gt_answer if isinstance(gt_answer, list) else [str(gt_answer)]
    ok = any(grade(pred, str(gt), False) for gt in gts)
    return {"score": 1.0 if ok else 0.0, "format_score": 1.0, "acc": bool(ok), "extracted_gt": gt_answer, "pred": pred,
            "answer_source": "answer_line"}


def reward_func(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # code problems: not gradable by the math grader (see the module docstring)
    if isinstance(data_source, str) and data_source.startswith("deepcoder"):
        return 0.0
    return compute_score_g2(solution_str, str(ground_truth))
