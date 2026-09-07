import pytest
import json
from pathlib import Path

from tinyreasonrl.rewards import correctness_reward, score_answer


@pytest.mark.parametrize("expression,numbers,target", [
    ("7 * 2 + 3", [2, 3, 7], 17),
    ("(7 + 3) * 2", [2, 3, 7], 20),
    ("8 / (3 - 8 / 3)", [8, 3, 8, 3], 24),
    ("(2 - 7) * 3", [2, 3, 7], -15),
    ("2 + 2 + 3", [2, 2, 3], 7),
])
def test_correct(expression, numbers, target):
    assert score_answer(f"<answer>{expression}</answer>", numbers, target).reward == 1


@pytest.mark.parametrize("output", [
    "<answer>7 * 2 - 3</answer>",
    "<answer>7 + 7 + 3</answer>",
    "<answer>7 + 2</answer>",
    "<answer>10 + 7</answer>",
    "<answer>2 ** 3 + 7</answer>",
    "<answer>7 // 2 + 3</answer>",
    "<answer>7 % 2 + 3</answer>",
    "<answer>7 * 2 + 3 = 17</answer>",
    "<answer>7 * 2 + 3",
    "7 * 2 + 3",
    "<answer>7 * 2 + 3</answer><answer>17</answer>",
    "<answer>__import__('os').system('echo unsafe')</answer>",
    "<answer>[x for x in range(17)]</answer>",
    "<answer>7 * 2 + True</answer>",
    "<answer>7 * 2 + 3.0</answer>",
    "<answer>7 * 2 - -3</answer>",
    "<answer>7 * +2 + 3</answer>",
    "<answer>(7 * 2 + 3</answer>",
    "<answer>" + "(" * 300 + "17" + ")" * 300 + "</answer>",
])
def test_invalid(output):
    assert score_answer(output, [2, 3, 7], 17).reward == 0


def test_division_by_zero():
    assert score_answer("<answer>7 / (2 - 2)</answer>", [7, 2, 2], 17).reward == 0


def test_valid_expression_wrong_target():
    score = score_answer("<answer>7 + 2 + 3</answer>", [2, 3, 7], 17)
    assert score.expression_valid and score.reason == "target" and score.reward == 0


def test_reasoning_ignored_and_batch_reward():
    output = "My guess is 900. <answer>7 * 2 + 3</answer>"
    assert correctness_reward([output, "bad"], [[2, 3, 7]] * 2, [17, 17]) == [1, 0]


def test_final_answer_after_tag_mentions_and_abandoned_opening():
    output = ('Expression: <answer>\nScratch work: use <answer> and </answer> tags. '
              'First guess: <answer>7 + 2 + 3</answer>. '
              'Checking gives 7 * 2 + 3 = 17. <answer>7 * 2 + 3</answer>')
    assert score_answer(output, [2, 3, 7], 17).reward == 1


def test_last_answer_wins_even_when_earlier_answer_is_correct():
    output = "<answer>7 * 2 + 3</answer> Actually: <answer>7 + 2 + 3</answer>"
    assert score_answer(output, [2, 3, 7], 17).reward == 0


@pytest.mark.parametrize("suffix", ["<answer>", "</answer>"])
def test_malformed_final_answer_cannot_fall_back_to_earlier_success(suffix):
    assert score_answer("<answer>7 * 2 + 3</answer>" + suffix, [2, 3, 7], 17).reward == 0


@pytest.mark.parametrize("numbers,target,expression", [
    ([3, 42, 95], 50, "95 - 42 - 3"),
    ([92, 36, 73], 55, "92 - 73 + 36"),
])
def test_long_rollout_regressions(numbers, target, expression):
    output = ("<answer>\n<think>Put the expression between <answer> and </answer>. "
              f"Candidate: <answer>{expression}</answer>. Checked.</think>\n"
              f"<answer>{expression}</answer>")
    assert score_answer(output, numbers, target).reward == 1


def test_actual_saved_false_negative_rollouts():
    path = Path(__file__).resolve().parents[1] / "results/long-completion-diagnostic/generations.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    fixed = [row for row in rows
             if score_answer(row["completion"], row["numbers"], row["target"]).reward]
    assert len(fixed) == 2
    assert all(row["reward"] == 0 for row in fixed)
