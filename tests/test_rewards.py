import pytest

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
    "<answer><answer>7 * 2 + 3</answer>",
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
