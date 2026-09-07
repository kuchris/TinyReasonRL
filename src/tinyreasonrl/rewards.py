"""Bounded arithmetic parsing; model output is never executed."""

import ast
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
import re

REWARD_PROTOCOL = "final-answer-v2"


@dataclass(frozen=True)
class Score:
    reward: float
    reason: str
    format_valid: bool = False
    expression_valid: bool = False
    expression: str | None = None

    def to_dict(self):
        return asdict(self)


def score_answer(completion: str, numbers: list[int], target: int) -> Score:
    """Judge the last answer block, positive integer leaves, and every input.

    Negative and fractional intermediate values are allowed. Unary signs,
    decimal literals, concatenation, and powers are not. Earlier answer tags
    are scratch work; an unfinished final answer never falls back to an earlier one.
    """
    start = completion.rfind("<answer>")
    end = completion.find("</answer>", start + len("<answer>"))
    if start < 0 or end < 0 or "</answer>" in completion[end + len("</answer>"):]:
        return Score(0.0, "format")
    expression = completion[start + len("<answer>"):end].strip()

    def invalid(reason):
        return Score(0.0, reason, True, False, expression)

    if not expression or len(expression) > 256:
        return invalid("expression_size")
    if not re.fullmatch(r"[0-9+*/()\s-]+", expression):
        return invalid("illegal_syntax")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        return invalid("parse")
    if sum(1 for _ in ast.walk(tree)) > 64:
        return invalid("expression_size")
    leaves = []

    def calculate(node, depth=0):
        if depth > 16:
            raise ValueError("depth")
        if isinstance(node, ast.Constant) and type(node.value) is int:
            if node.value <= 0:
                raise ValueError("literal")
            leaves.append(node.value)
            return Fraction(node.value)
        if not isinstance(node, ast.BinOp) or type(node.op) not in (
                ast.Add, ast.Sub, ast.Mult, ast.Div):
            raise ValueError("operator")
        left = calculate(node.left, depth + 1)
        right = calculate(node.right, depth + 1)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right

    try:
        value = calculate(tree.body)
    except (ValueError, ZeroDivisionError, RecursionError):
        return invalid("arithmetic")
    if Counter(leaves) != Counter(numbers):
        return invalid("numbers")
    correct = value == Fraction(target)
    return Score(float(correct), "correct" if correct else "target",
                 True, True, expression)


def correctness_reward(completions, numbers, target, **kwargs):
    """TRL passes dataset columns alongside plain-text completions."""
    return [score_answer(c, n, t).reward
            for c, n, t in zip(completions, numbers, target, strict=True)]
