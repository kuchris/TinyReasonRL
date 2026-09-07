import pytest

from tinyreasonrl.evaluate import evaluate, summarize


def test_pass_k_counts_puzzles_not_completions():
    records = [{"id": puzzle, "reward": reward, "reason": "correct" if reward else "target",
                "format_valid": True, "expression_valid": True, "tokens": 10,
                "truncated": False}
               for puzzle, rewards in [("a", [1, 0, 0, 0]), ("b", [0, 0, 0, 0])]
               for reward in rewards]
    metrics = summarize(records)
    assert metrics["sampled_pass@1"] == 0.125
    assert metrics["pass@4"] == 0.5
    assert metrics["mixed_reward_group_fraction"] == 0.5
    with pytest.raises(ValueError):
        summarize(records[:-1])


def test_evaluation_preserves_existing_generations(tmp_path):
    path = tmp_path / "generations.jsonl"
    path.write_text("existing evidence")
    with pytest.raises(FileExistsError, match="fresh output directory"):
        evaluate({}, "validation", 1, 4, tmp_path)
    assert path.read_text() == "existing evidence"
