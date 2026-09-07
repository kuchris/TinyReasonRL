from tinyreasonrl.data import puzzle_key, split_for_key
from tinyreasonrl.data import load_problems
import json
import pytest


def test_reordered_puzzles_stay_together():
    key = puzzle_key([2, 7, 3], 17)
    assert key == puzzle_key([7, 3, 2], 17)
    assert split_for_key(key, 42) == split_for_key(puzzle_key([3, 2, 7], 17), 42)
    assert key != puzzle_key([2, 2, 7], 17)


def test_data_revision_mismatch_is_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "dataset_id": "dataset", "revision": "old", "seed": 42}))
    with pytest.raises(ValueError, match="does not match"):
        load_problems("train", directory=tmp_path, config={
            "dataset_id": "dataset", "dataset_revision": "new", "seed": 42})


def test_load_problems_filters_by_number_count(tmp_path):
    rows = [
        {"id": "a", "numbers": [2, 3, 7], "target": 17, "prompt": "p"},
        {"id": "b", "numbers": [2, 3, 7, 5], "target": 17, "prompt": "p"},
        {"id": "c", "numbers": [1, 2], "target": 3, "prompt": "p"},
    ]
    for row in rows:
        (tmp_path / "train.jsonl").open("a", encoding="utf-8").write(json.dumps(row) + "\n")
    three_only = load_problems("train", directory=tmp_path, min_numbers=3, max_numbers=3)
    assert [r["id"] for r in three_only] == ["a"]
    up_to_three = load_problems("train", directory=tmp_path, max_numbers=3)
    assert [r["id"] for r in up_to_three] == ["a", "c"]
    # limit applies after filtering, so it stays inside the requested band.
    limited = load_problems("train", directory=tmp_path, limit=1, min_numbers=3, max_numbers=4)
    assert [r["id"] for r in limited] == ["a"]
