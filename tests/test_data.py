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
