from tinyreasonrl.data import puzzle_key, split_for_key


def test_reordered_puzzles_stay_together():
    key = puzzle_key([2, 7, 3], 17)
    assert key == puzzle_key([7, 3, 2], 17)
    assert split_for_key(key, 42) == split_for_key(puzzle_key([3, 2, 7], 17), 42)
    assert key != puzzle_key([2, 2, 7], 17)
