import json

import pytest

from tinyreasonrl.train import train


def test_cannot_silently_resume_with_different_reward_protocol(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"reward_protocol": "strict-single-answer-v1"}))
    with pytest.raises(ValueError, match="Reward protocol changed"):
        train({}, {}, str(tmp_path), str(tmp_path), resume="old-checkpoint")
