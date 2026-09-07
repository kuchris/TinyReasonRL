"""Rescore immutable saved generations; do not resample or overwrite old results."""

import argparse
import json
from pathlib import Path

from tinyreasonrl.evaluate import summarize
from tinyreasonrl.rewards import REWARD_PROTOCOL, score_answer

parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("output")
args = parser.parse_args()
source, output = Path(args.source), Path(args.output)
output.mkdir(parents=True, exist_ok=False)
rows = []
for line in (source / "generations.jsonl").read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    rows.append({**row, "previous_reward": row["reward"],
                 **score_answer(row["completion"], row["numbers"], row["target"]).to_dict()})
(output / "generations.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
metrics = summarize(rows)
metrics["by_number_count"] = {str(n): summarize(subset) for n in (3, 4)
                              if (subset := [r for r in rows if len(r["numbers"]) == n])}
metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
metadata.update(source=str(source), reward_protocol=REWARD_PROTOCOL, resampled=False)
(output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
(output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
