"""Export fixed, inspectable examples from recorded training and final evaluation."""

import argparse
import json
from pathlib import Path
import re


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


parser = argparse.ArgumentParser()
parser.add_argument("--training", required=True)
parser.add_argument("--after", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
training = read_rows(args.training)
after = read_rows(args.after)
last = max(row["step"] for row in training)
selected = [(f"Policy after {step} completed updates, before the next update",
             next(row for row in training if row["step"] == step))
            for step in sorted({0, min(1, last), last // 2, last})]
selected.append(("Final saved adapter: first validation sample", after[0]))
parts = ["# Recorded generation examples\n",
         "Selection is fixed: the first logged rollout at each listed stage and the first "
         "final validation sample. These examples are not selected for success. Training "
         "puzzles differ across stages, so this is not a controlled comparison of reasoning. "
         "Trailing line whitespace is removed for display; JSONL retains the raw text.\n",
         f"Sources: `{args.training}` and `{args.after}`.\n"]
for label, row in selected:
    completion = "\n".join(line.rstrip() for line in row["completion"].splitlines())
    fence = "`" * max(3, 1 + max((len(s) for s in re.findall(r"`+", row["completion"])), default=0))
    parts.extend([f"## {label}\n",
                  f"Numbers: {row['numbers']}; target: {row['target']}; "
                  f"reward: {row['reward']}; verdict: {row['reason']}.\n",
                  f"{fence}text\n{completion}\n{fence}\n"])
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(parts), encoding="utf-8")
