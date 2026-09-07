"""Check whether extra answer tags hid correct expressions; never change rewards."""

import argparse
import json
from pathlib import Path
import re

from tinyreasonrl.rewards import REWARD_PROTOCOL, score_answer


parser = argparse.ArgumentParser()
parser.add_argument("generations")
parser.add_argument("--output", required=True)
args = parser.parse_args()
rows = [json.loads(line) for line in Path(args.generations).read_text(encoding="utf-8").splitlines()]
hidden = []
for row in rows:
    if row["reward"]:
        continue
    candidates = re.findall(r"<answer>(.*?)</answer>", row["completion"], re.DOTALL)
    for expression in candidates:
        score = score_answer(f"<answer>{expression}</answer>", row["numbers"], row["target"])
        if score.reward:
            hidden.append({"id": row["id"], "sample": row["sample"], "expression": expression})
            break
report = {"source": args.generations, "completions": len(rows),
          "reward_protocol": REWARD_PROTOCOL,
          "correct_tagged_expression_in_rejected_output": len(hidden), "examples": hidden,
          "limitation": "Only complete answer-tag pairs are inspected; untagged scratch work is not judged."}
Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
