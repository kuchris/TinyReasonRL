"""Revision-pinned dataset with deterministic, puzzle-grouped splits."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from datasets import Dataset, load_dataset

from .prompts import make_prompt


def puzzle_key(numbers, target):
    return json.dumps([sorted(numbers), target], separators=(",", ":"))


def split_for_key(key, seed):
    bucket = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) % 100
    return "test" if bucket < 5 else "validation" if bucket < 10 else "train"


def prepare_data(config, directory="data"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(config["dataset_id"], revision=config["dataset_revision"], split="train")
    seen = set()
    counts = Counter()
    handles = {s: (directory / f"{s}.jsonl").open("w", encoding="utf-8")
               for s in ("train", "validation", "test")}
    examples = []
    try:
        for row in raw:
            numbers, target = row["nums"], row["target"]
            if len(numbers) not in (3, 4) or any(type(n) is not int or n <= 0 for n in numbers):
                raise ValueError(f"Unexpected dataset row: {row}")
            key = puzzle_key(numbers, target)
            if key in seen:
                continue
            seen.add(key)
            split = split_for_key(key, config["seed"])
            record = {"id": hashlib.sha256(key.encode()).hexdigest(),
                      "numbers": numbers, "target": target,
                      "prompt": make_prompt(numbers, target)}
            handles[split].write(json.dumps(record) + "\n")
            counts[f"{split}_{len(numbers)}_numbers"] += 1
            if len(examples) < 5:
                examples.append({"numbers": numbers, "target": target, "split": split})
    finally:
        for handle in handles.values():
            handle.close()
    manifest = {"dataset_id": config["dataset_id"], "revision": config["dataset_revision"],
                "seed": config["seed"], "raw_rows": len(raw), "unique_puzzles": len(seen),
                "duplicates_removed": len(raw) - len(seen), "counts": dict(counts),
                "examples": examples}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_problems(split, limit=None, directory="data", config=None):
    if config is not None:
        manifest = json.loads((Path(directory) / "manifest.json").read_text())
        expected = (config["dataset_id"], config["dataset_revision"], config["seed"])
        actual = (manifest["dataset_id"], manifest["revision"], manifest["seed"])
        if actual != expected:
            raise ValueError("Prepared data does not match configuration; rerun data preparation.")
    rows = []
    with (Path(directory) / f"{split}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return Dataset.from_list(rows)


if __name__ == "__main__":
    config = json.loads(Path("configs/experiment.json").read_text())
    print(json.dumps(prepare_data(config), indent=2))
