"""Save sampled rollouts and metrics with identical before/after settings."""

import argparse
from collections import Counter, defaultdict
import importlib.metadata
import json
from pathlib import Path
import time

import torch

from .data import load_problems
from .model import load_policy
from .prompts import render_prompt
from .rewards import score_answer


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["id"]].append(row)
    k = len(next(iter(groups.values())))
    if any(len(g) != k for g in groups.values()):
        raise ValueError("Each puzzle must have the same sample count.")
    return {
        "problems": len(groups), "completions": len(records),
        "sampled_pass@1": sum(r["reward"] for r in records) / len(records),
        f"pass@{k}": sum(any(r["reward"] for r in g) for g in groups.values()) / len(groups),
        "average_reward": sum(r["reward"] for r in records) / len(records),
        "exact_target_success_rate": sum(r["reward"] for r in records) / len(records),
        "format_valid_rate": sum(r["format_valid"] for r in records) / len(records),
        "valid_expression_rate": sum(r["expression_valid"] for r in records) / len(records),
        "average_completion_tokens": sum(r["tokens"] for r in records) / len(records),
        "truncation_rate": sum(r["truncated"] for r in records) / len(records),
        "mixed_reward_group_fraction": sum(0 < sum(r["reward"] for r in g) < k
                                            for g in groups.values()) / len(groups),
        "failure_reasons": dict(Counter(r["reason"] for r in records)),
    }


def evaluate(config, split, limit, samples, output, adapter=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    problems = load_problems(split, limit, config=config)
    model, tokenizer = load_policy(config, adapter)
    torch.cuda.reset_peak_memory_stats()
    records = []
    started = time.perf_counter()
    with (output / "generations.jsonl").open("w", encoding="utf-8") as handle:
        for index, problem in enumerate(problems):
            prompt = render_prompt(problem["prompt"], tokenizer, config)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=config["max_completion_length"],
                    do_sample=True, temperature=config["temperature"], top_p=config["top_p"],
                    num_return_sequences=samples,
                    pad_token_id=tokenizer.pad_token_id, use_cache=True,
                )[:, inputs["input_ids"].shape[1]:]
            eos_ids = model.generation_config.eos_token_id
            eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
            for sample, ids in enumerate(generated):
                stops = [i for i, token in enumerate(ids.tolist()) if token in eos_ids]
                ended = bool(stops)
                if ended:
                    ids = ids[:stops[0] + 1]
                completion = tokenizer.decode(ids, skip_special_tokens=True)
                score = score_answer(completion, problem["numbers"], problem["target"])
                row = {"id": problem["id"], "numbers": problem["numbers"],
                       "target": problem["target"], "prompt": prompt,
                       "sample": sample, "completion": completion,
                       "tokens": len(ids), "truncated": not ended,
                       **score.to_dict()}
                records.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            print(f"{index + 1}/{len(problems)}: rewards={[r['reward'] for r in records[-samples:]]}", flush=True)
    metrics = summarize(records)
    metrics["by_number_count"] = {str(n): summarize(subset) for n in (3, 4)
                                  if (subset := [r for r in records if len(r["numbers"]) == n])}
    metrics["elapsed_seconds"] = time.perf_counter() - started
    metrics["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    metrics["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    metadata = {"config": config, "split": split, "limit": limit, "samples": samples,
                "adapter": adapter, "gpu": torch.cuda.get_device_name(),
                "versions": {p: importlib.metadata.version(p) for p in (
                    "torch", "transformers", "trl", "peft", "datasets", "accelerate")}}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--output", default="results/milestone1")
    parser.add_argument("--adapter")
    args = parser.parse_args()
    if args.limit < 1 or args.samples < 1:
        parser.error("limit and samples must be positive")
    evaluate(json.loads(Path(args.config).read_text()), args.split, args.limit,
             args.samples, args.output, args.adapter)
