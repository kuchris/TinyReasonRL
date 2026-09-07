"""Direct GRPO with LoRA and correctness-only rewards; no SFT stage."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from .data import load_problems
from .model import load_policy
from .prompts import render_prompt
from .rewards import (REWARD_PROTOCOL, correctness_reward, score_answer,
                      shaped_reward)


class JsonLoggingCallback(TrainerCallback):
    def __init__(self, directory):
        self.directory = Path(directory)

    def on_log(self, args, state, control, logs=None, **kwargs):
        row = {"step": state.global_step, **(logs or {}),
               "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
        with (self.directory / "training.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")


def train(config, training, output, results, resume=None, reward="correctness",
          min_numbers=None, max_numbers=None):
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    if resume:
        previous = json.loads((results / "metadata.json").read_text(encoding="utf-8"))
        if previous.get("reward_protocol", "strict-single-answer-v1") != REWARD_PROTOCOL:
            raise ValueError("Reward protocol changed; start a fresh run instead of resuming.")
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(results / "tensorboard")
    model, tokenizer = load_policy(config)
    model.config.use_cache = False
    torch.cuda.reset_peak_memory_stats()
    args = GRPOConfig(
        output_dir=output, max_steps=training["max_steps"],
        learning_rate=training["learning_rate"], lr_scheduler_type="constant",
        per_device_train_batch_size=training["per_device_train_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        num_generations=training["num_generations"],
        max_completion_length=config["max_completion_length"],
        temperature=config["temperature"], top_p=config["top_p"], top_k=0,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        beta=training["beta"], loss_type=training["loss_type"],
        scale_rewards="group", use_vllm=False, seed=config["seed"],
        data_seed=config["seed"], report_to="tensorboard",
        logging_steps=1,
        save_steps=training["save_steps"], save_total_limit=4,
        dataloader_num_workers=0, optim="adamw_torch", max_grad_norm=1.0,
    )
    metadata = {"experiment": config, "training": training,
                "reward_protocol": REWARD_PROTOCOL,
                "reward": reward, "min_numbers": min_numbers,
                "max_numbers": max_numbers,
                "resolved_grpo": args.to_dict(), "resume": resume,
                "versions": {p: importlib.metadata.version(p) for p in (
                    "torch", "transformers", "trl", "peft", "datasets", "accelerate")}}
    metadata_path = results / ("resume-metadata.json" if resume else "metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    reward_func = correctness_reward if reward == "correctness" else shaped_reward

    def logged_correctness(completions, numbers, target, trainer_state=None, **kwargs):
        rewards = reward_func(completions, numbers, target)
        step = trainer_state.global_step if trainer_state else 0
        with (results / "rollouts.jsonl").open("a", encoding="utf-8") as handle:
            for completion, nums, goal in zip(completions, numbers, target, strict=True):
                handle.write(json.dumps({"step": step, "numbers": nums, "target": goal,
                                         "completion": completion,
                                         **score_answer(completion, nums, goal).to_dict()}) + "\n")
        return rewards

    dataset = load_problems("train", training["train_limit"], config=config,
                            min_numbers=min_numbers, max_numbers=max_numbers)
    dataset = dataset.map(lambda row: {"prompt": render_prompt(row["prompt"], tokenizer, config)})
    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer, args=args,
        reward_funcs=logged_correctness,
        train_dataset=dataset,
        peft_config=LoraConfig(r=training["lora_rank"], lora_alpha=training["lora_alpha"],
                               target_modules="all-linear", lora_dropout=0.0,
                               bias="none", task_type="CAUSAL_LM"),
        callbacks=[JsonLoggingCallback(results)],
    )
    trainer.model.print_trainable_parameters()
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(Path(output) / "final"))
    trainer.save_state()
    summary = {**result.metrics, "global_step": trainer.state.global_step,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    (results / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--training-config", default="configs/train.json")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output", default="checkpoints/main")
    parser.add_argument("--results", default="results/main-training")
    parser.add_argument("--resume")
    parser.add_argument("--reward", choices=["correctness", "shaped"],
                        default="correctness",
                        help="correctness-only (baseline) or shaped partial-progress reward")
    parser.add_argument("--min-numbers", type=int,
                        help="train only puzzles with at least this many inputs (curriculum)")
    parser.add_argument("--max-numbers", type=int,
                        help="train only puzzles with at most this many inputs (curriculum)")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    training = json.loads(Path(args.training_config).read_text())
    if args.max_steps is not None:
        training["max_steps"] = args.max_steps
        training["save_steps"] = min(training["save_steps"], args.max_steps)
    if training["max_steps"] < 1:
        parser.error("max-steps must be positive")
    if not args.resume and (Path(args.results) / "training.jsonl").exists():
        parser.error("Results already exist; use a new directory or --resume a checkpoint")
    train(config, training, args.output, args.results, args.resume,
          reward=args.reward, min_numbers=args.min_numbers,
          max_numbers=args.max_numbers)
