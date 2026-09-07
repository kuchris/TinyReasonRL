"""Plot saved measurements only; no synthetic experiment curves."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(training, before, after, output, intermediate=(), tensorboard_logdir=None):
    rows = [json.loads(line) for line in Path(training).read_text().splitlines()]
    rewards = [r for r in rows if "reward" in r]
    lengths = [r for r in rows if "completions/mean_length" in r]
    final_step = max(row["step"] for row in rows)
    before = json.loads((Path(before) / "metrics.json").read_text())
    after = json.loads((Path(after) / "metrics.json").read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for filename, data, field, ylabel in (
        ("reward", rewards, "reward", "Mean correctness reward"),
        ("completion-length", lengths, "completions/mean_length", "Completion tokens"),
    ):
        if not data:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([r["step"] for r in data], [r[field] for r in data], marker=".")
        ax.set(xlabel="Optimizer step", ylabel=ylabel)
        if field == "reward":
            ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / f"{filename}.png", dpi=160)
        plt.close(fig)
    keys = [k for k in before if "pass@" in k and k in after]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = list(range(len(keys)))
    base_bars = ax.bar([i - 0.18 for i in x], [before[k] for k in keys], width=0.36, label="Base")
    rl_bars = ax.bar([i + 0.18 for i in x], [after[k] for k in keys], width=0.36, label=f"After {final_step} updates")
    ax.bar_label(base_bars, labels=[f"{before[k]:.1%}" for k in keys], padding=3)
    ax.bar_label(rl_bars, labels=[f"{after[k]:.1%}" for k in keys], padding=3)
    ax.set(xticks=x, xticklabels=keys, ylabel="Success fraction", ylim=(0, 1))
    ax.set_title(f"Validation: {before['problems']} puzzles, {before['completions']} samples per condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "before-after.png", dpi=160)
    plt.close(fig)
    points = {0: before["sampled_pass@1"], final_step: after["sampled_pass@1"]}
    for step, directory in intermediate:
        step = int(step)
        if not 0 < step < final_step or step in points:
            raise ValueError("Intermediate steps must be unique and between zero and the final step.")
        metrics = json.loads((Path(directory) / "metrics.json").read_text())
        if (metrics["problems"], metrics["completions"]) != (before["problems"], before["completions"]):
            raise ValueError("Intermediate evaluations must use the same puzzle and sample counts.")
        points[step] = metrics["sampled_pass@1"]
    steps = sorted(points)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(steps, [points[step] for step in steps])
    ax.set_xticks(steps)
    ax.set(xlabel="Optimizer step", ylabel="Validation sampled pass@1", ylim=(0, 1),
           title="Measured checkpoints only")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "validation-accuracy.png", dpi=160)
    plt.close(fig)
    if tensorboard_logdir:
        from torch.utils.tensorboard import SummaryWriter
        with SummaryWriter(log_dir=tensorboard_logdir) as writer:
            for step in steps:
                writer.add_scalar("validation/sampled_pass@1", points[step], step)
            for row in rewards:
                writer.add_scalar("train/gpu_peak_allocated_gib", row["gpu_peak_allocated_gib"], row["step"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", required=True)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", default="results/plots")
    parser.add_argument("--intermediate", nargs=2, action="append", default=[], metavar=("STEP", "DIRECTORY"))
    parser.add_argument("--tensorboard-logdir")
    args = parser.parse_args()
    plot(args.training, args.before, args.after, args.output, args.intermediate, args.tensorboard_logdir)
