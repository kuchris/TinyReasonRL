"""Plot saved measurements only; no synthetic experiment curves."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(training, before, after, output):
    rows = [json.loads(line) for line in Path(training).read_text().splitlines()]
    rewards = [r for r in rows if "reward" in r]
    lengths = [r for r in rows if "completions/mean_length" in r]
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
    rl_bars = ax.bar([i + 0.18 for i in x], [after[k] for k in keys], width=0.36, label="After RL")
    ax.bar_label(base_bars, labels=[f"{before[k]:.1%}" for k in keys], padding=3)
    ax.bar_label(rl_bars, labels=[f"{after[k]:.1%}" for k in keys], padding=3)
    ax.set(xticks=x, xticklabels=keys, ylabel="Success fraction", ylim=(0, 1))
    ax.set_title(f"Smoke comparison: {before['problems']} puzzles, {before['completions']} samples per condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "before-after.png", dpi=160)
    plt.close(fig)
    final_step = max(row["step"] for row in rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter([0, final_step], [before["sampled_pass@1"], after["sampled_pass@1"]])
    ax.set(xlabel="Optimizer step", ylabel="Validation sampled pass@1", ylim=(0, 1),
           title="Measured endpoints only; no intermediate evaluations")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "validation-accuracy.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", required=True)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", default="results/plots")
    args = parser.parse_args()
    plot(args.training, args.before, args.after, args.output)
