# TinyReasonRL

Can direct reinforcement learning improve a small pretrained model's Countdown
accuracy without an intervening supervised fine-tuning (SFT) stage?

This is a local, single-GPU experiment using **Qwen3.5-0.8B-Base**, LoRA, and
verifiable arithmetic rewards. Increased reward is evidence of improved task
performance; it does not establish that genuine reasoning emerged. Pretraining
already supplies capabilities, and this experiment does not start from random weights.

## Setup (Windows PowerShell)

Python 3.11 and an NVIDIA GPU supporting BF16 are required for the experiment.
The initial host is an RTX 5070 Ti with 16 GB VRAM. No SFT data or training is used.

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -e .
.venv\Scripts\python.exe -m pytest -q
```

The CUDA wheel is important for this GPU. System Python is not modified.
Resolved package versions will be recorded with experiment results.

To reproduce the installed environment, use `requirements-lock.txt` after
installing the CUDA torch wheel, then install this project with `-e . --no-deps`.

## Run the first milestone

```powershell
.venv\Scripts\python.exe -m tinyreasonrl.data
.venv\Scripts\python.exe -m tinyreasonrl.evaluate --limit 1 --samples 4 --output results/milestone1
```

Data preparation found 490,364 source rows, 449,570 unique puzzles, and 40,794
duplicates. Splits contain 404,836 training, 22,389 validation, and 22,345 test
puzzles. `results/dataset-manifest.json` records counts and example rows.

The first actual run completed on the RTX 5070 Ti in 20.6 seconds of generation
and scoring, excluding loading. Peak PyTorch allocated memory was 1.47 GiB
(reserved: 1.53 GiB; these exclude other desktop applications).
Four samples on one validation puzzle produced zero correct answers: one
returned only the target and three were truncated at 256 tokens. These are
pipeline smoke results, **not a meaningful accuracy estimate**. Full generations,
metrics, and runtime metadata are in `results/milestone1/`.

Current tested versions: Python 3.11, PyTorch 2.11.0+cu128, Transformers 5.16.1,
TRL 1.12.0, PEFT 0.20.0, Datasets 5.0.1, Accelerate 1.14.0. The text-only
`Qwen3_5ForCausalLM` loader reports no missing or mismatched weights.

## Reward contract

Each problem supplies three or four positive integers and a target. The model
may write scratch work, then must produce exactly one `<answer>expression</answer>`.
The prompt contains no worked answers or reasoning demonstrations.

The evaluator accepts only integer leaves and binary `+`, `-`, `*`, `/` operations
with parentheses. Every input must appear exactly once, including repeated inputs.
Negative and fractional intermediate results are allowed; unary signs, decimal
literals, exponentiation, floor division, and concatenated input numbers are not.
Division uses exact rational arithmetic. Division by zero is invalid.

Generated text is never executed. A strict AST allowlist and expression size,
node-count, and depth limits bound parsing and evaluation. Correctness is the
only reward: **1.0** if all rules pass and the target is reached, otherwise **0.0**.
Format failures, invalid expressions, and valid expressions with wrong targets
are recorded separately.

## Concepts

- **Policy:** the model's distribution over possible next tokens.
- **Rollout:** one answer sampled from that policy for a puzzle.
- **Reward:** the verifier's numerical assessment of the completed answer.
- **Advantage:** how an answer's reward compares with other answers to the same prompt.
- **GRPO:** sample a group of answers, compute relative advantages, then adjust
  the policy to increase the likelihood of relatively successful answers.
  A group with identical rewards has no relative correctness signal.
- **KL divergence:** a measure of policy deviation from a reference model. Its
  penalty is optional and the chosen coefficient must be recorded explicitly.
- **LoRA:** train small low-rank adapter matrices while keeping base weights frozen.
- **Gradient accumulation:** combine gradients across smaller microbatches before
  updating parameters, reducing the memory needed per forward/backward pass.
- **pass@k:** the probability that at least one of k sampled answers succeeds.
  Here it is measured as the fraction of puzzles with a success among k samples;
  sampled pass@1 averages correctness across individual samples, not greedy decoding.

## Experiment design

Model and dataset revisions are pinned in `configs/experiment.json`. Split keys
use the sorted multiset of numbers and target, so reordered duplicate puzzles
cannot cross splits. Duplicate puzzles are removed. A seeded SHA-256 partition
assigns approximately 90% train, 5% validation, and 5% final test.

Use validation for development and checkpoint selection. Reserve test for the
final frozen before/after comparison. A held-out split cannot rule out exposure
during the model's original pretraining.

The first milestone is loading the text policy, generating four answers for one
puzzle, and calculating rewards. A later baseline across many puzzles will check
whether mixed-reward groups occur often enough for RL to learn. A handful of
examples cannot establish accuracy or reasoning emergence.

## Sources

- [Base model card](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
- [Countdown dataset](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)
- [Qwen3.5 text-only loading](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer)
- [PEFT LoRA documentation](https://huggingface.co/docs/peft/developer_guides/lora)
