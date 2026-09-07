# Corrected GRPO smoke: parameter updates verified, accuracy unchanged

This fresh run used Qwen3.5-0.8B-Base, a 1,024-token completion limit, plain prompts,
and the corrected `final-answer-v2` correctness-only verifier. No SFT or formatting
reward was used. Five optimizer updates consumed 40 sampled training answers.

| Optimizer update | Mean reward | Gradient norm |
|---|---:|---:|
| 1 | 0 | 0 |
| 2 | 0 | 0 |
| 3 | 0 | 0 |
| 4 | 0.125 | 0.421599 |
| 5 | 0.125 | 0.460564 |

Both correct training answers used three numbers. Their expressions were
`50 + 68 - 36` and `99 - 22 + 5`, both reaching target 82. All 186 LoRA B tensors
changed from zero, with maximum absolute value approximately 1.315e-5. All saved
adapter tensors were finite. This verifies an actual reward-driven update.

Training took 356.9 seconds. Peak allocated GPU memory was 5.37 GiB, with 6.66 GiB
reserved. Runtime and memory are local measurements, not universal estimates.

The final adapter was evaluated on the same eight validation puzzles and four
samples per puzzle as the rescored long base-model diagnostic. Before and after
both scored 2/32 correct (sampled pass@1 6.25%, pass@4 25%). Twenty-six completion
texts changed, but the aggregate score did not. There is no demonstrated accuracy
gain in this small pilot and no evidence that reasoning first emerged during RL.

Evidence:

- `metadata.json`: exact configuration, reward protocol, and package versions.
- `training.jsonl`: per-update loss, reward, gradient, and memory measurements.
- `rollouts.jsonl`: all training answers, with step counting completed updates
  before generation (step 3 feeds optimizer update 4).
- `checkpoint-verification.json`: finite tensors and nonzero LoRA B matrices.
- `summary.json`: completed step count, runtime, and memory.
- `EXAMPLES.md`: fixed-selection examples, not a collection selected for success.
- `../long-v2-after/`: complete adapter validation and comparison verification.
- `../long-v2-plots/`: four plots generated from the measured data.

The local final adapter is `../../checkpoints/long-v2-smoke/final/`; checkpoint 5
also contains optimizer, scheduler, RNG, and trainer state for continuation. These
large artifacts and TensorBoard event files are intentionally excluded from Git.

The next recommended test is continuing to 20 total updates, followed by matched
validation. The additional 15 steps would take roughly 18 minutes at the observed
training speed, plus evaluation. That extension and the full 200-step experiment
have not been run. The final test split remains unevaluated.
