# Pilot evidence: direct RL has no correctness signal yet

Date: 2026-09-07. Hardware: NVIDIA RTX 5070 Ti, 16 GB, native Windows.

## What actually ran

1. CUDA and BF16 tensor computation; revision-pinned text-only base-model loading
   with no missing or mismatched weights.
2. Download and schema inspection of all 490,364 Countdown rows. Group-key
   deduplication removed 40,794 repeated puzzles. A full ID-set comparison verified
   that training, validation, and test splits have no intersecting puzzle IDs.
3. Four sequential answers to one validation puzzle, safely scored.
4. A 100-puzzle validation probe, four samples per puzzle, generated in groups.
5. One GRPO optimizer step, checkpoint save, restart, and continuation through
   step five. All 40 training rollouts were saved.
6. A matched eight-puzzle evaluation of the final smoke adapter.
7. A separate eight-puzzle base-model chat-template diagnostic, without SFT or
   worked examples. This changes the prompt condition and is not a trained model.
8. Four plots generated from actual saved measurements.

## Findings

- Plain baseline: 0/400 correct, 0/100 puzzles solved within four samples.
- Legal expressions using exactly the supplied numbers: 11/400 (2.75%). All eleven
  reached the wrong target. Half the generations exhausted the token budget.
- Training: zero reward, zero reward variance within every group, and zero gradient
  norm at every step. The LoRA B matrices remained zero; saved adapters at steps
  two and five were identical. This is a stalled reward signal, not a successful
  learning curve.
- Matched before/after: 0/32 versus 0/32, with identical generated completions.
- Chat diagnostic: 0/32 correct; 29/32 truncated. Fluent arithmetic discussion did
  not produce valid solutions in this sample.
- Peak PyTorch memory: 1.59 GiB for the 100-puzzle baseline; 2.49 GiB for the resumed
  training smoke, reserving at most 3.05 GiB. These counters exclude other apps.

## Evidence files

| Artifact | Path |
|---|---|
| Dataset revision, deduplication, split counts | `dataset-manifest.json` |
| First milestone | `milestone1/` |
| 100-puzzle baseline metrics and raw outputs | `baseline-validation/` |
| First eight baseline puzzles, extracted without resampling | `baseline-validation-first8/` |
| GRPO configuration and package versions | `smoke-training/metadata.json` |
| Restart configuration | `smoke-training/resume-metadata.json` |
| Per-step reward and gradient evidence | `smoke-training/training.jsonl` |
| Every training answer | `smoke-training/rollouts.jsonl` |
| Adapter tensor and checkpoint verification | `smoke-training/checkpoint-verification.json` |
| Actual adapter evaluation | `after-smoke-validation/` |
| Separate template diagnostic | `chat-diagnostic/` |
| Plots | `smoke-plots/` |

Checkpoints and TensorBoard event files are local, not tracked by Git. The final
adapter is `../checkpoints/smoke/final/`. Restartable checkpoints include optimizer,
scheduler, RNG, and trainer state. Retention rotated away checkpoint one after
resumption; checkpoints two through five remain.

## Interpretation and next experiment

The implementation and checkpoint restart are verified. No reward improvement
or emergent reasoning was demonstrated. This result is limited to this prompt,
sampling configuration, model revision, short token budget, and small pilot.
It does not establish that the model cannot learn Countdown through RL.

The full 200-step experiment is configured but not executed. Test-set evaluation,
multiple training seeds, confidence intervals, intermediate validation checkpoints,
and a quantitative analysis of backtracking/self-checking remain future work.
The validation plot therefore shows only two measured endpoints, with no invented
intermediate curve. Some local runs overlapped; runtimes are not GPU throughput
benchmarks.

Before increasing training duration, test a longer completion budget on a fixed
development subset. Keep the reward and base weights unchanged so the effect of
that one change can be observed. Training needs mixed-reward groups; attractive
scratch work alone supplies no learning signal under this reward.
