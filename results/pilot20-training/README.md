# Twenty-step GRPO pilot

Completed 2026-09-07 on the RTX 5070 Ti. The model receives direct GRPO with
LoRA, without SFT. Configuration and reward protocol remain the same as the
corrected five-step run: `configs/long-completion.json`, 1,024 completion tokens,
and `final-answer-v2`.

## Outcome

| Matched validation checkpoint | Correct answers | Sampled pass@1 | pass@4 |
|---|---:|---:|---:|
| Base | 2/32 | 6.25% | 25% |
| Step 5 | 2/32 | 6.25% | 25% |
| Step 20 | 1/32 | 3.125% | 12.5% |

All conditions use the same eight validation puzzles, four samples per puzzle,
prompt, token cap, seed, and evaluation sampling settings. Raw rewards and
aggregate metrics were independently checked with the current verifier. At step
20, 29/32 completion texts differ from each earlier condition. Half the final
completions hit the token cap; only two expressions are valid, and one is correct.
The correct answer uses three numbers. No four-number puzzle was solved.

This probe shows no accuracy improvement. Eight fixed development puzzles and
one seed are insufficient for a reliable estimate of a training effect or any
claim of reasoning emergence. The final test split remains unevaluated. The
configured 200-step experiment has not run.

## Training and continuation checks

- Resumed `checkpoints/long-v2-smoke/checkpoint-5` with optimizer, scheduler, RNG,
  and trainer state; completed exactly 20 total updates.
- Copied the first five updates' metadata and JSON logs into this directory,
  then appended 15 new updates. Original five-step results and source checkpoint
  remain unchanged. `continuation.json` records source provenance and its hash;
  `resume-metadata.json` records the new invocation. `metadata.json` is the
  inherited original configuration, so its five-step limit is historical.
- Combined logs contain 160 rollouts, eight per update, on 40 distinct training
  puzzles. All belong to the configured training subset; none overlap validation.
- Six training answers earned reward, all on three-number puzzles. Four were
  among the 120 additional rollouts. Positive gradient updates: 4, 5, 6, 8, 14, 20.
  These changing training puzzles do not provide a matched accuracy trend.
- All 372 adapter tensors changed from step five and remained finite. Largest
  absolute parameter change: 0.0000884014. Source checkpoint hash is unchanged.
- Additional training took 906.8 seconds (15.1 minutes), excluding the original
  five steps. Peak PyTorch allocated/reserved memory: 5.37/9.55 GiB; other desktop
  allocations are excluded. Final validation took 350.3 seconds.
- Resumed trainer state retained checkpoint cadence five despite the requested
  save interval 20. Final checkpoint includes optimizer, scheduler, RNG, and
  trainer state; adapters and checkpoint files are local and excluded from Git.

## Evidence

- [Training verification](verification.json), [continuation provenance](continuation.json),
  [training summary](summary.json), [training metrics](training.jsonl), and
  [raw training rollouts](rollouts.jsonl).
- [Validation metrics](../pilot20-after/metrics.json),
  [matched comparison checks](../pilot20-after/comparison-verification.json), and
  [raw validation answers](../pilot20-after/generations.jsonl).
- [Fixed-selection readable examples](EXAMPLES.md).
- [Reward](../pilot20-plots/reward.png),
  [completion length](../pilot20-plots/completion-length.png),
  [measured validation checkpoints](../pilot20-plots/validation-accuracy.png), and
  [before/after comparison](../pilot20-plots/before-after.png).

Checks: 39 tests passed; the plotting CLI rendered all four figures successfully;
TensorBoard event inspection confirmed measured validation at steps 0, 5, and 20
and peak memory at all 20 steps. No interpolated validation scores are reported.
