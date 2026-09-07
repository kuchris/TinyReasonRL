# Longer-completion diagnostic under the original verifier

Only the completion limit changed from 256 to 1,024 tokens. The model revision,
plain prompt, seed, temperature, top-p, and evaluation top-k setting were unchanged.
These are four samples for each of the same first eight validation puzzles.

The original strict single-answer-tag verifier assigned zero reward to all 32
outputs. Average length increased from 146.97 to 444.75 tokens; truncation fell
from 50% to 34.375%. A post-generation audit then found two correct final answers
rejected because their scratch work contained earlier answer tags. The raw JSONL
and metrics preserve that original scoring; they must not be read as proof that
all final arithmetic expressions were wrong.

The affected outputs are sample 1 for numbers `[3, 42, 95]`, target 50, and sample
1 for `[92, 36, 73]`, target 55. Both final answers use every number correctly and
reach the target. Neither output came from an RL-trained checkpoint.

The corrected verifier and separately rescored measurements are recorded in the
subsequent final-answer-v2 change. No generated answers were edited or resampled.
