# Decoy-at-depth: a turbo2 exact-value cliff at long context

Date: 2026-06-17. Rig: RTX 4090, WSL2, CUDA 13.0. Fork: llama-cpp-turboquant
(`tom/catchup-from-feature` @ ed81ed0). Script: `runners/decoy-bench.py`. Data:
`results/decoy-at-depth-2026-06-17.csv`.

This is a found-object finding, reported with its own caveats. Whether the cliff is a
behavior of the cache or partly an artifact of the metric is the first thing the invention
front (`INVENTION-FRONT.md`) tests. Read that before treating this as settled.

## Task

A needle-in-haystack with distractors. One canonical entry ("the secret access code for unit
Orion is X") plus four decoy entries for other units, scattered through a long log. The model
must answer the exact code for Orion. We measure three things per cell (N positions of the
canonical needle):
- retrieval: the canonical code appears in the answer.
- decision: the answer IS the canonical code (exact match).
- decoy_rate: the answer is one of the decoy codes.

Filler is sized to a target token count per model (tokenizer density differs: Llama ~2.4
chars/token, Mistral/DeepSeek ~2.0; calibrated per model). Greedy decode, temp 0, seed 1,
flash attention on. Only the KV cache type changes within a model.

## Result: the depth curve (Llama-3.1-8B, N=5 per cell)

decision accuracy:

| depth | f16 | turbo4 | turbo3 | turbo2 |
|---|---|---|---|---|
| 4096  | 1.0 | 1.0 | 0.8 | 0.8 |
| 8192  | 1.0 | 1.0 | 1.0 | 1.0 |
| 16384 | 1.0 | 1.0 | 1.0 | 1.0 |
| 32768 | 1.0 | 1.0 | 1.0 | 0.4 |

f16, turbo4, turbo3 hold across depth. turbo2 (the most aggressive, 2-bit values) holds to
16k then drops to 0.4 at 32k. The failures are value corruption (digit flips, hallucinated
words, blanks), not decoy confusion (decoy_rate stays ~0).

## Confirmation at 32k (Llama-3.1-8B, N=16)

| KV | decision | failures | tok/s | VRAM |
|---|---|---|---|---|
| f16 | 1.0 | 0/16 | 84.7 | 9499 MiB |
| turbo3 | 1.0 | 0/16 | 80.7 | 5999 MiB |
| turbo2 | 0.438 | 9/16 | 89.7 | 5843 MiB |

The N=5 cliff is not low-N noise. At N=16, turbo2 fails 9 of 16 at 32k while f16 and turbo3
are perfect on the same prompts and positions.

## Cross-family: Mistral-7B-v0.3 (N=16, 32k)

| KV | decision | tok/s | VRAM |
|---|---|---|---|
| f16 | 1.0 | 91.3 | 8819 MiB |
| turbo3 | 0.938 | 85.3 | 5577 MiB |
| turbo2 | 0.375 | 103.0 | 5423 MiB |

Same shape on a different family. turbo2 cliffs (0.375), turbo3 nearly holds (0.938), f16
holds. The cliff is not Llama-specific.

## DeepSeek-Coder-V2-Lite (MLA): turbo impractical, not tested for the cliff

We could not get a clean cliff data point on the MLA architecture. turbo KV (quantized V)
requires flash attention; on DeepSeek-V2-Lite llama.cpp auto-disables FA, so turbo fails to
create a context (`quantized V cache was requested, but this requires Flash Attention`).
Forcing `-fa on` lets turbo load but decode is ~1.7 tok/s and 32k runs time out. So in the
current build, turbo KV on MLA is not usable. f16 on MLA works (0.938 at one N=16 cell before
we stopped the run). This is a build/architecture limitation, recorded, not a cliff
measurement.

## What this is and is not

- It is: a depth-resolved, cross-family demonstration that the most aggressive turbo KV mode
  loses exact value-recovery at 32k while average fidelity (and the milder turbo3) does not,
  and that turbo3 is the long-context sweet spot (decision-lossless at 32k, ~37% less KV than
  f16, comparable speed).
- It is not: a claim that turbo2 is bad in general (2-bit V is the most aggressive mode by
  design), nor a fidelity-quality claim. The metric is exact-code-match, a tail operator;
  perplexity (a body operator) stays flat. The gap between those two rulers is part of what
  we are calling a cliff, and the invention front tests whether the cliff is a structured
  behavior or partly a metric artifact before any method is built on it.

## Useful to implementers

- The `-fa`/MLA interaction: turbo KV silently fails on MLA models unless flash attention is
  forced on, with an error that does not point the user to the fix. Auto-enabling FA when
  turbo KV is requested, or a clearer error, would help.
- Mode selection for long context: turbo3 holds at 32k where turbo2 does not, at similar
  memory and speed. turbo2's extra aggressiveness buys little and costs exact recovery at
  depth.
