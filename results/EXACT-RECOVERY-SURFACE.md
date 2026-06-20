# KV value-cache exact recovery vs depth: a bits-vs-depth surface (and a self-correction)

Date: 2026-06-20. Rig: RTX 4090, WSL2, CUDA 13.0. Fork: llama-cpp-turboquant @ ed81ed0.
Model: Meta-Llama-3.1-8B-Instruct-Q4_K_M. Harness: `runners/decoy-bench-f0.py`.
Data: `results/f0-2026-06-20/` (logs, per-case dumps, `exact-recovery-surface.csv`).

## Self-correction up front

An earlier note in this repo (`results/decoy-at-depth-2026-06-17.md`) reported a "turbo2 (2-bit V)
exact-value cliff at 32k." That framing was wrong in shape. Controlled re-runs show the apparent
32k cliff was an artifact of **Boundary V**: this build auto-enables q8_0 on the first and last 2
V-layers when the V cache is turbo2 (opt-out `TURBO_LAYER_ADAPTIVE=0`), and that mitigation was on
in every prior run. It held 2-bit V at ~1.0 up to 16k and only gave out at 32k, manufacturing the
appearance of a cliff. With Boundary V OFF (pure 2-bit V), there is no 32k cliff: exact recovery is
already degraded by ~8k and plateaus. The real finding is a smooth bits-vs-depth surface, reported
below.

## Task and metric

Needle-in-haystack with distractors (decoy-at-depth): one canonical entry ("the secret access code
for unit Orion is X") plus four same-format decoys for other units, scattered through a long log.
The model must answer Orion's exact code. We score **exact recovery** = the answer IS the canonical
code (exact match), over N needle positions per cell. Greedy, temp 0, flash attention on. Only the
KV cache type changes within a model. We also record a continuous error (Levenshtein to the
canonical code) and the failure mode per case.

## The surface (Llama-3.1-8B, exact-recovery accuracy)

Pure codecs (turbo2 = 2-bit V with Boundary V OFF):

| depth | f16 | turbo3 (3-bit V) | turbo2 (2-bit V, pure) |
|---|---|---|---|
| 8k  | 1.0 | -   | 0.625 |
| 16k | 1.0 | -   | 0.44 (0.375-0.5) |
| 24k | 1.0 | -   | 0.375 |
| 32k | 1.0 | 1.0 | 0.44 (0.25-0.625, 6 runs/seeds) |
| 49k | 1.0 | 0.875\* | 0.25 |
| 65k | 1.0 | 1.0 | 0.25 |

\* turbo3@49k = one fail of 8; turbo3@65k is back to 1.0, so the 49k dip is instance noise.

- **f16 is exact (1.0) at every depth 8k to 65k**, across all prompt seeds. The ceiling and the
  control: the rig, the metric, and the prompts do not themselves produce failures.
- **3-bit V (turbo3) holds exact recovery to 65k** at roughly 37% less KV memory than f16. It is the
  practical floor for exact value recovery at long context.
- **2-bit V (pure) degrades early (~8k) and plateaus** around 0.25-0.45, with no cliff. The rate is
  instance-noisy (0.25-0.625 across seeds at 32k); any single number is one instance.

Boundary V (q8_0 on the 4 boundary layers) partially masks this: turbo2 at 32k reads 0.438 with
Boundary V ON vs 0.25 with it OFF. The mitigation delays the loss; it does not remove it.

## The dissociation (why average metrics miss it)

Perplexity (a body/average operator) stays flat across this whole surface for both turbo2 and
turbo3. Exact recovery (a tail operator) does not. The gap between those two rulers is the point:
2-bit V can look fine on average fidelity while losing the ability to return a specific value
verbatim at depth. The build's per-channel equalization (InnerQ) does not close the gap and can
widen it at full strength.

## Failure mode (per-case dump)

The failures are predominantly value corruption / confabulation, not distractor confusion. Examples
from the dumps: bare numbers ("4582", "1318"), a hallucinated unit word ("GIANT", not in the prompt
at all), confabulated sentences ("The final answer is: 335"), and partial recoveries (right digits,
wrong word; right word, mangled number). Decoy pickup is rare at 32k (~1 of 16) but rises at 65k
(decoy_rate ~0.25): at extreme depth, unable to recover the exact value, the model more often grabs
another unit's code.

## Scope and caveats

- Shown here: Llama-3.1-8B, one stack (this llama.cpp fork). The shape held cross-family in earlier
  (Boundary-V-on) runs on Mistral-7B; a clean cross-stack / cross-codec replication (e.g. a
  different 2-bit V codec) is not included here and is the natural next check.
- N = 8 to 16 needle positions per cell; rates are instance-noisy. f16 = 0 failures across all
  seeds/depths, so the turbo2 degradation is well outside the f16 band.
- Greedy decode; the seed varied is the prompt-construction seed (different codes/filler/positions),
  not the sampling seed.
- 65k required passing the prompt via file (`-f`); `-p` overflows ARG_MAX at ~157KB. Validated
  identical to `-p` at 49k.

## Takeaway

For workloads that need exact value recovery at long context (retrieval of specific codes / IDs /
numbers), 3-bit V is the practical floor and holds to 65k at ~37% less KV than f16; 2-bit V loses
exact recovery from moderate depth onward while average metrics stay flat. The earlier "32k cliff"
was a mitigation artifact, not an intrinsic property of 2-bit V.
