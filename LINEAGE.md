# Lineage: kv-score and turboquant-cuda-bench

kv-score did not start from zero. Its axes are the distilled, cross-stack form of
work done earlier in [sztlink/turboquant-cuda-bench](https://github.com/sztlink/turboquant-cuda-bench),
a dense single-GPU research repo. This file maps the connection so the earlier
evidence is reachable instead of buried.

## The two-layer split is the same one buun named

The kv-score **R axis** (LCP of greedy decode vs same-stack fp16) and the
decision-robustness view (logprob margin, from @spiritbuun's `margin_bench`) are
not competing metrics. They are two layers, and turboquant-cuda-bench already
stated the split:

- **R / trajectory layer == REFRACT.** "Does the token path match fp16." See
  [`bench-public/refract-trajectory/RESULTS.md`](https://github.com/sztlink/turboquant-cuda-bench/blob/main/bench-public/refract-trajectory). The repo's spine finding: KLD and token-match can pass while the generation **trajectory** collapses. A "lossless" KV claim measured only by KLD or exact-match is under-specified.
- **decision / action layer == KVFidelity.** "Does the degradation reach the
  answer." `GLOSSARY.md` states it directly: REFRACT detects token/trajectory
  degradation; KVFidelity asks whether it reaches the action trace.

buun's two-way caution ("trajectory fidelity is not decision goodness" and the
symmetric "routing decision is not fidelity goodness") is exactly this split. It
was measured before the conversation.

## The dissociation, in data, both directions

The AIME24 answer-trajectory atlas
([`06-publicable/kvfidelity/2026-05-trace-atlas-v4`](https://github.com/sztlink/turboquant-cuda-bench/tree/main/06-publicable/kvfidelity/2026-05-trace-atlas-v4),
decomposed into Discovery / Retention / Closure across KV configs) shows both
halves of buun's symmetry:

- **Trajectory faithful, decision wrong** (idx9, FullKV-4096): the correct answer
  `116` is emitted nine times late in the trajectory, then the run closes on `4`.
  See `08_label_birth_idx09_fullkv_4096.png` and
  `05-analysis/kvfidelity/2026-05-15-label-review-addendum-v0.md`.
- **Trajectory garbage, decision right** (idx7): a fully degenerate
  repetition-loop trajectory still lands `25`.

Caveat carried from that work: it is n=30, single-seed, extractor-labeled.
Treat as a demonstrated phenomenon, not a calibrated rate.

## The cap mechanism is net-new; the asymmetry is confirmed at the kernel

kv-score's finding that the **vLLM KVarN plugin** reserves a fixed fp16 sink/tail
pool that clamps `max_num_seqs` does not exist anywhere in turboquant-cuda-bench:
it is a new serving-layer result. What the older repo confirms is the other half:
`KERNEL-MAP.md` shows TurboQuant's vLLM path is Triton (not the llama.cpp `.cu`
kernels) and has **no reserved fp16 pool** (its "tail" is only head-dim alignment).
So the cap is genuinely KVarN-plugin-specific, with no TurboQuant analog.

## NIAH discipline (carry into any router-120 claim)

turboquant-cuda-bench already paid for these lessons:

- A single-needle NIAH under KV quant is **saturated** (25/25) and tells you
  nothing. Discrimination only appears with **decoys**, and the decoy *type*
  swings exact-match by ~48 points. `bench/longctx-decoy-isolation-2026-05-10`,
  `bench/evidence-utilization-phase-2026-05-17`.
- A verifier/gate that looked positive at N=100/300 **vanished at N=500**
  (p=1.0). `STATE.md`, `KEY-FINDINGS.md`. Margin is a confidence-flavored signal;
  do not claim a KV-quant decision effect from a small slice without a held-out
  scale check.

So router-120 exact-match and the margin are decision-robustness signals
**conditional on the distractor taxonomy, canonical rank, prompt, and model
family**, not trajectory claims and not calibrated-correctness claims.

## Do not resurface

Retired or falsified in the older repo, kept here so they are not repeated:
the gated verifier / answer-rerank control (falsified at N=500), "retrieved != used"
as a strong thesis, EPKV as a serving speedup, and RotorQuant throughput numbers
(a PPL proxy, not a clean decode benchmark).

## Pointers

- Head-to-head, already run: [`bench/kvarn-vs-turboquant-2026-06-07`](https://github.com/sztlink/turboquant-cuda-bench/tree/main/bench/kvarn-vs-turboquant-2026-06-07) (RESULTS-CORRECTED, THROUGHPUT). KVarN k4v2 near-lossless on Qwen3-4B at the lowest bit budget tested; saturates at ~96 tok/s under batched serving (`_sinkhorn_log_kernel` JIT recompile, filed `huawei-csl/KVarN#15`).
- KVarN GQA non-power-of-2 decode crash: filed `huawei-csl/KVarN#12`.
- Canonical state of the older repo: its `STATE.md` and `KEY-FINDINGS.md`.
