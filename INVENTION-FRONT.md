# Invention front: the Value-cache long-context cliff

Status: 2026-06-17. This is a roadmap, not a result. The method section is deliberately
blank. It will be written by what the experiments below return, not before.

## Origin (the human gesture)

Felipe (inventor) asked whether, from the position our measurement gives us, we could
build our own KV-quantization math rather than only benchmarking TheTom's TurboQuant and
Huawei's KVarN. This file is the disciplined answer: a falsification-first protocol that
decides, cheaply, whether there is a method to build at all, before any method is built.

## What we measured (the found object)

On the decoy-at-depth benchmark (`runners/decoy-bench.py`, results
`results/decoy-at-depth-2026-06-17.csv`): the most aggressive 2-bit Value-cache
quantization (turbo2) shows a sharp drop in exact value-recovery at 32k context while
perplexity stays flat. Confirmed cross-family (Llama-3.1-8B and Mistral-7B), N=16, with
f16 and the milder turbo3 holding. Failure mode is value corruption (digit flips,
hallucinated tokens), not distractor confusion.

## The discipline (Casey Reas review, 2026-06-17)

The plan that walked in carried a fully specified four-part codec (see Appendix B). Casey's
verdict: MODIFY. A method you can write down in full before running an experiment did not
come from the behavior; it came from the literature. Form must be the trace of behavior,
not a framework imposed on it. The corrections, adopted:

1. The cliff is not yet established as a behavior of the system rather than a form produced
   by our apparatus. We measured PPL (a body/average operator) against exact-digit-match (a
   tail/zero-tolerance operator) and named the gap a "cliff." Part of that gap may be the
   difference between the two rulers, not a property of the cache.
2. Therefore a falsifier-0 comes before everything: does the cliff exist as a *shape* in the
   continuous error distribution, before any pass/fail threshold?
3. The KVarN cross-stack reproduction moves early (cheapest artifact detector: its authors
   looked at V and reported no cliff).
4. One titrated experiment can let the cliff reveal its own form (Experiment 3).
5. The competitor (arXiv:2605.20868) shipped a *behavioral* solution (detect bad attention
   output, fall back to fp16), not an elegant codec. The phenomenon may be governable by a
   gate long before it is solvable by math. Do not assume a codec is the shape of the answer.

## Experiments (ordered; each has a kill criterion)

### F0 - Is the cliff a shape, or our metric slicing a fat tail?
- Measure per-entry value-output error as a CONTINUOUS distribution across context length
  (4k, 8k, 16k, 32k), per stack. Do not binarize.
- Look for structure: a bimodal distribution or a sharp knee = a real, separable population
  of catastrophically-wrong entries. A smooth fat tail that our threshold happened to slice
  = artifact.
- Controls: (a) null tasks at the same context length (summarization, multi-evidence
  aggregation, paraphrased retrieval) - if the cliff only exists on decoy-at-depth, it is a
  property of the instrument; (b) seed/placement jitter on f16 to establish the measurement
  noise floor - if turbo2's failure rate sits inside the f16-jitter band, the cliff is partly
  instrument variance.
- KILL: if the error is a smooth fat tail with no shape, and turbo2 sits in the f16 noise
  band, the cliff is (mostly) our apparatus. Stop the invention front; report the
  metric-dissociation honestly and move on.

### F1 - KVarN cross-stack cliff reproduction (run early, fused with the throughput re-bench)
- Bring up the KVarN vLLM stack (already needed for the throughput re-bench of philippebich's
  optimized main). Run decoy-at-depth on KVarN 2-bit V past 32k.
- KVarN's paper claims V is the easy case and reports no cliff (tested only to ~32k on
  Qwen3-4B). This is the cheapest possible artifact detector.
- KILL/REFRAME: if KVarN's fp8 per-token scale already kills the cliff, our finding is
  stack-specific (a KIVI/turbo artifact), not fundamental. Still publishable, but reframed.

### F2 - Titrated oracle-protect: the experiment that writes the method
- Protect the needle's value-bearing V-entries in fp16 in GRADED fractions, on two axes:
  (a) top-k by attention mass, (b) top-k by |V| magnitude. Sweep k from 0 to all. Two
  recovery curves on one plot.
- The shape of these two curves IS the form of the cliff. It answers simultaneously:
  is the cliff sparse-recoverable at all (how small is the protected set?), and is the right
  selection criterion attention or magnitude?
- This is the single highest-information experiment. It either earns the method's existence
  or evaporates it in an afternoon.
- KILL: if even a generous oracle protection set does not lift the cliff, it is not a
  sparse-recovery problem (likely accumulated decode dynamics) and the entire codec framing
  is the wrong tool.

### F3 - Un-rotated vs rotated V (only if F0/F1/F2 say the cliff is a structured sparse set)
- Same model, 2-bit V, with vs without Hadamard rotation, exact-recovery metric vs depth.
- Tests the claim (from the rotation sweep) that incoherence-via-rotation helps average error
  and hurts exact recovery of specific entries.
- KILL: if rotated V recovers as well or better, the rotation-hurts-exact-recovery thesis is
  dead.

## Method

TBD. To be written by F0-F3. If F2's curves show a small attention-selected set recovers the
cliff, a method earns its existence and its form (gate vs codec, selection criterion,
position-awareness) is dictated by the curves. Until then this section stays blank on purpose.

## Appendix A - The literature map (6-way sweep, 2026-06-17)

Convergent finding across six independent mathematical territories: the incumbents
(TurboQuant, KVarN, RateQuant, the field) optimize an AVERAGE objective (MSE/variance/PPL).
Our cliff lives in the tail they discard. The right objective for Values appears to be
attention-weighted (A^2) per-entry distortion (HeadQ: corr 0.984 vs 0.49 for raw MSE), and
"value-bearing" entries are the query-attended ones, not the large-magnitude outliers (so
magnitude-based selection chooses the wrong set). Rotation/Hadamard spreads outliers to lower
average error, which mathematically destroys per-entry exact recoverability (no measurement
redundancy in KV).

Key prior art to read before any build:
- arXiv:2605.20868 runtime-certified bounded-error attention (closest competitor; behavioral
  fp16 fallback, certified per-head value-error bound, targets NIAH/RULER 8k-128k).
- HeadQ (2605.03562) A^2-weighted value distortion; keys-only, leaves V unbuilt.
- SQuat (2503.24358) subspace-orthogonal quantization; keys-only.
- KVQuant (2401.18079) dense-and-sparse, magnitude-selected outliers.
- KVarN (2606.03458) Sinkhorn-style variance normalization; claims V is easy.
- CAOTE (2504.14051) value-output-error gate (value-aware, cheap online).
- MixKVQ (2512.19206) query-aware mixed precision; V left uniform 2-bit.

## Appendix B - Codec hypotheses (NOT a plan; do not build before F0-F2)

Held here as hypotheses the experiments may or may not earn. The honest novelty, if any, is
on the Value side (the field opened the Key side and left V): (a) SQuat-for-V (error
orthogonal to the attention row-space), (b) per-entry weight calibrated to decision-margin
(projection onto the answer-token unembedding direction), (c) position/depth-aware bit
allocation, (d) cheap online gate (CAOTE) keyed to value-output-error. The mechanism
(mixed-precision / top-k fp16) is well-trodden; any contribution is the selection criterion
(decision/attention-aware, not magnitude), position-awareness, and the objective
(decision-margin, not MSE). The most in-love and least-earned component is (b); delete first
if forced to cut.
