# kv-score

Score the KV-cache compression options on **your** rig, and read the trade-off instead of a leaderboard. KVarN and TurboQuant do not have a winner: each one leads a different axis, and which axis you cannot give up is the only question that matters.

## Where this comes from

This is not a vendor-neutral arbiter. It is one independent rig owner's measurement, run on a single RTX 4090 under WSL2, comparing two methods with very different structures behind them: **KVarN**, built and maintained by a corporate team ([Huawei](https://github.com/huawei-csl/KVarN)), and **TurboQuant**, an independent one-person fork ([TheTom](https://github.com/TheTom/turboquant_plus), @no_stp_on_snek). The numbers are reproducible and every correction is public. The framing, including which axes go into the headline score, is a choice. Read it as a position, not a view from nowhere.

## The four axes

- **Q** (behavior): does the model still *act* the same? Exact-match on 120 action-routing cases vs fp16. Behavior breaks before benchmark scores do.
- **T** (throughput): engine-reported tokens/s under a load that saturates the GPU, vs fp16.
- **C** (capacity): KV tokens that actually fit, measured from the engine, vs fp16.
- **R** (fidelity): how long greedy decoding stays on the exact fp16 token path. This is the axis that still discriminates at 14B, where Q saturates.

## The trade-off (Qwen2.5-7B; fidelity also at 14B)

One grid, one row per axis. Best per axis in **bold**. No column wins every row.

| axis | KVarN k4v2 | KVarN k4v4 | TurboQuant k8v4 | TurboQuant k4v2_nc |
|---|---|---|---|---|
| Q  behavior (vs fp16=100) | 92 | **100** | 95 | 91 |
| T  throughput (vs fp16=100) | 82 | 82 | **88** | 84 |
| C  capacity (vs fp16=100) | 199 | 153 | 214 | **322** |
| R  fidelity, 7B (LCP chars) | **601** | 357 | 134 | 55 |
| R  fidelity, 14B (LCP chars) | 1024 | **1034** | 199 | 53 |

**TurboQuant takes throughput (T) and capacity (C). KVarN takes behavior (Q) and fidelity (R).** The composite KV-Score (below) puts the headline number on TurboQuant because it weights T and C and leaves R out; that is a choice, and KVarN's win on R is real even though it does not enter the score. Pick by the axis your workload cannot give up, then rank.

## Two measurement traps (the part that outlives the numbers)

The tables above are a dated snapshot and will age within weeks. These two traps do not: anyone benchmarking KV-quant, on methods that do not exist yet, will hit them. They are the most transferable thing here.

**1. A throughput burst that does not saturate the GPU inverts the ranking.** My first throughput numbers used a client load that never filled the engine. fp16 looked slow because it was capacity-limited at low concurrency, and the order read KVarN ahead of TurboQuant. Under a load that actually saturates the card (200 concurrent decodes, GPU at full board power, verified by `nvidia-smi`), the order flips: TurboQuant sustains full concurrency at near-fp16 throughput. The corrected table above is the result. **Takeaway: use a load that saturates the GPU, and report the power draw that produced the number.**

**2. KVarN's fp16 tail pool caps concurrency, by design, and the cap is model-dependent.** KVarN keeps each request's sink and in-progress tail in fp16 forever, in a fixed-size pool sized against the post-weight memory envelope. The platform clamps `max_num_seqs` so the pool fits, and logs it: `capping max_num_seqs 256 -> N`. On a 24 GB card the cap is ~163 on Qwen2.5-7B but ~35 on Qwen3-8B (more kv-heads x layers, costlier slots), which throttles KVarN's throughput at 8B while TurboQuant runs uncapped. It is tunable (raise `--gpu-memory-utilization` or `KVARN_POOL_MEM_FRAC`: 0.80 to 0.90 doubled the cap and the throughput in my sweep) but bounded by VRAM. The [Mistral bug](#what-i-measured-wrong-and-fixed) is the same mechanism at its extreme: a doubled weight estimate drove the cap to 1. **Takeaway: when benchmarking KVarN throughput, read the `capping max_num_seqs` line first; a low cap there, not the kernel, may be your bottleneck.**

## The fidelity axis (R): how long the cache holds the fp16 trajectory

The behavior probe saturates at 14B (every method scores 118-120/120), so it cannot rank methods on the large models where serving actually happens. A fourth axis does. Decode the same prompt greedily (temperature 0) under fp16 and under a quant method on the same stack, and measure the longest common prefix (LCP): how many characters the quantized cache keeps on the exact fp16 token path before the first argmax flip. It is deterministic, exact, and it discriminates precisely where Q goes blind.

LCP in characters, median over 28 long-form prompts, 2048-token greedy generations, each method vs the fp16 of its own stack:

| model | KVarN k4v2 | KVarN k4v4 | TurboQuant k8v4 | TurboQuant k4v2_nc | KVarN / TQ gap |
|---|---|---|---|---|---|
| Qwen2.5-7B | 601 | 357 | 134 | 55 | ~4 to 11x |
| Qwen2.5-14B-AWQ | 1024 | 1034 | 199 | 53 | ~5 to 20x |

KVarN holds the fp16 trajectory far longer than TurboQuant: roughly 4x to 11x at 7B (bf16), depending on the TurboQuant preset, and the lead persists at 14B (AWQ), where the behavior probe saturates. KVarN's 2-bit-V preset (k4v2) even beats TurboQuant's higher-precision k8v4 (4-bit V). One caveat on the trend: the 14B row is AWQ-quantized, so its larger gap mixes model scale with weight quantization. A clean same-family bf16 scale comparison does not fit a single 24 GB card, so I report the 14B as "the lead holds at scale", not as a scale law.

**Note on what this axis measures.** LCP measures how long greedy decoding stays on the *exact* fp16 token path: a determinism and reproducibility metric, not a quality judgment. Both methods remain non-degenerate (4-gram diversity within noise of fp16 up to 2048 tokens). TurboQuant diverges earlier, but that is a different valid continuation, not a worse one. High LCP means "this cache faithfully reproduces your fp16 reference", not "this method generates better text". The measurement is determinism-checked per model. At 7B both stacks reproduce their own fp16 exactly across separate serves (8 of 8 prompts identical), so the LCP is purely the KV quantization. At 14B the stacks are not bit-identical run to run, but the fp16 self-agreement floor (median ~2000 to 3000 chars) sits well above the measured fidelity (1024 and 199), so the comparison still holds. Qwen3-8B is deliberately excluded from the table: the KVarN build is non-deterministic there (fp16 self-floor ~620 chars, on the order of the fidelity itself), which would cap the measurement, so no 8B fidelity number is reported. Control data: `results/rprobe-determinism-control-2026-06-13.csv`. Raw points: `results/rprobe-lcp-validated-n28-2026-06-12.csv`. Scope caveat: LCP measures the argmax path; a long-range-dependency task (needle recall, positional counting) is the next step to test whether quant error compounds over the horizon rather than just perturbing it early.

## The KV-Score (the reusable instrument)

When you do want one number for a single rig, the composite is the geometric mean of the three measured-on-vLLM axes:

```
KV-Score = (Q x T x C)^(1/3)        each axis: 100 = fp16 parity, on YOUR rig
```

Geometric mean on purpose: a method that wins capacity but collapses quality or throughput goes toward zero. You cannot buy score on one axis. **R sits outside this scalar on purpose**, and the consequence is explicit: the single headline number favors capacity-led methods. KVarN's strength lives in R.

Read it in two steps: first filter by the axis your workload cannot give up, then rank by score. For agent workloads that filter is Q:

- **Q >= 95**: behavior-safe. Rank these by score and pick.
- **Q 75-95**: degraded. Only if your workload tolerates behavior drift.
- **Q < 75**: broken for agent use, whatever the composite says.

A score above 100 means "the capacity gain outweighs the loss IF you value all axes equally". Most workloads do not, and the bands are the honest correction for that.

## Run it, and send your rig's row

```
# 1. T + C: serve each config, fire a saturating load, read the engine's own counters
runners/measure-tput.sh fp16      '<your vllm serve command, fp16 KV>'
runners/measure-tput.sh mymethod  '<same command with --kv-cache-dtype ...>'

# 2. Q: action-trace probe against each running config
python3 runners/probe-router.py http://localhost:8001/v1/completions 120

# 3. fill one CSV row per config, compute scores
python3 score.py results/myrig-mymodel-2026-06-12.csv
```

One rule: same model, same flags, same box for every row; fp16 on the same stack is the reference. `results/` is the point of the repo: each file is one rig's table, `results/<rig>-<model>-<date>.csv`. Methods score differently on native Linux vs WSL2, on a 4090 vs an H100; the catalog of rigs is the product, not any single table. Send a PR with yours.

## Per-model snapshot (dated, and meant to age)

These rankings are a snapshot on one 4090/WSL2 box, `gpu_memory_utilization 0.80`. What does not age is above (the traps, the laws below); what follows is the arithmetic of this date.

The behavior probe and the fidelity axis cover different parts of the size range, and the gap is the reason both exist:

| model | behavior probe (Q) | fidelity (R) |
|---|---|---|
| Qwen2.5-1.5B | floors, fp16 ~16/120 | not run |
| Qwen2.5-7B | discriminates, 91-100 | discriminates, KVarN >> TQ |
| Qwen3-8B | discriminates, 115-119 | not run |
| Qwen2.5-14B-AWQ | saturates, 118-120 | discriminates strongly |

Q has resolving power at 4-8B and saturates by 14B; R takes over exactly there. Detail:

- **Qwen2.5-7B** is the full table at the top. fp16 raw is ~97/120; KVarN k4v4 is the only behavior-lossless config (Q = fp16); TurboQuant leads T and C by a small margin.
- **Qwen3-8B**: Q stays discriminating (fp16 119/120, every method loses 2 to 4 points). Capacity favors TurboQuant k4v2_nc (3.26x). This is the model where KVarN's concurrency cap bites (clamped to 35 at gmu 0.80); see [trap 2](#two-measurement-traps-the-part-that-outlives-the-numbers). Qwen3 needs its chat template with thinking disabled; raw completion scores 0.
- **Qwen2.5-14B-AWQ**: every method 118-120/120. The routing task stops discriminating; this is what motivates the R axis.
- **Qwen2.5-1.5B** marks two edges. KVarN requires `head_dim` in {128, 256, 512}, so Qwen2.5-0.5B (head_dim 64) cannot run it at all, and the smallest comparable model is the 1.5B. And the probe floors: fp16 itself scores only ~16/120, so the behavior axis has no resolving power below ~4B. Directionally the most aggressive preset (TurboQuant k4v2_nc, 2-bit V) roughly halves at the floor while KVarN holds, but the signal-to-noise is low.

## Two laws beyond the numbers

The rankings will date. These two findings are the laws the rankings keep re-confirming, and they are what to cite a year from now.

**There is no universal best eviction method; there is a universal cliff.** Across SnapKV, TOVA, KeyDiff, CUR, ExpectedAttention and Knorm, behavior collapses between compression ratio 0.375 and 0.625 on the same cases, on three models in two families. The best press at 0.5 is model-dependent (SnapKV on Qwen2.5-7B, TOVA on Qwen3-4B, KeyDiff on Mistral-7B), and presses that survive one model collapse on another. The cliff does not move.

**In KV quantization, the normalization is the product, not the bit budget.** At the same 4-bit budget on an outlier-heavy model, naive per-channel int4 retains 0% behavior while every distribution-aware method is near-lossless. The technique that tames the outlier channels (KIVI's per-token V plus fp16 residual, KVarN's Sinkhorn normalization, TurboQuant's rotation) is the whole game; the bits are secondary.

## Supporting evidence

**Eviction family (kvpress @ ratio 0.5, transformers stack, Q/T/C vs its own fp16).**

| config | Q | T | C | KV-Score |
|---|---|---|---|---|
| fp16 transformers (reference) | 100.0 | 100.0 | 100.0 | **100.0** |
| SnapKV | 75.5 | 90.6 | 200.0 | **111.0** |
| ExpectedAttention | 56.1 | 95.7 | 200.0 | **102.4** |
| Knorm | 42.9 | 94.4 | 200.0 | **93.2** |

At 2x compression, eviction breaks 24-98 behavior points where quant breaks 4-6, and answer-level suites do not see it. Full ranking at 0.5 (of 120, baseline 98): SnapKV 74, TOVA 69, KeyDiff 61, CUR 60, ExpectedAttention 55, Knorm 42, LagKV 2, PyramidKV 1, StreamingLLM 0. The collapsed presses fail beautifully: output stays fluent while the decision or the target identity is wrong (LagKV answers "ignore;none", PyramidKV paraphrases the target, StreamingLLM deletes the middle where the payload lives). Fluency metrics pass all of them. The collapse curve (SnapKV by ratio, baseline 98): 0.125 -> 97, 0.25 -> 95, 0.375 -> 88, 0.5 -> 74, 0.625 -> 32, 0.75 -> 0. Behavior-safe for this workload is ~0.25 (1.33x capacity), well below the 0.5+ regime these methods advertise. Regime caveat: these are 1-2k token agent-routing prompts, the short end of what eviction is designed for. Throughput here is single-stream decode (transformers has no continuous batching), comparable only within this table. Raw points in `results/`. AdaKV is not-scored: its kvpress wrappers degenerate to repeated-token gibberish under both backends while the SnapKV scorer they wrap is clean, an integration failure of the adapter layer, not a behavioral verdict.

**The normalization cartography.** Same model (Qwen2.5-7B), same 4-bit KV budget, same 120-case probe, raw counts:

| method | bits | behavior (of 120) | stack (fp16 ref) |
|---|---|---|---|
| naive per-channel int4 (transformers QuantizedCache) | 4 | **0** | transformers (fp16 98) |
| KIVI (asymmetric, per-channel K / per-token V, fp16 residual) | 4 | **99** | transformers (fp16 98) |
| KIVI | 2 | 66 | transformers (fp16 98) |
| KVarN k4v2 / k4v4 (Sinkhorn normalization) | 4 | 90 / 99 | vLLM (fp16 98) |
| TurboQuant k4v2_nc / k8v4 (near-optimal rotation) | ~4 / 7 | 90 / 93 | vLLM (fp16 98) |

KIVI even at 2-bit (66) beats naive at 4-bit (0). Double-checked: KIVI identical across two passes; naive int4 = 0 reproduced across transformers 4.49 / 4.57 / 5.2. Naive per-channel int4 is its own cross-family data point: it retains 0% on Qwen2.5-7B (Qwen's outlier channels) but 100% (lossless) on Mistral-7B-v0.3, which has no outlier problem; the failure is model-dependent, settled by bisection, and the upstream first-token fix (huggingface/transformers#35760) does not repair it. KIVI was run via its own Triton kernels (group_size 64, 128-token fp16 residual), on Mistral natively and on Qwen2.5 through my adaptation of its attention class; it is my adaptation, not an official KIVI-Qwen reference. Behavior axis only.

## What I measured wrong and fixed

This repo corrects in public. Listing the mistakes is not an apology; it is why the numbers that remain are trustworthy.

- **Throughput, undercounted ~12x.** An early client-side tokens/wall-clock probe under-counted throughput badly and produced a public claim I retracted ([correction on #16](https://github.com/huawei-csl/KVarN/pull/16)). I now use vLLM's own `generation throughput` logger, steady-state windows only.
- **The 7B throughput order, inverted.** A non-saturating load read KVarN ahead of TurboQuant on T. Under a saturating load the order is TurboQuant ahead (trap 1). The corrected table is at the top.
- **A Mistral serving bug, found and fixed upstream.** On Mistral-7B-v0.3, KVarN capped concurrency to 1 (`capping max_num_seqs 256 -> 1`) and served single-stream only. Root cause: its weight-size estimate globbed and summed every `*.safetensors`, and Mistral ships a `consolidated.safetensors` (13.5 GB) alongside the HF shards (another 13.5 GB), doubling the estimate to 27 GB and flooring the pool budget. vLLM itself loads correctly (it reads the shard index); only the estimate double-counted. Fixed in [huawei-csl/KVarN#20](https://github.com/huawei-csl/KVarN/pull/20) (merged): prefer the `*.safetensors.index.json` manifest. After the fix the cap rises from 1 to 71. Hits Mistral, Mixtral, any repo shipping both formats.
- **A contaminated Mistral behavior table, retracted.** An earlier cross-method Mistral comparison mixed transformers-stack baselines that are version-sensitive (73 vs 83 on the same probe), and an earlier draft claimed a KIVI/TurboQuant weakness there; both were baseline artifacts and are withdrawn pending a single-stack re-run.

## Method notes (scars included)

- **Watch power draw, not util%**: on WSL2, exceeding the GPU residency limit makes Windows silently spill VRAM to system RAM; identical kernels run far slower while `nvidia-smi` shows 100% util at near-idle power ([investigation](https://github.com/huawei-csl/KVarN/issues/15)). A table from a spilled rig is garbage.
- **KVarN decode throughput is periodic** (a tile-flush cycle); a short window can land on a trough and read far below steady state. Use windows that span several flush cycles (120s x 2 reps), report the mean of the steady engine windows.
- **Quality probe**: greedy, temperature 0, errors reported separately (a timeout is not a wrong answer). Cases are model-agnostic (`data/router_cases.jsonl`, the action-trace half of KVFidelity). The probe is mode-sensitive: thinking-mode models need their chat template with thinking disabled or they score near zero on raw completion.

## Roadmap

- **R axis: done** (fidelity / fp16-trajectory LCP, validated n=28). Next: a long-range-dependency variant (needle recall, positional counting) to test whether quant error compounds over the horizon, plus 32B where Q is fully saturated.
- Re-measure the 7B `g64` variants and TurboQuant TriAttention under the saturating load; a clean Mistral throughput row on the [#20](https://github.com/huawei-csl/KVarN/pull/20)-fixed build.
- AdaKV and the flash-attention presses; ReasonAlloc when its code is released.

## Provenance

Built during the KVarN / TurboQuant benchmarking of June 2026. KVarN: [huawei-csl/KVarN](https://github.com/huawei-csl/KVarN) (issues [#12](https://github.com/huawei-csl/KVarN/issues/12), [#15](https://github.com/huawei-csl/KVarN/issues/15); PRs [#16](https://github.com/huawei-csl/KVarN/pull/16), [#19](https://github.com/huawei-csl/KVarN/pull/19), [#20](https://github.com/huawei-csl/KVarN/pull/20)). TurboQuant: [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) (@no_stp_on_snek). The original iso-bits quality comparison (gsm8k / MATH / HumanEval) lives in [turboquant-cuda-bench](https://github.com/sztlink/turboquant-cuda-bench). The Q instrument is the action-trace half of KVFidelity.
