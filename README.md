# kv-score

Score the KV-cache compression options available on **your** rig: one number you can compare, three axes you can argue with.

```
KV-Score = (Q × T × C)^(1/3)        each axis: 100 = fp16 parity, on YOUR rig
```

- **Q** (quality retention): does the model still *behave* the same? Action-trace exact-match (action / target / source-rank) on 120 routing cases vs fp16. Behavior breaks before benchmark scores do.
- **T** (throughput retention): engine-reported tokens/s under load (steady state, 128 concurrent decodes) vs fp16.
- **C** (capacity gain): KV tokens that actually fit (measured from the engine, not nominal bits) vs fp16.

Geometric mean on purpose: a method that wins capacity but collapses quality or throughput goes toward zero. You cannot buy score on one axis.

## Run it

```
# 1. T + C: serve each config, fire load, read the engine's own counters
runners/measure-tput.sh fp16      '<your vllm serve command, fp16 KV>'
runners/measure-tput.sh mymethod  '<same command with --kv-cache-dtype ...>'

# 2. Q: action-trace probe against each running config
python3 runners/probe-router.py http://localhost:8001/v1/completions 120

# 3. fill one CSV row per config, compute scores
python3 score.py results/myrig-mymodel-2026-06-10.csv
```

One rule: same model, same flags, same box for every row. fp16 on the same stack is the reference.

## PR your rig's row

`results/` is the point of this repo. Each file is one rig's table: `results/<rig>-<model>-<date>.csv`. Send a PR with yours; the catalog of rigs is the product, not any single table. Methods score differently on native Linux vs WSL2, on a 4090 vs an H100; the honest question is "what should I run on MY hardware", and that table only you can produce.

## Example: Qwen2.5-7B-Instruct, RTX 4090, WSL2, gmu 0.80

| config | Q | T | C | KV-Score |
|---|---|---|---|---|
| fp16 (reference) | 100.0 | 100.0 | 100.0 | **100.0** |
| KVarN k4v2_g128 | 96.0 | 91.0 | 257.2 | **131.0** |
| TurboQuant k8v4 | 102.0 | 83.7 | 214.4 | **122.3** |
| TurboQuant k4v2_nc | 94.0 | 80.6 | 322.0 | **134.6** |

Reading: all three compressed configs land in a 122-135 band, by different routes. TQ k8v4 holds behavior at full parity (102 is within ±1-case noise of 100) with 2.1x capacity. TQ k4v2_nc leads the composite via 3.2x capacity but pays 6 behavior points vs k8v4. KVarN k4v2 splits the difference: 91% throughput, 2.6x capacity, 4 behavior points. Which one wins depends on the axis your workload cannot give up; the composite only tells you none of them is dominated.

## Example 2: eviction family (kvpress @ compression_ratio 0.5), same model, same rig, transformers stack

| config | Q | T | C | KV-Score |
|---|---|---|---|---|
| fp16 transformers (reference) | 100.0 | 100.0 | 100.0 | **100.0** |
| SnapKV | 75.5 | 90.6 | 200.0 | **111.0** |
| ExpectedAttention | 56.1 | 95.7 | 200.0 | **102.4** |
| Knorm | 42.9 | 94.4 | 200.0 | **93.2** |

The behavior axis separates the families: at 2x effective compression the quant methods above lose 4-6 behavior points; the eviction presses at the same 2x lose 24 to 57 points on the identical cases, and Knorm scores below doing nothing at all. These same presses report near-lossless results on answer-level long-context suites; behavior collapses first. Regime caveat: these cases are 1-2k token agent-routing prompts, the short end of what eviction methods are designed for. AdaKV (SnapKV) is not scored: it requires flash-attention and degenerates under SDPA (harness limitation, not a method verdict).

Throughput note: this stack's T is single-stream decode tok/s (transformers has no continuous batching); it is only comparable within this table, normalized to its own fp16 row.

## Method notes (scars included)

- **Throughput**: vLLM's own `generation throughput` logger, steady-state windows only. Never client-side tokens/wall-clock: that probe under-counted 12x in our hands and produced a public claim we had to retract ([correction](https://github.com/huawei-csl/KVarN/pull/16)).
- **Windows must span flush cycles**: KVarN's decode throughput is periodic (tile-flush cycle); 70-second windows landed on troughs and read T=44.5 instead of 91.0. We use 120s windows × 2 reps and report the mean of all steady engine windows.
- **Watch power draw, not util%**: on WSL2, exceeding the GPU residency limit makes Windows silently spill VRAM to system RAM; identical cuBLAS kernels run 50-130x slower while `nvidia-smi` shows 100% util at 104 W ([investigation](https://github.com/huawei-csl/KVarN/issues/15)). A table produced on a spilled rig is garbage. Keep weights + KV + any method-specific pools under ~90% of physical VRAM.
- **Quality**: greedy, temperature 0, 8-way concurrency, generous timeouts; errors reported separately (a timeout is not a wrong answer). Cases are model-agnostic (`data/router_cases.jsonl`, from the KVFidelity instrument).

## Roadmap

- AdaKV and the flash-attention presses (needs flash-attn on this rig).
- KIVI (2-bit quant classic).
- ReasonAlloc when code is released.
- An R axis (robustness: degeneration rate on long generation).

## Provenance

Built during the KVarN/TurboQuant benchmarking of June 2026 ([#12](https://github.com/huawei-csl/KVarN/issues/12), [#15](https://github.com/huawei-csl/KVarN/issues/15), [#16](https://github.com/huawei-csl/KVarN/pull/16)). The Q instrument is the action-trace half of KVFidelity.
