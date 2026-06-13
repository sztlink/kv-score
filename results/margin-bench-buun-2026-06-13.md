# Margin bench (buun's harness) on vLLM, KVarN vs TurboQuant, depth 8192

Ran @spiritbuun's `margin_bench` against the vLLM KVarN / TurboQuant configs by pointing its OpenAI-compatible probe at the vLLM endpoint (no llama.cpp needed). The probe records `logprob(chosen) - logprob(runner-up)` per answer token; a case score is the minimum margin (calibrated distance-to-flip). Comparing two configs is a paired per-case difference, reported as a t-statistic. Cases: `rd_8192_c2.jsonl` (action / target / source-rank evidence buried at 8192-token depth behind distractors). Greedy, temperature 0, sequential.

## Exact-match saturates (buun's Trap 3); the margin does not

| metric | fp16 | kvarn-k4v2 | kvarn-k4v4 | tq-k8v4 | tq-k4v2nc |
|---|---|---|---|---|---|
| exact 7B (of 120) | 120 | 118 | 120 | 119 | 112 |
| exact 14B-AWQ (of 120) | 120 | 120 | 120 | 120 | 120 |
| worst min-margin 14B | 2.41 | 1.64 | 2.56 | 2.81 | 0.12 |

At 14B every config scores 120/120: exact-match gives zero discrimination. The min-margin (distance-to-flip) still separates them.

## Paired margin t-stats, 14B-AWQ (mean(A - B); positive = A more robust)

| A vs B | mean dA-B | t |
|---|---|---|
| fp16 vs tq-k4v2nc | +0.54 | 4.06 |
| fp16 vs tq-k8v4 | +0.11 | 3.34 |
| kvarn-k4v4 vs tq-k8v4 | +0.08 | 2.14 |
| kvarn-k4v2 vs tq-k8v4 | -0.00 | -0.03 |
| kvarn-k4v2 vs kvarn-k4v4 | -0.08 | -1.02 |
| tq-k4v2nc vs tq-k8v4 | -0.43 | -3.17 |

## Reading

- The margin discriminates at 14B where exact-match is saturated. buun's Trap 3, reproduced on the vLLM stack.
- The clear degrader is the aggressive 2-bit-V preset (tq-k4v2nc): lower margin than every other config, worst case 0.12 (nearly a flip).
- Among the accurate configs the decision robustness is comparable. KVarN k4v2 vs TurboQuant k8v4 is t = -0.03 (no difference). KVarN k4v4 edges tq-k8v4 (t = 2.14, tiny mean).
- This refines the fidelity (LCP) axis. LCP measures whole-trajectory reproducibility, where KVarN holds the fp16 path much longer. The task-grounded margin shows the routing decision itself is about as robust under KVarN as under TurboQuant. Trajectory fidelity is not decision goodness (buun's Trap 1).

Harness by @spiritbuun (`margin_bench`). Reproduce: serve a config on vLLM, `python3 probe_router.py --base-url http://localhost:PORT/v1 --model NAME --data rd_8192_c2.jsonl --out lp_X.jsonl`, then `paired_margins.py lp_*.jsonl`.
