# F0 (forma vs regua) + controles baratos — resultados 2026-06-20

Rig: RTX 4090, WSL2, fork llama-cpp-turboquant (TheTom) @ ed81ed0. Modelo: Meta-Llama-3.1-8B-Instruct-Q4_K_M. Harness: `runners/decoy-bench-f0.py` (decoy-at-depth + metricas continuas digit_err/lev + dump per-case + prompt-seed jitter + opt-out de Boundary V). Greedy temp0, -fa on. Prod pausada/restaurada (qwen2.5-7b-tq na 11435).

Motivacao: a revisao adversarial de 2026-06-20 (`research/2026-06-20-kv-github-deepdive-antagonize.md`) apontou que (a) o Boundary V (mode 7: q8_0 nas 2 primeiras+2 ultimas camadas, AUTO-ativado quando V=turbo2) estava ligado em TODOS os runs anteriores -> nunca mediamos 2-bit V puro; (b) o "penhasco a 32k" podia ser artefato do exact-match ou do Boundary V; (c) N=16 eram posicoes, nao seeds; (d) o modo de falha nunca foi dumpado.

## Controle do Boundary V (32k, N=16, mesmas prompts)
| celula | decisao | mean_lev | malformed |
|---|---|---|---|
| f16 | 1.0 (16/16) | 0.0 | 0.0 |
| turbo3 | 1.0 (16/16) | 0.0 | 0.0 |
| turbo2 Boundary-V AUTO (ligado) | 0.438 (9/16) | 4.56 | 0.375 |
| turbo2 Boundary-V OFF (2-bit puro) | 0.25 (12/16) | 5.56 | 0.50 |

**Boundary V estava mascarando, nao criando o problema.** 2-bit V puro (OFF) e PIOR (0.25) que com Boundary V (0.438). Nossos numeros anteriores SUBESTIMAVAM a perda. Confirmado via captura: o build auto-ativa mode 7 quando V=turbo2 (opt-out TURBO_LAYER_ADAPTIVE=0).

## A superficie completa (decisao = recuperacao exata; turbo2 = 2-bit V puro, Boundary OFF)
| profundidade | f16 | turbo3 (3-bit V) | turbo2 (2-bit V puro) |
|---|---|---|---|
| 8k | 1.0 | - | 0.625 |
| 16k | 1.0 | - | 0.375 |
| 24k | 1.0 | - | 0.375 |
| 32k | 1.0 | 1.0 | 0.375-0.625 (4 seeds, media ~0.47) |
| 49k | 1.0 | 0.875* | 0.25 |
| 65k | 1.0 | 1.0 | 0.25 |

\* turbo3@49k = 1 falha de 8; turbo3@65k volta a 1.0 -> a queda de 49k e ruido de instancia.
(65k so rodou apos corrigir o harness: prompt via -f arquivo; -p estoura ARG_MAX a ~157KB. -f validado identico ao -p no 49k turbo3=0.875.)

## Conclusoes F0 (forma vs regua: RESOLVIDO)
1. **Nao ha penhasco a 32k.** Era artefato do Boundary V (as camadas q8_0 seguravam turbo2 em 1.0 ate 16k e so estouravam a 32k, criando a *aparencia* de penhasco). O 2-bit V puro degrada CEDO (~8k) e faz PLATO (~0.25-0.4). Forma = "onset precoce + plato", nao cliff.
2. **f16 = exato (1.0) em TODAS as profundidades 8k->65k**, todos os seeds. Mata "rig quebrado", "f16 tambem cai", e "artefato de exact-match" (a regua nao quebra f16).
3. **turbo3 (3-bit V) segura ate 65k** (uma queda ruidosa a 49k). 3-bit V preserva recuperacao exata a 65k+ com ~37% menos KV que f16. **O sweet spot do turbo3 e MAIS forte do que a alegacao original "a 32k": ele segura mais fundo.**
4. **A acao da recuperacao exata vive no degrau 2-bit -> 3-bit.** PPL fica lisa pros dois; a dissociacao regua-de-cauda (exact-match) vs regua-de-corpo (PPL) e precisamente o regime de 2-bit V.
5. **Modo de falha (dump):** dominante = corrupcao de valor / confabulacao (numeros soltos "4582"/"1318", palavra ALUCINADA "GIANT" fora da lista, frases confabuladas "The final answer is: 335", parciais "7351-DELTA" digitos certos palavra errada). Decoy-pickup e RARO a 32k (~1/16) mas SOBE a 65k (decoy_rate 0.25): em profundidade extrema, sem conseguir recuperar o valor exato, o modelo mais frequentemente pega o codigo de outra unidade.
6. **Taxa e ruidosa por instancia** (turbo2 32k: 0.375-0.625 em 4 seeds). Qualquer numero unico ("9/16", "0.438") e uma instancia; o fenomeno e "degradacao substancial, taxa varia". f16 = 0 falhas em todos os seeds.

## O que isso faz com a tese de invencao
A pergunta deixa de ser "consertar um penhasco a 32k" e vira nitida:
> **O codebook Gaussiano fixo de 4 niveis (Lloyd-Max) do 2-bit V perde recuperacao exata desde ~8k, enquanto 3-bit segura a 65k. Um codec de 2-bit V melhor desenhado (objetivo = recuperacao exata / atencao, nao MSE medio) pode fechar o degrau 2-bit->3-bit, recuperando como turbo3 a custo de turbo2? Ou 3-bit e o piso pra recuperacao exata em profundidade?**

Essa pergunta unica decide se existe invencao (codec de 2-bit V melhor) ou se a contribuicao vira "use 3-bit V + a superficie + a metrica de dissociacao" (contribuicao de medicao, ainda valiosa e alinhada ao campo).

## Proximos passos
1. **F0 FECHADO.** Corrigir o registro publico (decoy-at-depth-2026-06-17.md): cliff -> superficie; turbo3 segura a 65k. Drafts X antigos ("cliffs at 32k") estao com forma errada, NAO postar; reformular.
2. **Probe de codebook (BARATO, mesmo engine, sem stack KVarN) = teste pivo:** o build expoe envs TURBO_INNERQ / TURBO_INNERQ_STRENGTH / TURBO_AUTO_ASYMMETRIC. Testar se uma variante de 2-bit V (asimetrica/inner-quant) move a curva no mesmo bit-budget. Se mover -> codec design e a alavanca (invencao real). Se nao -> 3-bit e o piso.
3. **F1 (KVarN k4v2 cross-stack):** padrao-ouro de comparacao de codec (fp8 per-token + Sinkhorn vs Lloyd-fixo), mesmo 2-bit V. Bloqueador: pool fp16 do KVarN nao cabe a 32k em 24GB -> Qwen3-4B, card maior, ou port beellama.cpp (mesmo engine, mata confound de stack).
4. **F2 (oracle-protect titulado):** so se 2/3 mostrarem que codec design recupera. Converte "flanco aberto" em "flanco com metodo".

Dados brutos nesta pasta: f0-stepB.{log,dump.jsonl} (controle Boundary V + 32k N=16), f0-stepC.{log,dump.jsonl} (sweep + multi-seed + 49k), f0-deep.{log,dump.jsonl} (49k validacao -f + 65k).
