# Qwen2.5-0.5B, W3 — BoA attention-metric study

Qwen/Qwen2.5-0.5B: 24 blocks, d=896, 14 query heads / 2 kv heads, d_h=64, q/k/v biases, RoPE.
All runs `--w_bits 3 --block_v --qparam_comput Hessian --act_order_col --act_order_row`, 128x2048 wikitext2 calibration.

**Bottom line: no paper on this line.** The hypothesis is confirmed and does not pay: BoA's q_proj row metric is ~96% softmax-invisible direction, centring removes it, and neither centring it nor deleting it changes perplexity measurably.

## Reproduction gates

| quantity | reference | measured | verdict |
|---|---|---|---|
| FP wiki2 | 13.07 | 13.078 | PASS |
| W3 best act-order wiki2 | 22.02 | 19.717 (ao_both) | PASS (10% better) |
| byte-identical default path | bit-exact | PASS | all 3 configs, after every patch |

## Stage B — act-order sweep (seed 0), complete

| act-order | wiki2 | c4-new |
|---|---|---|
| ao_both | 19.717 | 39.072 |
| ao_col | 20.293 | 39.612 |
| ao_row | 21.808 | 40.443 |
| ao_none | 22.911 | 44.309 |

`ao_both` fixed for everything below.

## All runs

| arm | seed | wiki2 | c4-new | wall (s) |
|---|---|---|---|---|
| boa | 0 | 19.717 | 39.072 | 5064 |
| boa | 1 | 19.993 | 38.698 | 5149 |
| boa | 2 | 19.825 | 39.132 | 5149 |
| combined | 0 | 19.778 | 38.678 | 6065 |
| combined | 1 | 19.893 | 39.895 | 6065 |
| combined | 2 | 19.736 | 39.935 | 6065 |
| q-centered | 0 | 20.144 | 40.052 | 4364 |
| q-centered | 1 | 19.989 | 38.685 | 4367 |
| q-centered | 2 | 20.217 | 40.259 | 4368 |
| q-identity | 0 | 19.808 | 39.2 | 4368 |
| q-identity | 1 | 19.878 | 38.865 | 4199 |
| q-identity | 2 | 20.263 | 40.27 | 4367 |

## Paired deltas vs same-seed `boa` (negative = better)

| arm | metric | n | per-seed | mean | std | all improve? | mean > 2*std? |
|---|---|---|---|---|---|---|---|
| q-centered | wikitext2 | 3 | s0: +0.427, s1: -0.004, s2: +0.392 | +0.272 | 0.239 | no | no |
| q-centered | c4-new | 3 | s0: +0.980, s1: -0.013, s2: +1.127 | +0.698 | 0.620 | no | no |
| q-identity | wikitext2 | 3 | s0: +0.091, s1: -0.115, s2: +0.438 | +0.138 | 0.279 | no | no |
| q-identity | c4-new | 3 | s0: +0.128, s1: +0.167, s2: +1.138 | +0.478 | 0.572 | no | no |
| combined | wikitext2 | 3 | s0: +0.061, s1: -0.100, s2: -0.089 | -0.043 | 0.090 | no | no |
| combined | c4-new | 3 | s0: -0.394, s1: +1.197, s2: +0.803 | +0.535 | 0.829 | no | no |

`boa` baseline seed spread: 19.717 / 19.993 / 19.825 — range 0.276, std 0.139. Every effect below is inside that.

## Finding 1 — BoA's q/k row metric is ~96% projection bias

BoA hooks the layer OUTPUT, so q_proj's row metric is `E[K K^T]` from BIASED keys and k_proj's is `E[Q Q^T]` from biased queries. Norm ratio `||b b^T||_F / ||E[Y Y^T]||_F`, all 24 blocks:

| metric | used for | mean | min | max |
|---|---|---|---|---|
| `E[Q Q^T]` | k_proj | 0.9573 | 0.8923 | 1.0289 |
| `E[K K^T]` | q_proj | 0.9602 | 0.6195 | 1.0591 |

Norm ratios, not an orthogonal decomposition — the cross terms are not orthogonal, so shares can sum past 1.

## Finding 2 — that mass is softmax-INVISIBLE

The exact softmax Jacobian is `J = diag(p) - p p^T`, whose quadratic form is the p-weighted COVARIANCE:

```
  sum_u p_tu (g.k_u)^2 - (sum_u p_tu g.k_u)^2  =  g^T Cov_{p_t}(k) g
```

Since `sum_u p_tu = 1`, writing `k_u = b + k'_u` cancels `b` EXACTLY. So a constant offset in the keys — overwhelmingly the k_proj bias here — is invisible to attention, and BoA spends its metric budget on it.

**This is also a correction to the existing OPT phase 1 numbers.** `G123j` weights by `p(1-p)`, only the DIAGONAL of `J`, which RETAINS `b b^T * sum_u p(1-p)`. On a bias-dominated metric it compares two matrices sharing an invisible dominant direction, finds them close, and reports a small gap that is an artefact. My first pass made exactly that error and concluded 'gap is small, stop'; that conclusion was wrong.

Centred (exact) vs BoA's `H_row` for q_proj, 24 blocks:

| measure | mean | median | min | max |
|---|---|---|---|---|
| **rel_fro centred (exact J)** | 2.1352 | 1.7228 | 0.927 | 5.5563 |
| rel_fro diagonal (G123j, superseded) | 0.4096 | 0.4286 | 0.0775 | 0.6184 |
| Frobenius cosine | 0.4046 | 0.4444 | 0.0173 | 0.6847 |
| BoA H_row top-1 share | 0.4251 | 0.3107 | 0.1832 | 0.9761 |
| centred metric top-1 share | 0.1234 | 0.0998 | 0.0695 | 0.4282 |
| BoA mass in centred-weak dirs | 0.1821 | 0.1303 | 0.047 | 0.9663 |

Block 0: BoA's metric is 95% one direction, 96.6% of its mass sits where the exact metric barely looks, cosine 0.017 — effectively orthogonal. Centring removes 95.5% of `H_row`'s trace and drops top-1 share 0.953 → 0.268.

`rel_fro` here is `||a-b||_F/||a||_F` and is unbounded; values > 1 arise because BoA's metric is near rank-1 while the exact one is spread over 64 directions. That concentration mismatch is the finding.

Implementation validated against OPT phase 1 (`tests/test_softmax_gap.py`): 7 of 8 reference values reproduce. The 8th, k_proj G12, diverges by design — phase 1 applies the same FORWARD cumsum to both layers, but for k_proj a key at `u` is seen by queries `t >= u`, a REVERSE cumsum.

## Verdict — stop

Pre-registered rule: (1) large gap + (2) `q-centered` beats `boa` on all three seeds → cheap paper; (1) large + (2) null → the invisible mass is cheap to protect and does not matter, stop; (1) small → stop.

**(1) is large** — rel_fro 2.135, cosine 0.405. **(2) is null** — `q-centered` is +0.272 wiki2 vs `boa` and beats it on no seed. → **stop.**

## Finding 3 — the metric does no measurable work anyway

`q-identity` discards BoA's q_proj row metric entirely (H_row = I, i.e. plain per-row GPTQ) and costs **+0.138 wiki2**, against a `boa` seed spread of 0.276. It is free.

Both halves of the hypothesis are confirmed and they cancel: the metric IS ~96% softmax-invisible, centring DOES remove it, and the metric buys nothing measurable at W3 either way — so fixing it gains nothing.

At n=3 with std ~0.24–0.28 none of these separations are significant. The defensible claim is that no arm is distinguishable from `boa`, and the effect of the whole q_proj row metric is bounded below ~0.3 ppl at n=3. `q-centered` coming out slightly WORSE is not a finding.

### Caveats and what is left open

- One model, one bit-width. Finding 3 especially may be a W3-on-0.5B statement.
- The k_proj softmax gap is heavy-tailed (median 0.199, max 0.867 at block 13). A few blocks do carry a large gap; that is a separate, per-block question.
- The earlier `combined` / `qk-quantK` / `v-rowmetric` arms are superseded: they were built on the uncentred metric and, per Finding 1, could only ever act on a few percent of it.

## Cost (measured)

One block, 128x2048 calibration, A100-40GB: compute_Hessian 3.0 s; q_proj solve 65.4 s; k_proj 68.3 s; v/o/mlp 21.4 s; **block total 158.1 s** → ~63 min/run (~89 min when v_proj also goes two-sided). The runbook's 6–10 min/run estimate is low by ~7x. Cost is the Python row loop, not Hessian collection. Runs are single-threaded and ~1.3–1.9 GB GPU; the shared A100 is the throughput bottleneck, not the 16 vCPUs.

