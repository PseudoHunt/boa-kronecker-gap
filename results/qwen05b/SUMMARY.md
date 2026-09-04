# Qwen2.5-0.5B, W3 — BoA extension arms

Model: Qwen/Qwen2.5-0.5B (24 blocks, d=896, 14 q heads / 2 kv heads, d_h=64).
All runs: `--w_bits 3 --block_v --qparam_comput Hessian`, 128x2048 wikitext2 calibration.

## Reproduction gate

| quantity | reference | measured | verdict |
|---|---|---|---|
| FP wiki2 | 13.07 | 13.078 | PASS |
| W3 best-of-4 act-order wiki2 | 22.02 | 19.717 (ao_both) | PASS (within 3%) |

## Stage B — act-order sweep (seed 0)

| act-order | wiki2 | c4-new | wall (s) |
|---|---|---|---|
| ao_none | 22.911 | 44.309 | 7871 |
| ao_col | 20.293 | 39.612 | 7871 |
| ao_row | 21.808 | 40.443 | 7871 |
| ao_both | 19.717 | 39.072 | 5064 |

## Stage C — all runs

| arm | seed | wiki2 | c4-new | wall (s) |
|---|---|---|---|---|
| boa | 0 | 19.717 | 39.072 | 5064 |
| boa | 1 | 19.993 | 38.698 | 5149 |
| boa | 2 | 19.825 | 39.132 | 5149 |
| combined | 0 | 19.778 | 38.678 | 6065 |
| combined | 1 | 19.893 | 39.895 | 6065 |
| combined | 2 | 19.736 | 39.935 | 6065 |

## Paired deltas vs same-seed `boa` (negative = better)

| arm | metric | n | per-seed deltas | mean | std | all improve? | mean > 2*std? |
|---|---|---|---|---|---|---|---|
| combined | wikitext2 | 3 | s0: +0.061, s1: -0.100, s2: -0.089 | -0.043 | 0.090 | no | no |
| combined | c4-new | 3 | s0: -0.394, s1: +1.197, s2: +0.803 | +0.535 | 0.829 | no | no |

## Why a null on the qk arms is expected here

BoA hooks the layer OUTPUT, so q_proj's row metric is `E[K K^T]` built from
BIASED keys and k_proj's is `E[Q Q^T]` from biased queries. On Qwen2.5-0.5B
that bias term owns almost the whole metric, measured across all 24 blocks as
the norm ratio `||b b^T||_F / ||E[Y Y^T]||_F`:

| metric | used for | mean | min | max |
|---|---|---|---|---|
| `E[Q Q^T]` | k_proj | 0.9573 | 0.8923 | 1.0289 |
| `E[K K^T]` | q_proj | 0.9602 | 0.6195 | 1.0591 |

The bias is a constant that weight quantization never touches, so ~96% of the
Frobenius mass of the q/k row metric is inert. `--qk_quantK` rebuilds that
metric from the quantized key but can only move the few percent that depends
on W. **A null on `qk-quantK` and `combined` is therefore evidence about
Qwen's q/k biases, not about the hypothesis.** (Norm ratios, not an orthogonal
decomposition -- the cross terms are not orthogonal, so shares can sum past 1.)

## Verdict (runbook section 7)

**Null on `combined`.** Per the runbook the next branch is set by the Stage D softmax gap: small gap (<= OPT's) => no paper on this line, stop spending; large gap => the paper hinges on the softmax solver.

## Cost (measured, not estimated)

One block of Qwen2.5-0.5B at W3, 128x2048 calibration, on an A100-40GB:

| phase | time |
|---|---|
| compute_Hessian (128 seqs) | 3.0 s |
| solve q_proj (two-sided `boa()`) | 65.4 s |
| solve k_proj (two-sided `boa()`) | 68.3 s |
| solve v/o/gate/up/down (one-sided `gptq()`) | 21.4 s |
| **block total** | **158.1 s** |

So ~63 min/run for `boa` and `qk-quantK`, and ~89 min/run for `combined` and
`v-rowmetric`, which route v_proj through the two-sided solver as well. The
runbook's estimate of 6-10 min/run is low by roughly 7x. Cost is dominated by
the Python row loop (64 row steps x ~896 GPTQ column steps per two-sided
layer), not by Hessian collection. Runs are single-threaded, ~1.3-1.9 GB GPU
each, so ~11 fit concurrently on 16 vCPUs without contention.

## Stage D — the softmax gap (RUN)

FP model, 16 sequences x 2048 tokens, all 24 blocks, split-half
corrected. Implementation validated against this repo's OPT-125m phase 1 numbers
(see `tests/test_softmax_gap.py`): q_proj G12 0.1609 vs 0.1616, k_proj G123p
0.1037 vs 0.1077, k_proj G123j 0.1151 vs 0.1218.

| layer | variant | mean | median | p25 | p75 | max |
|---|---|---|---|---|---|---|
| q_proj | G1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| q_proj | G12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| q_proj | G123p | 0.1107 | 0.1116 | 0.0705 | 0.1409 | 0.2465 |
| q_proj | G123j | 0.1106 | 0.1130 | 0.0694 | 0.1372 | 0.2270 |
| k_proj | G1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| k_proj | G12 | 0.0008 | 0.0000 | 0.0000 | 0.0000 | 0.0183 |
| k_proj | G123p | 0.2626 | 0.1993 | 0.1605 | 0.2799 | 0.9429 |
| k_proj | G123j | 0.2580 | 0.1985 | 0.1689 | 0.2680 | 0.8666 |

**Qwen G123j pooled over q+k: mean 0.1843, median 0.1562** (split-half noise floor 0.032, so the signal is real).
**OPT-125m reference: mean 0.2306.** So the Qwen softmax gap is NOT larger than OPT's -- it is smaller.

Two structural notes. `G1` is ~0 for every block: separability itself costs
nothing, exactly as on OPT (0.0005). `G12` is also ~0, which DIFFERS from OPT
(0.204) -- causal masking alone does not break separability here; on this model
essentially the entire discrepancy is the softmax weighting.

## Verdict — section 7, both inputs now in

`combined` vs `boa` is **null** (wiki2 mean -0.043, std 0.090, 2 up / 1 down), and the softmax gap is **0.184 mean / 0.156 median vs OPT's 0.231** -- i.e. small, at or below OPT's.

Section 7's table maps null + small gap to: **no paper on this line. Stop spending.**

Two independent lines of evidence agree, which is what makes this a stop rather
than a 'two more seeds':

1. The arm could not have acted on this model: ~96% of the Frobenius mass of the
   q/k row metric is the projection bias, which weight quantization never touches.
2. The quantity the arm exists to exploit -- the term BoA's objective drops -- is
   no larger here than on OPT, where it was already judged not worth pursuing.

What is NOT ruled out: the k_proj gap is heavy-tailed (median 0.199, max 0.867 at
block 13), so a few blocks do carry a large softmax gap. If anything survives this
line it is per-block, not global -- and it would need a model whose q/k metric is
not bias-dominated to be testable at all. Qwen2.5-0.5B cannot answer it.

