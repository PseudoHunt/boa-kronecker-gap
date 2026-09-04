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
| ao_both | 19.717 | 39.072 | 5064 |

## Stage C — all runs

| arm | seed | wiki2 | c4-new | wall (s) |
|---|---|---|---|---|
| boa | 0 | 19.717 | 39.072 | 5064 |

## Paired deltas vs same-seed `boa` (negative = better)

| arm | metric | n | per-seed deltas | mean | std | all improve? | mean > 2*std? |
|---|---|---|---|---|---|---|---|

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

**Incomplete** — `combined` has 0/3 paired seeds. No verdict; the decision table needs all three.

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

## Stage D (softmax-gap diagnostic): NOT RUN

`diag/phase1_runner.py` is OPT-shaped and does not support this model: it
indexes both the q and k row metrics by `n_heads` (under GQA the k metric has
`n_kv_heads` entries, so it raises), and it reconstructs Q/K as `W x + b` with
no RoPE, so the attention probabilities it derives would be wrong for Qwen.
A correct port also has to keep `R` in BoA's pre-RoPE (back-rotated) basis
while computing `A` from post-RoPE Q/K -- a convention split that is easy to
get plausibly but subtly wrong, which would produce a credible-looking gap
number driving a paper/no-paper call. It was left undone rather than guessed.

### The number Stage D has to produce, and what to compare it to

From this repo's existing OPT-125m phase 1 run (`results/phase1/summary.json`,
24 block x layer entries), the relevant reference values are:

| field | what it measures | OPT-125m mean | range |
|---|---|---|---|
| `G1_rel_fro` | pure separability (Kronecker) gap | 0.0005 | 0.0002 - 0.0020 |
| `G12_rel_fro` | + causal mask | 0.2041 | 0.0585 - 0.3092 |
| `G123p_rel_fro` | + attention-probability weighting | 0.2944 | 0.0810 - 0.4485 |
| `G123j_rel_fro` | + softmax-Jacobian weighting (**the softmax gap**) | **0.2306** | 0.0953 - 0.3402 |

So 'small (<= OPT's)' in section 7 means a Qwen `G123j_rel_fro` at or below
~0.23; materially above that is the 'large gap' branch. Note the contrast that
makes this the interesting quantity: the pure Kronecker gap is ~0.0005 (nil),
so essentially all of the discrepancy comes from the softmax weighting, not
from separability.

This matters more than its 'optional' billing suggests: if the arms come back
null, the bias result above says the null is uninformative about the
hypothesis, and section 7's branch turns entirely on whether the softmax gap
is small (stop) or large (the paper hinges on the softmax solver). **Build this
first next session, ahead of any Llama work.**

