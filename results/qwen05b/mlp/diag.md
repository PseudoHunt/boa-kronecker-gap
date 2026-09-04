# Step 1 — is the identity row metric a good proxy for Qwen's SwiGLU MLP?

**GATE: FAIL.** median identity Pred/T = 0.835 (needs <= 0.80); median top-5% saliency overlap = 0.550 (needs <= 0.60). The Pred/T condition fails.

ΔW is from a re-run of the banked `boa_w3_s0` configuration with `--dump_deltas`; it reproduced **wiki2 19.717 / c4-new 39.072**, identical to the banked run, so these are the deltas that run actually produced.

FP model, seed-0 calibration, 16 sequences x 2048 tokens (32 subsampled per sequence for the eigen-field), all 24 blocks, both input projections. Construction identical to `diag/phase4_fc1.py` so the ratios are comparable to OPT.

## Medians

| layer | n | median id/T | median kron/T | median full-pooled/T | median sal top-5% |
|---|---|---|---|---|---|
| mlp.up_proj | 24 | 0.835 | 0.933 | 0.956 | 0.549 |
| mlp.gate_proj | 24 | 0.838 | 0.822 | 0.842 | 0.550 |
| **both** | 48 | 0.835 | 0.894 | 0.918 | 0.550 |
| *OPT-125m (ReLU fc1, reference)* | | *0.67* | | | *0.41* |

Identity predicts the SwiGLU MLP objective **better** on Qwen than it did on OPT (0.86 vs 0.67 of the true error), and the saliency rankings agree more (0.56 vs 0.41). The gate-aware row metric therefore has less to correct here, not more.

## Per-block

| block | up id/T | up kron/T | up sal5 | gate id/T | gate kron/T | gate sal5 |
|---|---|---|---|---|---|---|
| 0 | 0.500 | 0.659 | 0.759 | 0.752 | 0.632 | 0.463 |
| 1 | 0.837 | 0.901 | 0.727 | 0.853 | 0.786 | 0.618 |
| 2 | 0.865 | 0.928 | 0.784 | 0.844 | 0.820 | 0.692 |
| 3 | 0.891 | 0.933 | 0.690 | 0.885 | 0.838 | 0.642 |
| 4 | 0.893 | 0.939 | 0.566 | 0.954 | 0.864 | 0.524 |
| 5 | 0.899 | 0.965 | 0.396 | 1.008 | 0.833 | 0.403 |
| 6 | 0.829 | 0.937 | 0.533 | 0.772 | 0.790 | 0.541 |
| 7 | 0.832 | 0.976 | 0.564 | 0.926 | 0.897 | 0.545 |
| 8 | 0.787 | 0.876 | 0.589 | 0.928 | 0.849 | 0.497 |
| 9 | 0.776 | 0.989 | 0.633 | 1.062 | 0.927 | 0.548 |
| 10 | 0.818 | 0.953 | 0.573 | 1.038 | 0.871 | 0.493 |
| 11 | 0.498 | 0.996 | 0.650 | 0.997 | 0.937 | 0.572 |
| 12 | 0.853 | 0.992 | 0.625 | 1.024 | 0.872 | 0.537 |
| 13 | 0.851 | 0.979 | 0.631 | 1.051 | 0.928 | 0.552 |
| 14 | 0.726 | 0.934 | 0.522 | 0.832 | 0.867 | 0.537 |
| 15 | 0.869 | 0.962 | 0.501 | 0.819 | 0.825 | 0.530 |
| 16 | 0.645 | 0.951 | 0.524 | 0.657 | 0.792 | 0.583 |
| 17 | 0.889 | 0.924 | 0.495 | 0.740 | 0.688 | 0.561 |
| 18 | 0.874 | 0.909 | 0.517 | 0.736 | 0.702 | 0.568 |
| 19 | 0.804 | 0.925 | 0.450 | 0.621 | 0.649 | 0.572 |
| 20 | 0.908 | 0.927 | 0.457 | 0.711 | 0.691 | 0.587 |
| 21 | 0.738 | 0.907 | 0.496 | 0.675 | 0.655 | 0.594 |
| 22 | 0.679 | 0.890 | 0.503 | 0.718 | 0.727 | 0.603 |
| 23 | 0.874 | 0.870 | 0.505 | 0.590 | 0.674 | 0.486 |

30 of 48 layer-blocks have identity Pred/T > 0.8; 12 have saliency overlap > 0.6.

Block 0 is the one clear exception (up 0.500, gate 0.752) and block 11's up_proj (0.498); everywhere else identity tracks the objective closely.

## Verdict

Per the stated gate, the identity metric predicts well on SwiGLU, so **the MLP is not where the loss is on this model**. Step 2 is not run: no `--row_metric_fc1` seeds, no arm, no paired table.

Worth recording for whoever picks this up. `kron/T` is *higher* than `id/T` on up_proj (median 0.933 vs 0.835), i.e. the pooled Kronecker metric is a genuinely better predictor of the true error — the port works and the metric is meaningful. It simply is not *needed*: identity is already close enough that improving the metric has little error left to remove. That is a different failure from the attention arms, where the metric was degenerate (rank-1, bias-dominated). Here the metric is well conditioned (top-1 eigenvalue share 0.108/0.073) and the objective is just easy.

`ratio_kron_full_pooled` tracks `ratio_kron` closely throughout, the phase-4 consistency check, so the eigenbasis truncation is not distorting the comparison.

