# Step 1 — does the tr(D) ordering score rank the act-orders like PPL?

**GATE: FAIL.** The score inverts `ao_row` and `ao_col`.

Toy Kronecker pivot check PASSED: max abs pivot diff 1.99e-13, relative sum diff 2.29e-16 (tol 1e-10). So `d_ij = d_row[i]*d_col[j]` is verified; the failure is not in the factorisation.

Qwen2.5-0.5B, FP model, seed-0 calibration (128x2048), all 24 blocks, every layer.

## Summed scores vs banked seed-0 perplexity

| ordering | score (weighted) | score (unweighted) | wiki2 | score rank | ppl rank |
|---|---|---|---|---|---|
| ao_none | 569,132 | 723,933,822 | 22.911 | 1 | 1 |
| ao_row | 120,882 | 465,616,187 | 21.808 | 3 | 2 |
| ao_col | 452,785 | 599,343,610 | 20.293 | 2 | 3 |
| ao_both | 100,482 | 395,422,319 | 19.717 | 4 | 4 |

score order (worst→best): `['ao_none', 'ao_col', 'ao_row', 'ao_both']`
ppl   order (worst→best): `['ao_none', 'ao_row', 'ao_col', 'ao_both']`
**match: False** — 2 of 4 positions correct (the endpoints only).

The score says row-ordering is worth far more than column-ordering (120,882 vs 452,785, a 3.75x gap). Perplexity says the opposite: `ao_col` 20.293 beats `ao_row` 21.808. The unweighted variant inverts the same way, so this is not an artefact of the s_i^2 weighting.

## Where the score mass sits

| ordering | q/k layers | all other layers | q/k share |
|---|---|---|---|
| ao_none | 522,731 | 46,401 | 0.918 |
| ao_row | 74,480 | 46,401 | 0.616 |
| ao_col | 411,407 | 41,378 | 0.909 |
| ao_both | 59,104 | 41,378 | 0.588 |

`ao_row` only touches the two-sided q/k layers, yet it moves the total from 569,132 to 120,882. The score is dominated by the row factor of two layers out of seven, which is exactly the sensitivity perplexity does not share.

## Per-block score(ao_both)/score(ao_none)

| block | q/k | other |
|---|---|---|
| 0 | 0.0956 | 0.8673 |
| 1 | 0.0191 | 0.9472 |
| 2 | 0.2866 | 0.9401 |
| 3 | 0.6649 | 0.9283 |
| 4 | 0.4403 | 0.9348 |
| 5 | 0.6608 | 0.9323 |
| 6 | 0.7602 | 0.9291 |
| 7 | 0.6566 | 0.9083 |
| 8 | 0.0273 | 0.8894 |
| 9 | 0.7029 | 0.9092 |
| 10 | 0.5832 | 0.8987 |
| 11 | 0.6234 | 0.9022 |
| 12 | 0.6095 | 0.8956 |
| 13 | 0.5888 | 0.8853 |
| 14 | 0.6241 | 0.8464 |
| 15 | 0.6147 | 0.8515 |
| 16 | 0.5901 | 0.8352 |
| 17 | 0.5801 | 0.8628 |
| 18 | 0.6077 | 0.8854 |
| 19 | 0.6346 | 0.8750 |
| 20 | 0.5990 | 0.8767 |
| 21 | 0.6296 | 0.8983 |
| 22 | 0.6628 | 0.9026 |
| 23 | 0.7124 | 0.8731 |

q/k mean ratio 0.5406, other mean ratio 0.8948.

## Verdict

The criterion does not order the four act-orders the way measured perplexity does, so it cannot be used to select an ordering. Step 2 is not run.

What the failure is NOT: the Kronecker pivot identity is verified to 2e-13, the orderings are BoA's own `reorder_col`/`reorder_row` on BoA's own damped Hessians, and the scales come from the same grid search BoA runs before reordering. The weighted and unweighted forms agree with each other and disagree with perplexity in the same way.

What it plausibly IS: `tr(D)` is derived under an equal-scale approximation, and the two Kronecker factors here are wildly unequal in scale — the q/k row metric on this model is ~96% projection bias and near rank-1 (top-1 eigenvalue share up to 0.976, see `diag/bias_share.json` and `softmax_gap_centred.json`). Sorting a near-rank-1 factor by its diagonal produces a large drop in `sum_i d_row[i]` that buys little real error reduction, so the row side dominates the score while contributing less to perplexity.

