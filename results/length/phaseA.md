# Phase A — length dependence of BoA's q/k row metric

Analytic, no quantisation. `H_out = E_D[R_D C R_D^T]` with `D = u - t` and the triangular pair weight `w_L(D) = (L-|D|)/L^2`. Closed form verified against the direct weighted sum at L=512 to **1.4e-16** (tolerance 1e-8), for both `R_D` and `R_D^T`; the split-half 2x2 construction reproduces `get_rotary_matrix` to 1.2e-07.

## Gate 1 — position independence

`rel_fro(H_out_analytic(w_2048), H_out_BoA)`, normalised by the analytic reference. **Pass: median < 0.05.**

| model | layer | n | median | p90 | max |
|---|---|---|---|---|---|
| qwen2.5-0.5b | q_proj | 336 | 0.0188 | 0.0363 | 0.0450 |
| qwen2.5-0.5b | k_proj | 48 | 0.0058 | 0.0129 | 0.0337 |
| llama-3.2-1b | q_proj | 512 | 0.0122 | 0.0219 | 0.0290 |
| llama-3.2-1b | k_proj | 128 | 0.0075 | 0.0141 | 0.0208 |

**Gate 1: PASS** — every median is below 0.05.

## Gate 2 — length sensitivity

`rel_fro(H_out(w_L), H_out(w_2048))`, normalised by `H_out(w_L)`. **Pass: median at 32k > 0.2, mass in the low-frequency bands.**

| model | layer | median L=8k | median L=32k | median L=128k |
|---|---|---|---|---|
| qwen2.5-0.5b | q_proj | 0.4251 | 0.7616 | — |
| qwen2.5-0.5b | k_proj | 0.2165 | 0.3814 | — |
| llama-3.2-1b | q_proj | 0.2032 | 0.3704 | 0.5707 |
| llama-3.2-1b | k_proj | 0.1580 | 0.3163 | 0.5135 |

### Band decomposition at 32k

Share of `||H(w_32k) - H(w_2048)||_F^2` by wavelength `2*pi/freq`. The (I,J) part of block pair (i,j) moves at `|theta_i - theta_j|` and the (K,L) part at `theta_i + theta_j`; the two are Frobenius-orthogonal so the split is exact.

| model | layer | <512 | 512-2048 | 2048-8192 | 8192-32768 | >32768 |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | q_proj | 0.000 | 0.000 | 0.137 | 0.615 | 0.230 |
| qwen2.5-0.5b | k_proj | 0.000 | 0.000 | 0.105 | 0.598 | 0.273 |
| llama-3.2-1b | q_proj | 0.000 | 0.001 | 0.145 | 0.441 | 0.389 |
| llama-3.2-1b | k_proj | 0.000 | 0.001 | 0.144 | 0.339 | 0.418 |

**Gate 2: PASS** — median rel_fro at 32k exceeds 0.2; the mass sits in the two longest-wavelength bands.

### Static pairs in calibration

| model | pairs rotating < 0.1 rad over 2047 tokens | of |
|---|---|---|
| qwen2.5-0.5b | 9 | 32 |
| llama-3.2-1b | 14 | 32 |

## Qwen key-bias visibility vs length

`visible_frac(L) = 1 - sum_i F_L(theta_i) ||b_i||^2 / ||b||^2` as specified. The column marked F^2 uses `sum_i F_L^2 ||b_i||^2`, which is what `||E_w[R_D] b||^2` literally equals since `E_w[R_D]` is block-diagonal with entries `F_L(theta_i) I_2`. Both are reported because the brief's formula and that identity differ; the qualitative trend is the same.

| L | median visible_frac (F) | median visible_frac (F^2) |
|---|---|---|
| 2048 | 0.2517 | 0.2772 |
| 8192 | 0.4072 | 0.4460 |
| 32768 | 0.5188 | 0.5414 |

At 2k the bias is still largely invisible, consistent with the earlier finding that centring removed ~95% of `H_row`'s trace; the question was how fast it rises.

