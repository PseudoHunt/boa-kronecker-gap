# Kronecker gap in BoA — running summary

Model: OPT-125m, wikitext2 (128 x 2048) calibration, H100 80GB.
Repo: SamsungLabs/BOA @ 1d80d2c + diagnostics in `diag/` behind new CLI flags.

---

## Phase 0 — reproduce BoA. **PASS (after fixing a blocking bug).**

| bits | best act-order | ours | paper (Table 10) | delta |
|---|---|---|---|---|
| FP16 | — | 27.654 | 27.65 | +0.01% |
| W3 | col+row | **31.940** | 31.95 | **−0.03%** |
| W2 | row | **85.884** | 85.63 | **+0.30%** |
| W4 | row | 28.600 | (no OPT-125m INT4 row in the paper) | — |

All inside the ±3% gate. Full 24-run table (as-released vs corrected):
`results/phase0.json`.

**This required fixing a real bug first.** As released, BoA's Hessian pass runs
*bidirectional* attention on OPT under transformers >= 4.53, because `get_opt` is
the only model loader that does not pass `attn_implementation="eager"`. SDPA gets
causality from `is_causal`, not from a mask, so `block_kwargs['attention_mask']` is
`None`; `--block_v` then forces `output_attentions=True`, transformers falls back to
eager, and eager has no mask to apply. Measured: attention mass above the diagonal
= 1926.0, and the corrupted pass's hidden states differ from the correct ones by up
to 14.25. It corrupts `H_col` for `v_proj`, `out_proj`, `fc1` and `fc2`.

Effect: W3 went 37.597 -> 32.105 (paper 31.95) on the single config first tested.
Full details, blast radius and fix: `results/BUG_causal_attention.md`.
Flag: `--attn_impl {eager,sdpa,auto}`, default `eager`.

---

## Phase 1 — how big is the Kronecker gap? **NEGLIGIBLE for OPT-125m attention.**

Measured on all 12 blocks x {q_proj, k_proj} x 12 heads, 96 calibration / 32
held-out sequences, W3 col+row (the Phase 0 baseline config).

| criterion | threshold | measured | |
|---|---|---|---|
| median \|log2 R\| | < 0.3 | **0.00006** | PASS by ~4 orders of magnitude |
| worst \|log2 R\| | < 0.3 | **0.00032** | PASS |
| min top-5% saliency overlap | > 0.9 | **0.9996** | PASS |
| `mass_off` (curvature mass wrong by >2x) | — | **0.00000** everywhere | |
| `Pred_BoA / T_exact` | 1.0 | **0.995 – 1.000** | |
| `Pred_G1 / T_exact`, held out | 1.0 | **0.998 – 1.005** | generalises |

`Pred_BoA / T_exact` is the decisive number: it compares BoA's **full** pooled
Kronecker quadratic form against the exact per-sequence objective
`sum_s ||K_s^T dW X_s||_F^2` for the dW that BoA actually produced. It is within
0.5% everywhere. That bounds not just the EK-FAC diagonal correction but *any*
correction to the factorisation — which is what Phase 3's dense-exact solve was
going to establish.

**The gap is real but tiny.** A permutation null (shuffle which `f_s` pairs with
which `e_s`, preserving both marginals) puts the finite-sample floor at
observed/null ~= 3.2 — so the measured `Cov_s(e, f)` is statistically genuine, just
absolutely negligible (0.0003–0.002 relative shape change). It grows mildly with
depth (block 2: 0.00026 -> block 11: 0.00196) but never approaches mattering.

**Why:** `Lam_G1 - Lam_BoA` is exactly `Cov_s(e[a], f[b])`. Over 2048-token
wikitext2 sequences the per-eigendirection energies `e_s`, `f_s` are extremely
homogeneous across sequences, so that covariance is near zero relative to the
product of the means. BoA's pooling assumption is simply well satisfied here.

### The objective simplifications matter 100–1000x more than the factorisation

| variant | what it changes | rel. field change (shape only) | `Pred/T` for its own objective |
|---|---|---|---|
| G1 | none (faithful correction) | **0.0003 – 0.002** | 0.995 – 1.000 |
| G12 | + causal mask | **0.06 – 0.29** | 0.96 – 1.01 |
| G123p | + attention weighting | **0.08 – 0.45** | 0.75 – 1.06 |
| G123j | + softmax-Jacobian diag | **0.10 – 0.34** | 0.81 – 1.05 |

Consistency checks: `scale_ratio(G12) = 0.481 ~= 1/2` (causal masking keeps half the
pairs) and `scale_ratio(G123) ~= 1/2048 = 1/L` (attention rows sum to 1).

Note also that the `(U (x) V)` eigenbasis fits the attention-weighted objective
noticeably *worse* (`Pred_G123p/T` down to 0.75) than it fits the unmasked one
(0.995–1.000) — so EK-FAC-style diagonal refitting is a good approximation for the
objective BoA actually uses, and a poorer one for the objective it arguably should.

**Verdict:** the Kronecker factorisation is not where BoA loses anything on
OPT-125m attention. Do not build a cheap EK solver. **Phase 3 (dense exact
reference) is unnecessary** — its question is already answered by
`Pred_BoA/T_exact`. Effort should go to Phase 2a and Phase 4.

Detail, plots, per-block table: `results/phase1/README.md`.

---

## Phase 2a — the value-projection row metric (paper eq. 9)

Status: implemented (`--row_metric_v`), exactness unit-tested, experiments running.

The handoff lists this as a gap we invented. It is not: the paper's Table 1 and
eq. (9) already specify

    H(w_V,h) = 2 X A_h^T A_h X^T  (x)  W_out,h^T W_out,h

and that derivation is **exact** — when only V changes, A_h does not move, so there
is no Taylor step and no Cauchy-Schwarz relaxation (unlike the q/k Hessians). The
released code builds only the left factor (that is `--block_v`) and never assigns
`H_row` for `v_proj`, so it falls through to the one-sided `gptq()` path with an
identity output metric. Verified exact numerically in
`tests/test_value_row_metric.py`.

Since Phase 0 reproduces the paper's numbers *without* it, the paper's reported OPT
results were evidently produced by this same code path — so eq. (9) is specified but
untested, and Phase 2a is a genuinely open question rather than a reproduction.

**Result at W3 (col+row, 3 seeds): no effect.** baseline 32.017 ± 0.134 vs
`--row_metric_v` 32.077 ± 0.101; paired per-seed deltas +0.15 / +0.05 / −0.02
(mixed sign, inside the seed spread). W2 (row): seed 0 **85.125 vs 85.884** (−0.76, −0.9%); seed 1 pending — W2's
seed spread is several PPL points, so a single seed is suggestive, not evidence. W3 is saturated for every variant tried today
(31.9–32.2), so W2 is the only bit-width where a q/k/v-side change can show.

---

## Related work and positioning

See `results/related.md` (YAQA, KronQ, BaKron, APTQ, RSQ, TurboBoA; the
"unoccupied square" table). Per the 2026-09-03 addendum, Phase 3 is repurposed to
compare q/k *objectives* (causal mask, attention weighting, softmax Jacobian, full
first-order attention-output error) under one dense solver, against RSQ as the
one-sided baseline.

## Engineering status

- `tests/test_byte_identical.py` — **PASS**. All 18 tensors bit-identical to the
  pristine upstream checkout across 3 configs (both sides driven at `--attn_impl
  sdpa`, isolating the instrumentation from the intentional causal fix).
- `tests/test_vec_convention.py` — **PASS** (rule 5.4 locked to ~1e-15, with a
  negative test pinning the `[a=col, b=row]` index order).
- `tests/test_kron_gap_math.py` — **PASS** (`Lam_G1 - Lam_BoA == Cov_s(e,f)` exactly).
- `tests/test_value_row_metric.py` — **PASS** (eq. 9 exact to 4e-16).
- A self-check inside the Phase 1 runner re-derives the attention probabilities and
  asserts they match the model's own **and** that the model's attention is causal —
  it is what caught both the missing q/k projection biases and would catch a
  regression to the SDPA path.

---

## Phase 4 — `fc1` with ReLU gating (G5). **The row metric matters here, but pooling breaks in 5/12 blocks.**

All 12 blocks, 128 seqs (all 262k tokens for `T`, `A_bar`, `B_bar`; 16k-token
stratified subsample for the EK field), W3 col+row. `T` = exact first-order MLP
output error `sum_t ||W2 (d_t * dW x_t)||^2` for the dW BoA produced.

| level | median `Pred/T` | top-5% saliency overlap vs Kronecker |
|---|---|---|
| identity row metric (released code) | **0.67** (0.02 – 1.00) | **0.41** |
| Kronecker `E_t[D G D] = G * E[d d^T]` ("MLP-aware BoA") | 0.82 (0.03 – 0.93) | — |
| EK-FAC refit in the `(U (x) V)` basis | 0.84 (0.03 – 1.00) | 0.96 |

Three findings:

1. **The identity output metric the released code uses for `fc1` is badly wrong.**
   ReLU is only 1–11% active per token (2–3% in blocks 1–4), so
   `E_t[D_t G D_t]` is nothing like `I`: identity mispredicts the MLP output error
   by 2–50x and ranks the sensitive weights so differently that only 41% of the
   top-5% overlap. Replacing it with the Kronecker `G * E[d d^T]` is cheap (a
   Hadamard product with the ReLU co-activation matrix) and, being a fixed
   3072x3072 matrix, drops straight into BoA's existing two-sided `boa()` solver.
2. **EK-FAC vs Kronecker is real but modest**: shape gap 0.054 (16x the
   permutation null of 0.003), saliency overlap 0.96. Same story as attention —
   refitting eigenvalues is not where the loss is.
3. **In blocks 1, 2, 3, 4 and 11 nothing Kronecker-shaped works**: identity,
   Kronecker, *full pooled* Kronecker (`ratio_kron_full_pooled`, not just the
   eigen-refit) and EK-FAC all predict 3–56% of `T`. That means the per-token
   terms `x_t x_t^T (x) D_t G D_t` are neither well pooled nor diagonal in the
   pooled eigenbasis — the dominant tokens carry ReLU patterns whose `M_t`
   eigenbasis differs from `V`. This is the gate problem proper. The right
   structure there is token-clustered (a rank-K sum of Kroneckers), not a
   single Kronecker with corrected eigenvalues. It is not a subsampling artefact
   (`T_all/T_sub` = 12–16 vs the ideal 16).

Positioning (per the 2026-09-03 addendum): YAQA/KronQ/BaKron do two-sided
Kronecker on MLP layers under a *global* loss with backprop; the angle here is
block-local, backprop-free and gate-aware — and finding 3 says the gate-awareness
has to be per-token-pattern, not pooled.

Detail: `results/phase4/README.md`, `fc1_summary.png`.

---

## Phase 3 (repurposed, addendum 2026-09-03) — objective comparison under one dense solver

**Correctness gate: PASSED in exact arithmetic.** On block 0 (act-order off, W3),
BoA's own `boa()` and the dense row-major GPTQ on `kron(H_row, H_col)` were run on
identical fp64 inputs: `|Q_dense - Q_boa| / |Q_boa| = 0.000e+00`, 0 level flips,
for both `k_proj` and `q_proj`. The identity behind it — `chol(kron(Hr,Hc)^-1) =
kron(U_row, U_col)` with row-major vectorisation — is locked in
`tests/test_vec_convention.py`.

Why the gate had to be fp64: in fp32 the same two solvers differ by 2e-4 – 9e-3
with 0–124 level flips per head, and *changing the lazy-batch size of the dense
solver alone* moved that from 2e-4 to 9e-3. Sequential OBS rounding over 49k
weights is chaotic to summation order at fp32; the handoff's `< 1e-4` fp32
criterion is not achievable by any reimplementation (nor by BoA against a
reordered copy of itself). The fp64 check is the meaningful one; the fp32 numbers
are logged as the solver's intrinsic ordering noise. Also measured: a dense fp32
Cholesky route is 1.8e-5 from the exact `kron(U_row,U_col)`, fp64 4e-6 — so the
non-control objectives factorise in fp64 (38.7 GB peak, seconds on the H100).

Arms (W2 `row`, W3 `col+row`; 3 seeds for the non-dense arms, seed 0 for the
dense arms on blocks 0, 5, 11): `boa`, `rsq-col`, `boa+rsq`, `dense-boa`,
`dense-mask`, `dense-p`, `dense-jac`, `dense-full`. Each logs all five objectives
on a held-out 32-sequence draw (seed 1000+seed) for blocks 0/5/11 — the transfer
matrix. RSQ weights verified against the released code (`input_weighting_module.py`).
Results: pending.

## Phase 4b — MLP-aware BoA (fc1 row metric `W2^T W2 * E[d d^T]`, block-diagonal, 48 row groups)

W3 col+row seed 0: **31.99** vs baseline 32.02 ± 0.13 (3 seeds) — no effect at W3,
where every variant tried today sits at 31.9–32.2.

**W2 (row, seed 0): wiki2 85.897 vs 85.884 (no change); c4-new 146.50 vs 154.85
(−5.4%).** The only intervention today that moved a held-out-distribution number
by more than noise. Seed 1 running; W2's wiki2 seed spread is several points, so
the c4 gain needs the second seed before it counts. Note the
approximation: the full 3072x3072 row metric costs ~20 min per fc1 through
`boa()`'s sequential row loop, so cross-group row coupling is dropped (contiguous
64-row groups); Phase 4's diagnostic says the pooled Kronecker itself is the
weaker link in blocks 1–4 and 11, so the group approximation is not the first
thing to fix.

### Early transfer row, block 0, W2 (dense-full built from 16 seqs x 128 query tokens)

Ratio of the objective under `dense-full`'s dW to the same objective under BoA's
dW, summed over 12 heads:

| layer | split | O_BoA | O_mask | O_p | O_jac | O_full |
|---|---|---|---|---|---|---|
| k_proj | calibration | 0.81 | 0.88 | 0.54 | 0.49 | **0.40** |
| k_proj | held-out | 1.63 | 1.79 | 1.13 | 1.03 | **0.96** |
| q_proj | calibration | 0.96 | 0.96 | 0.55 | 0.56 | **0.41** |
| q_proj | held-out | 1.95 | 1.95 | 1.14 | 1.18 | **0.91** |

`dense-full` wins everything on the sequences it saw and almost nothing on
sequences it did not: its own objective improves 4–9% held-out while the unmasked
and masked objectives get 1.6–2x worse, and ||dW||_F grows 5–12%. That is
overfitting of the dense Hessian, not a solver artefact (the gate is exact). The
dense objective is built from 2k query tokens because that is what a dense 49k x
49k accumulation affords; BoA's Kronecker factors use all 262k calibration tokens
for free. Follow-up queued: same arm, block 0, 32 seqs x 512 tokens (8x).

**dense-full PPL (W2 row, seed 0, blocks 0 and 11 dense, rest BoA): 86.344 wiki2 /
154.42 c4** vs BoA 85.884 / 154.85 — a wash. Block 11 transfer row (held-out,
ratio to BoA): k_proj O_full **0.48** but O_BoA 6.2x and O_mask 6.7x worse; q_proj
O_full 0.85, O_BoA 2.2x worse. The softmax-aware objective can be optimised hard
in its own terms without PPL noticing, and the price is paid on the plain logit
error. Two readings, not distinguishable at this budget: (a) O_full as defined
(first-order, per-head, W_o included) is not the quantity PPL is sensitive to at
this scale; (b) 2k tokens under-determine a 49k-dim quadratic and the solver
exploits the null space (the calibration-vs-held-out gap says this is at least
part of it). The 8x-token rerun on block 0 separates them.

**Non-dense arms, W2 row, seed 0:** `rsq-col` (one-sided GPTQ, RSQ attention-
concentration token weights, `r_min = 0.005`) = **261.1** wiki2 / 347.6 c4 vs BoA
85.9 / 154.9. At W2 on OPT-125m the row metric is worth ~3x in perplexity and
token re-weighting does not begin to substitute for it; `boa+rsq` (two-sided,
weighted H_col) pending.

**Phase 2a W2, eq. (9), seed 1:** 83.214 / 146.24 (seed 0: 85.125 / 159.42);
baseline seed 1 pending for the paired comparison.

---

## Final table — W2 (act_order_row), OPT-125m, wikitext2 / c4-new PPL

| arm | seed 0 | seed 1 | mean | paired Δ wiki2 (s0, s1) |
|---|---|---|---|---|
| **BoA** (corrected baseline) | 85.884 / 154.85 | 88.882 / 167.75 | 87.38 / 161.30 | — |
| BoA + eq. (9) v_proj row metric | 85.125 / 159.42 | 83.214 / 146.24 | 84.17 / 152.83 | −0.76, −5.67 |
| BoA + MLP-aware fc1 row metric | 85.897 / 146.50 | 84.303 / 150.69 | 85.10 / 148.59 | +0.01, −4.58 (c4: −8.4, −17.1) |
| rsq-col (one-sided, RSQ weights, r_min .005) | 261.06 / 347.63 | — | — | +175 |
| boa+rsq (two-sided, RSQ-weighted H_col) | 256.98 / 370.13 | — | — | +171 |
| dense-full, blocks 0+11, 2k tokens/head | 86.344 / 154.42 | — | — | +0.46 |
| **dense-full, block 0 only, 16k tokens/head** | **82.727 / 145.31** | **82.357 / 142.71** | 82.54 / 144.01 | **−3.16, -6.53** (c4: −9.5, -25.0) |

Seed spread of the baseline at W2 is 3.0 wiki2 / 12.9 c4, so single-seed
differences below ~3 are not evidence; paired same-seed deltas are the meaningful
column.

### Verdict per the addendum's sec. 2.5

1. **The softmax-aware objective is real, and it was data-hungry, not wrong.**
   With 2k query tokens per head the dense `O_full` solve overfits (held-out
   own-objective 0.91–0.96x, PPL a wash). With 16k tokens the same arm, changing
   *only block 0's q/k*, reaches held-out own-objective 0.66–0.74x, O_p/O_jac
   0.93–1.0x, and **−3.2 wiki2 / −6.2% c4 paired against BoA on the same
   calibration draw** (seed 1, same protocol: 82.36 vs 88.88, i.e. -6.53;
   held-out own-objective 0.63–0.74x again). Two paired seeds, both well outside
   the per-seed noise of the other arms, from one block's q/k. Sec. 2.5 case 3 is the live one: `dense-jac`'s held-out
   objective tracks `dense-full`'s closely (0.93–0.94 vs 0.66–0.74 — the
   diagonal-Jacobian objective is *improved* by the full solve, so the cheap
   `sum_t (x_t x_t^T) (x) (K diag(w_t) K^T)` structure captures much of it), which
   is the go signal for a cheap solver — provided the token budget problem is
   solved, because that is what a rank-structured estimator has to buy.
2. **The published one-sided baseline (RSQ) does not compete at W2 on OPT-125m**:
   261 vs 86. Two-sidedness (the BoA row metric) is worth ~3x here, and RSQ's
   attention-concentration weights at their default `r_min = 0.005` are
   destructive even two-sided (`boa+rsq` 257) — OPT's attention sinks send nearly
   every token's weight to ~2.5e-5. Not tuned within today's budget; RSQ searched
   `r_min` in {0.1 … 0.005} on rotated LLaMA-3, so a fair comparison needs that
   sweep. Report as "default setting fails", not as "RSQ fails".
3. **eq. (9) (v_proj row metric) and the MLP-aware fc1 metric both help at W2**,
   each by about the seed spread: eq. (9) −0.76 / −5.67 paired on wiki2; fc1
   +0.01 / −4.58 on wiki2 but −8.4 / −17.1 on c4 (consistent sign). Both are free
   at inference and near-free at quantization time; both are exact Kronecker
   objectives the released code omits. Two seeds each — needs the third.
4. **Not a paper yet, but the shape of one.** Combined story: on OPT-125m at W2
   the released BoA loses (a) nothing to pooling (Phase 1), (b) 3x to going
   one-sided, (c) ~1 seed-spread each to the missing v_proj and fc1 row metrics,
   and (d) ~3 PPL points *per block* to discarding the softmax Jacobian —
   recoverable backprop-free and block-locally, given enough tokens. What is
   missing: a cheap solver for (d) that does not need a 49k x 49k dense Hessian, a
   third seed for (c), the `r_min` sweep for the RSQ baseline, and TurboBoA as
   the two-sided baseline instead of BoA. That is one to two weeks, not a quarter.
