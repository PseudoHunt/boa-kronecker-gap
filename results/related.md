# Related work — skeleton (2026-09-03)

Citations checked against arXiv on 2026-09-03; where a claim about a paper's
content could not be verified from its abstract it is marked (unverified).

**YAQA** (Cornell RelaxML, arXiv 2505.22988, May 2025; code public). Kronecker-
factored *sketches* of each layer's Hessian with respect to the full-model KL:
Sketch A is the K-FAC independence assumption, Sketch B a one-round power
iteration toward the best Kronecker fit; two-sided rounding with theory; ~30% KL
reduction over layer-wise objectives. *Contrast:* the "K-FAC for quantization"
move is published and mature, and YAQA needs backprop through the whole model.
Nothing here is "first to use Kronecker/K-FAC ideas in PTQ"; our object is a
block-local, backprop-free Hessian.

**KronQ** (Lee, Li, Yin, Panda, arXiv 2607.07964, Jul 2026). K-FAC Kronecker
Hessian `H_X (x) H_G` from the empirical Fisher, i.e. the loss depends jointly on
activation and gradient covariances, plus bidirectional incoherence processing and
a sensitivity metric for mixed precision. *Contrast:* same lane as YAQA —
two-sided, gradient (backprop) based, global loss.

**BaKron** (Birnick & Saab, arXiv 2608.06291, Aug 2026). Fast solvers for any
`A (x) B` Hessian: anti-diagonal parallelism with a recursive divide-and-conquer
construction, `O(m^2 n^2) -> O(mn(m+n))` for an `m x n` weight, so two-sided
Kronecker rounding matches GPTQ's cubic scaling. Per the handoff it also contrasts
K-FAC-style (independence) with Shampoo-style (power-iteration best-fit) Kronecker
fits and names the "expectation of Kronecker products is not Kronecker" problem
(unverified from the abstract). *Contrast:* the solver we would need if a
sum-of-Kroneckers objective ever won; our Phase 1 measured that gap for BoA's q/k
Hessians and found it <= 0.5%, so for attention the solver question is moot.

**APTQ** (DAC 2024, arXiv 2402.14866). Gauss-Newton Hessian of the *attention-
block output*, formally including the softmax, obtained from backprop gradients;
one-sided (`d_in x d_in`) column Hessian only; Hessian-trace mixed precision.
Reported baselines are inconsistent with standard GPTQ numbers — cite, do not
rely on. *Contrast:* first to put the softmax in a PTQ Hessian, but one-sided and
backprop. The repurposed Phase 3 solves an APTQ-style objective (`O_full`) two-
sided and backprop-free.

**RSQ** (Sung et al., arXiv 2503.01820, 2025; code public, verified 2026-09-03).
GPTQ with token-weighted `H_col = 2 X R^2 X^T`, where the per-layer importance
`r_j = sum_{heads, queries} A[m, i, j]` ("attention concentration": attention a token
*receives*) is min-max normalised to `[r_min, 1]`, `r_min = 0.005`; the same
weights are shared by every weight in the block. Also shows that restricting the
loss to later tokens helps — the causal-mask effect (our G12) seen empirically.
*Contrast:* covers the cheap "attention-weighted `H_col`" fix, one-sided; it is the
published one-sided baseline (`rsq-col`) and the cheapest two-sided variant
(`boa+rsq`) in Phase 3.

**TurboBoA** (Kim et al., ICLR 2026, arXiv 2602.04929, Samsung). Same Kronecker
Hessian as BoA; joint out-channel quantization with a closed-form compensation
rule (3-6x faster), a correction for error propagated from previously quantized
layers, adaptive grids with attention-wise refinement. *Contrast:* still no softmax
Jacobian; it is the current Samsung baseline, so every objective change here is
stated relative to TurboBoA's objective.

**BoA** (Kim et al., ICML 2025, arXiv 2406.13474). Two-sided `H_col (x) H_row` for
q/k from an upper bound that discards the softmax Jacobian as a constant; eq. (9)
gives an exact row metric for v_proj that the released code omits (Phase 2a).

## The unoccupied square

| | two-sided | softmax-aware | backprop-free | block-local |
|---|---|---|---|---|
| APTQ | no | yes | no | yes |
| RSQ | no | partly (token weights) | yes | yes |
| YAQA / KronQ / BaKron | yes | via global loss | no | no |
| BoA / TurboBoA | yes | no | yes | yes |
| **Phase 3 target** | yes | yes | yes | yes |

For `fc1` (Phase 4): YAQA/KronQ/BaKron already do two-sided Kronecker on MLP
layers under a global loss with backprop; our angle is block-local, backprop-free,
and gate-aware — and Phase 4 found that gate-awareness has to be per-token-pattern,
not pooled (5/12 blocks where no single Kronecker fits).
