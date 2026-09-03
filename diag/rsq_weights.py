"""RSQ token weights (Sung et al., arXiv 2503.01820), "attention concentration".

Per block (RSQ: "token importance is kept consistent across all weights within a
layer"), from the block's own FP attention probabilities A [M, T, T]:

    r_j = sum_{m,i} A[m, i, j]                       (attention token j RECEIVES)
    r   = r_min + (r - min r) / (max r - min r) * (r_max - r_min)      (their eq. 4)
    H_RSQ = 2 X R^2 X^T                              (tokens scaled by r_j)

Defaults from their run script: r_max = 1, r_min = 0.005 (the released yaml has
[1, 3]; the paper searches r_min in {0.1, ..., 0.005} with r_max = 1 and the
script's --min_value/--max_value override the yaml). Normalisation is per
sequence, matching a per-sample Hessian accumulation.

Used two ways here (the 2026-09-03 addendum, Table in sec. 2.4):
    rsq-col : q/k through one-sided gptq() with the weighted H_col, H_row = I
    boa+rsq : q/k through boa() with the weighted H_col, H_row unchanged
`--rsq_all_layers` additionally applies the same weights to every layer's H_col
in the block, which is what RSQ itself does.
"""
import torch


def attention_concentration(A, r_min=0.005, r_max=1.0):
    """A: [M, T, T] probabilities (row = query). Returns r [T] in [r_min, r_max]."""
    r = A.float().sum(dim=(0, 1))                            # sum over heads and queries
    lo, hi = r.min(), r.max()
    r = r_min + (r - lo) / (hi - lo).clamp_min(1e-12) * (r_max - r_min)
    return r


def weighted_cov_update(cov, n_data, X, r):
    """Same running-mean convention as utils.hessian_utils.compute_cov (factor 2/n),
    with token t scaled by r_t: adds (2/n) * sum_t r_t^2 x_t x_t^T.

    X: [d, T] (or [H, d, T]); r: [T].
    """
    n_new = X.shape[-1]
    cov *= n_data / (n_data + n_new)
    n_data += n_new
    Xs = X.float() * r.to(X.device)[None, :] if X.dim() == 2 else X.float() * r.to(X.device)[None, None, :]
    cov += (2 / n_data) * (Xs @ Xs.transpose(-1, -2))
    return cov, n_data
