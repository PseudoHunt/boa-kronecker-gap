"""tr(D) ordering score for BoA's two-sided solver (Chen et al., arXiv 2507.18553).

GPTQ's column order is the pivot order of the Hessian's LDL and the expected layer
error is proportional to tr(D) under the equal-scale approximation. BoA runs the
same algorithm on kron(H_row, H_col), and Phase 3 established
chol(kron^-1) = kron(U_row, U_col), so the pivots factorise:

    d_ij = d_row[i] * d_col[j],      d_col[j] = 1/U_col[j,j]^2,
                                     d_row[i] = 1/U_row[i,i]^2

giving  score_layer = (sum_i s_i^2 d_row[i]) * (sum_j d_col[j])

with s_i the per-row grid-search scale. One-sided layers (H_row is None) take
d_row = 1. verify_kron_pivots() checks the factorisation against a dense kron.
"""
import torch

from quantizers.utils import get_cholesky_of_inverse, reorder_col, reorder_row


def _pivots(H):
    """d[k] = 1/U[k,k]^2 for U = chol(H^-1, upper), per head. H: [n, m, m]."""
    U = get_cholesky_of_inverse(H.clone())
    return 1.0 / U.diagonal(dim1=-2, dim2=-1).pow(2)          # [n, m]


@torch.no_grad()
def layer_score(W, H_col, H_row, scale, act_order_col, act_order_row):
    """score_layer and its unweighted twin, replicating BoA's ordering exactly.

    W, H_col, H_row, scale come straight out of BoA.preprocess() +
    find_params_H, i.e. already damped, dead-neuron filtered and head-reshaped.
    scale is computed ONCE on the unpermuted layer, as BoA does (it is derived
    before reorder_col/reorder_row run), and is permuted by reorder_row here.
    """
    Wc, Hc = W.clone(), H_col.clone()
    Hr = None if H_row is None else H_row.clone()
    sc = scale.clone()

    if act_order_col:
        Wc, Hc, _ = reorder_col(Wc, Hc)
    if Hr is not None and act_order_row:
        Wc, Hr, sc, _, _ = reorder_row(Wc, Hr, sc, torch.zeros_like(sc))

    d_col = _pivots(Hc)                                        # [n_col, hidden]
    n_heads = Wc.shape[0]
    if Hr is not None:
        d_row = _pivots(Hr)                                    # [n_heads, head_dim]
    else:
        d_row = torch.ones(n_heads, Wc.shape[1], dtype=Wc.dtype, device=Wc.device)

    s2 = sc.reshape(n_heads, -1).double() ** 2                 # [n_heads, head_dim]
    dr = d_row.double()
    dc = d_col.double()
    weighted = unweighted = 0.0
    for h in range(n_heads):
        c = dc[h] if dc.shape[0] > 1 else dc[0]
        weighted += (s2[h] * dr[h]).sum().item() * c.sum().item()
        unweighted += dr[h].sum().item() * c.sum().item()
    return weighted, unweighted


@torch.no_grad()
def verify_kron_pivots(d=8, d_h=4, seed=0, tol=1e-10):
    """sum_ij d_row[i] d_col[j] must equal the pivot sum of the dense kron."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g, dtype=torch.float64); H_col = A @ A.T + d * torch.eye(d, dtype=torch.float64)
    B = torch.randn(d_h, d_h, generator=g, dtype=torch.float64); H_row = B @ B.T + d_h * torch.eye(d_h, dtype=torch.float64)

    d_col = _pivots(H_col.unsqueeze(0))[0]
    d_row = _pivots(H_row.unsqueeze(0))[0]
    K = torch.kron(H_row, H_col)                               # row-major: index i*d + j
    d_K = _pivots(K.unsqueeze(0))[0]

    factorised = torch.outer(d_row, d_col).reshape(-1)         # (i,j) -> i*d + j
    max_abs = (d_K - factorised).abs().max().item()
    sum_dense = d_K.sum().item()
    sum_factored = d_row.sum().item() * d_col.sum().item()
    rel_sum = abs(sum_dense - sum_factored) / abs(sum_dense)
    return {"max_abs_pivot_diff": max_abs, "sum_dense": sum_dense,
            "sum_factored": sum_factored, "rel_sum_diff": rel_sum,
            "pass": (max_abs < tol) and (rel_sum < tol)}
