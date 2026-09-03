"""The single vectorisation convention for this project (engineering rule 5.4).

CONVENTION (column-major / Kronecker-standard), fixed once and for all:

    dW has shape [d_h, d]   (out_features_per_head, in_features), i.e. the shape
                            BoA already uses per head: W.view(n_heads, d_h, d).
    vec(dW)[a * d_h + b] = dW[b, a]          -> `vec(dW) == dW.t().reshape(-1)`
    H = kron(H_col, H_row)                   -> torch.kron(H_col, H_row)
                                                with H_col [d,d] FIRST.

With that pairing the three ways of writing the BoA objective coincide:

    vec(dW) @ kron(H_col, H_row) @ vec(dW)
      == trace(dW @ H_col @ dW.T @ H_row)                        (cheap form)
      == sum_{a,b} L[a,b] * (V.T @ dW @ U)[b,a]**2               (eigenbasis form)

where H_col = U diag(Lc) U.T, H_row = V diag(Lr) V.T and L = outer(Lc, Lr).
The eigenbasis form is what the EK-FAC correction generalises: replace the
rank-1 field L = outer(Lc, Lr) by a measured field L*.
"""
import torch


def vec(dW):
    """Column-major vectorisation matching torch.kron(H_col, H_row)."""
    return dW.transpose(-1, -2).reshape(*dW.shape[:-2], -1)


def unvec(v, d_h, d):
    return v.reshape(*v.shape[:-1], d, d_h).transpose(-1, -2)


def kron_hessian(H_col, H_row):
    """Dense kron(H_col, H_row) in the fixed convention. [d*d_h, d*d_h]."""
    return torch.kron(H_col, H_row)


def quad_dense(dW, H):
    v = vec(dW)
    return (v @ H @ v).item()


def quad_kron(dW, H_col, H_row):
    """trace(dW H_col dW^T H_row) -- the cheap BoA form, no dense Hessian."""
    return torch.einsum("ba,ac,dc,db->", dW, H_col, dW, H_row).item()


def quad_eig(dW, U, V, Lam):
    """sum_{a,b} Lam[a,b] * (V^T dW U)[b,a]^2 -- the EK-FAC form.

    U: [d, d] eigenvectors of H_col; V: [d_h, d_h] eigenvectors of H_row;
    Lam: [d, d_h] eigenvalue field indexed [a, b].
    """
    G = V.transpose(-1, -2) @ dW @ U          # [d_h, d]
    return (Lam * (G.transpose(-1, -2) ** 2)).sum().item()


# ---------------------------------------------------------------------------
# Row-major companion convention, needed by the dense reference solver.
#
# BoA's boa() eliminates ROW-major: outer loop over rows b of the per-head weight,
# inner GPTQ loop over columns a. Sequential OBS quantization depends on the
# elimination order, so a dense GPTQ that must reproduce boa() bit-for-bit has to
# walk indices in the same order:  idx = b * d + a  ==  dW.reshape(-1).
# With that ordering the matching Kronecker is kron(H_row, H_col) (H_row FIRST),
# and its upper Cholesky is kron(U_row, U_col) -- which is exactly what makes
# boa()'s two-stage compensation identical to one dense GPTQ pass.
# The quadratic form is the same number as in the column-major convention above.
# ---------------------------------------------------------------------------
def vec_rowmajor(dW):
    return dW.reshape(*dW.shape[:-2], -1)


def unvec_rowmajor(v, d_h, d):
    return v.reshape(*v.shape[:-1], d_h, d)


def kron_hessian_rowmajor(H_col, H_row):
    """kron(H_row, H_col): pairs with vec_rowmajor. [d_h*d, d_h*d]."""
    return torch.kron(H_row, H_col)


def quad_dense_rowmajor(dW, H):
    v = vec_rowmajor(dW)
    return (v @ H @ v).item()
