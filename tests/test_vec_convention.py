"""Rule 5.4 lock-in: vec / kron / trace / eigenbasis forms must agree exactly.

    python3 tests/test_vec_convention.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from diag.vec_conventions import vec, unvec, kron_hessian, quad_dense, quad_kron, quad_eig

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def spd(n, g):
    A = torch.randn(n, n, generator=g, dtype=torch.float64)
    return A @ A.T / n + torch.eye(n, dtype=torch.float64) * 0.1


def main():
    g = torch.Generator().manual_seed(0)
    d, d_h = 11, 5                      # deliberately non-square, d != d_h
    dW = torch.randn(d_h, d, generator=g, dtype=torch.float64)
    H_col, H_row = spd(d, g), spd(d_h, g)

    # 0. vec/unvec round trip
    assert torch.equal(unvec(vec(dW), d_h, d), dW), "vec/unvec round trip"

    H = kron_hessian(H_col, H_row)
    assert H.shape == (d * d_h, d * d_h)

    a = quad_dense(dW, H)
    b = quad_kron(dW, H_col, H_row)
    rel = abs(a - b) / abs(a)
    print(f"vec^T kron(H_col,H_row) vec = {a:.12f}")
    print(f"tr(dW H_col dW^T H_row)     = {b:.12f}   rel.diff = {rel:.3e}")
    assert rel < 1e-12, "dense-kron vs trace form disagree"

    # eigenbasis form with the rank-1 (BoA) eigenvalue field must reproduce it
    Lc, U = torch.linalg.eigh(H_col)
    Lr, V = torch.linalg.eigh(H_row)
    Lam_boa = torch.outer(Lc, Lr)               # [d, d_h]
    c = quad_eig(dW, U, V, Lam_boa)
    rel2 = abs(a - c) / abs(a)
    print(f"sum_ab Lam_BoA (V^T dW U)^2 = {c:.12f}   rel.diff = {rel2:.3e}")
    assert rel2 < 1e-12, "eigenbasis form disagrees with BoA Hessian"

    # (U kron V) diag(Lam) (U kron V)^T must rebuild kron(H_col,H_row) exactly
    UV = torch.kron(U, V)
    H_rebuilt = UV @ torch.diag(Lam_boa.reshape(-1)) @ UV.T
    err = (H_rebuilt - H).abs().max().item() / H.abs().max().item()
    print(f"max|(UxV)diag(Lam)(UxV)^T - kron(H_col,H_row)| / max|H| = {err:.3e}")
    assert err < 1e-12, "eigen-reconstruction of the Kronecker Hessian failed"

    # the ordering of Lam must be [a=col-eig, b=row-eig]: a transposed field must FAIL
    if d == d_h:
        raise SystemExit("choose d != d_h so the transpose check is meaningful")
    try:
        quad_eig(dW, U, V, Lam_boa.T)
        raise SystemExit("transposed eigenvalue field did not error -- index order is not pinned")
    except (RuntimeError, IndexError):
        print("transposed Lam correctly rejected (index order [a=col, b=row] is pinned)")

    print("\nVEC CONVENTION TEST: PASS")


if __name__ == "__main__":
    main()


def test_rowmajor():
    from diag.vec_conventions import (vec_rowmajor, unvec_rowmajor,
                                      kron_hessian_rowmajor, quad_dense_rowmajor)
    g = torch.Generator().manual_seed(1)
    d, d_h = 11, 5
    dW = torch.randn(d_h, d, generator=g, dtype=torch.float64)
    H_col, H_row = spd(d, g), spd(d_h, g)
    assert torch.equal(unvec_rowmajor(vec_rowmajor(dW), d_h, d), dW)
    a = quad_dense_rowmajor(dW, kron_hessian_rowmajor(H_col, H_row))
    b = quad_kron(dW, H_col, H_row)
    c = quad_dense(dW, kron_hessian(H_col, H_row))
    assert abs(a - b) / abs(b) < 1e-12 and abs(a - c) / abs(c) < 1e-12
    # Cholesky of the Kronecker is the Kronecker of the Choleskys (upper, row-major)
    Hi = torch.linalg.inv(kron_hessian_rowmajor(H_col, H_row))
    U = torch.linalg.cholesky(Hi, upper=True)
    U_row = torch.linalg.cholesky(torch.linalg.inv(H_row), upper=True)
    U_col = torch.linalg.cholesky(torch.linalg.inv(H_col), upper=True)
    err = (U - torch.kron(U_row, U_col)).abs().max().item() / U.abs().max().item()
    assert err < 1e-10, f"chol(kron) != kron(chol): {err:.2e}"
    print(f"row-major: quad forms agree ({a:.12f}); chol(kron(Hr,Hc)^-1) == kron(U_row,U_col) to {err:.1e}")
    print("ROW-MAJOR CONVENTION TEST: PASS")


if __name__ == "__main__":
    test_rowmajor()
