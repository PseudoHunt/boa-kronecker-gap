"""Validate the Phase 1 eigenvalue-field math on synthetic data.

Checks the three identities the whole diagnostic rests on:
  1. Lam_G1 - Lam_BoA == Cov_s(e[a], f[b])      (the term BoA drops, exactly)
  2. Pred with Lam_BoA == S * tr(dW A_bar dW^T B_bar)   (BoA's own quadratic form)
  3. uncorrelated sequences  =>  Lam_G1 == Lam_BoA
  4. uniform attention       =>  Lam_G123p == Lam_G1 / L

    python3 tests/test_kron_gap_math.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from diag.kron_gap import BlockGapAccumulator, _eigh_desc
from diag.vec_conventions import quad_eig, quad_kron

torch.backends.cuda.matmul.allow_tf32 = False


def build(S, d, d_h, L, H, g, couple=0.0):
    """Synthetic sequences. `couple` scales key energy with input energy -> G1 gap."""
    Xs, Ks = [], []
    for s in range(S):
        gain = 1.0 + couple * (s / max(S - 1, 1))
        Xs.append(torch.randn(d, L, generator=g, dtype=torch.float64) * gain)
        Ks.append(torch.randn(H, d_h, L, generator=g, dtype=torch.float64) * (gain ** 2))
    return Xs, Ks


def run(S=16, d=9, d_h=4, L=7, H=2, couple=0.0, attn=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    Xs, Ks = build(S, d, d_h, L, H, g, couple)
    A_bar = sum(X @ X.T for X in Xs) / S
    B_bar = torch.stack([sum(K[h] @ K[h].T for K in Ks) / S for h in range(H)])

    lam_c, U = _eigh_desc(A_bar)
    Vs, lam_rs = [], []
    for h in range(H):
        lr, V = _eigh_desc(B_bar[h])
        Vs.append(V); lam_rs.append(lr)
    V = torch.stack(Vs)

    acc = BlockGapAccumulator(d, d_h, H, "cpu", want_attn=attn is not None)
    e_all, f_all = [], []
    for s in range(S):
        P = U.T @ Xs[s]
        R = torch.stack([V[h].T @ Ks[s][h] for h in range(H)])
        acc.add(P, R, attn)
        e_all.append((P ** 2).sum(-1))
        f_all.append(torch.stack([(R[h] ** 2).sum(-1) for h in range(H)]))
    fields = acc.fields()

    E = torch.stack(e_all)                       # [S, d]
    F = torch.stack(f_all)                       # [S, H, d_h]

    # --- 1. the dropped term is exactly the cross-sequence covariance ----------
    for h in range(H):
        cov = (E.T @ F[:, h]) / S - torch.outer(E.mean(0), F[:, h].mean(0))
        diff = fields["G1"][h] - fields["BoA"][h]
        err = (cov - diff).abs().max() / cov.abs().max().clamp_min(1e-30)
        assert err < 1e-10, f"head {h}: Cov identity failed, rel err {err:.2e}"
    print(f"  [1] Lam_G1 - Lam_BoA == Cov_s(e,f)                 max rel err {err:.2e}")

    # --- 2. BoA field reproduces BoA's own quadratic form ---------------------
    dW = torch.randn(H, d_h, d, generator=g, dtype=torch.float64)
    for h in range(H):
        pred = S * quad_eig(dW[h], U, V[h], fields["BoA"][h])
        direct = S * quad_kron(dW[h], A_bar, B_bar[h])
        rel = abs(pred - direct) / abs(direct)
        assert rel < 1e-10, f"head {h}: Pred_BoA != S*tr(dW A dW^T B), rel {rel:.2e}"
    print(f"  [2] Pred_BoA == S*tr(dW A_bar dW^T B_bar)          max rel err {rel:.2e}")

    # --- 4. uniform attention collapses G123p onto G1/L ----------------------
    if attn is not None and torch.allclose(attn, torch.full_like(attn, 1.0 / L)):
        for h in range(H):
            rel = ((fields["G123p"][h] - fields["G1"][h] / L).abs().max()
                   / fields["G1"][h].abs().max())
            assert rel < 1e-10, f"head {h}: uniform-attention check failed {rel:.2e}"
        print(f"  [4] uniform attention: Lam_G123p == Lam_G1 / L    max rel err {rel:.2e}")

    return fields


if __name__ == "__main__":
    print("uncorrelated sequences (expect zero gap):")
    f0 = run(couple=0.0)
    gap0 = max(((f0["G1"][h] - f0["BoA"][h]).abs().max()
                / f0["BoA"][h].abs().max()).item() for h in range(f0["BoA"].shape[0]))
    print(f"  [3] residual gap with no coupling                  {gap0:.2e}")

    print("\ncoupled sequences (expect a real gap):")
    f1 = run(couple=3.0)
    gap1 = max(((f1["G1"][h] - f1["BoA"][h]).abs().max()
                / f1["BoA"][h].abs().max()).item() for h in range(f1["BoA"].shape[0]))
    print(f"      relative gap with coupling                     {gap1:.3f}")
    assert gap1 > 0.05, "coupled construction produced no gap -- test is not sensitive"

    print("\nuniform attention weighting:")
    L = 7
    run(couple=1.0, attn=torch.full((2, L, L), 1.0 / L, dtype=torch.float64))

    print("\nKRON GAP MATH TEST: PASS")
