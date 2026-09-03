"""Phase 2a: verify the value-projection Kronecker Hessian (paper eq. 9) is EXACT.

When only W_V changes, the attention weights A_h do not move, so

    ||dMHA_h||_F^2 = || W_out,h dW_V,h X A_h^T ||_F^2
                   = tr( dW (X A_h^T A_h X^T) dW^T (W_out,h^T W_out,h) )

with no Taylor step and no relaxation -- unlike the q/k Hessians. This test
asserts that equality numerically, which is what justifies calling --row_metric_v
an exact improvement rather than a heuristic.

    python3 tests/test_value_row_metric.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from diag.vec_conventions import quad_kron

torch.backends.cuda.matmul.allow_tf32 = False


def main():
    g = torch.Generator().manual_seed(0)
    d, d_h, L, H = 24, 6, 11, 4
    X = torch.randn(d, L, generator=g, dtype=torch.float64)
    W_out = torch.randn(d, H * d_h, generator=g, dtype=torch.float64)
    logits = torch.randn(H, L, L, generator=g, dtype=torch.float64)
    A = torch.softmax(logits.masked_fill(
        torch.ones(L, L, dtype=torch.bool).triu(1), -1e30), dim=-1)   # causal
    dW = torch.randn(H, d_h, d, generator=g, dtype=torch.float64)

    worst = 0.0
    for h in range(H):
        W_oh = W_out[:, h * d_h:(h + 1) * d_h]                 # [d, d_h]
        direct = (W_oh @ dW[h] @ X @ A[h].T).pow(2).sum().item()

        H_col = X @ A[h].T @ A[h] @ X.T                        # [d, d]
        H_row = W_oh.T @ W_oh                                  # [d_h, d_h]
        kron = quad_kron(dW[h], H_col, H_row)

        rel = abs(direct - kron) / abs(direct)
        worst = max(worst, rel)
        print(f"  head {h}: ||W_o dW X A^T||_F^2 = {direct:.10f}   "
              f"tr(dW H_col dW^T H_row) = {kron:.10f}   rel.diff = {rel:.2e}")
    assert worst < 1e-12, f"eq. (9) is not exact numerically (worst rel {worst:.2e})"

    # and the slicing convention used by quantize.value_row_metric must agree
    from quantize import value_row_metric

    class _Attn:
        pass

    class _Blk:
        pass

    blk, attn = _Blk(), _Attn()
    lin = torch.nn.Linear(H * d_h, d, bias=False).double()
    lin.weight.data = W_out.clone()
    attn.out_proj = lin
    blk.self_attn = attn
    got = value_row_metric(blk, H, d_h, device="cpu", dtype=torch.float64)
    for h in range(H):
        W_oh = W_out[:, h * d_h:(h + 1) * d_h]
        err = (got[h] - W_oh.T @ W_oh).abs().max().item()
        assert err < 1e-12, f"head {h}: value_row_metric slicing mismatch ({err:.2e})"
    print(f"  value_row_metric() slicing matches for all {H} heads")

    print("\nVALUE ROW METRIC TEST: PASS")


if __name__ == "__main__":
    main()
