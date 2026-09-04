"""The pooled SwiGLU row metric must equal the mean of the per-token metrics.

    H_row = W_down^T W_down (hadamard) E_t[d d^T]   vs   E_t[ D_t W_down^T W_down D_t ]

These are equal elementwise by construction, so this is a check on the
implementation (grouping, indexing, dtype), not on the algebra. Also checks the
SiLU derivative against autograd and PSD-ness after BoA's damping.

    python3 tests/test_swiglu_row_metric.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from diag.swiglu_metric import dsilu, gate_diagonals, grouped_row_metric, exact_per_token_mean, silu
from utils.quant_utils import damping


def main():
    torch.manual_seed(0)
    d_hidden, d_ff, groups, T = 24, 32, 4, 50        # r = 8
    W_down = torch.randn(d_hidden, d_ff, dtype=torch.float32)
    g = torch.randn(T, d_ff)
    u = torch.randn(T, d_ff)

    # 1. SiLU derivative vs autograd
    gg = g.clone().double().requires_grad_(True)
    silu(gg).sum().backward()
    err = (gg.grad - dsilu(g.double())).abs().max().item()
    print(f"  phi' vs autograd: max abs err {err:.3e}")
    assert err < 1e-10, err

    # 2. pooled == mean of per-token, for both layers
    d_up, d_gate = gate_diagonals(g, u)
    for name, d in (("up_proj", d_up), ("gate_proj", d_gate)):
        C = torch.zeros(groups, d_ff // groups, d_ff // groups, dtype=torch.float64)
        dg = d.view(T, groups, d_ff // groups).double()
        C += torch.einsum("tgi,tgj->gij", dg, dg)
        pooled = grouped_row_metric(W_down, C, T, groups)
        exact = exact_per_token_mean(W_down, d, groups)
        rel = ((pooled - exact).norm() / exact.norm()).item()
        print(f"  {name:9s} pooled vs per-token mean: rel {rel:.3e}  shape {tuple(pooled.shape)}")
        assert rel < 1e-6, f"{name}: {rel:.3e}"

        # 3. PSD after damping (it is PSD already; damping must not break it)
        H = damping(pooled.clone())
        mn = min(torch.linalg.eigvalsh(H[i]).min().item() for i in range(groups))
        mx = max(torch.linalg.eigvalsh(H[i]).max().item() for i in range(groups))
        print(f"  {name:9s} min eig after damping {mn:.3e} (max {mx:.3e}) -> "
              f"{'PSD' if mn >= -1e-9 * max(mx, 1e-30) else 'NOT PSD'}")
        assert mn >= -1e-9 * max(mx, 1e-30), f"{name} not PSD: {mn}"

    print("\nSWIGLU ROW METRIC TEST: PASS")


if __name__ == "__main__":
    main()
