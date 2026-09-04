"""Gate-aware row metric for SwiGLU MLPs (Qwen/Llama), the analogue of Phase 4's
ReLU fc1 metric.

    h = phi(g) * u,   g = W_gate x,  u = W_up x,  phi = SiLU,  y = W_down h

Perturbing one of the two input projections and holding the gate pattern fixed
(first order), the output error is

    T(dW) = sum_t || W_down ( d_t * (dW x_t) ) ||^2

with a per-token diagonal that differs by layer:

    up_proj   : d_t = phi(g_t)
    gate_proj : d_t = phi'(g_t) * u_t,   phi'(g) = sigma(g) (1 + g (1 - sigma(g)))

The exact per-token row metric is D_t W_down^T W_down D_t, whose entries are
(W_down^T W_down)_ij d_i d_j, so pooling over tokens is EXACT elementwise:

    H_row = W_down^T W_down  (hadamard)  E_t[d d^T]

Unlike ReLU, d is real-valued rather than a 0/1 mask, so gate_proj's metric also
carries the magnitude of u. Only the block-diagonal groups boa() actually consumes
are accumulated: Qwen's 4864 -> 76 groups of 64.
"""
import torch

GATE, UP, DOWN = "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"


def silu(g):
    return g * torch.sigmoid(g)


def dsilu(g):
    """phi'(g) = sigma(g) (1 + g (1 - sigma(g)))."""
    s = torch.sigmoid(g)
    return s * (1.0 + g * (1.0 - s))


def gate_diagonals(g, u):
    """Per-token diagonals for (up_proj, gate_proj). g, u: [..., d_ff]."""
    return silu(g), dsilu(g) * u


class SwiGLUCoactivation:
    """Block-diagonal E[d d^T] for up_proj and gate_proj.

    Hooks gate_proj and up_proj to capture g and u, then accumulates on down_proj's
    forward, which is guaranteed to fire after both. Only the `groups` diagonal
    blocks are formed -- the full 4864x4864 second moment is never materialised.
    """

    def __init__(self, d_ff, groups, device):
        assert d_ff % groups == 0, f"{d_ff} not divisible by {groups} groups"
        self.groups, self.r = groups, d_ff // groups
        self.C = {k: torch.zeros(groups, self.r, self.r, device=device, dtype=torch.float64)
                  for k in ("up", "gate")}
        self.n = 0
        self._buf = {}

    def hook_gate(self, _m, _i, o):
        self._buf["g"] = o.detach()

    def hook_up(self, _m, _i, o):
        self._buf["u"] = o.detach()

    def hook_down(self, _m, _i, _o):
        g = self._buf.pop("g", None)
        u = self._buf.pop("u", None)
        if g is None or u is None:
            return
        g = g.reshape(-1, g.shape[-1]).float()
        u = u.reshape(-1, u.shape[-1]).float()
        d_up, d_gate = gate_diagonals(g, u)
        for key, d in (("up", d_up), ("gate", d_gate)):
            dg = d.view(-1, self.groups, self.r).double()          # [T, groups, r]
            self.C[key] += torch.einsum("tgi,tgj->gij", dg, dg)
        self.n += g.shape[0]


def grouped_row_metric(W_down, C, n, groups):
    """H_row[i] = (W_down_i^T W_down_i) hadamard (C_i / n).  Returns [groups, r, r]."""
    d_ff = W_down.shape[1]
    r = d_ff // groups
    out = []
    for i in range(groups):
        sl = slice(i * r, (i + 1) * r)
        Wd = W_down[:, sl].float()
        out.append((Wd.T @ Wd).double() * (C[i] / max(n, 1)))
    return torch.stack(out)


@torch.no_grad()
def exact_per_token_mean(W_down, d, groups):
    """Mean over tokens of D_t W_down^T W_down D_t, block-diagonal. d: [T, d_ff]."""
    d_ff = W_down.shape[1]
    r = d_ff // groups
    T = d.shape[0]
    out = []
    for i in range(groups):
        sl = slice(i * r, (i + 1) * r)
        Wd = W_down[:, sl].float()
        G = (Wd.T @ Wd).double()                                   # [r, r]
        acc = torch.zeros(r, r, dtype=torch.float64, device=d.device)
        for t in range(T):
            dt = d[t, sl].double()
            acc += G * torch.outer(dt, dt)
        out.append(acc / T)
    return torch.stack(out)
