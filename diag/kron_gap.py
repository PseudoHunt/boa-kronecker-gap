"""Phase 1: measure the Kronecker gap in BoA's attention-aware Hessians.

BoA (arXiv 2406.13474, Table 1) uses, for the query projection of head h,

    H(w_Q,h) = 2 X X^T  (x)  K_h^T K_h                                   (eq. 11)

where X pools every calibration token and K_h pools every calibration token.
The objective it is meant to approximate is, per sequence s,

    T(dW) = sum_s || K_s^T dW X_s ||_F^2 = sum_s tr(dW A_s dW^T B_s)
    A_s = X_s X_s^T,  B_s = K_s K_s^T

which is an exact Kronecker product *per sequence* but not after summing:

    sum_s A_s (x) B_s   !=   (1/S) (sum_s A_s) (x) (sum_s B_s)      <-- gap G1

This module measures that gap in the EK-FAC sense: keep the eigenvectors of the
two pooled factors, and refit the eigenvalue field by direct measurement.

    H_col_bar = mean_s A_s = U diag(lam) U^T          U: [d, d]
    H_row_bar = mean_s B_s = V diag(mu)  V^T          V: [d_h, d_h]

    e_s[a] = u_a^T A_s u_a = sum_t P_s[a,t]^2         P_s = U^T X_s
    f_s[b] = v_b^T B_s v_b = sum_u R_s[b,u]^2         R_s = V^T K_s

    Lam_BoA[a,b] = mean_s(e_s[a]) * mean_s(f_s[b])
    Lam_G1 [a,b] = mean_s(e_s[a]  *  f_s[b])
    Lam_G1 - Lam_BoA = Cov_s(e[a], f[b])              <-- the dropped term, exactly

Masked (G2) and attention-weighted (G3) variants replace f_s[b] by a per-token
inner sum; see `_row_energy_variants`. NOTE: the paper's objective is the
*unmasked* surrogate (it upper-bounds the true error and drops the softmax
Jacobian as a constant, sec. 3.3), so Lam_G1 is the faithful correction to what
BoA actually optimises, while G2/G3 are changes of objective toward the true
attention reconstruction error.

Nothing here is imported on the default quantization path.
"""
import json
import os

import torch

from diag.vec_conventions import quad_eig
from diag.dump_utils import git_commit

VARIANTS = ("G1", "G12", "G123p", "G123j")


# ---------------------------------------------------------------------------
# eigen-decomposition of the two pooled factors
# ---------------------------------------------------------------------------
def _eigh_desc(H):
    """Eigendecomposition, eigenvalues descending. H is symmetric PSD."""
    lam, U = torch.linalg.eigh(H.double())
    idx = torch.argsort(lam, descending=True)
    return lam[idx].contiguous(), U[:, idx].contiguous()


# ---------------------------------------------------------------------------
# per-sequence energies in the eigenbasis
# ---------------------------------------------------------------------------
def _row_energy_variants(R, A_head):
    """Per-token row energies for each objective variant.

    R      : [d_h, L]  row-eigenbasis projection of the per-head keys (or queries)
    A_head : [L, L] or None -- attention probabilities for this head, row t = query t

    Returns dict variant -> [d_h, L] where entry [b, t] is the weight that
    eigen-direction b carries for query position t:

      G1    : total energy, independent of t (unmasked all-pairs)
      G12   : cumulative energy over u <= t (causal mask)
      G123p : sum_u p_{tu} R[b,u]^2            (attention-probability weighted)
      G123j : sum_u p_{tu}(1-p_{tu}) R[b,u]^2  (softmax-Jacobian diagonal weighted)
    """
    R2 = R ** 2                                     # [d_h, L]
    L = R2.shape[-1]
    out = {}
    out["G1"] = R2.sum(dim=-1, keepdim=True).expand(-1, L)
    out["G12"] = torch.cumsum(R2, dim=-1)
    if A_head is not None:
        out["G123p"] = R2 @ A_head.transpose(-1, -2)
        Aj = A_head * (1.0 - A_head)
        out["G123j"] = R2 @ Aj.transpose(-1, -2)
    return out


class BlockGapAccumulator:
    """Accumulates the eigenvalue fields for one (block, layer) over sequences."""

    def __init__(self, d, d_h, n_heads, device, want_attn):
        self.n_heads, self.d, self.d_h = n_heads, d, d_h
        self.device = device
        self.want_attn = want_attn
        z = lambda: torch.zeros(n_heads, d, d_h, dtype=torch.float64, device=device)
        self.sum_ef = {v: z() for v in (VARIANTS if want_attn else VARIANTS[:2])}
        self.sum_e = torch.zeros(n_heads, d, dtype=torch.float64, device=device)
        self.sum_f = torch.zeros(n_heads, d_h, dtype=torch.float64, device=device)
        self.n_seq = 0
        # full per-sequence e_s / f_s. Small (S*d + S*H*d_h floats) and needed for
        # the top-k correlation heat-maps (1c) and the permutation null.
        self.e_full, self.f_full = [], []

    def add(self, P, R, A, topk=8):
        """P: [d, L] (shared across heads).  R: [H, d_h, L].  A: [H, L, L] or None."""
        L = P.shape[-1]
        P2 = (P ** 2).double()                                    # [d, L]
        e_tot = P2.sum(dim=-1)                                    # [d]
        for h in range(self.n_heads):
            var = _row_energy_variants(R[h], A[h] if A is not None else None)
            f_tot = (R[h] ** 2).double().sum(dim=-1)              # [d_h]
            self.sum_e[h] += e_tot
            self.sum_f[h] += f_tot
            for name, M in var.items():
                if name not in self.sum_ef:
                    continue
                if name == "G1":
                    # separable: outer(e_tot, f_tot) without touching the L axis
                    self.sum_ef[name][h] += torch.outer(e_tot, f_tot)
                else:
                    self.sum_ef[name][h] += P2 @ M.double().transpose(-1, -2)  # [d, d_h]
        self.e_full.append(e_tot.cpu())
        self.f_full.append(torch.stack([(R[h] ** 2).double().sum(-1).cpu()
                                        for h in range(self.n_heads)]))
        self.n_seq += 1

    def fields(self):
        S = float(self.n_seq)
        lam_boa = torch.einsum("ha,hb->hab", self.sum_e / S, self.sum_f / S)
        out = {"BoA": lam_boa}
        for name, acc in self.sum_ef.items():
            out[name] = acc / S
        return out


# ---------------------------------------------------------------------------
# structural metrics (1c)
# ---------------------------------------------------------------------------
def structural_metrics(lam_star, lam_boa, eps_rel=1e-10):
    """All metrics on the ratio R = Lam* / Lam_BoA, per head.

    Rule 5.3(b): structural comparison is scale-free, so both fields are
    normalised to unit trace-sum first and the scale ratio reported separately.
    """
    s_star = lam_star.sum(dim=(-2, -1), keepdim=True)
    s_boa = lam_boa.sum(dim=(-2, -1), keepdim=True)
    A = lam_star / s_star
    B = lam_boa / s_boa

    floor = eps_rel * B.amax(dim=(-2, -1), keepdim=True)
    valid = B > floor
    ratio = torch.where(valid, A / B.clamp_min(floor), torch.ones_like(A))
    log2r = torch.log2(ratio.clamp_min(1e-30))

    # Shape-only (rule 5.3b): both fields are already normalised to unit sum, so
    # this cannot be inflated by a variant simply carrying less total curvature
    # (G12 keeps ~half the pairs, G123p reweights by probabilities summing to 1 --
    # on the RAW fields those show up as rel_fro ~1 and ~2000 respectively, which
    # says nothing about whether the eigenvalue SHAPE moved). The overall scale is
    # reported separately as scale_ratio.
    rel_fro = ((A - B).pow(2).sum(dim=(-2, -1)).sqrt()
               / A.pow(2).sum(dim=(-2, -1)).sqrt())
    rel_fro_raw = ((lam_star - lam_boa).pow(2).sum(dim=(-2, -1)).sqrt()
                   / lam_star.pow(2).sum(dim=(-2, -1)).sqrt())

    # mass-weighted percentiles: weight each cell by the curvature it carries.
    w = (A * valid).flatten(1)
    lr = log2r.flatten(1)
    qs = torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], dtype=lr.dtype, device=lr.device)

    def wq(vals, wts):
        order = torch.argsort(vals)
        v, ww = vals[order], wts[order]
        c = torch.cumsum(ww, 0)
        c = c / c[-1].clamp_min(1e-300)
        return torch.stack([v[torch.searchsorted(c, q).clamp(max=v.numel() - 1)] for q in qs])

    n_heads = lam_star.shape[0]
    pct_w = torch.stack([wq(lr[h], w[h]) for h in range(n_heads)])
    pct_u = torch.stack([torch.quantile(lr[h][valid.flatten(1)[h]], qs) for h in range(n_heads)])

    off = (ratio > 2) | (ratio < 0.5)
    mass_off = (A * off * valid).flatten(1).sum(1) / A.flatten(1).sum(1)

    return {
        "rel_fro": rel_fro,
        "rel_fro_raw": rel_fro_raw,
        "log2R_pct_massweighted": pct_w,
        "log2R_pct_unweighted": pct_u,
        "mass_off": mass_off,
        "scale_ratio": (s_star / s_boa).flatten(),
    }


# ---------------------------------------------------------------------------
# predictive metrics (1d)
# ---------------------------------------------------------------------------
def predictive_metrics(dW_heads, U, V, fields, per_seq_T):
    """dW_heads: [H, d_h, d]; U: [d,d]; V: [H,d_h,d_h]; fields: variant -> [H,d,d_h]."""
    S = per_seq_T["n_seq"]
    out = {}
    n_heads = dW_heads.shape[0]
    for name, lam in fields.items():
        pred = []
        for h in range(n_heads):
            pred.append(S * quad_eig(dW_heads[h].double(), U.double(), V[h].double(), lam[h]))
        out[f"Pred_{name}"] = pred
    return out


# ---------------------------------------------------------------------------
# permutation null for the G1 gap
# ---------------------------------------------------------------------------
def permutation_null(acc, seed=0, n_perm=16):
    """Finite-sample floor of the measured G1 gap.

    Lam_G1 - Lam_BoA is the sample covariance Cov_s(e[a], f[b]). With a finite
    number of sequences that estimate is nonzero even when e and f are genuinely
    independent, and the spurious part decays only as ~1/sqrt(S). Shuffling which
    f_s is paired with which e_s destroys any real coupling while preserving both
    marginals exactly, so the gap measured under the shuffle IS that floor.

    A measured gap is only evidence of real structure if it exceeds this.
    """
    E = torch.stack(acc.e_full)                 # [S, d]
    F = torch.stack(acc.f_full)                 # [S, H, d_h]
    S = E.shape[0]
    def _shape(x):
        return x / x.sum(dim=(-2, -1), keepdim=True)

    lam_boa = _shape(torch.einsum("a,hb->hab", E.mean(0), F.mean(0)))
    lam_g1 = _shape(torch.einsum("sa,shb->hab", E, F) / S)
    obs = ((lam_g1 - lam_boa).pow(2).sum(dim=(-2, -1)).sqrt()
           / lam_g1.pow(2).sum(dim=(-2, -1)).sqrt())

    g = torch.Generator().manual_seed(seed)
    null = []
    for _ in range(n_perm):
        perm = torch.randperm(S, generator=g)
        lam_n = _shape(torch.einsum("sa,shb->hab", E, F[perm]) / S)
        null.append((lam_n - lam_boa).pow(2).sum(dim=(-2, -1)).sqrt()
                    / lam_n.pow(2).sum(dim=(-2, -1)).sqrt())
    null = torch.stack(null)                    # [n_perm, H]
    return {
        "rel_fro_observed": obs.tolist(),
        "rel_fro_null_mean": null.mean(0).tolist(),
        "rel_fro_null_std": null.std(0).tolist(),
        "excess_ratio": (obs / null.mean(0).clamp_min(1e-30)).tolist(),
        "n_perm": n_perm, "n_seq": S,
    }


# ---------------------------------------------------------------------------
# OBS saliency ranking (1d, "direction gap")
# ---------------------------------------------------------------------------
def _spearman(a, b):
    ra = torch.argsort(torch.argsort(a)).double()
    rb = torch.argsort(torch.argsort(b)).double()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return (ra @ rb / (ra.norm() * rb.norm()).clamp_min(1e-30)).item()


def saliency_comparison(W_head, U, V, lam_boa, lam_star, percdamp=0.01, tops=(0.01, 0.05)):
    """Compare which weights GPTQ/BoA would treat as sensitive, under both fields.

    OBS saliency of a weight is w^2 / [H^-1]_{ii}. For H = (UxV) diag(Lam) (UxV)^T,

        [H^-1]_{(i,j),(i,j)} = sum_{a,b} U[j,a]^2 V[i,b]^2 / Lam[a,b]
                             = ((V**2) @ (1/Lam).T @ (U**2).T)[i, j]

    Damping: BoA inverts (H_col + eps_c I) and (H_row + eps_r I), so its damped
    field is exactly (lam_a + eps_c)(mu_b + eps_r). To keep the two fields
    comparable we add the SAME absolute increment to Lam*, rather than inventing a
    separate EK damping constant.

    If the two rankings agree, no solver that only changes the eigenvalue field can
    reorder the quantization -- i.e. the correction cannot matter.
    """
    W2 = (W_head.double() ** 2)                              # [d_h, d]
    U2 = (U.double() ** 2)                                   # [d, d]
    V2 = (V.double() ** 2)                                   # [d_h, d_h]

    def diag_inv(lam):
        return V2 @ (1.0 / lam).transpose(-1, -2) @ U2.transpose(-1, -2)   # [d_h, d]

    s_boa = W2 / diag_inv(lam_boa).clamp_min(1e-300)
    s_ek = W2 / diag_inv(lam_star).clamp_min(1e-300)

    a, b = s_boa.flatten(), s_ek.flatten()
    n = a.numel()
    out = {"spearman": _spearman(a, b)}
    for t in tops:
        k = max(1, int(round(t * n)))
        ia = set(torch.topk(a, k).indices.tolist())
        ib = set(torch.topk(b, k).indices.tolist())
        out[f"top{int(t*100)}pct_overlap"] = len(ia & ib) / k
    return out


def damp_field(lam_c, lam_r, percdamp=0.01):
    """BoA's damped rank-1 field and the absolute increment damping introduces."""
    eps_c = percdamp * lam_c.mean()
    eps_r = percdamp * lam_r.mean()
    undamped = torch.outer(lam_c, lam_r)
    damped = torch.outer(lam_c + eps_c, lam_r + eps_r)
    return damped, damped - undamped
