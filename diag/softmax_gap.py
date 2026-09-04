"""Stage D: the softmax gap for RoPE + GQA models.

phase1_runner.py measures this for OPT but cannot run here: it indexes both the q
and k row metrics by n_heads (under GQA the k metric has n_kv_heads entries), and
it reconstructs Q/K as `W x + b` with no RoPE. This module redoes the measurement
in a form that is exact for RoPE and GQA and reduces to the OPT case.

The derivation. With RoPE the logit is q_rot,t . k_rot,u, so a perturbation dW of
the query projection changes it by

    z_tu = (R_t dW x_t) . (R_u k_u) = g_t^T R_{u-t} k_u ,   g_t = dW x_t

and the weighted objective T = sum_tu w_tu z_tu^2 becomes sum_t g_t^T C_t g_t with

    C_t = sum_u w_tu (R_t^T k_rot,u)(R_t^T k_rot,u)^T = R_t^T D_t R_t ,
    D_t = sum_u w_tu k_rot,u k_rot,u^T .

So the row factor is genuinely per-query-position. Two consequences:

  * With w_tu = 1, mean_t C_t is EXACTLY BoA's H_row for q_proj -- the code does
    `(rot^T @ E[K_rot K_rot^T] @ rot).mean(0)`. The separable BoA field and the
    coupled field are therefore built from the same object, which is what makes
    the comparison meaningful rather than a change of basis.
  * With R_t = I (no RoPE) C_t collapses to a constant and this reduces to
    phase1's formulation, which is the validation used in test_softmax_gap.py.

In the eigenbases U (of H_col) and V (of BoA's H_row), writing P = U^T X:

    F[t,b]      = (R_t v_b)^T D_t (R_t v_b)
    Lam_var[a,b] = sum_t P[a,t]^2 F_var[t,b]          coupled field
    Lam_BoA[a,b] = (sum_t P[a,t]^2) * mean_t F_G1[t,b]   separable field

Variants differ only in w: G1 all-pairs, G12 causal, G123p attention probability,
G123j softmax-Jacobian p(1-p). G123j vs BoA is the softmax gap.

For k_proj the roles swap (the row index is the key position u and the source is
the queries), and under GQA a kv head's factor SUMS over the query heads that read
it, matching how the value row metric is built.
"""
import torch

from diag.kron_gap import _eigh_desc, structural_metrics

VARIANTS = ("G1", "G12", "G123p", "G123j")


def attn_probs(Qr, Kr, scaling, neg_inf=-1e30):
    """Causal softmax attention. Qr,Kr: [L, d_h] post-RoPE. Returns [L, L], row t."""
    L = Qr.shape[0]
    logits = (Qr.float() @ Kr.float().T) * scaling
    mask = torch.ones(L, L, dtype=torch.bool, device=Qr.device).triu(1)
    return torch.softmax(logits.masked_fill(mask, neg_inf), dim=-1)


def _weight_mats(A):
    """w_tu per variant for one head. A: [L, L] or None."""
    L = A.shape[-1] if A is not None else None
    out = {}
    out["G1"] = None                      # all-ones, handled analytically
    out["G12"] = None                     # causal, handled by cumsum
    if A is not None:
        out["G123p"] = A
        out["G123j"] = A * (1.0 - A)
    return out


@torch.no_grad()
def _D_fields(S, A, transpose_w):
    """D_t = sum_u w_tu S[u] S[u]^T for each variant.

    S : [L, d_h] the row source (post-RoPE keys for q_proj, queries for k_proj)
    transpose_w : sum over the FIRST index of w instead of the second (k_proj)
    Returns dict variant -> [L, d_h, d_h] (or [d_h, d_h] for the constant G1).
    """
    L, d_h = S.shape
    SS = (S[:, :, None] * S[:, None, :])                       # [L, d_h, d_h]
    flat = SS.reshape(L, d_h * d_h)
    out = {}
    out["G1"] = SS.sum(0)                                      # [d_h, d_h], constant in t
    # causal: for q_proj sum_{u<=t}; for k_proj sum_{t>=u} = reverse cumsum
    out["G12"] = (torch.flip(torch.cumsum(torch.flip(SS, [0]), 0), [0])
                  if transpose_w else torch.cumsum(SS, 0))
    for v in ("G123p", "G123j"):
        W = A[v]
        if W is None:
            continue
        Wm = (W.transpose(0, 1) if transpose_w else W).to(flat.dtype)   # [L(row), L(src)]
        out[v] = (Wm @ flat).reshape(L, d_h, d_h)
    return out


@torch.no_grad()
def _F_from_D(D, VR):
    """F[t,b] = (R_t v_b)^T D_t (R_t v_b).  VR: [L, d_h, d_h] columns R_t v_b."""
    if D.dim() == 2:                                           # constant in t
        return torch.einsum("tdb,de,teb->tb", VR, D, VR)
    return torch.einsum("tdb,tde,teb->tb", VR, D, VR)


@torch.no_grad()
def block_fields(P2, src, V, rot, A_head, transpose_w):
    """Accumulate one head's fields for one sequence.

    P2   : [d, L]  (U^T X)^2, shared across heads
    src  : [L, d_h] row source, post-RoPE
    V    : [d_h, d_h] row eigenbasis (BoA's, columns are v_b)
    rot  : [L, d_h, d_h] per-position rotation, or None
    Returns (dict variant -> [d, d_h] Lam, f_boa [d_h], e [d])
    """
    L = src.shape[0]
    VR = (rot @ V) if rot is not None else V.expand(L, *V.shape)   # [L, d_h, d_h]
    D = _D_fields(src, _weight_mats(A_head), transpose_w)
    lam, F1 = {}, None
    for v in VARIANTS:
        if v not in D:
            continue
        F = _F_from_D(D[v], VR)                                # [L, d_h]
        if v == "G1":
            F1 = F
        lam[v] = P2 @ F                                        # [d, d_h]
    e = P2.sum(-1)                                             # [d]
    f_boa = F1.mean(0)                                         # [d_h] = v^T H_row_boa v
    return lam, f_boa, e


def kish_n_eff(w):
    """Kish effective sample size (sum w)^2 / sum w^2 over all (t,u) pairs."""
    s1 = w.sum()
    s2 = (w * w).sum()
    return (s1 * s1 / s2.clamp_min(1e-30)).item()


def split_half_correct(obs, noise):
    """Remove the sampling floor: sqrt(max(0, obs^2 - noise^2))."""
    import math
    return math.sqrt(max(0.0, obs * obs - noise * noise))


# ---------------------------------------------------------------------------
# exact softmax Jacobian (centred).  J = diag(p) - p p^T, so
#     dlogit^T J dlogit = sum_u p_tu (g.k_u)^2 - (sum_u p_tu g.k_u)^2
#                       = g^T [ sum_u p_tu k_u k_u^T - kbar_t kbar_t^T ] g
#                       = g^T Cov_{p_t}(k) g
# Since sum_u p_tu = 1, writing k_u = b + k'_u cancels b EXACTLY: the metric is
# Cov_{p_t}(k'). G123j uses only diag(J) = p(1-p), which RETAINS b b^T -- decisive
# when the key second moment is bias-dominated, as it is on Qwen (~96%).
# ---------------------------------------------------------------------------
@torch.no_grad()
def centred_metric(K, A, rot):
    """M_t = R_t^T [ sum_u p_tu k_u k_u^T - kbar_t kbar_t^T ] R_t, back-rotated.

    K   : [L, d_h] post-RoPE keys, as attention consumes them
    A   : [L, L] attention probabilities, row t
    rot : [L, d_h, d_h] per-position rotation R_t, or None
    Returns [L, d_h, d_h].
    """
    L, d_h = K.shape
    Ad = A.to(K.dtype)
    KK = (K[:, :, None] * K[:, None, :]).reshape(L, d_h * d_h)
    second = (Ad @ KK).reshape(L, d_h, d_h)            # sum_u p_tu k_u k_u^T
    kbar = Ad @ K                                      # [L, d_h]
    M = second - kbar[:, :, None] * kbar[:, None, :]   # Cov_{p_t}(k)  (PSD)
    if rot is None:
        return M
    return rot.transpose(-1, -2) @ M @ rot             # back-rotate


@torch.no_grad()
def invisible_mass(H_row, Mbar, weak_frac=0.05):
    """How much of BoA's H_row lives where the exact softmax metric barely looks.

    A covariance is generically full rank, so the centred metric annihilates no
    exact subspace -- a hard null-space threshold reports 0 and says nothing. What
    matters is the SPECTRAL mismatch: sort the centred metric's eigendirections
    ascending and take the weakest ones carrying `weak_frac` of its trace; those
    are directions the exact Jacobian is nearly blind to. Report the share of
    H_row's trace sitting there, plus the concentration of each metric.

    Returns dict:
      h_mass_in_weak  H_row trace share in the centred metric's weakest directions
      h_top1          H_row trace share along its OWN top direction (rank-1-ness)
      m_top1          same for the centred metric
      cos             Frobenius cosine between the two, unit-trace normalised
    """
    mu, V = torch.linalg.eigh(Mbar.double())
    order = torch.argsort(mu)                      # ascending
    mu_s, V_s = mu[order], V[:, order]
    h = torch.einsum("db,de,eb->b", V_s, H_row.double(), V_s).clamp_min(0)
    cum = torch.cumsum(mu_s.clamp_min(0), 0) / mu_s.clamp_min(0).sum().clamp_min(1e-30)
    weak = cum <= weak_frac
    if not weak.any():
        weak[0] = True
    hm = (h[weak].sum() / h.sum().clamp_min(1e-30)).item()

    hh = torch.linalg.eigvalsh(H_row.double()).clamp_min(0)
    mm = mu.clamp_min(0)
    a = Mbar.double() / Mbar.double().trace().clamp_min(1e-30)
    b = H_row.double() / H_row.double().trace().clamp_min(1e-30)
    cos = ((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30)).item()
    return {"h_mass_in_weak": hm,
            "h_top1": (hh.max() / hh.sum().clamp_min(1e-30)).item(),
            "m_top1": (mm.max() / mm.sum().clamp_min(1e-30)).item(),
            "cos": cos}


def rel_fro_matrix(Astar, Bboa):
    """Scale-free discrepancy between two metrics, per head: both to unit trace."""
    a = Astar / Astar.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-30)[..., None, None]
    b = Bboa / Bboa.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-30)[..., None, None]
    return ((a - b).pow(2).sum((-2, -1)).sqrt() / a.pow(2).sum((-2, -1)).sqrt())
