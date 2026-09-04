"""Length dependence of BoA's post-RoPE q/k row metric.

BoA builds  H_out = mean_t R_t^T [ mean_u R_u C R_u^T ] R_t.  Since R_t^T R_u =
R_{u-t}, that is E_{(t,u)} R_D C R_D^T with D = u - t, i.e. the pre-RoPE covariance
conjugated by the RELATIVE rotation, averaged over all calibration pairs. For a
calibration length L every pair separation D in [-(L-1), L-1] occurs with the
triangular weight

    w_L(D) = (L - |D|) / L^2 ,   sum_D w_L(D) = 1

so if C is position independent, H_out(L) = sum_D w_L(D) R_D C R_D^T is computable
for ANY target length from short calibration.

Closed form. RoPE here is the split-half convention: coordinate i pairs with
i + d_h/2 and rotates by D*theta_i, so R_D is block diagonal in 2x2 blocks. Write
the (i,j) block of C as

    M = alpha I + beta J + gamma K + delta L
    I = [[1,0],[0,1]]  J = [[0,-1],[1,0]]  K = [[1,0],[0,-1]]  L = [[0,1],[1,0]]

Then R(a) M R(b)^T rotates the (I,J) part by (a-b) and the (K,L) part by (a+b):

    R(a) M R(b)^T = [a_cos(a-b) - b_sin(a-b)] I + [a_sin(a-b) + b_cos(a-b)] J
                  + [g_cos(a+b) - d_sin(a+b)] K + [g_sin(a+b) + d_cos(a+b)] L

With a = D theta_i, b = D theta_j and w_L symmetric in D, every sine term averages
to zero and every cosine term averages to the Fejer kernel

    F_L(phi) = (1/L^2) (sin(L phi/2) / sin(phi/2))^2 = sum_D w_L(D) cos(D phi)

so the whole average collapses to two scalars per block pair:

    E_w[R_D C R_D^T]_{ij} : (I,J) part scaled by F_L(theta_i - theta_j)
                            (K,L) part scaled by F_L(theta_i + theta_j)

Nothing here touches the solver or quantises anything.
"""
import torch


def fejer(L, phi):
    """F_L(phi) = (1/L^2)(sin(L phi/2)/sin(phi/2))^2, with F_L(0) = 1."""
    phi = torch.as_tensor(phi, dtype=torch.float64)
    half = phi / 2.0
    s = torch.sin(half)
    num = torch.sin(L * half)
    # |sin(phi/2)| small: the ratio tends to L, so F -> 1. Guard rather than divide.
    small = s.abs() < 1e-12
    ratio = torch.where(small, torch.full_like(s, float(L)), num / torch.where(small, torch.ones_like(s), s))
    return (ratio ** 2) / (L ** 2)


def _blocks(C):
    """Split [d_h, d_h] into the four [h, h] block components (split-half pairs)."""
    h = C.shape[-1] // 2
    m11, m12 = C[..., :h, :h], C[..., :h, h:]
    m21, m22 = C[..., h:, :h], C[..., h:, h:]
    alpha = (m11 + m22) / 2
    beta = (m21 - m12) / 2
    gamma = (m11 - m22) / 2
    delta = (m12 + m21) / 2
    return alpha, beta, gamma, delta


def _unblock(alpha, beta, gamma, delta):
    m11, m12 = alpha + gamma, -beta + delta
    m21, m22 = beta + delta, alpha - gamma
    return torch.cat([torch.cat([m11, m12], -1), torch.cat([m21, m22], -1)], -2)


def avg_closed(C, theta, L, transpose=False):
    """E_w[ R_D C R_D^T ] in closed form (transpose=True gives E_w[ R_D^T C R_D ])."""
    a, b, g, d = _blocks(C.double())
    th = theta.double()
    Fm = fejer(L, th[:, None] - th[None, :])
    Fp = fejer(L, th[:, None] + th[None, :])
    # R_D^T C R_D is the D -> -D relabelling; w_L is symmetric so the sine terms
    # still vanish and the result is identical. Written out rather than assumed.
    if transpose:
        Fm = fejer(L, -(th[:, None] - th[None, :]))
        Fp = fejer(L, -(th[:, None] + th[None, :]))
    return _unblock(Fm * a, Fm * b, Fp * g, Fp * d)


def avg_direct(C, theta, L, transpose=False, chunk=4096):
    """Same quantity by explicit weighted sum over D -- the reference implementation."""
    a, b, g, d = _blocks(C.double())
    th = theta.double()
    dif = th[:, None] - th[None, :]
    summ = th[:, None] + th[None, :]
    out = [torch.zeros_like(x) for x in (a, b, g, d)]
    D_all = torch.arange(-(L - 1), L, dtype=torch.float64, device=C.device)
    w_all = (L - D_all.abs()) / (L ** 2)
    for s in range(0, D_all.numel(), chunk):
        D = D_all[s:s + chunk][:, None, None]
        w = w_all[s:s + chunk][:, None, None]
        sgn = -1.0 if transpose else 1.0
        pm, pp = sgn * D * dif, sgn * D * summ
        cm, sm = torch.cos(pm), torch.sin(pm)
        cp, sp = torch.cos(pp), torch.sin(pp)
        out[0] += (w * (a * cm - b * sm)).sum(0)
        out[1] += (w * (a * sm + b * cm)).sum(0)
        out[2] += (w * (g * cp - d * sp)).sum(0)
        out[3] += (w * (g * sp + d * cp)).sum(0)
    return _unblock(*out)


def band_decomposition(C, theta, L_ref, L_new, edges=(512, 2048, 8192, 32768)):
    """Share of ||H(w_L_new) - H(w_L_ref)||_F^2 by wavelength band.

    The (I,J) part of block pair (i,j) moves with frequency |theta_i - theta_j| and
    the (K,L) part with theta_i + theta_j, and the two are Frobenius-orthogonal, so

        ||dM||_F^2 = 2[ (dF_-)^2 (alpha^2 + beta^2) + (dF_+)^2 (gamma^2 + delta^2) ]

    splits exactly. Each term is binned by its own wavelength 2*pi/frequency.
    """
    a, b, g, d = _blocks(C.double())
    th = theta.double()
    dif = (th[:, None] - th[None, :]).abs()
    summ = th[:, None] + th[None, :]
    dFm = fejer(L_new, dif) - fejer(L_ref, dif)
    dFp = fejer(L_new, summ) - fejer(L_ref, summ)
    e_m = 2 * (dFm ** 2) * (a ** 2 + b ** 2)
    e_p = 2 * (dFp ** 2) * (g ** 2 + d ** 2)
    lam_m = torch.where(dif > 0, 2 * torch.pi / dif.clamp_min(1e-30), torch.full_like(dif, float("inf")))
    lam_p = 2 * torch.pi / summ.clamp_min(1e-30)
    names = [f"<{edges[0]}"] + [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges) - 1)] + [f">{edges[-1]}"]
    tot = (e_m.sum() + e_p.sum()).clamp_min(1e-300)
    shares = {}
    for k, nm in enumerate(names):
        lo = -1.0 if k == 0 else edges[k - 1]
        hi = edges[k] if k < len(edges) else float("inf")
        m = ((lam_m > lo) & (lam_m <= hi)).double() * e_m + ((lam_p > lo) & (lam_p <= hi)).double() * e_p
        shares[nm] = (m.sum() / tot).item()
    return shares, tot.sqrt().item()


def rel_fro(A, B):
    """||A - B||_F / ||A||_F, A the reference."""
    return ((A - B).pow(2).sum((-2, -1)).sqrt() / A.pow(2).sum((-2, -1)).sqrt().clamp_min(1e-300))
