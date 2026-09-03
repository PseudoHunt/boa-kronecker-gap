"""Repurposed Phase 3: dense reference solves for different q/k OBJECTIVES.

Per head the vectorised weight has n = d_h * d = 49,152 entries and a dense fp32
Hessian is 9.7 GB. We build one dense Hessian per objective and run the SAME dense
GPTQ elimination on each, so that differences in PPL are differences in objective,
not in solver.

Vectorisation (rule 5.4, row-major companion): idx = b*d + a, dW.reshape(-1).
BoA's boa() eliminates row-major, and chol(kron(H_row,H_col)^-1) = kron(U_row,U_col),
so dense GPTQ on kron(H_row, H_col) reproduces boa() exactly -- that is the
correctness gate (`dense-boa` must match `boa()` to < 1e-4 relative).

Objectives (q_proj, head h; roles swap for k_proj, see `_pair_factors`):

    O_BoA  : sum_{t,u}            (k_u^T dW x_t)^2     -> kron(K K^T, X X^T)      control
    O_mask : sum_t sum_{u<=t}     (k_u^T dW x_t)^2     -> sum_t (x_t x_t^T) (x) C_t
    O_p    : + p_tu weights
    O_jac  : + p_tu (1 - p_tu) weights
    O_full : sum_t || W_o,h V_h J_t delta_t ||^2,  J_t = diag(p_t) - p_t p_t^T
             (first-order change of the block-output contribution of head h,
              including W_o,h; cross-head terms dropped, as in BoA)

Every per-token Hessian is kron(R_t, Cc_t) with R_t [d_h,d_h], Cc_t [d,d]:
    q_proj: Cc_t = x_t x_t^T (rank 1),  R_t = K diag(w_t) K^T  (or the J-form)
    k_proj: R_t  = q_t q_t^T (rank 1),  Cc_t = X diag(w_t) X^T (or the J-form)
and the dense matrix is accumulated as ONE matmul per token chunk:
    C[(b,b'),(a,a')] = sum_t vec(R_t)[(b,b')] vec(Cc_t)[(a,a')]  = Rvec^T @ Ccvec
then permuted to H[(b,a),(b',a')]. Never materialise kron per token.
"""
import math
import os

import torch

from utils.quant_utils import fake_quantize, filter_dead_neuron, damping

Q_NAME, K_NAME = "self_attn.q_proj", "self_attn.k_proj"
OBJECTIVES = ("boa", "mask", "p", "jac", "full")


# ---------------------------------------------------------------------------
# dense GPTQ on a vectorised weight
# ---------------------------------------------------------------------------
@torch.no_grad()
def chol_of_inverse_dense(holder):
    """Upper Cholesky factor of H^-1 for a dense fp64 [n,n] H64 (consumed: the
    caller must not keep another reference). Measured on block 0: an fp32 route is
    1.8e-5 off the exact kron(U_row,U_col), fp64 4e-6; fp64 costs nothing on an
    H100 and keeps the elimination's input as clean as BoA's own small factors.
    Peak memory 2 x n^2 x 8 B = 38.7 GB at n = 49,152.
    """
    # `holder` is a 1-element list: popping it is the only way to make sure no
    # caller-side name keeps the 19 GB fp64 matrix alive during the factorisation
    # (a plain `del` in the callee only drops the local binding).
    H64 = holder.pop()
    assert H64.dtype == torch.float64 and not holder
    L = torch.linalg.cholesky(H64)
    del H64
    Hinv = torch.cholesky_inverse(L)
    del L
    U = torch.linalg.cholesky(Hinv, upper=True)
    del Hinv
    return U.float()


@torch.no_grad()
def dense_gptq(w, H, scale_vec, zero_vec, maxq, blocksize=256, U=None):
    """GPTQ (lazy-batch) on one vector w [n] with dense Hessian H [n,n] (damped).

    scale_vec/zero_vec are per-index quantisation parameters (broadcast from the
    per-row scales). Returns q [n]. Elimination order = index order. If `U` (the
    upper Cholesky factor of H^-1) is given, H is ignored.
    """
    n = w.numel()
    W = w.clone()                       # dtype preserved (fp64 for the gate)
    Q = torch.zeros_like(W)
    if U is None:
        holder = [H.double()]; del H
        U = chol_of_inverse_dense(holder).to(W.dtype)
    for i1 in range(0, n, blocksize):
        i2 = min(i1 + blocksize, n)
        cnt = i2 - i1
        W1 = W[i1:i2].clone()
        Q1 = torch.zeros(cnt, device=W.device, dtype=W.dtype)
        Err1 = torch.zeros(cnt, device=W.device, dtype=W.dtype)
        U1 = U[i1:i2, i1:i2]
        for i in range(cnt):
            wq = W1[i]
            q = fake_quantize(wq, scale_vec[i1 + i], zero_vec[i1 + i], maxq)
            Q1[i] = q
            err = (wq - q) / U1[i, i]
            W1[i:] -= err * U1[i, i:]
            Err1[i] = err
        Q[i1:i2] = Q1
        W[i2:] -= Err1 @ U[i1:i2, i2:]
    return Q


# ---------------------------------------------------------------------------
# dense Hessian construction
# ---------------------------------------------------------------------------
def _perm_to_rowmajor(C, d_h, d):
    """C [(b,b'),(a,a')] -> H [(b,a),(b',a')]."""
    return C.reshape(d_h, d_h, d, d).permute(0, 2, 1, 3).reshape(d_h * d, d_h * d)


def kron_rowmajor(H_row, H_col):
    return torch.kron(H_row, H_col)


@torch.no_grad()
def attn_probs(Q, K, scaling):
    L = Q.shape[-1]
    logits = (Q.transpose(-1, -2) * scaling) @ K
    logits = logits.masked_fill(torch.ones(L, L, dtype=torch.bool, device=Q.device).triu(1), -1e30)
    return torch.softmax(logits.float(), dim=-1)            # [t, u]


def _pair_weights(obj, A_h, L, dev):
    """w[t,u] for the pair-weighted objectives. Causal for everything but 'boa'."""
    if obj == "boa":
        return torch.ones(L, L, device=dev)
    tri = torch.ones(L, L, device=dev).tril(0)
    if obj == "mask":
        return tri
    if obj == "p":
        return A_h
    if obj == "jac":
        return A_h * (1 - A_h)
    raise ValueError(obj)


@torch.no_grad()
def accumulate_dense(obj, layer, X, Qf, Kf, Vf, A_h, W_oh, scaling, tok_idx, C_acc, chunk=256):
    """Add this sequence's contribution (tokens tok_idx) to C_acc [(d_h^2),(d^2)].

    X [d,L]; Qf,Kf,Vf [d_h,L] FP (with bias); A_h [L,L]; W_oh [d, d_h]; tok_idx [T].
    """
    d, L = X.shape
    d_h = Kf.shape[0]
    dev = X.device
    T = tok_idx.numel()
    if obj == "full":
        G_v = Vf.T @ (W_oh.T @ W_oh) @ Vf                  # [L,L] value-output metric
    else:
        Wp = _pair_weights(obj, A_h, L, dev)               # [t,u]
    for c0 in range(0, T, chunk):
        idx = tok_idx[c0:c0 + chunk]
        if layer == Q_NAME:
            # Cc_t = x_t x_t^T (rank-1), R_t = M_t
            xt = X[:, idx].T                                # [T, d]
            Ccvec = (xt[:, :, None] * xt[:, None, :]).reshape(len(idx), d * d)
            if obj == "full":
                p = A_h[idx]                                # [T, L] rows = query t
                KJ = Kf[None] * p[:, None, :] - (Kf @ p.T).T[:, :, None] * p[:, None, :]  # [T,d_h,L]
                R = KJ @ G_v @ KJ.transpose(1, 2)           # [T,d_h,d_h]
            else:
                w = Wp[idx]                                 # [T, L] weights over keys u
                R = (Kf[None] * w[:, None, :]) @ Kf.T       # [T,d_h,d_h]
            Rvec = R.reshape(len(idx), d_h * d_h)
        else:
            # k_proj: R_t = q_t q_t^T (rank-1), Cc_t = X J-form X^T
            qt = Qf[:, idx].T                               # [T, d_h]
            Rvec = (qt[:, :, None] * qt[:, None, :]).reshape(len(idx), d_h * d_h)
            if obj == "full":
                p = A_h[idx]
                # delta_t[u] = q_t^T dW x_u ; output change = W_o V J_t delta_t
                # -> Cc_t = X J_t G_v J_t X^T
                XJ = X[None] * p[:, None, :] - (X @ p.T).T[:, :, None] * p[:, None, :]  # [T,d,L]
                Cc = XJ @ G_v @ XJ.transpose(1, 2)          # [T,d,d]
            else:
                w = Wp[idx]
                Cc = (X[None] * w[:, None, :]) @ X.T        # [T,d,d]
            Ccvec = Cc.reshape(len(idx), d * d)
        C_acc.addmm_(Rvec.T, Ccvec)                          # [(d_h^2),(d^2)]
    return C_acc


# ---------------------------------------------------------------------------
# objective evaluation (held-out), for any dW -- all five at once
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_objectives(layer, dW_h, X, Qf, Kf, Vf, A_h, W_oh):
    """dW_h [d_h,d]. Returns dict obj -> value for this sequence and head."""
    L = X.shape[-1]
    dev = X.device
    if layer == Q_NAME:
        Z = Kf.T @ (dW_h @ X)                               # Z[u,t] = k_u^T dW x_t
    else:
        Z = (Qf.T @ dW_h @ X).T                             # Z[u,t] = q_t^T dW x_u
    Z2 = Z ** 2
    tri = torch.ones(L, L, device=dev).tril(0)              # [u,t] u<=t
    W_p = A_h.T                                             # [u,t]
    out = {
        "boa": Z2.sum().item(),
        "mask": (Z2 * tri).sum().item(),
        "p": (Z2 * W_p).sum().item(),
        "jac": (Z2 * W_p * (1 - W_p)).sum().item(),
    }
    # full: for each query t, delta_t = Z[:,t] (u<=t enforced by p_t = 0 above diag)
    P = A_h                                                 # [t,u]
    Jz = P * Z.T - P * (P * Z.T).sum(1, keepdim=True)       # [t,u] = J_t delta_t
    out_v = Jz @ Vf.T                                       # [t,d_h]
    out["full"] = (out_v @ W_oh.T).pow(2).sum().item()
    return out


# ---------------------------------------------------------------------------
# glue: build + solve one layer in one block
# ---------------------------------------------------------------------------
@torch.no_grad()
def solve_layer_dense(obj, layer, wrapper, blockdata, n_heads, d_h, quantizer_cfg,
                      act_order_col, act_order_row, tokens_per_seq, gen, log):
    """Quantise q_proj/k_proj with the dense solver for objective `obj`.

    `blockdata` holds FP per-sequence X, Q, K, V, attention probs and W_o.
    Returns the fake-quantised weight [n_heads*d_h, d] in ORIGINAL coordinates.
    """
    from quantizers.utils import reorder_col, reorder_row
    dev = wrapper.layer.weight.device
    W = wrapper.layer.weight.data.clone().float()
    H_col = wrapper.H_col.clone()
    H_row = wrapper.H_row.clone()
    d = W.shape[-1]

    # --- identical preprocessing to BoA.preprocess (dead neurons + damping) ---
    W, H_col = filter_dead_neuron(W, H_col, replace=wrapper.hyperparams['replace'], apply_damping=True)
    H_row = damping(H_row)
    if H_col.dim() == 2:
        H_col = H_col.unsqueeze(0)
    W = W.view(n_heads, d_h, d)
    scale, zero = wrapper.quantizer.find_params_H(W, H_col, search=True)   # [H,d_h,1]
    maxq = wrapper.quantizer.maxq

    # --- act-order permutations, taken from the BoA factors for every arm -----
    invperm_col = invperm_row = None
    if act_order_col:
        W, H_col, invperm_col = reorder_col(W, H_col, )
        perm_col = torch.argsort(invperm_col, dim=-1)       # [1,d]
    if act_order_row:
        W, H_row, scale, zero, invperm_row = reorder_row(W, H_row, scale, zero)
        perm_row = torch.argsort(invperm_row, dim=-1)       # [H,d_h]

    Qout = torch.zeros_like(W)
    n = d_h * d
    for h in range(n_heads):
        Hc = H_col[0] if H_col.shape[0] == 1 else H_col[h]
        U_given = None
        if obj == "boa":
            # Control arm. chol(kron(Hr,Hc)^-1) == kron(U_row, U_col) exactly, so use
            # BoA's own small factors: this is the numerically faithful route and
            # makes the gate a test of the ELIMINATION, not of a 49k x 49k Cholesky.
            from quantizers.utils import get_cholesky_of_inverse
            U_col = get_cholesky_of_inverse(Hc.unsqueeze(0).clone())[0]
            U_row = get_cholesky_of_inverse(H_row[h].unsqueeze(0).clone())[0]
            U_given = torch.kron(U_row, U_col)
            if h == 0 and log is not None:
                # How far off is a dense factorisation route? Decides the precision
                # the non-control objectives need. Done one route at a time with
                # explicit frees: each route peaks at ~3 dense matrices.
                errs = {}
                for tag, dt in (("fp32", torch.float32), ("fp64", torch.float64)):
                    Hd = kron_rowmajor(H_row[h], Hc).to(dt)
                    Lc = torch.linalg.cholesky(Hd); del Hd
                    Hi = torch.cholesky_inverse(Lc); del Lc
                    Ud = torch.linalg.cholesky(Hi, upper=True).float(); del Hi
                    errs[tag] = ((Ud - U_given).norm() / U_given.norm()).item()
                    del Ud
                    torch.cuda.empty_cache()
                log(f"      [diag] |U_dense - kron(U_row,U_col)|/|U|: fp32 route {errs['fp32']:.3e}, fp64 route {errs['fp64']:.3e}")
            H = None
            if h == 0 and log is not None:
                gate64 = _fp64_gate(wrapper, W[h], Hc, H_row[h], scale[h], zero[h], maxq, d, log)
        else:
            C = torch.zeros(d_h * d_h, d * d, device=dev)
            for s, bd in enumerate(blockdata):
                L = bd["X"].shape[-1]
                idx = torch.randperm(L, generator=gen)[:tokens_per_seq].to(dev)
                accumulate_dense(obj, layer, bd["X"], bd["Q"][h], bd["K"][h], bd["V"][h],
                                 bd["A"][h].float(), bd["W_o"][:, h * d_h:(h + 1) * d_h],
                                 bd["scaling"], idx, C)
            H = _perm_to_rowmajor(C, d_h, d)
            del C
            torch.cuda.empty_cache()
            H.add_(H.T.clone()).mul_(0.5)          # symmetrise (one transient copy)
            # permute into the act-order coordinates of W
            if act_order_col or act_order_row:
                pr = perm_row[h] if act_order_row else torch.arange(d_h, device=dev)
                pc = perm_col[0] if act_order_col else torch.arange(d, device=dev)
                pidx = (pr[:, None] * d + pc[None, :]).reshape(-1)
                H = H[pidx][:, pidx]
            H.diagonal().add_(0.01 * H.diagonal().mean())    # BoA damping, 1% mean diag
        scale_vec = scale[h].expand(d_h, d).reshape(-1)
        zero_vec = zero[h].expand(d_h, d).reshape(-1)
        # blocksize = d: lazy batches aligned to rows, so the summation order matches
        # boa()'s row-by-row structure (within-row immediate, cross-row after the row).
        w_vec = W[h].reshape(-1)
        if H is not None:
            holder = [H.double()]; del H
            U_given = chol_of_inverse_dense(holder)
        q = dense_gptq(w_vec, None, scale_vec, zero_vec, maxq, U=U_given, blocksize=d)
        del U_given
        Qout[h] = q.view(d_h, d)
        torch.cuda.empty_cache()
        log(f"      head {h} done")

    solve_layer_dense.last_gate64 = locals().get("gate64")
    from quantizers.utils import reverse_reorder_col, reverse_reorder_row
    if act_order_row:
        Qout = reverse_reorder_row(Qout, invperm_row)
    if act_order_col:
        Qout = reverse_reorder_col(Qout, invperm_col)
    return Qout.reshape(n_heads * d_h, d)


@torch.no_grad()
def _fp64_gate(wrapper, W_h, Hc, Hr, scale_h, zero_h, maxq, d, log):
    """Exact-arithmetic correctness gate for the dense elimination.

    Runs BoA's own boa() and dense_gptq on IDENTICAL fp64 inputs (one head, no
    act-order). In fp64 the near-boundary rounding flips that make two fp32
    orderings diverge essentially vanish, so agreement here proves the dense
    machinery (indexing, row-major ordering, kron(U_row,U_col), scale
    broadcasting); the fp32 discrepancy reported alongside is then pure
    summation-order noise, which is also what BoA vs any other fp32 ordering of
    itself would show.
    """
    from quantizers.utils import get_cholesky_of_inverse
    W64 = W_h.double()
    Hc64, Hr64 = Hc.double().clone(), Hr.double().clone()
    sc64, ze64 = scale_h.double(), zero_h.double()
    # BoA reference, fp64, one head (boa() expects leading head axis)
    Q_boa = wrapper.boa(W64[None].clone(), Hc64[None].clone(), Hr64[None].clone(),
                        sc64[None], ze64[None])[0]
    U_col = get_cholesky_of_inverse(Hc64[None].clone())[0]
    U_row = get_cholesky_of_inverse(Hr64[None].clone())[0]
    U = torch.kron(U_row, U_col)
    d_h = W64.shape[0]
    q = dense_gptq(W64.reshape(-1), None, sc64.expand(d_h, d).reshape(-1),
                   ze64.expand(d_h, d).reshape(-1), maxq, U=U, blocksize=d)
    del U
    Q_dense = q.view(d_h, d)
    diff = (Q_dense - Q_boa).abs()
    rel = (diff.norm() / Q_boa.norm()).item()
    n_flip = (diff > 0.5 * sc64.abs()).sum().item()
    log(f"      [GATE fp64, head 0] |Q_dense - Q_boa|/|Q_boa| = {rel:.3e}, level flips = {n_flip}")
    return {"fp64_rel_err": rel, "fp64_n_flips": n_flip}
