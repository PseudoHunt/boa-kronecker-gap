"""Phase 4: the ReLU-gated MLP (fc1) -- gap G5.

OPT's MLP is  y = W2 relu(W1 x + b1) + b2.  Perturbing W1 by dW and holding the ReLU
pattern d_t = 1[W1 x_t + b1 > 0] fixed (first order), the output error is

    T(dW) = sum_t || W2 ( d_t * (dW x_t) ) ||^2
          = sum_t vec(dW)^T [ (x_t x_t^T)  (x)  (D_t G D_t) ] vec(dW),   G = W2^T W2

which is an exact Kronecker product PER TOKEN and, as for attention, not after
summing. Three levels of Hessian are compared, all in the eigenbasis (U, V) of the
pooled factors  A_bar = E_t[x x^T]  and  B_bar = E_t[D_t G D_t] = G * E_t[d d^T]:

    identity  (released code)  : Lam_id  [a,b] = lam_a * mean(mu)     (scaled I)
    Kronecker ("MLP-aware BoA"): Lam_kron[a,b] = lam_a * mu_b
    EK-FAC                     : Lam_ek  [a,b] = E_t[ e_t[a] f_t[b] ]

    e_t[a] = (U^T x_t)[a]^2            f_t[b] = || W2 (d_t * v_b) ||^2

E_t[e_t] = lam, E_t[f_t] = mu exactly, so Lam_ek - Lam_kron = Cov_t(e, f): the same
structure as Phase 1's G1, with the ReLU mask now making the per-token row factor
strongly input dependent.

Cost: f_t for every b is the column norms of  (W2 * d_t) @ V  -- one [768x3072]x
[3072x3072] matmul per token, batched. All tokens are used for T, A_bar, B_bar; a
fixed subsample is used for the EK field (and T is also reported on that subsample
so Pred/T compares like with like).
"""
import json
import os

import torch

from diag.dump_utils import git_commit
from diag.kron_gap import _eigh_desc, structural_metrics, saliency_comparison
from diag.vec_conventions import quad_eig, quad_kron

FC1, FC2 = "fc1", "fc2"


class Phase4Collector:
    def __init__(self, out_dir, tokens_per_seq=128, seed=0, batch_tokens=32):
        self.out_dir = out_dir
        self.tokens_per_seq = tokens_per_seq
        self.seed = seed
        self.batch_tokens = batch_tokens
        os.makedirs(out_dir, exist_ok=True)
        self.state = {}

    # ------------------------------------------------------------------ stage A
    @torch.no_grad()
    def on_block_hessians(self, block, block_idx, wrappers, quant_inps, block_kwargs):
        fc1, fc2 = block.fc1, block.fc2
        cap = {}
        h = fc1.register_forward_hook(lambda m, i, o: cap.__setitem__("X", i[0].detach()))
        Xs = []
        kw = dict(block_kwargs); kw["output_attentions"] = False
        for s in range(len(quant_inps)):
            block(quant_inps[s].unsqueeze(0), **kw)
            Xs.append(cap["X"].reshape(-1, fc1.in_features).T.contiguous().cpu())  # [d, L] fp16
        h.remove()
        self.state[block_idx] = {
            "Xs": Xs,
            "H_col": wrappers[FC1].H_col.detach().double().clone(),
            "W1": fc1.weight.detach().float().clone(),
            "b1": fc1.bias.detach().float().clone() if fc1.bias is not None else None,
            "W2": fc2.weight.detach().float().clone(),
            "dW": None, "W_orig": None,
        }

    @torch.no_grad()
    def on_layer_quantized(self, block_idx, name, W_orig, W_quant):
        st = self.state.get(block_idx)
        if st is None or name != FC1:
            return
        st["dW"] = (W_quant.detach().float() - W_orig.detach().float())
        st["W_orig"] = W_orig.detach().float().clone()

    # ------------------------------------------------------------------ stage B
    @torch.no_grad()
    def finish_block(self, block_idx):
        st = self.state.pop(block_idx, None)
        if st is None or st["dW"] is None:
            return
        dev = st["W1"].device
        W1, b1, W2, dW = st["W1"], st["b1"], st["W2"], st["dW"]
        d_ff, d = W1.shape
        G = W2.T @ W2                                               # [d_ff, d_ff]
        S = len(st["Xs"])
        g = torch.Generator().manual_seed(self.seed)

        # ---- pass 1: exact T on all tokens, co-activation C = sum_t d d^T,
        #              and the token subsample ------------------------------------
        T_all = 0.0
        C = torch.zeros(d_ff, d_ff, device=dev)
        n_tok = 0
        sub_X, sub_D = [], []
        for s in range(S):
            X = st["Xs"][s].to(dev).float()                         # [d, L]
            L = X.shape[-1]
            pre = W1 @ X + (b1[:, None] if b1 is not None else 0)
            Dm = (pre > 0).float()                                  # [d_ff, L]
            E = W2 @ (Dm * (dW @ X))                                # [d, L]
            T_all += E.pow(2).sum().item()
            C += Dm @ Dm.T
            n_tok += L
            idx = torch.randperm(L, generator=g)[: self.tokens_per_seq]
            sub_X.append(X[:, idx].cpu()); sub_D.append(Dm[:, idx].cpu())
        Xsub = torch.cat(sub_X, dim=1).to(dev)                      # [d, N_sub]
        Dsub = torch.cat(sub_D, dim=1).to(dev)                      # [d_ff, N_sub]
        N_sub = Xsub.shape[1]

        B_bar = (G * (C / n_tok)).double()                          # E_t[D G D]
        A_bar = st["H_col"] / 2.0                                   # BoA's XXT is 2*E[xx^T]
        lam_c, U = _eigh_desc(A_bar)
        mu, V = _eigh_desc(B_bar)
        Uf, Vf = U.float(), V.float()

        # ---- pass 2: EK field on the subsample --------------------------------
        P = Uf.T @ Xsub                                             # [d, N_sub]
        Esq = (P ** 2).double()                                     # e_t   [d, N_sub]
        F = torch.empty(d_ff, N_sub, device=dev, dtype=torch.float64)
        for i in range(0, N_sub, self.batch_tokens):
            dm = Dsub[:, i:i + self.batch_tokens].T                 # [B, d_ff]
            M = (W2[None] * dm[:, None, :]) @ Vf                    # [B, d, d_ff]
            F[:, i:i + self.batch_tokens] = M.pow(2).sum(1).T.double()
        e_bar = Esq.mean(1); f_bar = F.mean(1)
        lam_ek = (Esq @ F.T) / N_sub                                # [d, d_ff]
        lam_kron = torch.outer(e_bar, f_bar)
        lam_id = torch.outer(e_bar, torch.full_like(f_bar, f_bar.mean()))

        # exact T on the subsample, and the identity/kron/ek predictions on it
        Esub = W2 @ (Dsub * (dW @ Xsub))
        T_sub = Esub.pow(2).sum().item()
        dWd = dW.double()
        pred = {k: N_sub * quad_eig(dWd, U, V, lam)
                for k, lam in (("id", lam_id), ("kron", lam_kron), ("ek", lam_ek))}
        # Reference: the prediction "MLP-aware BoA" itself would make, i.e. the
        # FULL pooled Kronecker quadratic form N_sub * tr(dW A_bar dW^T B_bar), all
        # tokens, no eigenbasis truncation. Differs from Pred_kron only through the
        # off-diagonal part of U^T A_sub U (subsample vs full-token basis).
        pred_kron_direct = N_sub * quad_kron(dWd, A_bar, B_bar)

        # permutation null over tokens
        null = []
        for _ in range(8):
            perm = torch.randperm(N_sub, generator=g).to(dev)
            ln = (Esq @ F[:, perm].T) / N_sub
            a = ln / ln.sum(); b = lam_kron / lam_kron.sum()
            null.append(((a - b).pow(2).sum().sqrt() / a.pow(2).sum().sqrt()).item())

        def _S(x):  # structural_metrics wants a leading head axis
            return structural_metrics(x[None], lam_kron[None])
        s_ek = {k: v[0].tolist() if v.dim() > 1 else v.tolist() for k, v in _S(lam_ek).items()}
        s_id = {k: v[0].tolist() if v.dim() > 1 else v.tolist() for k, v in _S(lam_id).items()}

        # damped saliency rankings for the three levels
        eps_c, eps_r = 0.01 * e_bar.mean(), 0.01 * f_bar.mean()
        incr = torch.outer(e_bar + eps_c, f_bar + eps_r) - lam_kron
        W_o = st["W_orig"].double()
        sal = {
            "kron_vs_id": saliency_comparison(W_o, U, V, lam_kron + incr, lam_id + incr),
            "ek_vs_kron": saliency_comparison(W_o, U, V, lam_kron + incr, lam_ek + incr),
            "ek_vs_id": saliency_comparison(W_o, U, V, lam_id + incr, lam_ek + incr),
        }

        rec = {
            "git_commit": git_commit(), "block": block_idx, "d": d, "d_ff": d_ff,
            "n_tokens_all": n_tok, "n_tokens_sub": N_sub,
            "relu_active_frac": (C.diagonal().sum() / (n_tok * d_ff)).item(),
            "T_all_tokens": T_all, "T_subsample": T_sub,
            "Pred_id_sub": pred["id"], "Pred_kron_sub": pred["kron"], "Pred_ek_sub": pred["ek"],
            "Pred_kron_full_pooled": pred_kron_direct,
            "ratio_kron_full_pooled": pred_kron_direct / T_sub,
            "ratio_id": pred["id"] / T_sub, "ratio_kron": pred["kron"] / T_sub,
            "ratio_ek": pred["ek"] / T_sub,
            "struct_ek_vs_kron": s_ek, "struct_id_vs_kron": s_id,
            "null_rel_fro_mean": sum(null) / len(null),
            "saliency": sal,
            "mu_top": mu[:8].tolist(), "mu_bottom": mu[-8:].tolist(),
            "lam_c_top": lam_c[:8].tolist(),
        }
        path = os.path.join(self.out_dir, f"block{block_idx:02d}.json")
        json.dump(rec, open(path, "w"), indent=2)
        print(f"[phase4] block {block_idx}: T_sub={T_sub:.4e}  id/T={rec['ratio_id']:.3f}  "
              f"kron/T={rec['ratio_kron']:.3f}  ek/T={rec['ratio_ek']:.3f}  "
              f"ek-vs-kron relF={s_ek['rel_fro'][0]:.4f} (null {rec['null_rel_fro_mean']:.4f})  "
              f"sal kron-vs-id top5={sal['kron_vs_id']['top5pct_overlap']:.3f}  "
              f"ek-vs-kron top5={sal['ek_vs_kron']['top5pct_overlap']:.3f}")
