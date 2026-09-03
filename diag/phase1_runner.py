"""Phase 1 driver: measure the Kronecker gap in BoA's attention-aware Hessians.

Wired into `boa_fwrd` behind `--phase1`; see diag/kron_gap.py for the math.

Per block the work is:

  stage A  (on_block_hessians, run right after compute_Hessian while every weight
            in the block is still FP)
      - eigendecompose the two pooled factors: H_col = U diag(lam_c) U^T (shared by
        q and k) and H_row = V diag(lam_r) V^T (per head, per layer)
      - stash the FP q/k weights and the per-sequence inputs X_s (fp16, on CPU --
        ~400 MB for 128x2048x768, freed at the end of the block)

  stage B  (finish_block, after every layer of the block is quantized, so that
            BOTH dW_q and dW_k are known and the data pass is made only once)
      - one pass over the calibration sequences accumulating, per head:
          * the four eigenvalue fields (G1 / G12 / G123p / G123j)
          * the FOUR matching ground-truth objectives, each scored against its own
            objective rather than against the unmasked one

Ground truths. With P = U^T X, R = V^T K and G = V^T dW U, and U, V orthogonal,

    Z[u, t] = R[:,u]^T G P[:,t] = k_u^T dW x_t

is the exact logit error for the pair (query t, key u). So a single Z gives all
four objectives by reweighting the same [L, L] matrix:

    T_unmasked = sum_{t,u} Z^2                     <- what BoA actually optimises
    T_masked   = sum_t sum_{u<=t} Z^2              <- G2
    T_attn_p   = sum_{t,u} p_{tu}      Z^2         <- G3, attention-probability
    T_attn_j   = sum_{t,u} p_{tu}(1-p_{tu}) Z^2    <- G3, softmax-Jacobian diagonal

Rule 5.3: the predictive metrics (1d) match normalisation exactly -- Pred_* and
T_* are both sums over the same sequences with no per-token rescaling. The
structural metrics (1c) are scale-free (both fields are normalised to unit sum and
the scale ratio is reported separately).

Rule 5.5: the 96/32 calibration/held-out split applies to Phase 1 only.
"""
import json
import math
import os

import torch

from diag.kron_gap import (VARIANTS, BlockGapAccumulator, _eigh_desc,
                           structural_metrics, permutation_null,
                           saliency_comparison)
from diag.dump_utils import git_commit
from diag.vec_conventions import quad_eig, quad_kron

Q_NAME, K_NAME = "self_attn.q_proj", "self_attn.k_proj"


def attn_probs(Q, K, scaling, neg_inf=-1e30):
    """Causal softmax attention probabilities for one head.

    Q, K: [d_h, L]. Returns [L, L] with row t = query t, column u = key u.
    Matches transformers' OPTAttention: the query is pre-scaled by head_dim**-0.5.
    """
    L = Q.shape[-1]
    logits = (Q.transpose(-1, -2) * scaling) @ K            # [L(query t), L(key u)]
    mask = torch.ones(L, L, dtype=torch.bool, device=Q.device).triu(1)
    logits = logits.masked_fill(mask, neg_inf)
    return torch.softmax(logits.float(), dim=-1)


class Phase1Collector:
    def __init__(self, out_dir, n_calib=96, want_attn=True, topk=8, seed=0,
                 max_blocks=None):
        self.out_dir = out_dir
        self.n_calib = n_calib
        self.want_attn = want_attn
        self.topk = topk
        self.seed = seed
        self.max_blocks = max_blocks
        os.makedirs(out_dir, exist_ok=True)
        self.state = {}
        self._validated = False

    # ------------------------------------------------------------------ stage A
    @torch.no_grad()
    def on_block_hessians(self, block, block_idx, wrappers, quant_inps,
                          block_kwargs, n_heads, head_dim):
        if self.max_blocks is not None and block_idx >= self.max_blocks:
            return
        d = block.self_attn.q_proj.weight.shape[-1]

        H_col = wrappers[Q_NAME].H_col.detach().double()
        H_row_q = wrappers[Q_NAME].H_row.detach().double()      # E[k k^T] per head
        H_row_k = wrappers[K_NAME].H_row.detach().double()      # E[q q^T] per head

        lam_c, U = _eigh_desc(H_col)
        Vq = torch.stack([_eigh_desc(H_row_q[h])[1] for h in range(n_heads)])
        Vk = torch.stack([_eigh_desc(H_row_k[h])[1] for h in range(n_heads)])

        # capture X_s = input to q_proj (post-LayerNorm); LayerNorm is never
        # quantized, so this is identical before and after the block is quantized.
        cap = {}
        h = block.self_attn.q_proj.register_forward_hook(
            lambda m, i, o: cap.__setitem__("X", i[0].detach()))
        Xs = []
        kw = dict(block_kwargs)
        kw["output_attentions"] = False
        A_ref = None
        for s in range(len(quant_inps)):
            if s == 0 and not self._validated:
                kw_a = dict(kw); kw_a["output_attentions"] = True
                A_ref = block(quant_inps[s].unsqueeze(0), **kw_a)[-1][0].float().cpu()
            block(quant_inps[s].unsqueeze(0), **kw)
            Xs.append(cap["X"].reshape(-1, d).T.contiguous().cpu())   # [d, L] fp16
        h.remove()

        self.state[block_idx] = {
            "U": U, "Vq": Vq, "Vk": Vk,
            "n_heads": n_heads, "d": d, "d_h": head_dim,
            "scaling": head_dim ** -0.5,
            "W_q_fp": block.self_attn.q_proj.weight.detach().float().clone(),
            "W_k_fp": block.self_attn.k_proj.weight.detach().float().clone(),
            # OPT's projections carry biases, and BoA hooks the layer OUTPUT, so
            # H_row = E[k k^T] is built from biased keys. The reconstruction must
            # match or the attention self-check below fires.
            "b_q_fp": (block.self_attn.q_proj.bias.detach().float().clone()
                       if block.self_attn.q_proj.bias is not None else None),
            "b_k_fp": (block.self_attn.k_proj.bias.detach().float().clone()
                       if block.self_attn.k_proj.bias is not None else None),
            "Xs": Xs, "dW": {},
            "A_ref": A_ref,
        }

    # ------------------------------------------------------------- record dW
    @torch.no_grad()
    def on_layer_quantized(self, block_idx, name, W_orig, W_quant):
        st = self.state.get(block_idx)
        if st is None or name not in (Q_NAME, K_NAME):
            return
        H, d, d_h = st["n_heads"], st["d"], st["d_h"]
        st["dW"][name] = (W_quant.detach().float() - W_orig.detach().float()).view(H, d_h, d)
        st.setdefault("W_orig", {})[name] = W_orig.detach().float().view(H, d_h, d)

    # ------------------------------------------------------------------ stage B
    @torch.no_grad()
    def finish_block(self, block_idx):
        st = self.state.pop(block_idx, None)
        if st is None or len(st["dW"]) < 2:
            return
        dev = st["W_q_fp"].device
        H, d, d_h = st["n_heads"], st["d"], st["d_h"]
        U, Vq, Vk = st["U"].to(dev), st["Vq"].to(dev), st["Vk"].to(dev)
        S_all = len(st["Xs"])
        n_cal = min(self.n_calib, S_all)
        Uf = U.float()

        layers = {Q_NAME: (Vq.float(), st["dW"][Q_NAME]),
                  K_NAME: (Vk.float(), st["dW"][K_NAME])}

        acc = {n: BlockGapAccumulator(d, d_h, H, dev, self.want_attn) for n in layers}
        acc_ho = {n: BlockGapAccumulator(d, d_h, H, dev, self.want_attn) for n in layers}
        OBJ = ("unmasked", "masked", "attn_p", "attn_j")
        T = {n: {o: torch.zeros(H, 2, dtype=torch.float64, device=dev) for o in OBJ}
             for n in layers}

        for s in range(S_all):
            X = st["Xs"][s].to(dev).float()                     # [d, L]
            L = X.shape[-1]
            Qf = st["W_q_fp"] @ X
            Kf = st["W_k_fp"] @ X
            if st["b_q_fp"] is not None:
                Qf = Qf + st["b_q_fp"][:, None]
            if st["b_k_fp"] is not None:
                Kf = Kf + st["b_k_fp"][:, None]
            Qf = Qf.view(H, d_h, L)
            Kf = Kf.view(H, d_h, L)
            P = Uf.T @ X                                        # [d, L]
            tri = torch.ones(L, L, dtype=torch.bool, device=dev).tril(0)  # [u,t]: u<=t

            A = torch.stack([attn_probs(Qf[h], Kf[h], st["scaling"]) for h in range(H)]) \
                if self.want_attn else None
            if A is not None and s == 0 and st.get("A_ref") is not None and not self._validated:
                self._validate_attention(st, A)

            col = 0 if s < n_cal else 1
            for name, (V, dW) in layers.items():
                src = Kf if name == Q_NAME else Qf              # q is scored against keys
                R = torch.stack([V[h].T @ src[h] for h in range(H)])   # [H, d_h, L]
                (acc if s < n_cal else acc_ho)[name].add(P, R, A, topk=self.topk)

                for h in range(H):
                    G = V[h].T @ dW[h] @ Uf                     # [d_h, d]
                    Y = G @ P                                   # [d_h, L]
                    Z = R[h].T @ Y                              # [L(u), L(t)]
                    Z2 = Z ** 2
                    T[name]["unmasked"][h, col] += Z2.sum().double()
                    T[name]["masked"][h, col] += (Z2 * tri).sum().double()
                    if A is not None:
                        W_p = A[h].T                            # [u, t]
                        T[name]["attn_p"][h, col] += (Z2 * W_p).sum().double()
                        T[name]["attn_j"][h, col] += (Z2 * W_p * (1 - W_p)).sum().double()
            del X, Qf, Kf, P, A

        results = {}
        for name, (V, dW) in layers.items():
            fields = acc[name].fields()
            fields_ho = acc_ho[name].fields()
            dWd = dW.double()
            rec = {"n_calib": n_cal, "n_heldout": S_all - n_cal}
            for o in OBJ:
                rec[f"T_{o}_calib"] = T[name][o][:, 0].cpu().tolist()
                rec[f"T_{o}_heldout"] = T[name][o][:, 1].cpu().tolist()
            for fname, lam in fields.items():
                rec[f"Pred_{fname}_calib"] = [
                    n_cal * quad_eig(dWd[h], U, V[h].double(), lam[h]) for h in range(H)]
            for fname, lam in fields_ho.items():
                rec[f"Pred_{fname}_heldout"] = [
                    (S_all - n_cal) * quad_eig(dWd[h], U, V[h].double(), lam[h])
                    for h in range(H)]
            for fname, lam in fields.items():
                if fname == "BoA":
                    continue
                m = structural_metrics(lam, fields["BoA"])
                rec[f"struct_{fname}"] = {k: v.tolist() for k, v in m.items()}
            # permutation null: destroys any real e_s/f_s coupling but keeps the
            # marginals, so it measures the finite-sample floor of the G1 gap.
            rec["null_G1"] = permutation_null(acc[name], seed=self.seed)

            # 1d "direction gap": would the EK-FAC field reorder which weights the
            # solver treats as sensitive? Damping is applied to BOTH fields as the
            # same absolute increment (see diag.kron_gap.saliency_comparison).
            e_bar = (acc[name].sum_e / acc[name].n_seq)          # [H, d]
            f_bar = (acc[name].sum_f / acc[name].n_seq)          # [H, d_h]
            W_o = st["W_orig"][name]
            sal = []
            for h in range(H):
                eps_c = 0.01 * e_bar[h].mean()
                eps_r = 0.01 * f_bar[h].mean()
                lam_boa_d = torch.outer(e_bar[h] + eps_c, f_bar[h] + eps_r)
                incr = lam_boa_d - torch.outer(e_bar[h], f_bar[h])
                sal.append(saliency_comparison(W_o[h], U, V[h].double(),
                                               lam_boa_d, fields["G1"][h] + incr))
            rec["saliency_G1"] = {k: [x[k] for x in sal] for k in sal[0]}
            results[name] = rec

        path = os.path.join(self.out_dir, f"block{block_idx:02d}.json")
        with open(path, "w") as f:
            json.dump({"git_commit": git_commit(), "block": block_idx,
                       "n_heads": H, "d": d, "d_h": d_h, "layers": results}, f, indent=2)
        print(f"[phase1] wrote {path}")

    # ------------------------------------------------------------------ checks
    def _validate_attention(self, st, A):
        """Assert our recomputed attention matches the model's own, once per run.

        Guards two things at the same time: that our causal mask and head_dim**-0.5
        scaling match OPTAttention, and that the model itself is running CAUSAL
        attention (see results/BUG_causal_attention.md -- with attn_impl=sdpa the
        reference is bidirectional and this check fires).
        """
        ref = st["A_ref"].to(A.device)
        err = (ref - A).abs().max().item()
        above = ref.triu(1).abs().max().item()
        print(f"[phase1] attention self-check: max|recomputed - model| = {err:.3e}; "
              f"model mass above diagonal = {above:.3e}")
        assert above < 1e-6, (
            "the model's own attention is NOT causal -- run with --attn_impl eager "
            "(see results/BUG_causal_attention.md)")
        assert err < 1e-2, f"recomputed attention disagrees with the model ({err:.3e})"
        self._validated = True
