"""Step 1: is the identity row metric a good proxy for the SwiGLU MLP objective?

Same construction as diag/phase4_fc1.py (so ratios are comparable to OPT's), with
the ReLU mask replaced by the SwiGLU diagonals:
    up_proj   d_t = phi(g_t)
    gate_proj d_t = phi'(g_t) * u_t

    identity  Lam_id  [a,b] = lam_a * mean(mu)
    Kronecker Lam_kron[a,b] = lam_a * mu_b
"""
import json, os, sys, time
import torch

sys.path.insert(0, "/home/boa-kronecker-gap")
from diag.kron_gap import _eigh_desc, saliency_comparison
from diag.swiglu_metric import GATE, UP, DOWN, gate_diagonals
from diag.vec_conventions import quad_eig, quad_kron
from quantize import compute_Hessian, get_rotary_matrix
from quantizers.boa import BoA
from quantizers.minmax import MinMaxQuantizer
from utils.model_utils import (cache_first_transformer_input, get_head_info, get_model,
                               get_rotary_emb, get_transformer_blocks)
from utils.utils import find_layers

MODEL  = "/home/models/qwen2.5-0.5b"
CALIB  = "/home/jl_fs/calib/calib_qwen2.5_wikitext2_128_2048_0.cache"
DELTAS = os.environ.get("BOA_DELTAS", "/home/jl_fs/deltas/boa_w3_s0")
NSEQ   = int(os.environ.get("BOA_NSEQ", "16"))
TPS    = int(os.environ.get("BOA_TPS", "32"))       # subsample tokens per sequence
BATCH  = int(os.environ.get("BOA_BATCH", "8"))
OUT    = "/home/boa-kronecker-gap/results/qwen05b/mlp"


@torch.no_grad()
def main():
    calib = torch.load(CALIB)[:NSEQ]
    llm = get_model(MODEL); llm.seqlen = 2048; llm.eval(); llm.config.use_cache = False
    nh, nkv, dh = get_head_info(llm)
    qi, bk = cache_first_transformer_input(llm, calib)
    re_ = get_rotary_emb(llm)
    rm = get_rotary_matrix(re_, llm.config, bk["position_ids"].cpu()) if re_ is not None else None
    blocks = get_transformer_blocks(llm)
    rows = []

    NB = int(os.environ.get("BOA_NBLOCK", "0")) or len(blocks)
    for bi in range(NB):
        t0 = time.time()
        blk = blocks[bi].cuda(); layers = find_layers(blk)
        wr = {}
        for nm, lay in layers.items():
            w = BoA(lay, {"qparam_comput": "Hessian", "block_v": True, "act_order_col": False,
                          "act_order_row": False, "row_metric_v": False, "row_metric_fc1": False},
                    {"replace": 1 / 2048})
            w.quantizer = MinMaxQuantizer(); w.quantizer.configure(3, per_channel=True, sym=False, mse=False)
            w.quantizer.find_params(w.layer.weight.data); wr[nm] = w
        compute_Hessian(blk, nh, nkv, dh, wr, qi, bk, True, rm)
        H_col = wr[UP].H_col.detach().double().clone()
        if H_col.dim() == 3: H_col = H_col[0]
        del wr

        W_down = layers[DOWN].weight.data.float()                  # [d, d_ff]
        d_ff = W_down.shape[1]; d = W_down.shape[0]
        dW = {}
        for nm in (UP, GATE):
            p = os.path.join(DELTAS, f"block{bi:02d}", f"{nm}.pt")
            blob = torch.load(p, map_location="cuda")
            dW[nm] = blob["delta"].float().cuda(); W_orig = blob.get("W_orig")
            dW[nm + "_W"] = W_orig.float().cuda()

        cap = {}
        hs = [layers[UP].register_forward_hook(lambda m, i, o: cap.__setitem__("X", i[0].detach())),
              layers[GATE].register_forward_hook(lambda m, i, o: cap.__setitem__("g", o.detach())),
              layers[UP].register_forward_hook(lambda m, i, o: cap.__setitem__("u", o.detach()))]
        kw = {k: v for k, v in bk.items() if k != "output_attentions"}

        C = {UP: torch.zeros(d_ff, d_ff, device="cuda"), GATE: torch.zeros(d_ff, d_ff, device="cuda")}
        T_all = {UP: 0.0, GATE: 0.0}
        sub = {UP: [], GATE: []}; subX = []
        n_tok = 0
        g_ = torch.Generator().manual_seed(0)
        for s in range(len(qi)):
            blk(qi[s].unsqueeze(0), **kw)
            X = cap["X"].reshape(-1, d).T.contiguous().float()      # [d, L]
            gg = cap["g"].reshape(-1, d_ff).float()
            uu = cap["u"].reshape(-1, d_ff).float()
            d_up, d_gate = gate_diagonals(gg, uu)                   # [L, d_ff]
            L = X.shape[-1]; n_tok += L
            idx = torch.randperm(L, generator=g_)[:TPS]
            subX.append(X[:, idx].cpu())
            for nm, dt in ((UP, d_up), (GATE, d_gate)):
                Dm = dt.T                                           # [d_ff, L]
                E = W_down @ (Dm * (dW[nm] @ X))
                T_all[nm] += E.pow(2).sum().item()
                C[nm] += Dm @ Dm.T
                sub[nm].append(Dm[:, idx].cpu())
            del X, gg, uu, d_up, d_gate
        for h in hs: h.remove()

        Xsub = torch.cat(subX, 1).cuda()                            # [d, N]
        N = Xsub.shape[1]
        A_bar = H_col / 2.0
        lam_c, U = _eigh_desc(A_bar); Uf = U.float()
        P = Uf.T @ Xsub; Esq = (P ** 2).double()                    # [d, N]
        e_bar = Esq.mean(1)
        G = (W_down.T @ W_down)                                     # [d_ff, d_ff] fp32

        for nm in (UP, GATE):
            Dsub = torch.cat(sub[nm], 1).cuda()                     # [d_ff, N]
            B_bar = (G.double() * (C[nm].double() / n_tok))
            mu, V = _eigh_desc(B_bar); Vf = V.float()
            F = torch.empty(d_ff, N, device="cuda", dtype=torch.float64)
            for i in range(0, N, BATCH):
                dm = Dsub[:, i:i + BATCH].T                         # [B, d_ff]
                M = (W_down[None] * dm[:, None, :]) @ Vf            # [B, d, d_ff]
                F[:, i:i + BATCH] = M.pow(2).sum(1).T.double()
            f_bar = F.mean(1)
            lam_kron = torch.outer(e_bar, f_bar)
            lam_id = torch.outer(e_bar, torch.full_like(f_bar, f_bar.mean()))
            Esubm = W_down @ (Dsub * (dW[nm] @ Xsub))
            T_sub = Esubm.pow(2).sum().item()
            dWd = dW[nm].double()
            pred_id = N * quad_eig(dWd, U, V, lam_id)
            pred_kr = N * quad_eig(dWd, U, V, lam_kron)
            pred_full = N * quad_kron(dWd, A_bar, B_bar)
            eps_c, eps_r = 0.01 * e_bar.mean(), 0.01 * f_bar.mean()
            incr = torch.outer(e_bar + eps_c, f_bar + eps_r) - lam_kron
            sal = saliency_comparison(dW[nm + "_W"].double(), U, V, lam_kron + incr, lam_id + incr)
            rec = {"block": bi, "layer": nm, "n_tokens_all": n_tok, "n_tokens_sub": N,
                   "T_all_tokens": T_all[nm], "T_subsample": T_sub,
                   "ratio_id": pred_id / T_sub, "ratio_kron": pred_kr / T_sub,
                   "ratio_kron_full_pooled": pred_full / T_sub,
                   "sal_top5pct_overlap": sal["top5pct_overlap"],
                   "sal_top1pct_overlap": sal["top1pct_overlap"],
                   "sal_spearman": sal["spearman"],
                   "mu_top1_share": (mu.clamp_min(0).max() / mu.clamp_min(0).sum()).item()}
            rows.append(rec)
            print(f"  b{bi:2d} {nm:14s} id/T={rec['ratio_id']:.3f} kron/T={rec['ratio_kron']:.3f} "
                  f"full/T={rec['ratio_kron_full_pooled']:.3f} sal5={rec['sal_top5pct_overlap']:.3f}",
                  flush=True)
            del Dsub, B_bar, mu, V, Vf, F, lam_kron, lam_id
            torch.cuda.empty_cache()

        for j in range(len(qi)):
            qi[j] = blk(qi[j].unsqueeze(0), **bk)[0]
        blocks[bi] = blk.cpu(); del C, G, dW, Xsub, U, Uf, P, Esq
        torch.cuda.empty_cache()
        print(f"  block {bi} done ({time.time()-t0:.1f}s)", flush=True)

    os.makedirs(OUT, exist_ok=True)
    json.dump({"model": MODEL, "deltas": DELTAS, "n_seq": NSEQ, "per_layer": rows},
              open(os.path.join(OUT, "diag.json"), "w"), indent=2)
    print("wrote", os.path.join(OUT, "diag.json"))


if __name__ == "__main__":
    main()
