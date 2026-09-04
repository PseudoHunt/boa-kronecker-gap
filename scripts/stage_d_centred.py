"""Stage D, centred form: BoA's q_proj row metric vs the EXACT softmax Jacobian.

Reports, per block, averaged over heads:
  rel_fro_centred   scale-free discrepancy between mean_t M_t and BoA's H_row,
                    where M_t = R_t^T Cov_{p_t}(k_rot) R_t   (exact Jacobian)
  rel_fro_diagonal  the same against mean_t R_t^T [sum_u p(1-p) k k^T] R_t,
                    i.e. the G123j variant, for direct comparison
  invisible_frac    share of BoA's H_row trace lying in directions the centred
                    metric annihilates -- the softmax-invisible mass
"""
import json, os, sys, time
import torch

sys.path.insert(0, "/home/boa-kronecker-gap")
from diag.softmax_gap import (attn_probs, centred_metric, invisible_mass, rel_fro_matrix)
from quantize import QKV_NAMES, compute_Hessian, get_rotary_matrix
from quantizers.boa import BoA
from quantizers.minmax import MinMaxQuantizer
from utils.model_utils import (cache_first_transformer_input, get_head_info, get_model,
                               get_rotary_emb, get_transformer_blocks)
from utils.utils import find_layers

MODEL  = os.environ.get("BOA_MODEL", "/home/models/qwen2.5-0.5b")
CALIB  = os.environ.get("BOA_CALIB", "/home/jl_fs/calib/calib_qwen2.5_wikitext2_128_2048_0.cache")
NSEQ   = int(os.environ.get("BOA_NSEQ", "8"))
SEQLEN = int(os.environ.get("BOA_SEQLEN", "2048"))
NBLOCK = int(os.environ.get("BOA_NBLOCK", "0"))
OUT    = os.environ.get("BOA_OUT", "/home/boa-kronecker-gap/results/qwen05b/diag/softmax_gap_centred.json")


def _heads(t, n, d_h):
    t = t.detach().float()
    if t.dim() == 4:
        return t[0]
    t = t.reshape(-1, t.shape[-1])
    return t.view(-1, n, d_h).transpose(0, 1).contiguous()


@torch.no_grad()
def main():
    calib = torch.load(CALIB)[:NSEQ]
    llm = get_model(MODEL); llm.seqlen = SEQLEN; llm.eval(); llm.config.use_cache = False
    n_heads, n_kv, d_h = get_head_info(llm)
    n_shared = n_heads // n_kv
    scaling = d_h ** -0.5
    qi, bk = cache_first_transformer_input(llm, calib)
    re_ = get_rotary_emb(llm)
    rm = get_rotary_matrix(re_, llm.config, bk["position_ids"].cpu()) if re_ is not None else None
    rot = rm.squeeze(1).cuda().double() if rm is not None else None
    blocks = get_transformer_blocks(llm)
    nb = NBLOCK or len(blocks)
    kw = {k: v for k, v in bk.items() if k != "output_attentions"}
    print(f"{MODEL} heads={n_heads}/{n_kv} d_h={d_h} seqs={len(qi)} blocks={nb}", flush=True)

    rows = []
    for bi in range(nb):
        t0 = time.time()
        blk = blocks[bi].cuda(); layers = find_layers(blk)
        wr = {}
        for nm, lay in layers.items():
            w = BoA(lay, {"qparam_comput": "Hessian", "block_v": True, "act_order_col": False,
                          "act_order_row": False, "row_metric_v": False, "row_metric_fc1": False},
                    {"replace": 1 / SEQLEN})
            w.quantizer = MinMaxQuantizer(); w.quantizer.configure(3, per_channel=True, sym=False, mse=False)
            w.quantizer.find_params(w.layer.weight.data); wr[nm] = w
        compute_Hessian(blk, n_heads, n_kv, d_h, wr, qi, bk, True, rm)
        H_row = wr[QKV_NAMES["query"]].H_row.double().clone()      # [n_heads, d_h, d_h]
        del wr

        cap = {}
        hs = []
        if rot is not None:
            hs += [blk.self_attn.rot_out_Q.register_forward_hook(lambda m, i, o: cap.__setitem__("Q", o)),
                   blk.self_attn.rot_out_K.register_forward_hook(lambda m, i, o: cap.__setitem__("K", o))]
        else:
            hs += [layers[QKV_NAMES["query"]].register_forward_hook(lambda m, i, o: cap.__setitem__("Q", o)),
                   layers[QKV_NAMES["key"]].register_forward_hook(lambda m, i, o: cap.__setitem__("K", o))]

        Mc = torch.zeros(n_heads, d_h, d_h, dtype=torch.float64, device="cuda")
        Md = torch.zeros_like(Mc); n_acc = 0
        for s in range(len(qi)):
            blk(qi[s].unsqueeze(0), **kw)
            Q = _heads(cap["Q"], n_heads, d_h).double()
            K = _heads(cap["K"], n_kv, d_h).double()
            Kx = K.repeat_interleave(n_shared, 0) if n_kv != n_heads else K
            for h in range(n_heads):
                A = attn_probs(Q[h], Kx[h], scaling).double()
                Mc[h] += centred_metric(Kx[h], A, rot).mean(0)
                # diagonal-Jacobian analogue, same pooling, for comparison
                Aj = (A * (1 - A))
                L = Kx[h].shape[0]
                KK = (Kx[h][:, :, None] * Kx[h][:, None, :]).reshape(L, d_h * d_h)
                Dd = (Aj @ KK).reshape(L, d_h, d_h)
                Md[h] += (rot.transpose(-1, -2) @ Dd @ rot).mean(0) if rot is not None else Dd.mean(0)
            n_acc += 1
        for hh in hs: hh.remove()
        Mc /= n_acc; Md /= n_acc

        rf_c = rel_fro_matrix(Mc, H_row)
        rf_d = rel_fro_matrix(Md, H_row)
        iv = [invisible_mass(H_row[h], Mc[h]) for h in range(n_heads)]
        def _m(k): return sum(x[k] for x in iv) / len(iv)
        rec = {"block": bi,
               "rel_fro_centred": round(rf_c.mean().item(), 4),
               "rel_fro_centred_min": round(rf_c.min().item(), 4),
               "rel_fro_centred_max": round(rf_c.max().item(), 4),
               "rel_fro_diagonal_G123j": round(rf_d.mean().item(), 4),
               "h_mass_in_weak": round(_m("h_mass_in_weak"), 4),
               "h_top1": round(_m("h_top1"), 4),
               "m_top1": round(_m("m_top1"), 4),
               "cos": round(_m("cos"), 4)}
        rows.append(rec)
        print(f"  block {bi:2d} centred={rec['rel_fro_centred']:.4f} "
              f"diagonal(G123j)={rec['rel_fro_diagonal_G123j']:.4f} "
              f"h_weak={rec['h_mass_in_weak']:.3f} h_top1={rec['h_top1']:.3f} "
              f"m_top1={rec['m_top1']:.3f} cos={rec['cos']:.3f} ({time.time()-t0:.1f}s)", flush=True)
        blocks[bi] = blk.cpu(); torch.cuda.empty_cache()

    import statistics as st
    summ = {k: {"mean": round(st.mean([r[k] for r in rows]), 4),
                "median": round(st.median([r[k] for r in rows]), 4),
                "min": round(min(r[k] for r in rows), 4),
                "max": round(max(r[k] for r in rows), 4)}
            for k in ("rel_fro_centred", "rel_fro_diagonal_G123j", "h_mass_in_weak",
                      "h_top1", "m_top1", "cos")}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"model": MODEL, "n_seq": len(qi), "seqlen": SEQLEN,
               "summary": summ, "per_block": rows}, open(OUT, "w"), indent=2)
    print(json.dumps(summ, indent=2)); print("wrote", OUT)


if __name__ == "__main__":
    main()
