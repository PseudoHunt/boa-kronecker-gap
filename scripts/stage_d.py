"""Stage D driver: per-block softmax gap on the FP model, split-half.

  BOA_MODEL=/home/models/qwen2.5-0.5b BOA_NSEQ=16 BOA_OUT=... python scripts/stage_d.py

Runs on the FP model only -- no quantization, no dW. Streams attention per
sequence and per head (never caches [n_heads, L, L] for all sequences).
"""
import json, os, sys, time
import torch

sys.path.insert(0, "/home/boa-kronecker-gap")
from diag.kron_gap import _eigh_desc, structural_metrics
from diag.softmax_gap import (VARIANTS, attn_probs, block_fields, kish_n_eff,
                              split_half_correct)
from quantize import QKV_NAMES, compute_Hessian, get_rotary_matrix
from quantizers.boa import BoA
from quantizers.minmax import MinMaxQuantizer
from utils.model_utils import (cache_first_transformer_input, get_head_info,
                               get_model, get_rotary_emb, get_transformer_blocks)
from utils.utils import find_layers

MODEL   = os.environ.get("BOA_MODEL", "/home/models/qwen2.5-0.5b")
CALIB   = os.environ.get("BOA_CALIB", "/home/jl_fs/calib/calib_qwen2.5_wikitext2_128_2048_0.cache")
NSEQ    = int(os.environ.get("BOA_NSEQ", "16"))      # total; split into two halves
SEQLEN  = int(os.environ.get("BOA_SEQLEN", "2048"))
NBLOCK  = int(os.environ.get("BOA_NBLOCK", "0"))     # 0 = all
OUT     = os.environ.get("BOA_OUT", "/home/boa-kronecker-gap/results/qwen05b/diag/softmax_gap.json")
LAYERS  = os.environ.get("BOA_LAYERS", "q_proj,k_proj").split(",")


def _heads_of(t, n_heads, d_h):
    """Normalise a captured activation to [H, L, d_h]."""
    t = t.detach().float()
    if t.dim() == 4:                      # [B, H, L, d_h]
        return t[0]
    t = t.reshape(-1, t.shape[-1])        # [L, H*d_h]
    return t.view(-1, n_heads, d_h).transpose(0, 1).contiguous()


@torch.no_grad()
def main():
    dev = "cuda"
    calib = torch.load(CALIB)[:NSEQ]
    llm = get_model(MODEL); llm.seqlen = SEQLEN; llm.eval(); llm.config.use_cache = False
    n_heads, n_kv, d_h = get_head_info(llm)
    scaling = d_h ** -0.5
    n_shared = n_heads // n_kv
    quant_inps, block_kwargs = cache_first_transformer_input(llm, calib)
    rot_emb = get_rotary_emb(llm)
    rot_mat = (get_rotary_matrix(rot_emb, llm.config, block_kwargs["position_ids"].cpu())
               if rot_emb is not None else None)
    rot = rot_mat.squeeze(1).to(dev).double() if rot_mat is not None else None
    blocks = get_transformer_blocks(llm)
    nb = NBLOCK or len(blocks)
    S = len(quant_inps); half = S // 2
    print(f"model={MODEL} heads={n_heads}/{n_kv} d_h={d_h} seqs={S} (halves {half}+{S-half}) "
          f"blocks={nb} rope={'yes' if rot is not None else 'no'}", flush=True)

    rows = []
    for bi in range(nb):
        t0 = time.time()
        blk = blocks[bi].to(dev)
        layers = find_layers(blk)
        wr = {}
        for nm, lay in layers.items():
            w = BoA(lay, {"qparam_comput": "Hessian", "block_v": True, "act_order_col": False,
                          "act_order_row": False, "row_metric_v": False,
                          "row_metric_fc1": False}, {"replace": 1 / SEQLEN})
            w.quantizer = MinMaxQuantizer()
            w.quantizer.configure(3, per_channel=True, sym=False, mse=False)
            w.quantizer.find_params(w.layer.weight.data)
            wr[nm] = w
        compute_Hessian(blk, n_heads, n_kv, d_h, wr, quant_inps, block_kwargs, True, rot_mat)
        H_col = wr[QKV_NAMES["query"]].H_col.double()
        if H_col.dim() == 3:
            H_col = H_col[0]
        _, U = _eigh_desc(H_col)                                    # [d, d]
        Vq = torch.stack([_eigh_desc(wr[QKV_NAMES["query"]].H_row.double()[h])[1]
                          for h in range(n_heads)])                 # [H, d_h, d_h]
        Vk = torch.stack([_eigh_desc(wr[QKV_NAMES["key"]].H_row.double()[g])[1]
                          for g in range(n_kv)])                    # [n_kv, d_h, d_h]
        d = U.shape[0]
        del wr

        # capture X, post-RoPE Q/K per sequence
        cap = {}
        hs = [layers[QKV_NAMES["query"]].register_forward_hook(
                  lambda m, i, o: cap.__setitem__("X", i[0].detach()))]
        if rot is not None:
            hs += [blk.self_attn.rot_out_Q.register_forward_hook(lambda m, i, o: cap.__setitem__("Q", o)),
                   blk.self_attn.rot_out_K.register_forward_hook(lambda m, i, o: cap.__setitem__("K", o))]
        else:
            hs += [layers[QKV_NAMES["query"]].register_forward_hook(lambda m, i, o: cap.__setitem__("Q", o)),
                   layers[QKV_NAMES["key"]].register_forward_hook(lambda m, i, o: cap.__setitem__("K", o))]
        kw = {k: v for k, v in block_kwargs.items() if k != "output_attentions"}

        acc = {lyr: [{"lam": {}, "e": 0.0, "f": 0.0, "n": 0} for _ in range(2)] for lyr in LAYERS}
        neff = []
        for s in range(S):
            blk(quant_inps[s].unsqueeze(0), **kw)
            X = cap["X"].detach().float().reshape(-1, d).T                 # [d, L]
            P2 = (U.T @ X.double()) ** 2                                   # [d, L]
            Q = _heads_of(cap["Q"], n_heads, d_h).double()                 # [n_heads, L, d_h]
            K = _heads_of(cap["K"], n_kv, d_h).double()                    # [n_kv, L, d_h]
            Kx = K.repeat_interleave(n_shared, 0) if n_kv != n_heads else K
            hlf = 0 if s < half else 1
            for lyr in LAYERS:
                nH = n_heads if lyr == "q_proj" else n_kv
                lamsum = acc[lyr][hlf]["lam"]; 
                for hh in range(nH):
                    if lyr == "q_proj":
                        A = attn_probs(Q[hh], Kx[hh], scaling)
                        lam, f, e = block_fields(P2, Kx[hh], Vq[hh], rot, A, False)
                        if bi == 0 and s == 0:
                            neff.append(kish_n_eff(A * (1 - A)))
                    else:
                        # kv head hh: SUM over the query heads that read it
                        lam, f, e = None, 0.0, 0.0
                        for hq in range(hh * n_shared, (hh + 1) * n_shared):
                            A = attn_probs(Q[hq], Kx[hq], scaling)
                            l2, f2, e2 = block_fields(P2, Q[hq], Vk[hh], rot, A, True)
                            if lam is None:
                                lam, f, e = {k: v.clone() for k, v in l2.items()}, f2.clone(), e2
                            else:
                                for k in lam: lam[k] += l2[k]
                                f = f + f2
                    for k, v in lam.items():
                        key = (k, hh)
                        lamsum[key] = lamsum.get(key, 0) + v
                    acc[lyr][hlf]["f"] = acc[lyr][hlf].get("f", 0)
                    acc[lyr][hlf].setdefault("fv", {})
                    acc[lyr][hlf]["fv"][hh] = acc[lyr][hlf]["fv"].get(hh, 0) + f
                    acc[lyr][hlf].setdefault("ev", {})
                    acc[lyr][hlf]["ev"][hh] = acc[lyr][hlf]["ev"].get(hh, 0) + e
                acc[lyr][hlf]["n"] += 1
            del X, P2, Q, K, Kx
        for h in hs: h.remove()

        for lyr in LAYERS:
            nH = n_heads if lyr == "q_proj" else n_kv
            per_half = []
            for hlf in (0, 1):
                a = acc[lyr][hlf]; n = max(a["n"], 1)
                lam_boa = torch.stack([torch.outer(a["ev"][h] / n, a["fv"][h] / n) for h in range(nH)])
                fields = {}
                for v in VARIANTS:
                    if (v, 0) not in a["lam"]:
                        continue
                    fields[v] = torch.stack([a["lam"][(v, h)] / n for h in range(nH)])
                per_half.append((lam_boa, fields))
            (boaA, fA), (boaB, fB) = per_half
            rec = {"block": bi, "layer": lyr}
            for v in fA:
                mA = structural_metrics(fA[v], boaA)["rel_fro"]
                mB = structural_metrics(fB[v], boaB)["rel_fro"]
                noise = structural_metrics(fA[v], fB[v])["rel_fro"]        # A vs B: sampling floor
                obs = ((mA + mB) / 2)
                corr = torch.tensor([split_half_correct(o.item(), nz.item())
                                     for o, nz in zip(obs, noise)])
                rec[f"{v}_rel_fro"] = round(obs.mean().item(), 6)
                rec[f"{v}_rel_fro_splithalf_noise"] = round(noise.mean().item(), 6)
                rec[f"{v}_rel_fro_corrected"] = round(corr.mean().item(), 6)
            rows.append(rec)
            print(f"  block {bi:2d} {lyr:7s} " + "  ".join(
                f"{v}={rec.get(v+'_rel_fro_corrected', float('nan')):.4f}" for v in VARIANTS), flush=True)
        if neff:
            rows[-1]["kish_n_eff_head_mean"] = round(sum(neff) / len(neff), 2)
        blocks[bi] = blk.cpu(); torch.cuda.empty_cache()
        print(f"  block {bi} done in {time.time()-t0:.1f}s", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    summ = {}
    for lyr in LAYERS:
        for v in VARIANTS:
            k = f"{v}_rel_fro_corrected"
            vals = [r[k] for r in rows if r["layer"] == lyr and k in r]
            if vals:
                summ[f"{lyr}.{v}"] = {"mean": round(sum(vals) / len(vals), 4),
                                      "min": round(min(vals), 4), "max": round(max(vals), 4)}
    json.dump({"model": MODEL, "n_seq": S, "seqlen": SEQLEN, "summary": summ,
               "per_block": rows}, open(OUT, "w"), indent=2)
    print(json.dumps(summ, indent=2)); print("wrote", OUT)


if __name__ == "__main__":
    main()
