"""Phase A: is BoA's q/k row metric position-independent, and is it length-sensitive?

Gate 1: analytic H_out(w_2048) built from the pooled PRE-RoPE covariance must match
        BoA's own H_out. Median rel_fro < 0.05.
Gate 2: rel_fro(H_out(w_L), H_out(w_2048)) at 8k/32k(/128k), median at 32k > 0.2,
        with the mass in low-frequency bands.
No quantisation anywhere.
"""
import json, os, sys, time
import torch

sys.path.insert(0, "/home/boa-kronecker-gap")
from diag.rope_length import avg_closed, band_decomposition, fejer, rel_fro
from quantize import QKV_NAMES, compute_Hessian, get_rotary_matrix
from quantizers.boa import BoA
from quantizers.minmax import MinMaxQuantizer
from utils.hessian_utils import CovarianceCollector
from utils.model_utils import (cache_first_transformer_input, get_head_info, get_model,
                               get_rotary_emb, get_transformer_blocks)
from utils.utils import find_layers
import functools

OUT = "/home/boa-kronecker-gap/results/length"
MODELS = {
    "qwen2.5-0.5b": dict(path="/home/models/qwen2.5-0.5b",
                         calib="/home/jl_fs/calib/calib_qwen2.5_wikitext2_128_2048_0.cache",
                         lengths=[8192, 32768]),
    "llama-3.2-1b": dict(path="/home/models/llama-3.2-1b",
                         calib="/home/jl_fs/calib/calib_llama_wikitext2_128_2048_0.cache",
                         lengths=[8192, 32768, 131072]),
}
L_REF = 2048
NSEQ = int(os.environ.get("BOA_NSEQ", "128"))


@torch.no_grad()
def run_model(name, cfg):
    calib = torch.load(cfg["calib"])[:NSEQ]
    llm = get_model(cfg["path"]); llm.seqlen = 2048; llm.eval(); llm.config.use_cache = False
    n_heads, n_kv, d_h = get_head_info(llm)
    n_shared = n_heads // n_kv
    qi, bk = cache_first_transformer_input(llm, calib)
    rot_emb = get_rotary_emb(llm)
    rm = get_rotary_matrix(rot_emb, llm.config, bk["position_ids"].cpu())
    theta = rot_emb.inv_freq.detach().double().cpu().clone()          # [d_h/2]
    assert theta.numel() == d_h // 2, (theta.numel(), d_h)

    # sanity: my 2x2 split-half construction must reproduce get_rotary_matrix
    t_chk = 7
    R_chk = torch.zeros(d_h, d_h, dtype=torch.float64)
    h = d_h // 2
    c, s = torch.cos(t_chk * theta), torch.sin(t_chk * theta)
    R_chk[:h, :h] = torch.diag(c); R_chk[:h, h:] = torch.diag(-s)
    R_chk[h:, :h] = torch.diag(s); R_chk[h:, h:] = torch.diag(c)
    rot_err = (R_chk - rm[t_chk, 0].double().cpu()).abs().max().item()

    blocks = get_transformer_blocks(llm)
    static_pairs = int((2047 * theta < 0.1).sum().item())
    print(f"[{name}] heads={n_heads}/{n_kv} d_h={d_h} blocks={len(blocks)} "
          f"rot_check={rot_err:.2e} static_pairs={static_pairs}/{theta.numel()}", flush=True)
    assert rot_err < 1e-6, "split-half rotation does not match get_rotary_matrix"

    rows, bias_rows = [], []
    for bi in range(len(blocks)):
        t0 = time.time()
        blk = blocks[bi].cuda(); layers = find_layers(blk)
        # pre-RoPE covariances, same normalisation as compute_cov (2 * E[x x^T])
        cq = CovarianceCollector(layers[QKV_NAMES["query"]])
        ck = CovarianceCollector(layers[QKV_NAMES["key"]])
        hq = layers[QKV_NAMES["query"]].register_forward_hook(
            functools.partial(cq.compute_cov_out_batch, n_heads=n_heads))
        hk = layers[QKV_NAMES["key"]].register_forward_hook(
            functools.partial(ck.compute_cov_out_batch, n_heads=n_kv))

        wr = {}
        for nm, lay in layers.items():
            w = BoA(lay, {"qparam_comput": "Hessian", "block_v": True, "act_order_col": False,
                          "act_order_row": False, "row_metric_v": False, "row_metric_fc1": False},
                    {"replace": 1 / 2048})
            w.quantizer = MinMaxQuantizer(); w.quantizer.configure(3, per_channel=True, sym=False, mse=False)
            w.quantizer.find_params(w.layer.weight.data); wr[nm] = w
        compute_Hessian(blk, n_heads, n_kv, d_h, wr, qi, bk, True, rm)
        hq.remove(); hk.remove()

        H_q_boa = wr[QKV_NAMES["query"]].H_row.double().cpu()          # [n_heads, d_h, d_h]
        H_k_boa = wr[QKV_NAMES["key"]].H_row.double().cpu()            # [n_kv,    d_h, d_h]
        C_k = ck.YYT.double().cpu()                                    # [n_kv,    d_h, d_h]
        C_q = cq.YYT.double().cpu()                                    # [n_heads, d_h, d_h]
        C_q_grp = C_q.reshape(n_kv, n_shared, d_h, d_h).mean(1)        # BoA's group average
        del wr, cq, ck

        # ---- Gate 1 + Gate 2 ------------------------------------------------
        for lname, Cs, H_boa, nH in (("q_proj", C_k, H_q_boa, n_heads),
                                     ("k_proj", C_q_grp, H_k_boa, n_kv)):
            for hh in range(nH):
                C = Cs[hh // n_shared] if lname == "q_proj" else Cs[hh]
                Href = avg_closed(C, theta, L_REF, transpose=(lname == "k_proj"))
                rec = {"model": name, "block": bi, "layer": lname, "head": hh,
                       "gate1_rel_fro": rel_fro(Href, H_boa[hh]).item()}
                for L in cfg["lengths"]:
                    HL = avg_closed(C, theta, L, transpose=(lname == "k_proj"))
                    rec[f"rel_fro_L{L}"] = rel_fro(HL, Href).item()
                    if L == 32768:
                        sh, _ = band_decomposition(C, theta, L_REF, L)
                        rec["bands_32k"] = sh
                    if L == 131072:
                        sh, _ = band_decomposition(C, theta, L_REF, L)
                        rec["bands_128k"] = sh
                rows.append(rec)

        # ---- Step 3: Qwen key-bias visibility --------------------------------
        kb = getattr(layers[QKV_NAMES["key"]], "bias", None)
        if kb is not None:
            b = kb.detach().double().cpu().reshape(n_kv, d_h)
            for g in range(n_kv):
                bi_sq = b[g, :d_h // 2] ** 2 + b[g, d_h // 2:] ** 2     # ||b_i||^2 per pair
                tot = bi_sq.sum().clamp_min(1e-300)
                r = {"model": name, "block": bi, "kv_head": g, "b_norm2": tot.item()}
                for L in (2048, 8192, 32768):
                    F = fejer(L, theta)
                    r[f"visible_frac_L{L}"] = (1 - (F * bi_sq).sum() / tot).item()
                    r[f"visible_frac_F2_L{L}"] = (1 - (F ** 2 * bi_sq).sum() / tot).item()
                bias_rows.append(r)

        for j in range(len(qi)):
            qi[j] = blk(qi[j].unsqueeze(0), **bk)[0]
        blocks[bi] = blk.cpu(); torch.cuda.empty_cache()
        print(f"  [{name}] block {bi:2d} ({time.time()-t0:.1f}s)", flush=True)

    return {"model": name, "n_heads": n_heads, "n_kv": n_kv, "d_h": d_h,
            "theta": theta.tolist(), "static_pairs_lt_0.1rad": static_pairs,
            "n_pairs": theta.numel(), "rot_check_max_abs": rot_err,
            "lengths": cfg["lengths"], "rows": rows, "bias_rows": bias_rows}


def main():
    out = {}
    for name, cfg in MODELS.items():
        out[name] = run_model(name, cfg)
        os.makedirs(OUT, exist_ok=True)
        json.dump(out, open(os.path.join(OUT, "phaseA.json"), "w"))
        print(f"[{name}] written", flush=True)
    print("done")


if __name__ == "__main__":
    main()
