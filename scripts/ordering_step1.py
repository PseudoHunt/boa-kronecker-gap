"""Step 1: does the tr(D) ordering score rank the four act-orders like PPL does?

FP model, seed-0 calibration. For every layer of all 24 blocks, score under
ao_none / ao_row / ao_col / ao_both, then compare the summed ranking against the
banked seed-0 perplexities.
"""
import json, os, sys, time
import torch

sys.path.insert(0, "/home/boa-kronecker-gap")
from diag.ordering_score import layer_score, verify_kron_pivots
from quantize import QKV_NAMES, compute_Hessian, get_rotary_matrix
from quantizers.boa import BoA
from quantizers.minmax import MinMaxQuantizer
from utils.model_utils import (cache_first_transformer_input, get_head_info, get_model,
                               get_rotary_emb, get_transformer_blocks)
from utils.utils import find_layers

MODEL  = "/home/models/qwen2.5-0.5b"
CALIB  = "/home/jl_fs/calib/calib_qwen2.5_wikitext2_128_2048_0.cache"
SEQLEN = 2048
OUTDIR = "/home/boa-kronecker-gap/results/qwen05b/ordering"
# banked seed-0 perplexities
PPL = {"ao_none": 22.911, "ao_row": 21.808, "ao_col": 20.293, "ao_both": 19.717}
ORDERS = {"ao_none": (False, False), "ao_row": (False, True),
          "ao_col": (True, False), "ao_both": (True, True)}
TWO_SIDED = {QKV_NAMES["query"], QKV_NAMES["key"]}


@torch.no_grad()
def main():
    toy = verify_kron_pivots(8, 4)
    print("toy kron pivot check:", json.dumps(toy), flush=True)
    assert toy["pass"], "Kronecker pivot factorisation failed the toy check"

    calib = torch.load(CALIB)
    llm = get_model(MODEL); llm.seqlen = SEQLEN; llm.eval(); llm.config.use_cache = False
    n_heads, n_kv, d_h = get_head_info(llm)
    qi, bk = cache_first_transformer_input(llm, calib)
    re_ = get_rotary_emb(llm)
    rm = get_rotary_matrix(re_, llm.config, bk["position_ids"].cpu()) if re_ is not None else None
    blocks = get_transformer_blocks(llm)
    opts = {"qparam_comput": "Hessian", "block_v": True, "act_order_col": False,
            "act_order_row": False, "row_metric_v": False, "row_metric_fc1": False}
    hyper = {"replace": 1 / SEQLEN}

    rows = []
    for bi in range(len(blocks)):
        t0 = time.time()
        blk = blocks[bi].cuda()
        layers = find_layers(blk)
        wr = {}
        for nm, lay in layers.items():
            w = BoA(lay, opts, hyper)
            w.quantizer = MinMaxQuantizer()
            w.quantizer.configure(3, per_channel=True, sym=False, mse=False)
            w.quantizer.find_params(w.layer.weight.data)
            wr[nm] = w
        compute_Hessian(blk, n_heads, n_kv, d_h, wr, qi, bk, True, rm)

        for nm in layers:
            W, H_col, H_row = wr[nm].preprocess()
            scale, _ = wr[nm].quantizer.find_params_H(W, H_col, search=True)
            rec = {"block": bi, "layer": nm, "two_sided": H_row is not None}
            for tag, (ac, ar) in ORDERS.items():
                wgt, unw = layer_score(W, H_col, H_row, scale, ac, ar)
                rec[tag] = wgt
                rec[tag + "_unweighted"] = unw
            rows.append(rec)
            wr[nm].free()

        # FP propagation: this is the FP model, so feed forward the UNquantized output
        for j in range(len(qi)):
            qi[j] = blk(qi[j].unsqueeze(0), **bk)[0]
        blocks[bi] = blk.cpu(); del wr; torch.cuda.empty_cache()
        print(f"  block {bi:2d} done ({time.time()-t0:.1f}s)", flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    json.dump({"model": MODEL, "toy_check": toy, "ppl_seed0": PPL, "per_layer": rows},
              open(os.path.join(OUTDIR, "step1.json"), "w"), indent=2)

    tot = {t: sum(r[t] for r in rows) for t in ORDERS}
    tot_u = {t: sum(r[t + "_unweighted"] for r in rows) for t in ORDERS}
    print(json.dumps({"summed_score": tot, "summed_unweighted": tot_u}, indent=2))
    by_score = sorted(ORDERS, key=lambda t: tot[t], reverse=True)
    by_ppl = sorted(ORDERS, key=lambda t: PPL[t], reverse=True)
    print("score order (worst->best):", by_score)
    print("ppl   order (worst->best):", by_ppl)
    print("RANKING MATCH:", by_score == by_ppl)


if __name__ == "__main__":
    main()
