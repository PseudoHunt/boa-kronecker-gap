"""How much of BoA's q/k row metric is the projection BIAS, per block?

BoA hooks the layer OUTPUT, so q_proj's row metric is E[K K^T] built from BIASED
keys (and k_proj's from biased queries). The bias is a constant that weight
quantization never touches, so whatever share of the metric it owns is inert:
--qk_quantK can only act on the remainder.

Reports, per block and per projection:
  bias_share   ||b b^T||_F / ||E[Y Y^T]||_F      (1.0 => metric is the bias alone)
  wx_share     ||E[(Wx)(Wx)^T]||_F / ||E[Y Y^T]||_F
  norm_ratio   mean_t ||W x_t|| / ||b||
Measured pre-RoPE, on real calibration data.
"""
import json, os, sys
import torch

sys.path.insert(0, "/home/boa-kronecker-gap")
from utils.model_utils import get_model, get_transformer_blocks, cache_first_transformer_input
from utils.utils import find_layers

MODEL = "/home/models/qwen2.5-0.5b"
CALIB = "/home/jl_fs/calib/calib_qwen2.5_wikitext2_128_2048_0.cache"
NSEQ = int(os.environ.get("NSEQ", "8"))
OUT = "/home/boa-kronecker-gap/results/qwen05b/diag/bias_share.json"


@torch.no_grad()
def main():
    calib = torch.load(CALIB)[:NSEQ]
    llm = get_model(MODEL); llm.seqlen = 2048; llm.eval(); llm.config.use_cache = False
    quant_inps, block_kwargs = cache_first_transformer_input(llm, calib)
    kw = {k: v for k, v in block_kwargs.items() if k != "output_attentions"}
    blocks = get_transformer_blocks(llm)

    rows = []
    for bi in range(len(blocks)):
        blk = blocks[bi].cuda()
        layers = find_layers(blk)
        cap = {}
        hs = []
        for nm in ("self_attn.q_proj", "self_attn.k_proj"):
            hs.append(layers[nm].register_forward_hook(
                lambda m, i, o, n=nm: cap.__setitem__(n, i[0].detach().float())))
        for s in range(len(quant_inps)):
            blk(quant_inps[s].unsqueeze(0), **kw)
            for nm in ("self_attn.q_proj", "self_attn.k_proj"):
                X = cap[nm].reshape(-1, cap[nm].shape[-1])
                lay = layers[nm]
                W, b = lay.weight.data.float(), lay.bias.data.float()
                Wx = X @ W.T
                Y = Wx + b
                acc = cap.setdefault("_acc", {}).setdefault((nm,), [0, 0, 0, 0.0, 0])
                acc[0] = acc[0] + Y.T @ Y
                acc[1] = acc[1] + Wx.T @ Wx
                acc[2] = acc[2] + Y.shape[0]
                acc[3] = acc[3] + Wx.norm(dim=-1).sum().item()
                acc[4] = acc[4] + Y.shape[0]
        for h in hs:
            h.remove()

        for nm in ("self_attn.q_proj", "self_attn.k_proj"):
            EYY, EWW, n, wxsum, nn_ = cap["_acc"][(nm,)]
            EYY, EWW = EYY / n, EWW / n
            b = layers[nm].bias.data.float()
            bb = torch.outer(b, b)
            rows.append({
                "block": bi, "proj": nm.split(".")[-1],
                "bias_share": round((bb.norm() / EYY.norm()).item(), 4),
                "wx_share": round((EWW.norm() / EYY.norm()).item(), 4),
                "norm_ratio_wx_over_b": round((wxsum / nn_) / b.norm().item(), 4),
            })
        cap.pop("_acc", None)
        blocks[bi] = blk.cpu(); torch.cuda.empty_cache()
        print(f"  block {bi:2d} done", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    summ = {}
    for p in ("q_proj", "k_proj"):
        v = [r["bias_share"] for r in rows if r["proj"] == p]
        summ[p] = {"bias_share_min": min(v), "bias_share_max": max(v),
                   "bias_share_mean": round(sum(v) / len(v), 4)}
    json.dump({"model": "Qwen/Qwen2.5-0.5B", "n_seq": NSEQ,
               "summary": summ, "per_block": rows}, open(OUT, "w"), indent=2)
    print(json.dumps(summ, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
