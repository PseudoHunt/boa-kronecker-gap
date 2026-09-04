"""--qk_quantK: q_proj's row metric must be re-measured through the QUANTIZED key.

Two properties, on a real Qwen2.5-0.5B block (14 query heads over 2 kv heads, RoPE):

  1. CONSISTENCY -- with k_proj left at FP, requantized_key_row_metric() must
     reproduce the H_row that compute_Hessian() already assigns to q_proj. That
     pins the hook target, the GQA kv->query expansion and the RoPE back-rotation
     to the existing path rather than to my reading of it.

  2. NOT A NO-OP -- perturb k_proj and the metric must move. Without this a silent
     no-op would look exactly like a null result in the decision table.

    python3 tests/test_qk_quantK.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from quantize import (QKV_NAMES, compute_Hessian, get_rotary_matrix,
                      requantized_key_row_metric)
from quantizers.boa import BoA
from quantizers.minmax import MinMaxQuantizer
from utils.model_utils import (get_model, get_head_info, get_rotary_emb,
                               get_transformer_blocks, cache_first_transformer_input)
from utils.utils import find_layers

MODEL = "/home/models/qwen2.5-0.5b"
NSAMP, SEQLEN = 2, 256


def main():
    torch.manual_seed(0)
    llm = get_model(MODEL)
    llm.seqlen = SEQLEN
    llm.eval()
    llm.config.use_cache = False          # boa_fwrd does this before caching inputs;
                                          # without it the KV cache grows across calls
    n_heads, n_kv_heads, head_dim = get_head_info(llm)
    print(f"  model: {n_heads} query heads / {n_kv_heads} kv heads, d_h={head_dim}")
    assert n_kv_heads < n_heads, "expected GQA"

    data = [(torch.randint(0, llm.config.vocab_size, (1, SEQLEN)),) for _ in range(NSAMP)]
    quant_inps, block_kwargs = cache_first_transformer_input(llm, data)

    rotary_emb = get_rotary_emb(llm)
    rotary_matrix = (get_rotary_matrix(rotary_emb, llm.config, block_kwargs["position_ids"].cpu())
                     if rotary_emb is not None else None)
    print(f"  RoPE matrix: {'present' if rotary_matrix is not None else 'none'}")

    blk = get_transformer_blocks(llm)[0].cuda()
    opts = {"qparam_comput": "Hessian", "block_v": True, "act_order_col": False,
            "act_order_row": False, "row_metric_v": False, "row_metric_fc1": False,
            "qk_quant_k": True}
    hyper = {"replace": 1 / SEQLEN}

    layers = find_layers(blk)
    wrappers = {}
    for nm, lay in layers.items():
        w = BoA(lay, opts, hyper)
        w.quantizer = MinMaxQuantizer()
        w.quantizer.configure(3, per_channel=True, sym=False, mse=False)
        w.quantizer.find_params(w.layer.weight.data)
        wrappers[nm] = w

    compute_Hessian(blk, n_heads, n_kv_heads, head_dim, wrappers, quant_inps,
                    block_kwargs, True, rotary_matrix)
    ref = wrappers[QKV_NAMES["query"]].H_row.detach().double().clone()
    print(f"  compute_Hessian q_proj H_row: {tuple(ref.shape)}")
    assert ref.shape == (n_heads, head_dim, head_dim), tuple(ref.shape)

    # 1. same measurement, k_proj untouched
    got = requantized_key_row_metric(blk, n_heads, n_kv_heads, head_dim,
                                     quant_inps, block_kwargs, rotary_matrix).double()
    rel = (got - ref).norm() / ref.norm()
    print(f"  FP k_proj: rel_fro(requantized, compute_Hessian) = {rel:.3e}")
    assert rel < 1e-6, f"does not reproduce the existing q_proj metric (rel {rel:.3e})"

    # 2. the metric must genuinely track K.
    #
    # Perturbing W_k alone is a WEAK test on Qwen: k_proj carries a bias, BoA hooks
    # the layer OUTPUT, so H_row is built from biased keys and E[K K^T] is dominated
    # by the constant rank-1 b_k b_k^T. Measured on wikitext2 calibration data,
    # ||b b^T||_F / ||E[K K^T]||_F is 0.95-1.03 for every block of Qwen2.5-0.5B
    # (block 0: ||b||=367 vs ||W x||=7.8). So halving W_k moves the metric by ~1e-2
    # and a no-op would sail past a naive threshold.
    #
    # Zero the bias and the weight dependence is exposed: halving W_k must then
    # quarter E[K K^T], i.e. rel_fro ~ 0.75.
    kp = layers[QKV_NAMES["key"]]
    saved_w = kp.weight.data.clone()
    saved_b = kp.bias.data.clone() if kp.bias is not None else None

    kp.weight.data = saved_w * 0.5
    moved = requantized_key_row_metric(blk, n_heads, n_kv_heads, head_dim,
                                       quant_inps, block_kwargs, rotary_matrix).double()
    rel_biased = ((moved - ref).norm() / ref.norm()).item()
    print(f"  halved W_k, bias intact : rel_fro = {rel_biased:.3e}  (bias-dominated)")

    if saved_b is not None:
        kp.weight.data = saved_w
        kp.bias.data = torch.zeros_like(saved_b)
        base0 = requantized_key_row_metric(blk, n_heads, n_kv_heads, head_dim,
                                           quant_inps, block_kwargs, rotary_matrix).double()
        kp.weight.data = saved_w * 0.5
        half0 = requantized_key_row_metric(blk, n_heads, n_kv_heads, head_dim,
                                           quant_inps, block_kwargs, rotary_matrix).double()
        rel0 = ((half0 - base0).norm() / base0.norm()).item()
        print(f"  halved W_k, bias zeroed : rel_fro = {rel0:.3e}  (expect ~0.75)")
        assert rel0 > 0.5, f"metric does not track K -- --qk_quantK is a no-op ({rel0:.3e})"
        kp.bias.data = saved_b
    kp.weight.data = saved_w

    print("\nQK_QUANTK TEST: PASS")


if __name__ == "__main__":
    main()
