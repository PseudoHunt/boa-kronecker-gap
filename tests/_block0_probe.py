"""Quantize block 0 of OPT-125m with whatever BoA checkout is passed via --repo.

Used by test_byte_identical.py to prove that the diagnostic patches leave the
default quantization path bit-for-bit unchanged (engineering rule 5.1). This file
must not import anything from `diag/`, so that it runs unchanged against the
pristine upstream checkout.
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llm_path", default="/home/models/opt-125m")
    ap.add_argument("--w_bits", type=int, default=3)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default="/home/BOA/cache")
    ap.add_argument("--act_order_col", action="store_true")
    ap.add_argument("--act_order_row", action="store_true")
    ap.add_argument("--attn_impl", default="sdpa",
                    help="Attention impl. The byte-identical test drives BOTH sides with "
                         "'sdpa' (upstream's implicit default) so it isolates the "
                         "instrumentation from the intentional causal-attention fix.")
    args_probe = ap.parse_args()

    sys.path.insert(0, args_probe.repo)

    import torch
    from types import SimpleNamespace
    from utils.model_utils import get_model, get_head_info, get_transformer_blocks, cache_first_transformer_input
    from utils.data_utils import get_calib_data
    from utils.utils import find_layers
    from quantizers.boa import BoA
    from quantizers.minmax import MinMaxQuantizer
    from quantize import compute_Hessian

    args = SimpleNamespace(
        llm_path=args_probe.llm_path, tokenizer_path=args_probe.llm_path,
        llm_name="opt-125m", llm_type="opt", cache_dir=args_probe.cache_dir,
        calib_data="wikitext2", nsamples=args_probe.nsamples, seqlen=args_probe.seqlen,
        seed=args_probe.seed, print_memory_usage=False,
    )

    import inspect
    if "attn_implementation" in inspect.signature(get_model).parameters:
        llm = get_model(args.llm_path, attn_implementation=args_probe.attn_impl)
    else:  # pristine upstream checkout: no such parameter, transformers picks sdpa
        llm = get_model(args.llm_path)
    llm.seqlen = args.seqlen
    llm.eval()
    calib_data = get_calib_data(args)

    qconfigs = {"w_bits": args_probe.w_bits, "w_sym": False}
    boa_opts = {"qparam_comput": "Hessian", "block_v": True,
                "act_order_col": args_probe.act_order_col,
                "act_order_row": args_probe.act_order_row}
    hyperparams = {"replace": 1 / args.seqlen}

    llm.config.use_cache = False
    quant_inps, block_kwargs = cache_first_transformer_input(llm, calib_data)
    n_heads, n_kv_heads, head_dim = get_head_info(llm)

    blocks = get_transformer_blocks(llm)
    block = blocks[0].to("cuda")
    fp_layers = find_layers(block)

    wrappers = {}
    for name, fp_layer in fp_layers.items():
        wrappers[name] = BoA(fp_layer, boa_opts, hyperparams)
        wrappers[name].quantizer = MinMaxQuantizer()
        wrappers[name].quantizer.configure(qconfigs["w_bits"], per_channel=True,
                                           sym=qconfigs["w_sym"], mse=False)
        wrappers[name].quantizer.find_params(wrappers[name].layer.weight.data)

    compute_Hessian(block, n_heads, n_kv_heads, head_dim, wrappers,
                    quant_inps, block_kwargs, boa_opts["block_v"], None)

    for name in fp_layers:
        wrappers[name].quant(False)
        wrappers[name].free()

    state = {name: fp_layers[name].weight.data.detach().cpu().clone() for name in fp_layers}
    torch.save(state, args_probe.out)
    print(f"[probe] repo={args_probe.repo} wrote {args_probe.out} ({len(state)} layers)")


if __name__ == "__main__":
    main()
