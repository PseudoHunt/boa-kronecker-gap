import argparse
from pathlib import Path

def get_boa_arguments(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument("--cache_dir", type=str, default='cache')
    parser.add_argument("--print_memory_usage", action='store_true')
    
    ## Model
    parser.add_argument("--llm_path", type=str, default='facebook/opt-125m')
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--eval_fp", action='store_true', help='Whether to evaluate the original fp model performance')
    parser.add_argument("--attn_impl", type=str, default='eager', choices=['eager', 'sdpa', 'auto'],
                        help="Attention implementation for OPT. Default 'eager' so that the "
                             "output_attentions=True pass used by --block_v stays CAUSAL; "
                             "'sdpa' reproduces the (non-causal, incorrect) upstream behaviour "
                             "under transformers>=4.53.")
    
    ## Calib. Data
    parser.add_argument('--calib_data', type=str, default="wikitext2", choices=["c4", "wikitext2"])
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--seqlen', type=int, default=2048, help='Length of input sequences')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')

    ## Quant. Configs.
    parser.add_argument('--w_bits', type=int, default=2)
    parser.add_argument('--w_sym', action="store_true")
    
    ## BoA Options
    parser.add_argument('--qparam_comput', type=str, default='Hessian', choices=['MinMax', 'MMSE', 'Hessian'], help="How to determine Quant. Params")
    parser.add_argument('--block_v', action="store_true", help="Whether to apply block-wise objective for the value projection. In memory-limited cases, we can significantly reduce memory by de-activating this option, but at the expense of a slight performance degradation.")
    parser.add_argument('--act_order_col', action='store_true', help='Whether to reorder columns based on column-wise Hessian diagonals')
    parser.add_argument('--act_order_row', action='store_true', help='Whether to reorder rows based on row-wise Hessian diagonals')
    parser.add_argument('--rsq_weights', action='store_true', help="RSQ attention-concentration token weights (arXiv 2503.01820, eq. 4) folded into H_col of q_proj/k_proj: the 'boa+rsq' arm. With --rsq_col_only it becomes the one-sided 'rsq-col' baseline.")
    parser.add_argument('--rsq_col_only', action='store_true', help='With --rsq_weights: drop H_row for q/k (one-sided GPTQ with the RSQ-weighted H_col). This is the published RSQ baseline.')
    parser.add_argument('--rsq_all_layers', action='store_true', help='With --rsq_weights: also weight H_col of v/out/fc1/fc2, as RSQ does for every weight in a block.')
    parser.add_argument('--rsq_min_value', type=float, default=0.005, help='RSQ r_min (their default 0.005; r_max = 1).')
    parser.add_argument('--row_metric_fc1', action='store_true', help="Phase 4: give fc1 the ReLU-gated output metric W2^T W2 * E[d d^T] (Kronecker level) and route it through the two-sided boa() solver. Released code uses an identity output metric for fc1.")
    parser.add_argument('--row_metric_fc1_groups', type=int, default=48, help='Block-diagonal approximation of the fc1 row metric: number of contiguous row groups (48 -> 64-row groups, same per-group cost as an attention head).')
    parser.add_argument('--row_metric_v', action='store_true', help="Use the paper's exact value-projection row Hessian W_out,h^T W_out,h (eq. 9), which the released code omits. Routes v_proj through the two-sided boa() solver instead of gptq().")

    parser.add_argument('--qk_quantK', action='store_true', help="Quantize k_proj BEFORE q_proj and rebuild q_proj's output metric from the QUANTIZED key's post-RoPE covariance. BoA measures E[K K^T] on the FP key, but inference forms logits against the quantized K; this closes that mismatch.")

    parser.add_argument('--q_centered', action='store_true', help="q_proj's row metric = CENTRED post-RoPE key covariance (second moment minus the mean outer product), back-rotated as usual. The exact softmax Jacobian is Cov_p(k), so a constant offset in k -- mostly the k_proj bias -- is invisible to attention; BoA's uncentred metric spends its budget there.")

    parser.add_argument('--q_identity', action='store_true', help="Control: q_proj's row metric = I, reducing its two-sided solve to plain per-row GPTQ. Tests whether BoA's q_proj metric buys anything at all.")

    parser.add_argument('--replace', type=float, default=1, help='Value to be replaced for the Hessian diagonal elements corresponding to dead neurons')

    ## Diagnostics for the Kronecker-gap study (arXiv 2406.13474 follow-up).
    ## These MUST NOT alter quantization numerics -- see tests/test_byte_identical.py.
    parser.add_argument('--dump_deltas', action='store_true', help='Dump per-layer quantization error dW = Q - W (and the original W) for later Hessian-gap analysis.')
    parser.add_argument('--dump_dir', type=str, default=None, help='Destination for --dump_deltas. Default: <cache_dir>/deltas/<config tag>.')
    parser.add_argument('--phase1', action='store_true', help='Run the Kronecker-gap diagnostic (EK-FAC eigenvalue fields) alongside quantization.')
    parser.add_argument('--phase1_dir', type=str, default='results/phase1', help='Output directory for the --phase1 per-block JSONs.')
    parser.add_argument('--phase1_ncalib', type=int, default=96, help='Sequences used to BUILD the eigenvalue fields; the rest are held out (Phase 1 only, rule 5.5).')
    parser.add_argument('--phase1_no_attn', action='store_true', help='Skip the attention-weighted (G3) variants, which need the [L,L] attention maps.')
    parser.add_argument('--phase4', action='store_true', help='Run the fc1/ReLU (G5) Kronecker-gap diagnostic alongside quantization.')
    parser.add_argument('--phase4_dir', type=str, default='results/phase4', help='Output directory for the --phase4 per-block JSONs.')
    parser.add_argument('--phase4_tokens_per_seq', type=int, default=128, help='Tokens subsampled per sequence for the EK-FAC field in --phase4.')
    parser.add_argument('--dense_arm', type=str, default='none', choices=['none', 'boa', 'mask', 'p', 'jac', 'full'], help="Repurposed Phase 3: quantize q/k of --dense_blocks with a DENSE GPTQ solve of this objective. 'boa' is the correctness gate (must reproduce boa()).")
    parser.add_argument('--dense_blocks', type=str, default='0,5,11', help='Comma-separated block indices for --dense_arm / --obj_eval.')
    parser.add_argument('--dense_nsamples', type=int, default=32, help='Calibration sequences used to build the dense Hessians.')
    parser.add_argument('--dense_tokens_per_seq', type=int, default=256, help='Query tokens subsampled per sequence for the dense Hessians.')
    parser.add_argument('--dense_dir', type=str, default='results/phase3', help='Output directory for dense-arm / objective-eval JSONs.')
    parser.add_argument('--obj_eval', action='store_true', help='Evaluate all five q/k objectives on held-out data for --dense_blocks, whatever solver produced the weights (the objective-transfer matrix).')
    parser.add_argument('--heldout_nsamples', type=int, default=32, help='Held-out sequences (a separate calibration draw, seed 1000+seed) for --obj_eval.')
    
    # LM Eval Arguments
    parser.add_argument("--lm_eval", action="store_true", help="Evaluate the model on LM Eval tasks.")
    parser.add_argument('--tasks', nargs='+', default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada_openai", "lambada_standard", "openbookqa", "boolq"])
    parser.add_argument('--lm_eval_batch_size', type=int, default=16, help='Batch size for evaluating with lm eval harness.')
    
    args = parser.parse_args()

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    if args.tokenizer_path is None:
        args.tokenizer_path = args.llm_path
    args.llm_name = args.tokenizer_path.split('/')[-1]
    args.llm_type = args.llm_name.split('-')[0]

    args.replace = 1 / args.seqlen

    return args


def get_boa_weight_quant_infos(args):
    qconfigs = {
        "w_bits": args.w_bits,
        "w_sym": args.w_sym,
    }
    boa_opts = {
        "qparam_comput": args.qparam_comput,
        "block_v": args.block_v,
        'act_order_col': args.act_order_col, 
        'act_order_row': args.act_order_row, 
        'row_metric_v': args.row_metric_v, 'row_metric_fc1': args.row_metric_fc1, 'row_metric_fc1_groups': args.row_metric_fc1_groups,
        'qk_quant_k': args.qk_quantK,
        'q_centered': args.q_centered, 'q_identity': args.q_identity,
        'rsq_weights': args.rsq_weights, 'rsq_col_only': args.rsq_col_only,
        'rsq_all_layers': args.rsq_all_layers, 'rsq_min_value': args.rsq_min_value,
    }
    hyperparams = {"replace": args.replace}
    
    return qconfigs, boa_opts, hyperparams