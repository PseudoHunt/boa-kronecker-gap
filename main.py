import time
from contextlib import redirect_stdout
import io
from utils.model_utils import get_model
from utils.data_utils import get_calib_data
from utils.eval_utils import evaluate
from utils.process_args import get_boa_arguments, get_boa_weight_quant_infos
from quantize import boa_fwrd

if __name__ == '__main__':
    args = get_boa_arguments()
    
    # load model
    with redirect_stdout(io.StringIO()) as f:
        llm = get_model(args.llm_path, attn_implementation=args.attn_impl)
    llm.seqlen = args.seqlen
    llm.eval()

    # evaluate the fp model performance
    if args.eval_fp:
        results = evaluate(llm, args)
        print(results)
        exit(0)

    # load calib. data
    calib_data = get_calib_data(args)

    # quantize
    qconfigs, boa_opts, hyperparams = get_boa_weight_quant_infos(args)
    print("Start quantization")
    tick = time.time()
    boa_fwrd(llm, calib_data, qconfigs, boa_opts, hyperparams, args)
    process_time = round(time.time() - tick, 3)
    print(f"Quantization processing time: {process_time}")

    if args.save_ckpt:
        import os as _os, torch as _torch
        _os.makedirs(_os.path.dirname(args.save_ckpt), exist_ok=True)
        _torch.save({k: v.cpu() for k, v in llm.state_dict().items()}, args.save_ckpt)
        print(f"[save_ckpt] wrote {args.save_ckpt}")
    
    # evaluate
    print(args)
    results = evaluate(llm, args)
    results['time'] = process_time
    print(results)