"""Standard 2k wiki2/c4 PPL for every Phase B checkpoint, evaluated identically.

Arm 4 (long-calibration) runs at seqlen 8192, so its in-run eval is not comparable
to the banked numbers; everything is re-evaluated here at seqlen 2048.
"""
import json, os, sys
from types import SimpleNamespace
import torch

sys.path.insert(0, "/home/boa-kronecker-gap")
from utils.eval_utils import evaluate
from utils.model_utils import get_model

MODEL = "/home/models/qwen2.5-0.5b"
CKPT = {
    "FP": None,
    "two-sided-2048": "/home/jl_fs/ckpt/phaseB/twosided2048.pt",
    "one-sided": "/home/jl_fs/ckpt/phaseB/onesided.pt",
    "two-sided-Lext": "/home/jl_fs/ckpt/phaseB/lext32k.pt",
    "two-sided-longcalib": "/home/jl_fs/ckpt/phaseB/longcalib.pt",
}
OUT = "/home/boa-kronecker-gap/results/length/phaseB_ppl2k.json"


@torch.no_grad()
def main():
    args = SimpleNamespace(cache_dir="/home/jl_fs/calib", llm_type="qwen2.5", seqlen=2048,
                           tokenizer_path=MODEL, lm_eval=False, tasks=None,
                           lm_eval_batch_size=1)
    res = {}
    for name, path in CKPT.items():
        if path is not None and not os.path.exists(path):
            print(f"[skip] {name}"); continue
        llm = get_model(MODEL)
        if path:
            llm.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
        llm.seqlen = 2048; llm.eval()
        r = evaluate(llm, args)
        res[name] = {k: v for k, v in r.items() if k in ("wikitext2", "c4-new")}
        print(name, res[name], flush=True)
        del llm; torch.cuda.empty_cache()
        json.dump(res, open(OUT, "w"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
