"""Serially build every calibration / test cache the concurrent runs will read.

Stage C launches ~11 jobs at once; if they all miss the same cache file they race
on torch.save and can hand each other a truncated pickle. Warming first makes
every later run a pure cache read.
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, "/home/boa-kronecker-gap")
from utils.data_utils import get_calib_data, get_testdata

BASE = dict(cache_dir="/home/jl_fs/calib", llm_type="qwen2.5", calib_data="wikitext2",
            nsamples=128, seqlen=2048, tokenizer_path="/home/models/qwen2.5-0.5b")

for s in (0, 1, 2):
    get_calib_data(SimpleNamespace(seed=s, **BASE))
    print(f"[warm] calib seed {s} ready", flush=True)
for ds in ("wikitext2", "c4-new"):
    get_testdata(ds, SimpleNamespace(seed=0, **BASE))
    print(f"[warm] testloader {ds} ready", flush=True)
print("[warm] done")
