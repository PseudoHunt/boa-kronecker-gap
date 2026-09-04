"""Phase B eval: per-position loss and KL(FP||arm) at 32k, plus standard 2k PPL.

Logits at 32k would be 32768 x 151936 (~10 GB fp16), so the final hidden states are
cached per sequence (56 MB) and the lm_head is applied in position chunks. FP is run
once; each arm is then compared against the cached FP hidden states.
"""
import json, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/boa-kronecker-gap")
from utils.model_utils import get_model

MODEL = "/home/models/qwen2.5-0.5b"
CKPT = {
    "two-sided-2048": "/home/jl_fs/ckpt/phaseB/twosided2048.pt",
    "one-sided": "/home/jl_fs/ckpt/phaseB/onesided.pt",
    "two-sided-Lext": "/home/jl_fs/ckpt/phaseB/lext32k.pt",
    "two-sided-longcalib": "/home/jl_fs/ckpt/phaseB/longcalib.pt",
}
SEQLEN = 32768
NDOC = int(os.environ.get("BOA_NDOC", "16"))
CHUNK = 512
BUCKETS = [(0, 2048), (2048, 4096), (4096, 8192), (8192, 16384), (16384, 32768)]
OUT = "/home/boa-kronecker-gap/results/length"


def build_sequences(tok):
    """32k-token sequences from PG19 test (>=32k tokens) and concatenated wikitext2."""
    from datasets import load_from_disk, load_dataset
    seqs = {}
    ds = load_from_disk("/home/jl_fs/pg19_test")
    pg = []
    for r in ds:
        ids = tok(r["text"], return_tensors="pt").input_ids[0]
        if ids.numel() >= SEQLEN:
            pg.append(ids[:SEQLEN])
        if len(pg) >= NDOC:
            break
    seqs["pg19"] = pg
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(wt["text"]), return_tensors="pt").input_ids[0]
    n = min(NDOC, ids.numel() // SEQLEN)
    seqs["wikitext2"] = [ids[i * SEQLEN:(i + 1) * SEQLEN] for i in range(n)]
    return seqs


@torch.no_grad()
def hidden_states(llm, ids):
    """Final pre-lm_head hidden states for one 32k sequence, [L, d] fp16 on CPU."""
    out = llm.model(ids.unsqueeze(0).cuda(), use_cache=False)
    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return h[0].half().cpu()


@torch.no_grad()
def bucket_stats(h_fp, h_arm, ids, W_lm):
    """Per-bucket mean NLL and mean KL(FP||arm). Positions predict the NEXT token."""
    L = ids.numel()
    tgt = ids[1:].cuda()
    acc = {b: {"nll_fp": 0.0, "nll_arm": 0.0, "kl": 0.0, "n": 0} for b in range(len(BUCKETS))}
    for s in range(0, L - 1, CHUNK):
        e = min(s + CHUNK, L - 1)
        lf = (h_fp[s:e].cuda().float() @ W_lm.T)
        la = (h_arm[s:e].cuda().float() @ W_lm.T) if h_arm is not None else None
        t = tgt[s:e]
        nll_f = F.cross_entropy(lf, t, reduction="none")
        if la is not None:
            nll_a = F.cross_entropy(la, t, reduction="none")
            kl = F.kl_div(F.log_softmax(la, -1), F.log_softmax(lf, -1),
                          reduction="none", log_target=True).sum(-1)
        pos = torch.arange(s, e, device="cuda")
        for bi, (lo, hi) in enumerate(BUCKETS):
            m = (pos >= lo) & (pos < hi)
            if not m.any():
                continue
            a = acc[bi]; a["n"] += int(m.sum())
            a["nll_fp"] += nll_f[m].sum().item()
            if la is not None:
                a["nll_arm"] += nll_a[m].sum().item(); a["kl"] += kl[m].sum().item()
        del lf, la
    return acc


def merge(dst, src):
    for b, v in src.items():
        for k in v:
            dst.setdefault(b, {kk: 0.0 for kk in v})[k] += v[k]


@torch.no_grad()
def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    seqs = build_sequences(tok)
    print({k: len(v) for k, v in seqs.items()}, flush=True)

    llm = get_model(MODEL); llm.eval().cuda()
    W_lm = llm.lm_head.weight.data.float()
    fp_h = {ds: [hidden_states(llm, s) for s in ss] for ds, ss in seqs.items()}
    print("FP hidden states cached", flush=True)

    res = {}
    fp_acc = {ds: {} for ds in seqs}
    for ds in seqs:
        for i, s in enumerate(seqs[ds]):
            merge(fp_acc[ds], bucket_stats(fp_h[ds][i], None, s, W_lm))
    res["FP"] = {ds: {f"{BUCKETS[b][0]}-{BUCKETS[b][1]}":
                      {"nll": v["nll_fp"] / max(v["n"], 1), "n": v["n"]}
                      for b, v in fp_acc[ds].items()} for ds in seqs}
    print("FP:", json.dumps(res["FP"], indent=1)[:400], flush=True)

    for name, path in CKPT.items():
        if not os.path.exists(path):
            print(f"[skip] {name}: no checkpoint", flush=True); continue
        t0 = time.time()
        sd = torch.load(path, map_location="cpu")
        llm.load_state_dict(sd, strict=True); llm.eval().cuda(); del sd
        arm = {}
        for ds in seqs:
            acc = {}
            for i, s in enumerate(seqs[ds]):
                merge(acc, bucket_stats(fp_h[ds][i], hidden_states(llm, s), s, W_lm))
            arm[ds] = {f"{BUCKETS[b][0]}-{BUCKETS[b][1]}": {
                "nll": v["nll_arm"] / max(v["n"], 1),
                "nll_fp": v["nll_fp"] / max(v["n"], 1),
                "ratio_to_fp": (v["nll_arm"] / max(v["n"], 1)) / max(v["nll_fp"] / max(v["n"], 1), 1e-12),
                "kl": v["kl"] / max(v["n"], 1), "n": v["n"]} for b, v in acc.items()}
        res[name] = arm
        print(f"[{name}] {time.time()-t0:.0f}s "
              + " ".join(f"{k}:{v['ratio_to_fp']:.4f}" for k, v in arm['pg19'].items()), flush=True)
        json.dump(res, open(os.path.join(OUT, "phaseB_long.json"), "w"), indent=2)
    json.dump(res, open(os.path.join(OUT, "phaseB_long.json"), "w"), indent=2)
    print("wrote phaseB_long.json")


if __name__ == "__main__":
    main()
