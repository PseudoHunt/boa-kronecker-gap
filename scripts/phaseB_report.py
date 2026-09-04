import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/home/boa-kronecker-gap/results/length"
ppl = json.load(open(os.path.join(D, "phaseB_ppl2k.json")))
lng = json.load(open(os.path.join(D, "phaseB_long.json")))
ARMS = ["two-sided-2048", "one-sided", "two-sided-Lext", "two-sided-longcalib"]
BK = ["0-2048", "2048-4096", "4096-8192", "8192-16384", "16384-32768"]
DS = "wikitext2"

L = ["# Phase B — does H_out's length dependence show up in loss at long context?\n",
     "Qwen2.5-0.5B, W3, seed 0. Arms differ ONLY in q/k H_out; damping, grid, ordering, "
     "calibration budget all identical to the banked `boa_w3_s0`.\n",
     "**PG19 was skipped by request** (the DeepMind loader was still fetching after 40 min), "
     "so the long-context result rests on concatenated wikitext2 alone. One corpus, not two.\n",
     "## Standard 2k PPL\n", "| arm | wiki2 | c4-new | wiki2 vs arm 1 |", "|---|---|---|---|"]
base = ppl.get("two-sided-2048", {}).get("wikitext2")
for k in ["FP"] + ARMS:
    if k not in ppl: continue
    v = ppl[k]
    d = "" if k in ("FP",) or base is None else f"{v['wikitext2']-base:+.3f}"
    L.append(f"| {k} | {v['wikitext2']} | {v['c4-new']} | {d} |")
L.append("")

# Gate 0
fp = lng["FP"][DS]
b0 = fp[BK[0]]["nll"]
L += ["## Gate 0 — is FP itself flat to 32k?\n",
      "| bucket | FP nll | vs 0-2k |", "|---|---|---|"]
for b in BK:
    L.append(f"| {b} | {fp[b]['nll']:.4f} | {fp[b]['nll']/b0:.3f}x |")
flat = max(fp[b]["nll"] / b0 for b in BK) < 1.15
L += ["", f"**Gate 0: {'PASS' if flat else 'CAP REQUIRED'}** — FP's worst bucket is "
      f"{max(fp[b]['nll']/b0 for b in BK):.3f}x its 0-2k loss.\n",
      "## Per-position loss, ratio to FP in the same bucket\n",
      "| arm | " + " | ".join(BK) + " |", "|---|" + "---|" * len(BK)]
for a in ARMS:
    if a not in lng: continue
    r = lng[a][DS]
    L.append(f"| {a} | " + " | ".join(f"{r[b]['ratio_to_fp']:.4f}" for b in BK) + " |")
L += ["", "## Per-position KL(FP || arm)\n",
      "| arm | " + " | ".join(BK) + " |", "|---|" + "---|" * len(BK)]
for a in ARMS:
    if a not in lng: continue
    r = lng[a][DS]
    L.append(f"| {a} | " + " | ".join(f"{r[b]['kl']:.4f}" for b in BK) + " |")
L.append("")

open(os.path.join(D, "phaseB.md"), "w").write("\n".join(L) + "\n")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
x = range(len(BK))
for a in ARMS:
    if a not in lng: continue
    ax[0].plot(x, [lng[a][DS][b]["ratio_to_fp"] for b in BK], marker="o", label=a)
    ax[1].plot(x, [lng[a][DS][b]["kl"] for b in BK], marker="o", label=a)
for i, (t, yl) in enumerate((("loss ratio to FP", "nll_arm / nll_FP"), ("KL(FP||arm)", "nats"))):
    ax[i].set_xticks(list(x)); ax[i].set_xticklabels(BK, rotation=25, fontsize=8)
    ax[i].set_title(t); ax[i].set_ylabel(yl); ax[i].legend(fontsize=7); ax[i].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(D, "phaseB.png"), dpi=120)
print("\n".join(L))
