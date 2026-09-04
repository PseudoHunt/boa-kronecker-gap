import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/home/boa-kronecker-gap/results/length"
ppl = json.load(open(os.path.join(D, "phaseB_ppl2k.json")))
wt  = json.load(open(os.path.join(D, "phaseB_long.json")))     # wikitext2 (earlier run)
pg  = json.load(open(os.path.join(D, "phaseB_pg19.json")))
ARMS = ["two-sided-2048", "one-sided", "two-sided-Lext", "two-sided-longcalib"]
BK = ["0-2048", "2048-4096", "4096-8192", "8192-16384", "16384-32768"]

def tbl(src, ds, key, fmt="{:.4f}"):
    out = ["| arm | " + " | ".join(BK) + " | slope |", "|---|" + "---|" * (len(BK) + 1)]
    for a in ARMS:
        if a not in src or ds not in src[a]: continue
        r = src[a][ds]
        sl = r[BK[-1]]["ratio_to_fp"] / r[BK[0]]["ratio_to_fp"]
        out.append(f"| {a} | " + " | ".join(fmt.format(r[b][key]) for b in BK) + f" | {sl:.4f} |")
    return out

L = ["# Phase B — long-document eval on PG19, against wikitext2\n",
     "Qwen2.5-0.5B W3 seed 0. Arms differ only in q/k H_out. PG19 test via the "
     "emozilla parquet mirror, 16 books, first 32768 tokens of each. proof-pile "
     "skipped: its HF loader errored, and the brief allowed at most five minutes.\n"]

fp = pg["FP"]["pg19"]; b0 = fp[BK[0]]["nll"]
worst = max(fp[b]["nll"] / b0 for b in BK)
L += ["## Gate 0 on PG19 — is FP flat to 32k?\n", "| bucket | FP nll | vs 0-2k |", "|---|---|---|"]
for b in BK:
    L.append(f"| {b} | {fp[b]['nll']:.4f} | {fp[b]['nll']/b0:.3f}x |")
L += ["", f"**Gate 0: {'PASS' if worst < 1.15 else 'CAP REQUIRED'}** — worst bucket {worst:.3f}x the 0-2k loss.\n"]

for nm, src, ds in (("PG19 (16 books)", pg, "pg19"), ("wikitext2 (9 seqs)", wt, "wikitext2")):
    L += [f"## Loss ratio to FP — {nm}\n"] + tbl(src, ds, "ratio_to_fp") + [""]
for nm, src, ds in (("PG19", pg, "pg19"), ("wikitext2", wt, "wikitext2")):
    L += [f"## KL(FP || arm) — {nm}\n"] + tbl(src, ds, "kl") + [""]

L += ["## Standard 2k PPL\n", "| arm | wiki2 | c4-new |", "|---|---|---|"]
for k in ["FP"] + ARMS:
    if k in ppl: L.append(f"| {k} | {ppl[k]['wikitext2']} | {ppl[k]['c4-new']} |")
L.append("")

# pre-registered reading
def order_ok(src, ds):
    r = {a: src[a][ds][BK[-1]]["ratio_to_fp"] for a in ARMS if a in src}
    return r["one-sided"] < r["two-sided-Lext"] < r["two-sided-2048"], r
ok_pg, r_pg = order_ok(pg, "pg19"); ok_wt, r_wt = order_ok(wt, "wikitext2")
gap_pg = r_pg["two-sided-2048"] - r_pg["one-sided"]; gap_wt = r_wt["two-sided-2048"] - r_wt["one-sided"]
L += ["## Pre-registered reading\n",
      "The finding stands only if the 16-32k ordering `one-sided < Lext < two-sided-2048` "
      "reproduces on PG19 AND the arm1-vs-arm2 gap is at least as large as on wikitext2.\n",
      "| corpus | one-sided | Lext | two-sided-2048 | ordering holds | arm1-arm2 gap |",
      "|---|---|---|---|---|---|"]
for nm, ok, r, g in (("PG19", ok_pg, r_pg, gap_pg), ("wikitext2", ok_wt, r_wt, gap_wt)):
    L.append(f"| {nm} | {r['one-sided']:.4f} | {r['two-sided-Lext']:.4f} | "
             f"{r['two-sided-2048']:.4f} | {'yes' if ok else 'NO'} | {g:+.4f} |")
verdict = ok_pg and (gap_pg >= gap_wt)
L += ["", f"**Verdict: {'HOLDS' if verdict else 'DOES NOT HOLD'}** — ordering on PG19 "
      f"{'reproduces' if ok_pg else 'does NOT reproduce'}; gap {gap_pg:+.4f} vs {gap_wt:+.4f} on "
      f"wikitext2 ({'>=' if gap_pg >= gap_wt else '<'} required).\n"]
open(os.path.join(D, "phaseB_pg19.md"), "w").write("\n".join(L) + "\n")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for i, (nm, src, ds) in enumerate((("PG19", pg, "pg19"), ("wikitext2", wt, "wikitext2"))):
    for a in ARMS:
        if a not in src or ds not in src[a]: continue
        ax[i].plot(range(len(BK)), [src[a][ds][b]["ratio_to_fp"] for b in BK], marker="o", label=a)
    ax[i].set_xticks(range(len(BK))); ax[i].set_xticklabels(BK, rotation=25, fontsize=8)
    ax[i].set_title(nm); ax[i].grid(alpha=0.3)
ax[0].set_ylabel("loss ratio to FP"); ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(D, "phaseB_pg19.png"), dpi=120)
print("\n".join(L))
