import json, os, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/home/boa-kronecker-gap/results/qwen05b/ordering"
d = json.load(open(os.path.join(D, "step1.json")))
rows, PPL = d["per_layer"], d["ppl_seed0"]
ORD = ["ao_none", "ao_row", "ao_col", "ao_both"]
QK = {"self_attn.q_proj", "self_attn.k_proj"}

tot = {t: sum(r[t] for r in rows) for t in ORD}
tot_u = {t: sum(r[t + "_unweighted"] for r in rows) for t in ORD}
by_score = sorted(ORD, key=lambda t: tot[t], reverse=True)
by_ppl = sorted(ORD, key=lambda t: PPL[t], reverse=True)
match = by_score == by_ppl

# contribution split
def s(sel, t): return sum(r[t] for r in rows if sel(r))
qk = lambda r: r["layer"] in QK
rest = lambda r: r["layer"] not in QK

L = ["# Step 1 — does the tr(D) ordering score rank the act-orders like PPL?\n",
     "**GATE: FAIL.** The score inverts `ao_row` and `ao_col`.\n",
     f"Toy Kronecker pivot check PASSED: max abs pivot diff "
     f"{d['toy_check']['max_abs_pivot_diff']:.2e}, relative sum diff "
     f"{d['toy_check']['rel_sum_diff']:.2e} (tol 1e-10). "
     "So `d_ij = d_row[i]*d_col[j]` is verified; the failure is not in the factorisation.\n",
     "Qwen2.5-0.5B, FP model, seed-0 calibration (128x2048), all 24 blocks, every layer.\n",
     "## Summed scores vs banked seed-0 perplexity\n",
     "| ordering | score (weighted) | score (unweighted) | wiki2 | score rank | ppl rank |",
     "|---|---|---|---|---|---|"]
for t in sorted(ORD, key=lambda x: PPL[x], reverse=True):
    L.append(f"| {t} | {tot[t]:,.0f} | {tot_u[t]:,.0f} | {PPL[t]} | "
             f"{by_score.index(t)+1} | {by_ppl.index(t)+1} |")
L += ["",
      f"score order (worst→best): `{by_score}`",
      f"ppl   order (worst→best): `{by_ppl}`",
      f"**match: {match}** — 2 of 4 positions correct (the endpoints only).\n",
      "The score says row-ordering is worth far more than column-ordering "
      f"({tot['ao_row']:,.0f} vs {tot['ao_col']:,.0f}, a {tot['ao_col']/tot['ao_row']:.2f}x gap). "
      "Perplexity says the opposite: `ao_col` 20.293 beats `ao_row` 21.808. The unweighted "
      "variant inverts the same way, so this is not an artefact of the s_i^2 weighting.\n",
      "## Where the score mass sits\n",
      "| ordering | q/k layers | all other layers | q/k share |", "|---|---|---|---|"]
for t in ORD:
    a, b = s(qk, t), s(rest, t)
    L.append(f"| {t} | {a:,.0f} | {b:,.0f} | {a/(a+b):.3f} |")
L += ["",
      "`ao_row` only touches the two-sided q/k layers, yet it moves the total from "
      f"{tot['ao_none']:,.0f} to {tot['ao_row']:,.0f}. The score is dominated by the row "
      "factor of two layers out of seven, which is exactly the sensitivity perplexity does "
      "not share.\n",
      "## Per-block score(ao_both)/score(ao_none)\n",
      "| block | q/k | other |", "|---|---|---|"]
ratios = []
for b in range(24):
    qa = sum(r["ao_both"] for r in rows if r["block"] == b and qk(r))
    qn = sum(r["ao_none"] for r in rows if r["block"] == b and qk(r))
    oa = sum(r["ao_both"] for r in rows if r["block"] == b and rest(r))
    on = sum(r["ao_none"] for r in rows if r["block"] == b and rest(r))
    ratios.append((b, qa / qn if qn else float("nan"), oa / on if on else float("nan")))
    L.append(f"| {b} | {ratios[-1][1]:.4f} | {ratios[-1][2]:.4f} |")
L += ["",
      f"q/k mean ratio {st.mean(r[1] for r in ratios):.4f}, "
      f"other mean ratio {st.mean(r[2] for r in ratios):.4f}.\n",
      "## Verdict\n",
      "The criterion does not order the four act-orders the way measured perplexity does, so "
      "it cannot be used to select an ordering. Step 2 is not run.\n",
      "What the failure is NOT: the Kronecker pivot identity is verified to 2e-13, the "
      "orderings are BoA's own `reorder_col`/`reorder_row` on BoA's own damped Hessians, and "
      "the scales come from the same grid search BoA runs before reordering. The weighted and "
      "unweighted forms agree with each other and disagree with perplexity in the same way.\n",
      "What it plausibly IS: `tr(D)` is derived under an equal-scale approximation, and the "
      "two Kronecker factors here are wildly unequal in scale — the q/k row metric on this "
      "model is ~96% projection bias and near rank-1 (top-1 eigenvalue share up to 0.976, see "
      "`diag/bias_share.json` and `softmax_gap_centred.json`). Sorting a near-rank-1 factor by "
      "its diagonal produces a large drop in `sum_i d_row[i]` that buys little real error "
      "reduction, so the row side dominates the score while contributing less to perplexity.\n"]
open(os.path.join(D, "step1.md"), "w").write("\n".join(L) + "\n")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for sel, nm, c in ((qk, "q/k (two-sided)", "tab:red"), (rest, "other (one-sided)", "tab:blue")):
    xs = [r["block"] for r in rows if sel(r)]
    ys = [r["ao_both"] / r["ao_none"] for r in rows if sel(r)]
    ax[0].scatter(xs, ys, s=18, alpha=0.7, label=nm, color=c)
ax[0].set_xlabel("block index"); ax[0].set_ylabel("score(ao_both)/score(ao_none)")
ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_title("per-layer score reduction")
ax[0].grid(alpha=0.3)
xs = [PPL[t] for t in ORD]; ys = [tot[t] for t in ORD]
ax[1].scatter(xs, ys, s=60)
for t in ORD:
    ax[1].annotate(t, (PPL[t], tot[t]), textcoords="offset points", xytext=(5, 5), fontsize=9)
ax[1].set_xlabel("wiki2 ppl (seed 0)"); ax[1].set_ylabel("summed score")
ax[1].set_title("score vs ppl — monotone would pass"); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(D, "step1_scatter.png"), dpi=120)
print("\n".join(L[:14])); print("...wrote step1.md + step1_scatter.png")
