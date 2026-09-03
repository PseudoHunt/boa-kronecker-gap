"""Aggregate the Phase 1 per-block JSONs into a summary table, plots and a verdict."""
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAYERS = ["self_attn.q_proj", "self_attn.k_proj"]
SHORT = {"self_attn.q_proj": "q_proj", "self_attn.k_proj": "k_proj"}
VARIANTS = ["G1", "G12", "G123p", "G123j"]
OBJ_FOR = {"BoA": "unmasked", "G1": "unmasked", "G12": "masked",
           "G123p": "attn_p", "G123j": "attn_j"}


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "block*.json"))):
        j = json.load(open(f))
        out[j["block"]] = j
    return out


def main(indir="results/phase1", outdir="results/phase1"):
    blocks = load(indir)
    if not blocks:
        raise SystemExit(f"no block JSONs in {indir}")
    os.makedirs(outdir, exist_ok=True)
    nb = sorted(blocks)

    rows = []
    for b in nb:
        for ln in LAYERS:
            r = blocks[b]["layers"][ln]
            row = {"block": b, "layer": SHORT[ln]}
            for v in VARIANTS:
                key = f"struct_{v}"
                if key not in r:
                    continue
                s = r[key]
                row[f"{v}_rel_fro"] = st.median(s["rel_fro"])
                row[f"{v}_mass_off"] = st.median(s["mass_off"])
                row[f"{v}_abs_log2R_med"] = st.median([abs(p[2]) for p in s["log2R_pct_massweighted"]])
                row[f"{v}_scale_ratio"] = st.median(s["scale_ratio"])
            for v in ["BoA"] + VARIANTS:
                pk, tk = f"Pred_{v}_calib", f"T_{OBJ_FOR[v]}_calib"
                if pk in r and tk in r:
                    row[f"{v}_pred_over_T"] = st.median(
                        [p / t for p, t in zip(r[pk], r[tk]) if t > 0])
                pk, tk = f"Pred_{v}_heldout", f"T_{OBJ_FOR[v]}_heldout"
                if pk in r and tk in r:
                    row[f"{v}_pred_over_T_ho"] = st.median(
                        [p / t for p, t in zip(r[pk], r[tk]) if t > 0])
            n = r.get("null_G1", {})
            if n:
                row["G1_null"] = st.median(n["rel_fro_null_mean"])
                row["G1_excess"] = st.median(n["excess_ratio"])
            sal = r.get("saliency_G1", {})
            if sal:
                row["spearman"] = st.median(sal["spearman"])
                row["top1"] = st.median(sal["top1pct_overlap"])
                row["top5"] = st.median(sal["top5pct_overlap"])
            rows.append(row)

    # ---------------- summary table ----------------
    hdr = (f"{'blk':>3} {'layer':>7} | {'G1 relF':>8} {'null':>7} {'xs':>5} "
           f"{'|log2R|':>8} {'massoff':>8} | {'BoA/T':>7} {'G1/T':>7} {'G1/T(ho)':>9} | "
           f"{'spear':>6} {'top1%':>6} {'top5%':>6}")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(
            f"{r['block']:>3} {r['layer']:>7} | {r.get('G1_rel_fro',0):>8.5f} "
            f"{r.get('G1_null',0):>7.5f} {r.get('G1_excess',0):>5.2f} "
            f"{r.get('G1_abs_log2R_med',0):>8.5f} {r.get('G1_mass_off',0):>8.5f} | "
            f"{r.get('BoA_pred_over_T',0):>7.4f} {r.get('G1_pred_over_T',0):>7.4f} "
            f"{r.get('G1_pred_over_T_ho',0):>9.4f} | "
            f"{r.get('spearman',0):>6.4f} {r.get('top1',0):>6.4f} {r.get('top5',0):>6.4f}")
    table = "\n".join(lines)
    print(table)

    # ---------------- objective-change variants ----------------
    hdr2 = (f"{'blk':>3} {'layer':>7} | " +
            " ".join(f"{v+' relF':>11}" for v in VARIANTS) + " | " +
            " ".join(f"{v+'/T':>9}" for v in VARIANTS))
    lines2 = [hdr2, "-" * len(hdr2)]
    for r in rows:
        lines2.append(f"{r['block']:>3} {r['layer']:>7} | " +
                      " ".join(f"{r.get(v+'_rel_fro',float('nan')):>11.5f}" for v in VARIANTS) +
                      " | " + " ".join(f"{r.get(v+'_pred_over_T',float('nan')):>9.4f}" for v in VARIANTS))
    table2 = "\n".join(lines2)
    print("\n" + table2)

    # ---------------- verdict ----------------
    med_log2 = st.median([r.get("G1_abs_log2R_med", 0) for r in rows])
    max_log2 = max(r.get("G1_abs_log2R_med", 0) for r in rows)
    min_top5 = min(r.get("top5", 1) for r in rows)
    med_excess = st.median([r.get("G1_excess", 0) for r in rows])
    negligible = (max_log2 < 0.3) and (min_top5 > 0.9)
    verdict = (
        f"median |log2 R| over all blocks/layers : {med_log2:.5f}\n"
        f"worst   |log2 R|                       : {max_log2:.5f}   (threshold 0.3)\n"
        f"min top-5% saliency overlap            : {min_top5:.4f}    (threshold 0.9)\n"
        f"median observed/null gap ratio         : {med_excess:.2f}     (1.0 = pure sampling noise)\n"
        f"\nSTOP/GO: G1 Kronecker gap is "
        f"{'NEGLIGIBLE -> skip the EK solver; do Phase 2a and Phase 4' if negligible else 'MATERIAL -> continue'}")
    print("\n" + verdict)

    # ---------------- plots ----------------
    def _plot(fig, name):
        p = os.path.join(outdir, name)
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {p}")

    fig, ax = plt.subplots(figsize=(7, 4))
    for ln in ["q_proj", "k_proj"]:
        sub = [r for r in rows if r["layer"] == ln]
        ax.plot([r["block"] for r in sub], [r.get("G1_rel_fro", 0) for r in sub],
                "o-", label=f"{ln} observed")
        ax.plot([r["block"] for r in sub], [r.get("G1_null", 0) for r in sub],
                "s--", alpha=.6, label=f"{ln} permutation null")
    ax.set_xlabel("block"); ax.set_ylabel("rel. Frobenius gap (shape only)")
    ax.set_title("G1 Kronecker gap vs its finite-sample noise floor")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    _plot(fig, "g1_gap_vs_null.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for v in VARIANTS:
        for ln in ["q_proj"]:
            sub = [r for r in rows if r["layer"] == ln]
            ax.plot([r["block"] for r in sub], [r.get(f"{v}_rel_fro", 0) for r in sub],
                    "o-", label=v)
    ax.set_xlabel("block"); ax.set_ylabel("rel. Frobenius gap (shape only)")
    ax.set_yscale("log"); ax.set_title("q_proj: eigenvalue-field change by variant")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    _plot(fig, "variants_rel_fro.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for ln in ["q_proj", "k_proj"]:
        sub = [r for r in rows if r["layer"] == ln]
        ax.plot([r["block"] for r in sub], [r.get("top5", 1) for r in sub], "o-", label=f"{ln} top-5%")
        ax.plot([r["block"] for r in sub], [r.get("top1", 1) for r in sub], "s--", label=f"{ln} top-1%")
    ax.axhline(0.9, color="r", ls=":", label="threshold 0.9")
    ax.set_xlabel("block"); ax.set_ylabel("saliency top-k overlap")
    ax.set_ylim(0, 1.05); ax.set_title("Does EK-FAC reorder which weights matter?")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    _plot(fig, "saliency_overlap.png")

    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write("# Phase 1 - Kronecker gap diagnostic\n\n## G1 (the faithful correction)\n\n```\n")
        f.write(table + "\n```\n\n## Objective-change variants (G2 mask, G3 softmax weighting)\n\n```\n")
        f.write(table2 + "\n```\n\n## Verdict\n\n```\n" + verdict + "\n```\n\n")
        for p in ["g1_gap_vs_null.png", "variants_rel_fro.png", "saliency_overlap.png"]:
            f.write(f"![{p}]({p})\n\n")
    json.dump(rows, open(os.path.join(outdir, "summary.json"), "w"), indent=2)
    print(f"\nwrote {outdir}/README.md and summary.json")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
