"""Aggregate Phase 4 (fc1 / ReLU) per-block JSONs into a table, plot and verdict."""
import glob, json, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(indir="results/phase4", outdir="results/phase4"):
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(indir, "block*.json")))]
    if not rows:
        raise SystemExit(f"no block JSONs in {indir}")
    hdr = (f"{'blk':>3} {'relu%':>6} | {'id/T':>6} {'kron/T':>7} {'ek/T':>6} | "
           f"{'id~kron relF':>12} {'massoff':>7} | {'ek~kron relF':>12} {'null':>7} {'massoff':>7} | "
           f"{'sal kron-vs-id':>14} {'ek-vs-kron':>10}")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        si, se, sal = r["struct_id_vs_kron"], r["struct_ek_vs_kron"], r["saliency"]
        lines.append(
            f"{r['block']:>3} {100*r['relu_active_frac']:>6.1f} | {r['ratio_id']:>6.3f} {r['ratio_kron']:>7.3f} {r['ratio_ek']:>6.3f} | "
            f"{si['rel_fro'][0]:>12.3f} {si['mass_off'][0]:>7.3f} | {se['rel_fro'][0]:>12.4f} {r['null_rel_fro_mean']:>7.4f} {se['mass_off'][0]:>7.3f} | "
            f"{sal['kron_vs_id']['top5pct_overlap']:>14.3f} {sal['ek_vs_kron']['top5pct_overlap']:>10.3f}")
    table = "\n".join(lines); print(table)

    med = lambda k: st.median(k(r) for r in rows)
    verdict = (
        f"identity row metric (released code):  median Pred/T = {med(lambda r: r['ratio_id']):.3f}, "
        f"median top-5% saliency overlap vs Kronecker = {med(lambda r: r['saliency']['kron_vs_id']['top5pct_overlap']):.3f}\n"
        f"Kronecker E[D G D] ('MLP-aware BoA'):  median Pred/T = {med(lambda r: r['ratio_kron']):.3f}\n"
        f"EK-FAC refit:                          median Pred/T = {med(lambda r: r['ratio_ek']):.3f}, "
        f"median ek-vs-kron shape gap = {med(lambda r: r['struct_ek_vs_kron']['rel_fro'][0]):.4f} "
        f"(null {med(lambda r: r['null_rel_fro_mean']):.4f}), "
        f"median top-5% overlap = {med(lambda r: r['saliency']['ek_vs_kron']['top5pct_overlap']):.3f}")
    print("\n" + verdict)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    b = [r["block"] for r in rows]
    ax[0].plot(b, [r["ratio_id"] for r in rows], "o-", label="identity (released)")
    ax[0].plot(b, [r["ratio_kron"] for r in rows], "s-", label="Kronecker E[DGD]")
    ax[0].plot(b, [r["ratio_ek"] for r in rows], "^-", label="EK-FAC")
    ax[0].axhline(1, color="k", ls=":"); ax[0].set_xlabel("block"); ax[0].set_ylabel("Pred / T_exact")
    ax[0].set_title("fc1: predicted vs exact MLP output error"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(b, [r["saliency"]["kron_vs_id"]["top5pct_overlap"] for r in rows], "o-", label="Kronecker vs identity")
    ax[1].plot(b, [r["saliency"]["ek_vs_kron"]["top5pct_overlap"] for r in rows], "^-", label="EK-FAC vs Kronecker")
    ax[1].axhline(0.9, color="r", ls=":"); ax[1].set_ylim(0, 1.05); ax[1].set_xlabel("block")
    ax[1].set_ylabel("top-5% saliency overlap"); ax[1].set_title("fc1: does the row metric reorder the weights?")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.savefig(os.path.join(outdir, "fc1_summary.png"), dpi=110, bbox_inches="tight")
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write("# Phase 4 - fc1 (ReLU-gated) Kronecker gap\n\n```\n" + table + "\n```\n\n```\n" + verdict + "\n```\n\n![fc1](fc1_summary.png)\n")
    print(f"\nwrote {outdir}/README.md, fc1_summary.png")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
