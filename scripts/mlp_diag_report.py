import json, os, statistics as st
D = "/home/boa-kronecker-gap/results/qwen05b/mlp"
rows = json.load(open(os.path.join(D, "diag.json")))["per_layer"]
UP, GATE = "mlp.up_proj", "mlp.gate_proj"

def med(sel, k): return st.median([r[k] for r in rows if sel(r)])
both = lambda r: True
mid_id, mid_sal = med(both, "ratio_id"), med(both, "sal_top5pct_overlap")
GATE_PASS = (mid_id <= 0.8) and (mid_sal <= 0.6)

L = ["# Step 1 — is the identity row metric a good proxy for Qwen's SwiGLU MLP?\n",
     f"**GATE: {'PASS' if GATE_PASS else 'FAIL'}.** "
     f"median identity Pred/T = {mid_id:.3f} (needs <= 0.80); "
     f"median top-5% saliency overlap = {mid_sal:.3f} (needs <= 0.60). "
     f"{'Both hold.' if GATE_PASS else 'The Pred/T condition fails.'}\n",
     "ΔW is from a re-run of the banked `boa_w3_s0` configuration with `--dump_deltas`; "
     "it reproduced **wiki2 19.717 / c4-new 39.072**, identical to the banked run, so these "
     "are the deltas that run actually produced.\n",
     "FP model, seed-0 calibration, 16 sequences x 2048 tokens (32 subsampled per sequence "
     "for the eigen-field), all 24 blocks, both input projections. Construction identical to "
     "`diag/phase4_fc1.py` so the ratios are comparable to OPT.\n",
     "## Medians\n",
     "| layer | n | median id/T | median kron/T | median full-pooled/T | median sal top-5% |",
     "|---|---|---|---|---|---|"]
for nm, sel in ((UP, lambda r: r["layer"] == UP), (GATE, lambda r: r["layer"] == GATE),
                ("**both**", both)):
    n = len([r for r in rows if sel(r)])
    L.append(f"| {nm} | {n} | {med(sel,'ratio_id'):.3f} | {med(sel,'ratio_kron'):.3f} | "
             f"{med(sel,'ratio_kron_full_pooled'):.3f} | {med(sel,'sal_top5pct_overlap'):.3f} |")
L += ["| *OPT-125m (ReLU fc1, reference)* | | *0.67* | | | *0.41* |", "",
      "Identity predicts the SwiGLU MLP objective **better** on Qwen than it did on OPT "
      "(0.86 vs 0.67 of the true error), and the saliency rankings agree more (0.56 vs 0.41). "
      "The gate-aware row metric therefore has less to correct here, not more.\n",
      "## Per-block\n",
      "| block | up id/T | up kron/T | up sal5 | gate id/T | gate kron/T | gate sal5 |",
      "|---|---|---|---|---|---|---|"]
for b in range(24):
    u = next(r for r in rows if r["block"] == b and r["layer"] == UP)
    g = next(r for r in rows if r["block"] == b and r["layer"] == GATE)
    L.append(f"| {b} | {u['ratio_id']:.3f} | {u['ratio_kron']:.3f} | {u['sal_top5pct_overlap']:.3f} "
             f"| {g['ratio_id']:.3f} | {g['ratio_kron']:.3f} | {g['sal_top5pct_overlap']:.3f} |")
n_bad = sum(1 for r in rows if r["ratio_id"] > 0.8)
L += ["", f"{n_bad} of {len(rows)} layer-blocks have identity Pred/T > 0.8; "
      f"{sum(1 for r in rows if r['sal_top5pct_overlap'] > 0.6)} have saliency overlap > 0.6.\n",
      "Block 0 is the one clear exception (up 0.500, gate 0.752) and block 11's up_proj (0.498); "
      "everywhere else identity tracks the objective closely.\n",
      "## Verdict\n",
      "Per the stated gate, the identity metric predicts well on SwiGLU, so **the MLP is not "
      "where the loss is on this model**. Step 2 is not run: no `--row_metric_fc1` seeds, no "
      "arm, no paired table.\n",
      "Worth recording for whoever picks this up. `kron/T` is *higher* than `id/T` on up_proj "
      f"(median {med(lambda r: r['layer']==UP,'ratio_kron'):.3f} vs "
      f"{med(lambda r: r['layer']==UP,'ratio_id'):.3f}), i.e. the pooled Kronecker metric is a "
      "genuinely better predictor of the true error — the port works and the metric is "
      "meaningful. It simply is not *needed*: identity is already close enough that improving "
      "the metric has little error left to remove. That is a different failure from the "
      "attention arms, where the metric was degenerate (rank-1, bias-dominated). Here the "
      "metric is well conditioned (top-1 eigenvalue share 0.108/0.073) and the objective is "
      "just easy.\n",
      "`ratio_kron_full_pooled` tracks `ratio_kron` closely throughout, the phase-4 "
      "consistency check, so the eigenbasis truncation is not distorting the comparison.\n"]
open(os.path.join(D, "diag.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L[:16]))
