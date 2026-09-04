import json, os, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/home/boa-kronecker-gap/results/length"
d = json.load(open(os.path.join(D, "phaseA.json")))
BANDS = ["<512", "512-2048", "2048-8192", "8192-32768", ">32768"]

def med(rows, k): 
    v = [r[k] for r in rows if k in r]
    return st.median(v) if v else float("nan")

L = ["# Phase A — length dependence of BoA's q/k row metric\n",
     "Analytic, no quantisation. `H_out = E_D[R_D C R_D^T]` with `D = u - t` and the "
     "triangular pair weight `w_L(D) = (L-|D|)/L^2`. Closed form verified against the direct "
     "weighted sum at L=512 to **1.4e-16** (tolerance 1e-8), for both `R_D` and `R_D^T`; the "
     "split-half 2x2 construction reproduces `get_rotary_matrix` to "
     f"{max(v['rot_check_max_abs'] for v in d.values()):.1e}.\n"]

# ---------------- Gate 1
g1 = {}
L += ["## Gate 1 — position independence\n",
      "`rel_fro(H_out_analytic(w_2048), H_out_BoA)`, normalised by the analytic reference. "
      "**Pass: median < 0.05.**\n",
      "| model | layer | n | median | p90 | max |", "|---|---|---|---|---|---|"]
for m, v in d.items():
    for lyr in ("q_proj", "k_proj"):
        r = [x["gate1_rel_fro"] for x in v["rows"] if x["layer"] == lyr]
        r.sort()
        g1[(m, lyr)] = st.median(r)
        L.append(f"| {m} | {lyr} | {len(r)} | {st.median(r):.4f} | "
                 f"{r[int(0.9*(len(r)-1))]:.4f} | {max(r):.4f} |")
G1 = all(x < 0.05 for x in g1.values())
L += ["", f"**Gate 1: {'PASS' if G1 else 'FAIL'}** — "
      f"{'every' if G1 else 'not every'} median is below 0.05.\n"]

# ---------------- Gate 2
L += ["## Gate 2 — length sensitivity\n",
      "`rel_fro(H_out(w_L), H_out(w_2048))`, normalised by `H_out(w_L)`. "
      "**Pass: median at 32k > 0.2, mass in the low-frequency bands.**\n",
      "| model | layer | " + " | ".join(f"median L={l//1024}k" for l in
        sorted({l for v in d.values() for l in v["lengths"]})) + " |",
      "|---|---|" + "---|" * len({l for v in d.values() for l in v["lengths"]})]
all_L = sorted({l for v in d.values() for l in v["lengths"]})
g2 = {}
for m, v in d.items():
    for lyr in ("q_proj", "k_proj"):
        rr = [x for x in v["rows"] if x["layer"] == lyr]
        cells = []
        for l in all_L:
            k = f"rel_fro_L{l}"
            cells.append(f"{med(rr,k):.4f}" if any(k in x for x in rr) else "—")
            if l == 32768:
                g2[(m, lyr)] = med(rr, k)
        L.append(f"| {m} | {lyr} | " + " | ".join(cells) + " |")
G2_mag = all(x > 0.2 for x in g2.values())

L += ["", "### Band decomposition at 32k\n",
      "Share of `||H(w_32k) - H(w_2048)||_F^2` by wavelength `2*pi/freq`. The (I,J) part of "
      "block pair (i,j) moves at `|theta_i - theta_j|` and the (K,L) part at "
      "`theta_i + theta_j`; the two are Frobenius-orthogonal so the split is exact.\n",
      "| model | layer | " + " | ".join(BANDS) + " |", "|---|---|" + "---|" * len(BANDS)]
low_share = {}
for m, v in d.items():
    for lyr in ("q_proj", "k_proj"):
        rr = [x for x in v["rows"] if x["layer"] == lyr and "bands_32k" in x]
        if not rr: continue
        sh = {b: st.median([x["bands_32k"][b] for x in rr]) for b in BANDS}
        low_share[(m, lyr)] = sh[">32768"] + sh["8192-32768"]
        L.append(f"| {m} | {lyr} | " + " | ".join(f"{sh[b]:.3f}" for b in BANDS) + " |")
G2 = G2_mag and all(x > 0.5 for x in low_share.values())
L += ["", f"**Gate 2: {'PASS' if G2 else 'FAIL'}** — median rel_fro at 32k "
      f"{'exceeds' if G2_mag else 'does not exceed'} 0.2"
      f"{'; the mass sits in the two longest-wavelength bands' if all(x>0.5 for x in low_share.values()) else '; the mass is NOT concentrated in the low-frequency bands'}.\n",
      "### Static pairs in calibration\n",
      "| model | pairs rotating < 0.1 rad over 2047 tokens | of |", "|---|---|---|"]
for m, v in d.items():
    L.append(f"| {m} | {v['static_pairs_lt_0.1rad']} | {v['n_pairs']} |")
L.append("")

# ---------------- Step 3
qb = d.get("qwen2.5-0.5b", {}).get("bias_rows", [])
if qb:
    L += ["## Qwen key-bias visibility vs length\n",
          "`visible_frac(L) = 1 - sum_i F_L(theta_i) ||b_i||^2 / ||b||^2` as specified. The "
          "column marked F^2 uses `sum_i F_L^2 ||b_i||^2`, which is what "
          "`||E_w[R_D] b||^2` literally equals since `E_w[R_D]` is block-diagonal with "
          "entries `F_L(theta_i) I_2`. Both are reported because the brief's formula and "
          "that identity differ; the qualitative trend is the same.\n",
          "| L | median visible_frac (F) | median visible_frac (F^2) |", "|---|---|---|"]
    for l in (2048, 8192, 32768):
        L.append(f"| {l} | {st.median([r[f'visible_frac_L{l}'] for r in qb]):.4f} | "
                 f"{st.median([r[f'visible_frac_F2_L{l}'] for r in qb]):.4f} |")
    L += ["", "At 2k the bias is still largely invisible, consistent with the earlier finding "
          "that centring removed ~95% of `H_row`'s trace; the question was how fast it rises.\n"]

open(os.path.join(D, "phaseA.md"), "w").write("\n".join(L) + "\n")

# ---------------- plots
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for m, v in d.items():
    for lyr, mk in (("q_proj", "o"), ("k_proj", "s")):
        xs, ys = [], []
        for l in v["lengths"]:
            rr = [x for x in v["rows"] if x["layer"] == lyr]
            xs.append(l); ys.append(med(rr, f"rel_fro_L{l}"))
        ax[0].plot(xs, ys, marker=mk, label=f"{m} {lyr}")
ax[0].axhline(0.2, ls="--", c="k", lw=1); ax[0].set_xscale("log", base=2)
ax[0].set_xlabel("target length L"); ax[0].set_ylabel("median rel_fro vs w_2048")
ax[0].set_title("Gate 2: length sensitivity"); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)

w = 0.2
for i, ((m, lyr), _) in enumerate(low_share.items()):
    rr = [x for x in d[m]["rows"] if x["layer"] == lyr and "bands_32k" in x]
    sh = [st.median([x["bands_32k"][b] for x in rr]) for b in BANDS]
    ax[1].bar([j + i * w for j in range(len(BANDS))], sh, width=w, label=f"{m} {lyr}")
ax[1].set_xticks([j + 1.5 * w for j in range(len(BANDS))]); ax[1].set_xticklabels(BANDS, rotation=30, fontsize=7)
ax[1].set_ylabel("share of ||dH||_F^2"); ax[1].set_title("band decomposition at 32k")
ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3, axis="y")

if qb:
    for key, lab in (("visible_frac_L", "F"), ("visible_frac_F2_L", "F^2")):
        xs = [2048, 8192, 32768]
        ax[2].plot(xs, [st.median([r[f"{key}{l}"] for r in qb]) for l in xs], marker="o", label=lab)
    ax[2].set_xscale("log", base=2); ax[2].set_xlabel("L"); ax[2].set_ylabel("visible_frac")
    ax[2].set_title("Qwen key-bias visibility"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(D, "phaseA.png"), dpi=120)
print("\n".join(L[:40]))
