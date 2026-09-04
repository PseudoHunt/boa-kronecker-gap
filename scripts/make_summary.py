"""Regenerate results/qwen05b/SUMMARY.md from the per-run JSONs.

Paired deltas are taken against the SAME-SEED boa baseline, which is the only
comparison the seeds support: run-to-run spread across seeds is much larger than
the effect being measured, so unpaired means would drown it.
"""
import glob, json, os, statistics as st

RES = os.environ.get("BOA_RES_DIR", "/home/boa-kronecker-gap/results/qwen05b")
ARMS = ["combined", "qk-quantK", "v-rowmetric"]
METRICS = ["wikitext2", "c4-new"]

runs = {}
for f in sorted(glob.glob(os.path.join(RES, "*.json"))):
    d = json.load(open(f))
    if d.get("rc") == 0 and d.get("wikitext2") is not None:
        runs[(d["tag"], d["seed"])] = d

def deltas(arm, metric):
    """Per-seed (arm - boa) at fixed seed. Negative = arm is better (lower ppl)."""
    out = []
    for s in (0, 1, 2):
        a, b = runs.get((arm, s)), runs.get(("boa", s))
        if a and b:
            out.append((s, a[metric] - b[metric]))
    return out

L = []
L.append("# Qwen2.5-0.5B, W3 — BoA extension arms\n")
L.append(f"Model: Qwen/Qwen2.5-0.5B (24 blocks, d=896, 14 q heads / 2 kv heads, d_h=64).")
L.append("All runs: `--w_bits 3 --block_v --qparam_comput Hessian`, 128x2048 wikitext2 calibration.\n")

fp = runs.get(("__fp__", 0))
L.append("## Reproduction gate\n")
L.append("| quantity | reference | measured | verdict |")
L.append("|---|---|---|---|")
if fp:
    ok = abs(fp["wikitext2"] - 13.07) <= 0.05
    L.append(f"| FP wiki2 | 13.07 | {fp['wikitext2']} | {'PASS' if ok else 'FAIL'} |")
ao = {t: runs.get((t, 0)) for t in ("ao_none", "ao_col", "ao_row", "ao_both")}
best = min((v for v in ao.values() if v), key=lambda d: d["wikitext2"], default=None)
if best:
    ok = best["wikitext2"] <= 22.02 * 1.03
    L.append(f"| W3 best-of-4 act-order wiki2 | 22.02 | {best['wikitext2']} "
             f"({best['tag']}) | {'PASS (within 3%)' if ok else 'FAIL'} |")
L.append("")

L.append("## Stage B — act-order sweep (seed 0)\n")
L.append("| act-order | wiki2 | c4-new | wall (s) |")
L.append("|---|---|---|---|")
for t in ("ao_none", "ao_col", "ao_row", "ao_both"):
    d = ao.get(t)
    if d:
        L.append(f"| {t} | {d['wikitext2']} | {d['c4-new']} | {d['wall_s']} |")
L.append("")

L.append("## Stage C — all runs\n")
L.append("| arm | seed | wiki2 | c4-new | wall (s) |")
L.append("|---|---|---|---|---|")
for (tag, seed) in sorted(runs, key=lambda k: (k[0], k[1])):
    if tag.startswith("ao_") or tag == "__fp__":
        continue
    d = runs[(tag, seed)]
    L.append(f"| {d['tag']} | {seed} | {d['wikitext2']} | {d['c4-new']} | {d['wall_s']} |")
L.append("")

L.append("## Paired deltas vs same-seed `boa` (negative = better)\n")
L.append("| arm | metric | n | per-seed deltas | mean | std | all improve? | mean > 2*std? |")
L.append("|---|---|---|---|---|---|---|---|")
verdict_input = {}
for arm in ARMS:
    for m in METRICS:
        ds = deltas(arm, m)
        if not ds:
            continue
        vals = [v for _, v in ds]
        mean = st.mean(vals)
        sd = st.stdev(vals) if len(vals) > 1 else float("nan")
        allimp = all(v < 0 for v in vals)
        strong = (mean < 0) and (abs(mean) > 2 * sd) if len(vals) > 1 else False
        per = ", ".join(f"s{s}: {v:+.3f}" for s, v in ds)
        L.append(f"| {arm} | {m} | {len(vals)} | {per} | {mean:+.3f} | {sd:.3f} | "
                 f"{'yes' if allimp else 'no'} | {'yes' if strong else 'no'} |")
        if arm == "combined" and m == "wikitext2":
            verdict_input = dict(n=len(vals), allimp=allimp, strong=strong, mean=mean, sd=sd)
L.append("")

# --- what the q/k row metric actually contains ------------------------------
bs_path = os.path.join(RES, "diag", "bias_share.json")
if os.path.exists(bs_path):
    bs = json.load(open(bs_path))["summary"]
    L.append("## Why a null on the qk arms is expected here\n")
    L.append("BoA hooks the layer OUTPUT, so q_proj's row metric is `E[K K^T]` built from")
    L.append("BIASED keys and k_proj's is `E[Q Q^T]` from biased queries. On Qwen2.5-0.5B")
    L.append("that bias term owns almost the whole metric, measured across all 24 blocks as")
    L.append("the norm ratio `||b b^T||_F / ||E[Y Y^T]||_F`:\n")
    L.append("| metric | used for | mean | min | max |")
    L.append("|---|---|---|---|---|")
    L.append(f"| `E[Q Q^T]` | k_proj | {bs['q_proj']['bias_share_mean']} | "
             f"{bs['q_proj']['bias_share_min']} | {bs['q_proj']['bias_share_max']} |")
    L.append(f"| `E[K K^T]` | q_proj | {bs['k_proj']['bias_share_mean']} | "
             f"{bs['k_proj']['bias_share_min']} | {bs['k_proj']['bias_share_max']} |")
    L.append("")
    L.append("The bias is a constant that weight quantization never touches, so ~96% of the")
    L.append("Frobenius mass of the q/k row metric is inert. `--qk_quantK` rebuilds that")
    L.append("metric from the quantized key but can only move the few percent that depends")
    L.append("on W. **A null on `qk-quantK` and `combined` is therefore evidence about")
    L.append("Qwen's q/k biases, not about the hypothesis.** (Norm ratios, not an orthogonal")
    L.append("decomposition -- the cross terms are not orthogonal, so shares can sum past 1.)\n")

L.append("## Verdict (runbook section 7)\n")
if not verdict_input or verdict_input["n"] < 3:
    L.append(f"**Incomplete** — `combined` has {verdict_input.get('n', 0)}/3 paired seeds. "
             "No verdict; the decision table needs all three.")
elif verdict_input["allimp"] and verdict_input["strong"]:
    L.append("**`combined` improves on all 3 seeds with mean > 2*std.** Per the runbook: "
             "*paper, cheap version* — \"the terms BoA's objective drops\". "
             "Next: Llama-1B with per-block resume, then TurboBoA baseline. "
             "Gate this on the Stage D softmax gap before committing.")
elif verdict_input["allimp"]:
    L.append("**Two up, one down / weak effect.** Per the runbook this is "
             "\"two more seeds\", not \"yes\".")
else:
    L.append("**Null on `combined`.** Per the runbook the next branch is set by the "
             "Stage D softmax gap: small gap (<= OPT's) => no paper on this line, stop "
             "spending; large gap => the paper hinges on the softmax solver.")
L.append("")

L.append("## Cost (measured, not estimated)\n")
L.append("One block of Qwen2.5-0.5B at W3, 128x2048 calibration, on an A100-40GB:\n")
L.append("| phase | time |")
L.append("|---|---|")
L.append("| compute_Hessian (128 seqs) | 3.0 s |")
L.append("| solve q_proj (two-sided `boa()`) | 65.4 s |")
L.append("| solve k_proj (two-sided `boa()`) | 68.3 s |")
L.append("| solve v/o/gate/up/down (one-sided `gptq()`) | 21.4 s |")
L.append("| **block total** | **158.1 s** |")
L.append("")
L.append("So ~63 min/run for `boa` and `qk-quantK`, and ~89 min/run for `combined` and")
L.append("`v-rowmetric`, which route v_proj through the two-sided solver as well. The")
L.append("runbook's estimate of 6-10 min/run is low by roughly 7x. Cost is dominated by")
L.append("the Python row loop (64 row steps x ~896 GPTQ column steps per two-sided")
L.append("layer), not by Hessian collection. Runs are single-threaded, ~1.3-1.9 GB GPU")
L.append("each, so ~11 fit concurrently on 16 vCPUs without contention.\n")

L.append("## Stage D (softmax-gap diagnostic): NOT RUN\n")
L.append("`diag/phase1_runner.py` is OPT-shaped and does not support this model: it")
L.append("indexes both the q and k row metrics by `n_heads` (under GQA the k metric has")
L.append("`n_kv_heads` entries, so it raises), and it reconstructs Q/K as `W x + b` with")
L.append("no RoPE, so the attention probabilities it derives would be wrong for Qwen.")
L.append("A correct port also has to keep `R` in BoA's pre-RoPE (back-rotated) basis")
L.append("while computing `A` from post-RoPE Q/K -- a convention split that is easy to")
L.append("get plausibly but subtly wrong, which would produce a credible-looking gap")
L.append("number driving a paper/no-paper call. It was left undone rather than guessed.\n")
L.append("### The number Stage D has to produce, and what to compare it to\n")
L.append("From this repo's existing OPT-125m phase 1 run (`results/phase1/summary.json`,")
L.append("24 block x layer entries), the relevant reference values are:\n")
L.append("| field | what it measures | OPT-125m mean | range |")
L.append("|---|---|---|---|")
L.append("| `G1_rel_fro` | pure separability (Kronecker) gap | 0.0005 | 0.0002 - 0.0020 |")
L.append("| `G12_rel_fro` | + causal mask | 0.2041 | 0.0585 - 0.3092 |")
L.append("| `G123p_rel_fro` | + attention-probability weighting | 0.2944 | 0.0810 - 0.4485 |")
L.append("| `G123j_rel_fro` | + softmax-Jacobian weighting (**the softmax gap**) | **0.2306** | 0.0953 - 0.3402 |")
L.append("")
L.append("So 'small (<= OPT's)' in section 7 means a Qwen `G123j_rel_fro` at or below")
L.append("~0.23; materially above that is the 'large gap' branch. Note the contrast that")
L.append("makes this the interesting quantity: the pure Kronecker gap is ~0.0005 (nil),")
L.append("so essentially all of the discrepancy comes from the softmax weighting, not")
L.append("from separability.\n")
L.append("This matters more than its 'optional' billing suggests: if the arms come back")
L.append("null, the bias result above says the null is uninformative about the")
L.append("hypothesis, and section 7's branch turns entirely on whether the softmax gap")
L.append("is small (stop) or large (the paper hinges on the softmax solver). **Build this")
L.append("first next session, ahead of any Llama work.**\n")

open(os.path.join(RES, "SUMMARY.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
