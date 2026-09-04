"""Regenerate results/qwen05b/SUMMARY.md from the per-run JSONs.

Paired deltas are taken against the SAME-SEED boa baseline, which is the only
comparison the seeds support: run-to-run spread across seeds is much larger than
the effect being measured, so unpaired means would drown it.
"""
import glob, json, os, statistics as st

RES = "/home/boa-kronecker-gap/results/qwen05b"
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

open(os.path.join(RES, "SUMMARY.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
