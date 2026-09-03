"""Phase 2a: --row_metric_v (paper eq. 9) vs baseline, 3 seeds, per bit-width."""
import ast, glob, json, os, re, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diag.dump_utils import git_commit


def res(path):
    m = re.findall(r"^\{'wikitext2'.*\}$", open(path, errors="replace").read(), flags=re.M)
    return ast.literal_eval(m[-1]) if m else None


def main(log_dir="logs/phase2a"):
    runs = {}
    for f in sorted(glob.glob(os.path.join(log_dir, "w*_*_s*.log"))):
        m = re.match(r"w(\d)_(\w+)_(base|rowv)_s(\d)\.log", os.path.basename(f))
        if not m:
            continue
        r = res(f)
        if r is None:
            continue
        wb, act, arm, seed = int(m[1]), m[2], m[3], int(m[4])
        runs.setdefault(wb, {}).setdefault(arm, {})[seed] = {"act": act, **r}

    out = {"git_commit": git_commit(), "per_bit": {}}
    hdr = f"{'bits':>4} {'act':>7} {'arm':>5} | {'seed0':>8} {'seed1':>8} {'seed2':>8} | {'mean':>8} {'spread':>7} || {'c4 mean':>8}"
    print(hdr); print("-" * len(hdr))
    for wb in sorted(runs, reverse=True):
        row = {}
        for arm in ("base", "rowv"):
            d = runs[wb].get(arm, {})
            w = [d[s]["wikitext2"] for s in sorted(d)]
            c = [d[s]["c4-new"] for s in sorted(d)]
            if not w:
                continue
            act = next(iter(d.values()))["act"]
            spread = max(w) - min(w)
            row[arm] = {"wiki2": w, "c4": c, "wiki2_mean": st.mean(w), "wiki2_spread": spread,
                        "c4_mean": st.mean(c), "seeds": sorted(d)}
            cells = " ".join(f"{d[s]['wikitext2']:>8.3f}" if s in d else f"{'-':>8}" for s in (0, 1, 2))
            print(f"{wb:>4} {act:>7} {arm:>5} | {cells} | {st.mean(w):>8.3f} {spread:>7.3f} || {st.mean(c):>8.3f}")
        if "base" in row and "rowv" in row:
            delta = row["rowv"]["wiki2_mean"] - row["base"]["wiki2_mean"]
            noise = max(row["base"]["wiki2_spread"], row["rowv"]["wiki2_spread"])
            # paired per-seed deltas are the cleaner test: same calibration set both arms
            paired = [runs[wb]["rowv"][s]["wikitext2"] - runs[wb]["base"][s]["wikitext2"]
                      for s in row["base"]["seeds"] if s in runs[wb]["rowv"]]
            sign = "IMPROVES" if delta < 0 else "WORSENS"
            consistent = all(x < 0 for x in paired) or all(x > 0 for x in paired)
            row["delta_wiki2_mean"] = delta
            row["paired_deltas"] = paired
            row["seed_spread"] = noise
            row["verdict"] = (f"{sign} by {abs(delta):.3f} (seed spread {noise:.3f}); paired deltas "
                              f"{[round(x,3) for x in paired]} -> "
                              f"{'consistent sign' if consistent else 'MIXED sign: within noise'}")
            print(f"     row_metric_v {row['verdict']}\n")
        out["per_bit"][str(wb)] = row
    json.dump(out, open("results/phase2a.json", "w"), indent=2)
    print("wrote results/phase2a.json")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
