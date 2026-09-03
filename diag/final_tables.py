"""Collect every arm's PPL (W2 first) plus the Phase 3 objective-transfer matrix."""
import ast, glob, json, os, re, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OBJ = ["boa", "mask", "p", "jac", "full"]


def ppl(path):
    if not os.path.exists(path): return None
    m = re.findall(r"^\{'wikitext2'.*\}$", open(path, errors="replace").read(), flags=re.M)
    return ast.literal_eval(m[-1]) if m else None


def row(label, paths, ref=None):
    rs = [ppl(p) for p in paths]; rs = [r for r in rs if r]
    if not rs: return f"{label:>34} | {'(pending)':>9}"
    w = [r["wikitext2"] for r in rs]; c = [r["c4-new"] for r in rs]
    d = f"{st.mean(w)-ref:+8.3f}" if ref is not None else "        "
    return f"{label:>34} | {st.mean(w):>9.3f} {('±%.3f' % ((max(w)-min(w))/2) if len(w)>1 else '      '):>7} n={len(w)} | {st.mean(c):>8.3f} | {d}"


def main():
    print(f"{'arm':>34} | {'wiki2':>9} {'spread':>7}     | {'c4-new':>8} | vs boa")
    print("-" * 90)
    for wb, act in ((2, "row"), (3, "colrow")):
        base = ppl(f"logs/phase0_eager/w{wb}_{act}.log")["wikitext2"]
        print(f"--- W{wb} ({act}) ---")
        print(row(f"boa (Phase 0, seed 0)", [f"logs/phase0_eager/w{wb}_{act}.log"], base))
        print(row(f"boa (3 seeds)", glob.glob(f"logs/phase2a/w{wb}_{act}_base_s*.log"), base))
        print(row(f"boa + row_metric_v  [eq. 9]", glob.glob(f"logs/phase2a/w{wb}_{act}_rowv_s*.log"), base))
        print(row(f"boa + row_metric_fc1 [MLP-aware]", glob.glob(f"logs/phase4b/w{wb}_{act}_fc1_s*.log"), base))
        for arm in ("rsq-col", "boa+rsq", "dense-mask", "dense-p", "dense-jac", "dense-full"):
            print(row(arm, glob.glob(f"logs/phase3/{arm}_w{wb}_{act}_s*.log"), base))
        print()
    # transfer matrix (held-out objectives on the dense blocks, normalised to the boa arm)
    root = "results/phase3"; tot = {}
    for d in sorted(glob.glob(os.path.join(root, "*_w2_s0"))):
        arm = os.path.basename(d).replace("_w2_s0", "")
        t = {o: 0.0 for o in OBJ}; n = 0
        for f in glob.glob(os.path.join(d, "block*.json")):
            for ln, r in json.load(open(f))["layers"].items():
                if "obj_heldout" in r:
                    for o in OBJ: t[o] += sum(r["obj_heldout"][o])
                    n += 1
        if n: tot[arm] = (t, n)
    if "boa" in tot:
        print("Held-out objective transfer matrix, W2 seed 0, normalised to the boa arm (row = arm, col = objective it is scored on):")
        print(f"{'arm':>12} | " + " ".join(f"{o:>8}" for o in OBJ) + "  | blocks x layers scored")
        b = tot["boa"][0]
        for arm, (t, n) in tot.items():
            print(f"{arm:>12} | " + " ".join(f"{t[o]/b[o]:>8.4f}" for o in OBJ) + f"  | {n}")
        print("(dense arms only quantized blocks 0 and 11 densely; their other blocks are standard BoA)")


if __name__ == "__main__":
    main()
