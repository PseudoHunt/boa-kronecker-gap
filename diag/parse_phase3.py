"""Repurposed Phase 3: objective-transfer matrix (arms x objectives) + PPL table."""
import ast, glob, json, os, re, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OBJ = ["boa", "mask", "p", "jac", "full"]


def ppl(path):
    if not os.path.exists(path):
        return None
    m = re.findall(r"^\{'wikitext2'.*\}$", open(path, errors="replace").read(), flags=re.M)
    return ast.literal_eval(m[-1]) if m else None


def main(root="results/phase3"):
    arms = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    print(f"{'arm':>10} | " + " ".join(f"{o:>9}" for o in OBJ) + "   (held-out objective, summed over blocks/layers/heads; row-normalised to the 'boa' arm)")
    base = {}
    table = {}
    for arm in arms:
        tot = {o: 0.0 for o in OBJ}; n = 0
        for f in glob.glob(os.path.join(root, arm, "block*.json")):
            j = json.load(open(f))
            for ln, r in j["layers"].items():
                if "obj_heldout" not in r:
                    continue
                for o in OBJ:
                    tot[o] += sum(r["obj_heldout"][o])
                n += 1
        if n == 0:
            continue
        table[arm] = tot
        if arm.startswith("boa") and not base:
            base = dict(tot)
    for arm, tot in table.items():
        cells = " ".join(f"{tot[o]/base[o]:>9.4f}" if base else f"{tot[o]:>9.3e}" for o in OBJ)
        print(f"{arm:>10} | {cells}")
    print("\nPPL (wikitext2 / c4-new):")
    for arm in arms:
        for f in sorted(glob.glob(os.path.join("logs/phase3", f"{arm}_*.log"))):
            r = ppl(f)
            if r:
                print(f"  {os.path.basename(f):>32}: {r['wikitext2']:>8.3f} / {r['c4-new']:>8.3f}")
    json.dump({"objective_matrix": table, "base_arm_totals": base}, open(os.path.join(root, "transfer_matrix.json"), "w"), indent=2)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
