import json, os
D = "/home/boa-kronecker-gap/results/length"
s0 = json.load(open(os.path.join(D, "phaseB_pg19.json")))
s1 = json.load(open(os.path.join(D, "phaseB_seed1.json")))
BK = ["0-2048", "2048-4096", "4096-8192", "8192-16384", "16384-32768"]
L = ["# Seed-1 replication — arms 1 and 3 at long context\n",
     "Paired within seed: `two-sided-Lext(32k)` minus `two-sided-2048`, same seed, "
     "same corpus. Negative = Lext better.\n"]
for ds in ("pg19", "wikitext2"):
    L += [f"## {ds}\n", "| pair | " + " | ".join(BK) + " | slope |", "|---|" + "---|" * (len(BK)+1)]
    for tag, src, a1, a3 in (("seed 0", s0, "two-sided-2048", "two-sided-Lext"),
                             ("seed 1", s1, "two-sided-2048_s1", "two-sided-Lext_s1")):
        if a1 not in src or a3 not in src: continue
        r1, r3 = src[a1][ds], src[a3][ds]
        d = [r3[b]["ratio_to_fp"] - r1[b]["ratio_to_fp"] for b in BK]
        sl3 = r3[BK[-1]]["ratio_to_fp"]/r3[BK[0]]["ratio_to_fp"]
        sl1 = r1[BK[-1]]["ratio_to_fp"]/r1[BK[0]]["ratio_to_fp"]
        L.append(f"| {tag} Lext-base | " + " | ".join(f"{x:+.4f}" for x in d) + f" | {sl3-sl1:+.4f} |")
    if "two-sided-Lext8k_s0" in s1:
        r1 = s0["two-sided-2048"][ds]; r8 = s1["two-sided-Lext8k_s0"][ds]
        d = [r8[b]["ratio_to_fp"] - r1[b]["ratio_to_fp"] for b in BK]
        L.append("| seed 0 Lext(8k)-base | " + " | ".join(f"{x:+.4f}" for x in d)
                 + f" | {r8[BK[-1]]['ratio_to_fp']/r8[BK[0]]['ratio_to_fp'] - r1[BK[-1]]['ratio_to_fp']/r1[BK[0]]['ratio_to_fp']:+.4f} |")
    L.append("")
L += ["## Raw 16-32k loss ratios\n", "| arm | pg19 | wikitext2 |", "|---|---|---|"]
for src, keys in ((s0, ["two-sided-2048", "one-sided", "two-sided-Lext", "two-sided-longcalib"]),
                  (s1, ["two-sided-2048_s1", "two-sided-Lext_s1", "two-sided-Lext8k_s0"])):
    for a in keys:
        if a in src:
            L.append(f"| {a} | {src[a]['pg19'][BK[-1]]['ratio_to_fp']:.4f} | "
                     f"{src[a]['wikitext2'][BK[-1]]['ratio_to_fp']:.4f} |")
open(os.path.join(D, "phaseB_seed1.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
