"""Alias the winning Stage B act-order run as `boa` seed 0.

Stage C runs the boa baseline at seeds 1 and 2 only: seed 0's baseline is the
Stage B sweep winner, which is the identical configuration (W3, block_v, Hessian,
the chosen act-order, seed 0). Without this alias `combined`/`qk-quantK`/
`v-rowmetric` at seed 0 have no same-seed baseline and the paired table loses a
third of its rows.
"""
import glob, json, os, sys

RES = "/home/boa-kronecker-gap/results/qwen05b"
EXPECTED_AO = sys.argv[1] if len(sys.argv) > 1 else None

rows = []
for f in glob.glob(os.path.join(RES, "ao_*_w3_s0.json")):
    d = json.load(open(f))
    if d.get("rc") == 0 and d.get("wikitext2") is not None:
        rows.append(d)
if not rows:
    raise SystemExit("no completed Stage B runs")

rows.sort(key=lambda d: d["wikitext2"])
best = rows[0]
gate = 22.02 * 1.03
print(f"Stage B winner: {best['tag']} wiki2={best['wikitext2']} "
      f"(gate <= {gate:.3f}) -> {'PASS' if best['wikitext2'] <= gate else 'FAIL'}")
for d in rows:
    print(f"  {d['tag']:<9} wiki2={d['wikitext2']:<8} c4={d['c4-new']}")

if EXPECTED_AO and best["tag"] != EXPECTED_AO:
    raise SystemExit(f"\nWINNER IS {best['tag']}, NOT {EXPECTED_AO}. "
                     "Speculative Stage C runs used the wrong act-order -- discard them.")

out = os.path.join(RES, "boa_w3_s0.json")
rec = dict(best)
rec["tag"] = "boa"
rec["aliased_from"] = best["tag"]
rec["note"] = ("Stage B sweep winner, reused as the seed-0 boa baseline: identical "
               "configuration to the Stage C boa arm at seeds 1 and 2.")
json.dump(rec, open(out, "w"), indent=2)
print(f"\nwrote {out} (aliased from {best['tag']})")
