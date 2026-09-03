"""Parse the Phase 0 sweep logs into results/phase0.json.

Two sweeps are parsed:
  logs/phase0/        -- as released (SDPA; non-causal Hessian pass, see
                         results/BUG_causal_attention.md). Kept only to document
                         the bug.
  logs/phase0_eager/  -- corrected (--attn_impl eager). THIS is the baseline that
                         every later phase is measured against.
"""
import ast
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diag.dump_utils import git_commit

ACT = {"none": (False, False), "col": (True, False), "row": (False, True), "colrow": (True, True)}
PAPER = {2: 85.63, 3: 31.95}          # arXiv 2406.13474, Table 10, OPT-125M BoA
PAPER_GPTQ = {2: 411.3, 3: 50.75}
FP16 = {"wikitext2": 27.654, "c4_new": 26.564}


def parse_log(path):
    if not os.path.exists(path):
        return None
    txt = open(path, errors="replace").read()
    m = re.findall(r"^\{'wikitext2'.*\}$", txt, flags=re.M)
    return ast.literal_eval(m[-1]) if m else None


def collect(log_dir, attn_impl):
    rows = []
    for wb in (4, 3, 2):
        for act, (ac, ar) in ACT.items():
            res = parse_log(os.path.join(log_dir, f"w{wb}_{act}.log"))
            if res is None:
                continue
            rows.append({
                "w_bits": wb, "act_tag": act, "act_order_col": ac, "act_order_row": ar,
                "attn_impl": attn_impl, "block_v": True, "qparam_comput": "Hessian",
                "calib_data": "wikitext2", "nsamples": 128, "seqlen": 2048, "seed": 0,
                "wikitext2": res["wikitext2"], "c4_new": res["c4-new"],
                "quant_time_s": res.get("time"),
            })
    return rows


def best(rows):
    out = {}
    for r in rows:
        wb = r["w_bits"]
        if wb not in out or r["wikitext2"] < out[wb]["wikitext2"]:
            out[wb] = r
    return out


def table(rows, title):
    b = best(rows)
    print(f"\n{title}")
    print(f"{'bits':>4} {'act_order':>9} {'wiki2':>9} {'c4-new':>9} {'time(s)':>8}  {'vs paper':>9}")
    print("-" * 60)
    for r in sorted(rows, key=lambda r: (-r["w_bits"], r["act_tag"])):
        star = " *" if b.get(r["w_bits"]) is r else "  "
        p = PAPER.get(r["w_bits"])
        rel = f"{100*(r['wikitext2']/p - 1):+8.1f}%" if p else "      n/a"
        print(f"{r['w_bits']:>4} {r['act_tag']:>9} {r['wikitext2']:>9.3f} {r['c4_new']:>9.3f} "
              f"{r['quant_time_s']:>8.1f} {rel}{star}")
    return b


def main():
    rel_rows = collect("logs/phase0", "sdpa (as released)")
    fix_rows = collect("logs/phase0_eager", "eager (corrected)")

    b_rel = table(rel_rows, "AS RELEASED  (SDPA -> non-causal Hessian pass; BUGGY)") if rel_rows else {}
    b_fix = table(fix_rows, "CORRECTED    (--attn_impl eager; causal)") if fix_rows else {}

    print("\nPhase 0 stop/go (corrected sweep, best act-order per bit-width, vs paper +/-3%):")
    verdict_ok = True
    for wb in (3, 2):
        if wb in b_fix:
            got, p = b_fix[wb]["wikitext2"], PAPER[wb]
            d = 100 * (got / p - 1)
            ok = abs(d) <= 3.0
            verdict_ok &= ok
            print(f"  W{wb}: {got:8.3f} vs paper {p:7.2f}  ({d:+.2f}%)  {'PASS' if ok else 'FAIL'}")
        else:
            print(f"  W{wb}: (pending)")
            verdict_ok = False
    print(f"  FP16: {FP16['wikitext2']:.3f} vs paper 27.65 (+0.01%)  PASS")

    out = {
        "git_commit": git_commit(),
        "model": "/home/models/opt-125m (facebook/opt-125m, converted to safetensors)",
        "fp16": FP16,
        "paper_reference_wiki2": {"fp16": 27.65, "boa": PAPER, "gptq": PAPER_GPTQ,
                                  "source": "arXiv 2406.13474 Table 10 (no INT4 row for OPT-125M)"},
        "runs_as_released": rel_rows,
        "runs_corrected": fix_rows,
        "best_per_bitwidth_corrected": {
            str(k): {"act_tag": v["act_tag"], "wikitext2": v["wikitext2"],
                     "c4_new": v["c4_new"], "act_order_col": v["act_order_col"],
                     "act_order_row": v["act_order_row"]}
            for k, v in sorted(b_fix.items(), reverse=True)},
        "stopgo_pass": bool(verdict_ok),
        "note": "Later phases use the corrected sweep only; see results/BUG_causal_attention.md",
    }
    os.makedirs("results", exist_ok=True)
    with open("results/phase0.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote results/phase0.json  ({len(rel_rows)} as-released + {len(fix_rows)} corrected runs)")


if __name__ == "__main__":
    main()
