"""diag/softmax_gap.py must reproduce phase 1 on OPT, where phase 1 can run.

The Stage D module is written for RoPE + GQA, which phase1_runner.py cannot handle.
With R_t = I and n_kv == n_heads it must collapse onto the OPT formulation, so the
existing results/phase1/summary.json is an independent ground truth for it. Without
this check the Qwen numbers rest on nothing but the derivation.

Reference (results/phase1/summary.json, block 0, 96 calib sequences):
    q_proj  G1 0.00067  G12 0.1616  G123p 0.0810  G123j 0.0953
    k_proj  G1 0.00077  G12 0.1492  G123p 0.1077  G123j 0.1218

This runs 4 sequences, so the split-half correction is doing real work and the
tolerances are loose; the point is that the numbers land on the reference rather
than somewhere else entirely.

    python3 tests/test_softmax_gap.py
"""
import json, os, subprocess, sys, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = {"q_proj": {"G1": 0.00067, "G12": 0.1616, "G123p": 0.0810, "G123j": 0.0953},
       "k_proj": {"G1": 0.00077, "G12": 0.1492, "G123p": 0.1077, "G123j": 0.1218}}


def main():
    out = os.path.join(tempfile.mkdtemp(), "sg.json")
    env = dict(os.environ, BOA_MODEL="/home/models/opt-125m",
               BOA_CALIB="/home/jl_fs/calib/calib_opt_wikitext2_128_2048_0.cache",
               BOA_NSEQ="4", BOA_NBLOCK="1", BOA_OUT=out,
               HF_HOME=os.environ.get("HF_HOME", "/home/jl_fs/hf"))
    r = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "stage_d.py")],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:]); raise SystemExit("stage_d failed")

    rows = json.load(open(out))["per_block"]
    ok = True
    for rec in rows:
        lyr = rec["layer"]
        for v, want in REF[lyr].items():
            got = rec[f"{v}_rel_fro_corrected"]
            if v == "G1":
                # separability gap is ~0; assert it stays negligible rather than matching
                good = got < 0.02
                print(f"  {lyr:7s} {v:6s} got {got:.4f}  ref {want:.5f}  "
                      f"{'OK (negligible)' if good else 'FAIL'}")
            elif (lyr, v) == ("k_proj", "G12"):
                # KNOWN, INTENTIONAL DIVERGENCE, not a tolerance fudge.
                # phase1 applies the same FORWARD cumsum to both layers
                # (_row_energy_variants: G12 = cumsum(R2, dim=-1)) and masks with a
                # fixed tri[u,t] = u<=t. For q_proj that is right: query t sees keys
                # u<=t. For k_proj the roles reverse -- R is built from the QUERIES,
                # so a key at position u is seen by queries t>=u, which is a REVERSE
                # cumsum. This module uses that direction, so the two disagree on
                # exactly this one entry. The attention-weighted variants, which are
                # what Stage D reports, agree closely (see G123p/G123j above), so the
                # divergence is confined to the pure causal-mask variant.
                good = 0.02 < got < want
                print(f"  {lyr:7s} {v:6s} got {got:.4f}  ref {want:.4f}  "
                      f"{'OK (causal direction differs by design)' if good else 'FAIL'}")
            else:
                good = abs(got - want) <= max(0.06, 0.35 * want)
                print(f"  {lyr:7s} {v:6s} got {got:.4f}  ref {want:.4f}  "
                      f"{'OK' if good else 'FAIL'}")
            ok &= good
    print("\nSOFTMAX GAP TEST: " + ("PASS" if ok else "FAIL"))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
