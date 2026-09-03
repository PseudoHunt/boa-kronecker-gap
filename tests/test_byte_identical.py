"""Engineering rule 5.1: the default path must be byte-identical to upstream.

Runs block 0 of OPT-125m at W3 through the pristine upstream checkout and through
this (patched) checkout in separate subprocesses, and asserts the resulting
quantized weights are bit-for-bit equal.

    python3 tests/test_byte_identical.py
"""
import os
import subprocess
import sys
import tempfile

import torch

REF_REPO = os.environ.get("BOA_REF_REPO", "/home/BOA_ref")
CUR_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE = os.path.join(CUR_REPO, "tests", "_block0_probe.py")


def run(repo, out, extra):
    cmd = [sys.executable, PROBE, "--repo", repo, "--out", out] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=tempfile.gettempdir())
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise SystemExit(f"probe failed for {repo}")
    return out


def compare(cfg_name, extra):
    with tempfile.TemporaryDirectory() as td:
        a = run(REF_REPO, os.path.join(td, "ref.pt"), extra)
        b = run(CUR_REPO, os.path.join(td, "cur.pt"), extra)
        sa, sb = torch.load(a), torch.load(b)
        assert set(sa) == set(sb), (set(sa), set(sb))
        ok = True
        for k in sa:
            if not torch.equal(sa[k], sb[k]):
                d = (sa[k].float() - sb[k].float()).abs().max().item()
                print(f"  MISMATCH {k}: max|diff| = {d:.3e}")
                ok = False
            else:
                print(f"  identical {k}  {tuple(sa[k].shape)}")
        print(f"[{cfg_name}] {'PASS' if ok else 'FAIL'}")
        return ok


if __name__ == "__main__":
    print(f"reference: {REF_REPO}\npatched  : {CUR_REPO}\n")
    results = []
    results.append(compare("W3 block_v Hessian, no act-order", ["--w_bits", "3"]))
    results.append(compare("W3 block_v Hessian, act_order_col+row",
                           ["--w_bits", "3", "--act_order_col", "--act_order_row"]))
    results.append(compare("W2 block_v Hessian, no act-order", ["--w_bits", "2"]))
    print()
    if all(results):
        print("BYTE-IDENTICAL TEST: PASS")
    else:
        print("BYTE-IDENTICAL TEST: FAIL")
        raise SystemExit(1)
