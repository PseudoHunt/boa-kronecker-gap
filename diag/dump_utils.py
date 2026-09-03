"""Diagnostic dumping helpers for the Kronecker-gap study.

Nothing in this module is imported on the default quantization path; it is only
reached when an explicit diagnostic CLI flag is set. See results/SUMMARY.md.
"""
import json
import os
import subprocess

import torch


def git_commit(repo_dir=None):
    """Short git hash of the checkout, for provenance in results JSONs (rule 5.2)."""
    repo_dir = repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "describe", "--always", "--dirty"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def default_dump_dir(args):
    tag = f"{args.llm_name}_w{args.w_bits}_{args.calib_data}_n{args.nsamples}_s{args.seed}"
    tag += f"_bv{int(args.block_v)}_ac{int(args.act_order_col)}_ar{int(args.act_order_row)}"
    return os.path.join(args.cache_dir, "deltas", tag)


def write_manifest(dump_dir, args):
    os.makedirs(dump_dir, exist_ok=True)
    manifest = {
        "git_commit": git_commit(),
        "args": {k: (str(v) if not isinstance(v, (int, float, bool, str, type(None), list)) else v)
                 for k, v in vars(args).items()},
        "torch_version": torch.__version__,
    }
    with open(os.path.join(dump_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def delta_path(dump_dir, block_idx, name):
    return os.path.join(dump_dir, f"block{block_idx:02d}", f"{name}.pt")


def save_delta(dump_dir, block_idx, name, W_orig, W_quant):
    """Persist dW = Q - W for one layer.

    Both operands are the values actually held by the module (fp16 for OPT), so
    the fp32 difference is exact -- this is the perturbation the deployed model
    carries, which is what the Phase 1d predictive metrics must be scored against.
    """
    path = delta_path(dump_dir, block_idx, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "delta": (W_quant.detach().float() - W_orig.detach().float()).cpu(),
            "W_orig": W_orig.detach().cpu(),
            "block": block_idx,
            "layer": name,
        },
        path,
    )
