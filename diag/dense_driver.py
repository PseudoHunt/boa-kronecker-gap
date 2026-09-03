"""Driver glue for the repurposed Phase 3 (see diag/dense_ref.py).

Hooks into boa_fwrd behind --dense_arm / --obj_eval:

  on_block_start   : for a block in --dense_blocks, cache FP per-sequence data
                     (X, Q, K, V with biases, attention probs, W_o) for the first
                     --dense_nsamples calibration sequences AND for a disjoint
                     held-out set that is propagated through the quantized model
                     alongside the calibration inputs.
  quantize_layer   : for q/k in a dense block, replace wrapper.quant() by the
                     dense solve for the chosen objective. For arm 'boa' the dense
                     solve is ALSO compared in-process against boa() -- the
                     correctness gate -- and must agree to < 1e-4 relative.
  on_layer_done    : evaluate all five objectives on the held-out set (and on the
                     dense calibration subset) for the dW that was produced,
                     whichever solver produced it. This is the 8x5 transfer matrix.
  on_block_end     : write JSON; propagate the held-out inputs.
"""
import json
import os

import torch

from diag.dense_ref import (Q_NAME, K_NAME, OBJECTIVES, attn_probs, eval_objectives,
                            solve_layer_dense)
from diag.dump_utils import git_commit


class DenseCollector:
    def __init__(self, arm, blocks, out_dir, heldout_inps, n_dense_seqs=32,
                 tokens_per_seq=256, seed=0, obj_eval=True):
        self.arm = arm                      # 'none' | one of OBJECTIVES
        self.blocks = set(blocks)
        self.out_dir = out_dir
        self.heldout_inps = heldout_inps    # [S_ho, L, d] fp16 on device, or None
        self.n_dense_seqs = n_dense_seqs
        self.tokens_per_seq = tokens_per_seq
        self.gen = torch.Generator().manual_seed(seed)
        self.obj_eval = obj_eval
        os.makedirs(out_dir, exist_ok=True)
        self.state = {}

    # ------------------------------------------------------------------ data
    @torch.no_grad()
    def _collect(self, block, inps, n_take, n_heads, d_h):
        attn = block.self_attn
        cap = {}
        h = attn.q_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("X", i[0].detach()))
        Wq, bq = attn.q_proj.weight.float(), attn.q_proj.bias.float()
        Wk, bk = attn.k_proj.weight.float(), attn.k_proj.bias.float()
        Wv, bv = attn.v_proj.weight.float(), attn.v_proj.bias.float()
        W_o = attn.out_proj.weight.float().clone()
        scaling = d_h ** -0.5
        out = []
        for s in range(min(n_take, len(inps))):
            block(inps[s].unsqueeze(0), **self.kw)
            X = cap["X"].reshape(-1, Wq.shape[1]).T.float()             # [d, L]
            L = X.shape[-1]
            Qf = (Wq @ X + bq[:, None]).view(n_heads, d_h, L)
            Kf = (Wk @ X + bk[:, None]).view(n_heads, d_h, L)
            Vf = (Wv @ X + bv[:, None]).view(n_heads, d_h, L)
            # attention maps are 200 MB/seq in fp32; keep them in fp16 (probabilities in
            # [0,1]) and upcast at use.
            A = torch.stack([attn_probs(Qf[hh], Kf[hh], scaling) for hh in range(n_heads)]).half()
            out.append({"X": X, "Q": Qf, "K": Kf, "V": Vf, "A": A, "W_o": W_o, "scaling": scaling})
        h.remove()
        return out

    @torch.no_grad()
    def on_block_start(self, block, block_idx, quant_inps, block_kwargs, n_heads, d_h):
        if block_idx not in self.blocks:
            return
        self.kw = dict(block_kwargs); self.kw["output_attentions"] = False
        st = {"n_heads": n_heads, "d_h": d_h, "results": {}}
        st["calib"] = self._collect(block, quant_inps, self.n_dense_seqs, n_heads, d_h)
        st["heldout"] = (self._collect(block, self.heldout_inps, len(self.heldout_inps), n_heads, d_h)
                         if self.heldout_inps is not None else [])
        self.state[block_idx] = st
        print(f"[dense] block {block_idx}: cached {len(st['calib'])} calib + {len(st['heldout'])} held-out sequences")

    # ------------------------------------------------------------------ solve
    @torch.no_grad()
    def quantize_layer(self, block_idx, name, wrapper, boa_opts, log=print):
        """Returns True if it handled the quantization (else caller runs wrapper.quant())."""
        st = self.state.get(block_idx)
        if st is None or name not in (Q_NAME, K_NAME) or self.arm == "none":
            return False
        n_heads, d_h = st["n_heads"], st["d_h"]
        W_orig = wrapper.layer.weight.data.clone()
        org_dtype = W_orig.dtype

        Q_dense = solve_layer_dense(
            self.arm, name, wrapper, st["calib"], n_heads, d_h, None,
            boa_opts["act_order_col"], boa_opts["act_order_row"],
            self.tokens_per_seq, self.gen, log)

        rec = st["results"].setdefault(name, {})
        if self.arm == "boa":
            # correctness gate: same inputs through BoA's own solver
            wrapper.quant(False)
            Q_boa = wrapper.layer.weight.data.float()
            rel = ((Q_dense - Q_boa).norm() / Q_boa.norm()).item()
            diff = (Q_dense - Q_boa).abs()
            n_diff = (diff > 0).sum().item()
            # separate genuine quantisation-level flips from fp32 summation noise:
            # a flip moves a weight by one grid step, i.e. by about its row's scale.
            sc = wrapper.quantizer.scale.reshape(-1, 1).abs().expand_as(diff)
            n_flip = (diff > 0.5 * sc).sum().item()
            max_noise = (diff[diff <= 0.5 * sc] / sc[diff <= 0.5 * sc]).max().item() if n_diff > n_flip else 0.0
            rec.update({"gate_rel_err": rel, "gate_n_diff": n_diff, "gate_n_flips": n_flip,
                        "gate_max_nonflip_over_scale": max_noise, "n_weights": diff.numel()})
            log(f"[dense] GATE {name}: |Q_dense - Q_boa|/|Q_boa| = {rel:.3e}; {n_diff}/{diff.numel()} entries differ, "
                f"{n_flip} are level flips, max non-flip |diff|/scale = {max_noise:.2e}")
            g64 = getattr(solve_layer_dense, "last_gate64", None) or {}
            rec.update(g64)
            assert g64.get("fp64_rel_err", 1.0) < 1e-6 and g64.get("fp64_n_flips", 1) == 0, \
                f"dense-boa fp64 gate FAILED ({g64})"
            log(f"[dense] GATE PASSED in exact arithmetic; fp32 ordering noise: rel {rel:.2e}, {n_flip} flips")
        else:
            wrapper.H_col = None; wrapper.H_row = None
        wrapper.layer.weight.data = Q_dense.to(org_dtype)
        return True

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def on_layer_done(self, block_idx, name, W_orig, W_quant):
        st = self.state.get(block_idx)
        if st is None or name not in (Q_NAME, K_NAME) or not self.obj_eval:
            return
        n_heads, d_h = st["n_heads"], st["d_h"]
        d = W_orig.shape[-1]
        dW = (W_quant.float() - W_orig.float()).view(n_heads, d_h, d)
        rec = st["results"].setdefault(name, {})
        for split in ("calib", "heldout"):
            tot = {o: [0.0] * n_heads for o in OBJECTIVES}
            for bd in st[split]:
                for h in range(n_heads):
                    v = eval_objectives(name, dW[h], bd["X"], bd["Q"][h], bd["K"][h], bd["V"][h],
                                        bd["A"][h].float(), bd["W_o"][:, h * d_h:(h + 1) * d_h])
                    for o in OBJECTIVES:
                        tot[o][h] += v[o]
            rec[f"obj_{split}"] = tot
            rec[f"n_{split}"] = len(st[split])
        rec["dW_fro"] = dW.norm().item()

    @torch.no_grad()
    def on_block_end(self, block, block_idx):
        st = self.state.pop(block_idx, None)
        if st is not None:
            path = os.path.join(self.out_dir, f"block{block_idx:02d}.json")
            json.dump({"git_commit": git_commit(), "arm": self.arm, "block": block_idx,
                       "layers": st["results"]}, open(path, "w"), indent=2)
            print(f"[dense] wrote {path}")
        # propagate held-out inputs through the (now quantized) block, like quant_inps
        if self.heldout_inps is not None:
            for j in range(len(self.heldout_inps)):
                self.heldout_inps[j] = block(self.heldout_inps[j].unsqueeze(0), **self.kw)[0]
