# Phase 0 blocker: the Hessian pass runs non-causal attention under transformers >= 4.53

**Severity: invalidates 4 of 6 layers' Hessians in every `--block_v` run on OPT.**
Found while investigating why Phase 0 missed the paper's OPT-125m numbers by 18-240%.

## Symptom

| bits | act_order | as-released | with fix | paper (Table 10) |
|------|-----------|-------------|----------|------------------|
| W3   | row       | 37.597      | **32.105** | **31.95** |

The as-released number is 17.7% above the paper. With the fix it is +0.49%, inside
the +/-3% stop/go band. The gap was entirely this bug, not hyperparameters.

## Mechanism

1. `utils/model_utils.get_opt` loads OPT **without** `attn_implementation`.
   `get_llama`, `get_qwen2` and `get_qwen3` all pass `attn_implementation="eager"`;
   `get_opt` is the only loader that does not.
2. transformers 4.53 therefore selects **SDPA** for OPT. With no padding,
   `_update_causal_mask` returns `None` and SDPA gets causality from its
   `is_causal=True` argument instead of from a mask tensor. Confirmed: the
   `block_kwargs` captured by `cache_first_transformer_input` contain
   `attention_mask: None`.
3. `compute_Hessian` sets `output_attentions=True` when `--block_v` is on
   (quantize.py, `block_kwargs['output_attentions'] = True`). SDPA cannot return
   attention probabilities, so transformers **falls back to eager** for that call
   and warns.
4. Eager attention applies causality *only* through the mask tensor. The mask is
   `None`, so nothing is applied: the Hessian pass runs **bidirectional**.

Measured on OPT-125m block 0, sequence 0:

```
attn mass above the diagonal   = 1926.0        (0.0 if causal)
max above-diagonal probability = 6.66e-01
max|hidden_sdpa - hidden_eager| = 1.425e+01    (0.0 if the paths agreed)
```

## Blast radius

Collected during that single corrupted forward pass:

| Hessian | Affected? | Why |
|---|---|---|
| `H_col` for q/k/v (input covariance) | **No** | input to the attn projections is pre-attention (block input + LayerNorm) |
| `H_row` for q/k (`E[kk^T]`, `E[qq^T]`) | **No** | outputs of `k_proj`/`q_proj`, also pre-attention |
| `H_col` for `v_proj` under `--block_v` | **Yes** | it is `E[(A_h X)(A_h X)^T]`, built from the non-causal `A` |
| `H_col` for `out_proj`, `fc1`, `fc2` | **Yes** | their inputs are downstream of attention, so they come from the corrupted pass |

Not affected: the inter-block activation propagation (`quantize.py` line ~53) runs
with the original `block_kwargs` (`output_attentions=False`), so it uses SDPA and
stays causal. This is why the FP16 baseline is exactly right (27.654) and the bug
is invisible outside `--block_v`.

Also note the two passes silently disagreed with each other: Hessians were built on
bidirectional activations while the activations actually propagated to the next
block were causal.

## Fix

Thread an `attn_implementation` through `get_opt`/`get_model` and default it to
`eager`, matching every other loader in the repo. With eager, transformers builds
an explicit `[1, 1, L, L]` causal mask, so `block_kwargs['attention_mask']` is a
real tensor and both passes agree:

```
attn mass above the diagonal = 0.000e+00
max|hidden(output_attentions=True) - hidden(False)| = 0.000e+00
```

CLI: `--attn_impl {eager,sdpa,auto}`, default `eager`. `--attn_impl sdpa`
reproduces the incorrect upstream behaviour for comparison.

Cost: eager is slower than SDPA end to end (168 s -> 435 s per OPT-125m run).

## Why the paper is unaffected

requirements.txt pins `transformers==4.53.0`, but the OPT experiments predate it
(paper mid-2024; the requirements pin comes from a later "Update requirements.txt"
commit). Older transformers built an explicit 4-D causal mask for OPT
unconditionally, so the eager fallback stayed causal and the bug could not fire.
It is an environment-induced regression, not an error in the paper.

## Consequence for this project

Every phase depends on these Hessians, so **all Phase 0 baselines were re-run with
`--attn_impl eager`**. The as-released numbers are kept in
`results/phase0_asreleased.json` purely to document the bug. This should be
reported upstream.
