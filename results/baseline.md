# Phase 0 — Reference numbers for OPT-125m

Source: BoA, ICML 2025, arXiv 2406.13474. Numbers below are **Table 10**
("Weight-only quantization performance on OPT models without transformation"),
which is the table matching our setting (no rotation/Hadamard transform).

Paper setup (§4.1), verified to match ours:
- 128 random sequences of length 2048 from WikiText-2 as calibration data
- WikiText-2 test PPL, seqlen 2048
- single NVIDIA H100 80GB
- act-order: "we conduct experiments with and without this heuristic and report
  the better results" → our 4-way act_order sweep per bit-width is the right protocol.

## Paper numbers — OPT-125M, WikiText-2 PPL (lower better)

| Precision | Method   | Wiki2 PPL |
|-----------|----------|-----------|
| FP16      | Baseline | 27.65     |
| INT2      | RTN      | 1.0e4     |
| INT2      | GPTQ     | 411.3     |
| INT2      | **BoA**  | **85.63** |
| INT3      | RTN      | 233.9     |
| INT3      | GPTQ     | 50.75     |
| INT3      | **BoA**  | **31.95** |

Notes:
- The paper reports **no INT4 row for OPT-125M**. Our W4 runs therefore have no
  paper reference; they are recorded as our own baseline only.
- Handoff quoted the GPTQ paper's anchors (W4 ~31, W3 ~54). The BoA paper's own
  GPTQ W3 is 50.75. Both are in the same range; we anchor to the BoA paper since
  the stop/go criterion is stated against "the paper's OPT-125m rows".
- The paper reports no C4 PPL for OPT; our c4-new column is recorded for
  completeness and is not part of the stop/go check.

## Our measured FP16 baseline

| Metric    | Ours   | Paper | Delta |
|-----------|--------|-------|-------|
| wikitext2 | 27.654 | 27.65 | +0.004 ✓ |
| c4-new    | 26.564 | n/a   | n/a   |

FP16 stop/go criterion (27.65 +/- 0.05): **MET**.

## Environment deviations from requirements.txt (recorded for reproducibility)

| Package      | requirements.txt | installed | why |
|--------------|------------------|-----------|-----|
| torch        | 2.1.0+cu121      | 2.5.1+cu124 | preinstalled on the H100 image; not changed to avoid a CUDA-stack rebuild |
| transformers | 4.53.0           | 4.53.0    | pinned as specified |
| datasets     | 3.5.0            | 3.5.0     | pinned as specified |
| accelerate   | 0.33.0           | 0.33.0    | pinned as specified |
| numpy        | 1.26.4           | 1.26.4    | pinned as specified |

Model loading: `facebook/opt-125m` ships only `pytorch_model.bin` (no safetensors),
and transformers 4.53 refuses `torch.load` unless torch>=2.6 (CVE-2025-32434).
Rather than upgrade torch, the checkpoint was converted once to safetensors at
`/home/models/opt-125m` (bit-identical weights; `lm_head.weight` deduplicated as a
shared tensor with `model.decoder.embed_tokens.weight`). All runs use
`--llm_path /home/models/opt-125m`, which keeps `llm_type == "opt"` so cache
filenames and code paths are unchanged.

Note: OPT is loaded without `attn_implementation="eager"` (upstream `get_opt` does
not pass it, unlike `get_llama`/`get_qwen2`). With `--block_v`, transformers 4.53
silently falls back to eager for the `output_attentions=True` call and warns. The
attention weights are correct; the only consequence is that the Hessian pass uses
eager attention while the activation-propagation pass uses SDPA.
