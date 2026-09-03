#!/bin/bash
# Repurposed Phase 3, dense arms on blocks 0,5,11 at W2 seed 0 (addendum sec. 2.6).
# Run only after the fp64 gate has passed (logs/phase3_gate6.log).
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=/home/models/opt-125m
mkdir -p logs/phase3 results/phase3
WB=${WB:-2}; ACT=${ACT:---act_order_row}; TAG=${TAG:-row}
for arm in boa mask p jac full; do
  OUT=logs/phase3/dense-${arm}_w${WB}_${TAG}_s0.log
  if [ -s "$OUT" ] && grep -q "^{'wikitext2'" "$OUT"; then echo "skip $OUT"; continue; fi
  echo "=== dense-${arm} w${WB} seed0 ==="
  python3 -u main.py --llm_path $MODEL --w_bits $WB --block_v --qparam_comput Hessian --attn_impl eager \
    --seed 0 $ACT --dense_arm $arm --dense_blocks 0,5,11 --dense_nsamples 32 --dense_tokens_per_seq 256 \
    --heldout_nsamples 32 --dense_dir results/phase3/dense-${arm}_w${WB}_s0 > "$OUT" 2>&1
  echo "  -> $(grep -E "^\{'wikitext2'" "$OUT" | tail -1)"
done
echo "ALL DONE"
