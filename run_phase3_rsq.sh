#!/bin/bash
# Repurposed Phase 3, non-dense arms: boa (objective-eval baseline), rsq-col, boa+rsq.
# Best act-order per bit-width from corrected Phase 0. Held-out objective eval on blocks 0,5,11.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=/home/models/opt-125m
mkdir -p logs/phase3 results/phase3
declare -A ACT=( [3]="--act_order_col --act_order_row" [2]="--act_order_row" )
declare -A TAG=( [3]="colrow" [2]="row" )
for wb in 2 3; do
  for arm in boa rsq-col boa+rsq; do
    case $arm in
      boa)     EXTRA="" ;;
      rsq-col) EXTRA="--rsq_weights --rsq_col_only" ;;
      boa+rsq) EXTRA="--rsq_weights" ;;
    esac
    for seed in 0 1 2; do
      OUT=logs/phase3/${arm}_w${wb}_${TAG[$wb]}_s${seed}.log
      if [ -s "$OUT" ] && grep -q "^{'wikitext2'" "$OUT"; then echo "skip $OUT"; continue; fi
      echo "=== ${arm} w${wb} seed${seed} ==="
      python3 -u main.py --llm_path $MODEL --w_bits $wb --block_v --qparam_comput Hessian --attn_impl eager \
        --seed $seed ${ACT[$wb]} $EXTRA --obj_eval --dense_blocks 0,5,11 --heldout_nsamples 32 \
        --dense_dir results/phase3/${arm}_w${wb}_s${seed} > "$OUT" 2>&1
      echo "  -> $(grep -E "^\{'wikitext2'" "$OUT" | tail -1)"
    done
  done
done
echo "ALL DONE"
