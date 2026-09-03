#!/bin/bash
# Phase 2a: paper eq.(9) value row metric (--row_metric_v) vs baseline.
# Best act-order config per bit-width, fixed from the corrected Phase 0 sweep.
set -u
MODEL=/home/models/opt-125m
mkdir -p logs/phase2a
declare -A ACT=( [4]="--act_order_col --act_order_row" [3]="--act_order_col --act_order_row" [2]="--act_order_row" )
declare -A TAG=( [4]="colrow" [3]="colrow" [2]="row" )
for wb in 3 2 4; do
  for arm in base rowv; do
    EXTRA=""; [ "$arm" = "rowv" ] && EXTRA="--row_metric_v"
    for seed in 0 1 2; do
      OUT=logs/phase2a/w${wb}_${TAG[$wb]}_${arm}_s${seed}.log
      if [ -s "$OUT" ] && grep -q "^{'wikitext2'" "$OUT"; then echo "skip $OUT"; continue; fi
      echo "=== w${wb} ${arm} seed${seed} ==="
      python3 -u main.py --llm_path $MODEL --w_bits $wb --block_v --qparam_comput Hessian \
          --attn_impl eager --seed $seed ${ACT[$wb]} $EXTRA > "$OUT" 2>&1
      echo "  -> $(grep -E "^\{'wikitext2'" "$OUT" | tail -1)"
    done
  done
done
echo "ALL DONE"
