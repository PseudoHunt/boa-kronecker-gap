#!/bin/bash
# Phase 0: BoA baseline sweep on OPT-125m, wikitext2 calib, W2/W3/W4 x 4 act-order configs
set -u
MODEL=/home/models/opt-125m
mkdir -p logs/phase0
for wb in 4 3 2; do
  for act in "none" "col" "row" "colrow"; do
    case $act in
      none)   FLAGS="" ;;
      col)    FLAGS="--act_order_col" ;;
      row)    FLAGS="--act_order_row" ;;
      colrow) FLAGS="--act_order_col --act_order_row" ;;
    esac
    OUT=logs/phase0/w${wb}_${act}.log
    if [ -s "$OUT" ] && grep -q "^{'wikitext2'" "$OUT"; then echo "skip w${wb}_${act}"; continue; fi
    echo "=== RUN w_bits=$wb act=$act ==="
    python3 main.py --llm_path $MODEL --w_bits $wb --block_v --qparam_comput Hessian $FLAGS > "$OUT" 2>&1
    echo "  -> $(grep -E "^\{'wikitext2'" "$OUT" | tail -1)"
  done
done
echo "ALL DONE"
