#!/usr/bin/env bash
# stage_c.sh "<act-order flags>"
# Wave 1 = the five decision runs (combined x3, boa s1/s2). Wave 2 = the six
# attribution runs. Two waves rather than one so the decision table is banked
# before the lower-priority attribution arms start, per runbook section 5.
set -uo pipefail
AO="${1:-}"
R=/home/boa-kronecker-gap/scripts/run_qwen.sh
D=/home/jl_fs/results/qwen05b_mirror

echo "[stageC] act-order: '${AO}'"

echo "[stageC] wave 1: combined s0,s1,s2 + boa s1,s2"
for s in 0 1 2; do
  nohup $R combined $s $AO --qk_quantK --row_metric_v > $D/drv_combined_$s.txt 2>&1 &
done
for s in 1 2; do
  nohup $R boa $s $AO > $D/drv_boa_$s.txt 2>&1 &
done
wait
echo "[stageC] wave 1 complete"

echo "[stageC] wave 2: qk-quantK s0-2, v-rowmetric s0-2"
for s in 0 1 2; do
  nohup $R qk-quantK   $s $AO --qk_quantK    > $D/drv_qk_$s.txt 2>&1 &
  nohup $R v-rowmetric $s $AO --row_metric_v > $D/drv_vrm_$s.txt 2>&1 &
done
wait
echo "[stageC] wave 2 complete"
