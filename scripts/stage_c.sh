#!/usr/bin/env bash
# stage_c.sh "<act-order flags>"
#
# All 11 runs in ONE wave. Each run is a single-threaded Python row loop pinned to
# one core (measured: 100% of one core, ~1.3 GB GPU), so 11 jobs on 16 vCPUs do not
# contend -- the five decision runs finish no later than they would in a wave of
# five, and the six attribution runs come free alongside them.
# Launch order still follows runbook section 5 priority.
set -uo pipefail
AO="${1:-}"
R=/home/boa-kronecker-gap/scripts/run_qwen.sh
D=/home/jl_fs/results/qwen05b_mirror
echo "[stageC] act-order: '${AO}'  ($(date -u))"

for s in 0 1 2; do nohup $R combined    $s $AO --qk_quantK --row_metric_v > $D/drv_combined_$s.txt 2>&1 & done
for s in 1 2;   do nohup $R boa         $s $AO                            > $D/drv_boa_$s.txt       2>&1 & done
for s in 0 1 2; do nohup $R qk-quantK   $s $AO --qk_quantK                > $D/drv_qk_$s.txt        2>&1 & done
for s in 0 1 2; do nohup $R v-rowmetric $s $AO --row_metric_v             > $D/drv_vrm_$s.txt       2>&1 & done
wait
echo "[stageC] all runs complete ($(date -u))"
