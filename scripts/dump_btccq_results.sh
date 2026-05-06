#!/usr/bin/env bash
# Dump key results from a completed cell's log + checkpoint dir.
# Usage:
#   bash scripts/dump_btccq_results.sh /workspace/logs/btccq_ld_seed0/reduced_btccq_seed0.log

set -e

LOG="$1"
if [ -z "$LOG" ] || [ ! -f "$LOG" ]; then
    echo "Usage: bash $(basename $0) /path/to/cell.log"
    exit 1
fi

NAME=$(basename "$LOG" .log)

echo "==============================================="
echo "Result dump for: $NAME"
echo "Log file:        $LOG"
echo "Log size:        $(du -h $LOG | cut -f1)"
echo "==============================================="

echo ""
echo "----- BT-CCQ Calibration (step 0) -----"
grep -E "BT-CCQ calibration|q_hat|q_tail|q_scale|q_min|positive_frac|b_iqr" "$LOG" | head -5

echo ""
echo "----- Final progress -----"
grep -E "it/s|Saved checkpoint|Switching" "$LOG" | tail -10

echo ""
echo "----- Eval scores (raw) -----"
grep -E "average_normalized_return|average_return|success_rate" "$LOG" | head -20

echo ""
echo "----- Last 30 lines of log -----"
tail -30 "$LOG"

echo ""
echo "----- Errors? -----"
grep -B1 -A5 "Traceback\|Error" "$LOG" | head -30 || echo "(no errors found)"
