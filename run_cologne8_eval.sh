#!/bin/bash
# run_cologne8_eval.sh — Chay evaluate_cologne8.py tren Vast.ai
#
# Usage (sau khi setup_vast.sh da chay xong):
#   cd /workspace/rl-traffic && bash run_cologne8_eval.sh

set -e

WORK_DIR="/workspace/rl-traffic"
cd "$WORK_DIR"

source .venv/bin/activate

export SUMO_HOME="/usr/share/sumo"
export LIBSUMO_AS_TRACI=1

echo "=== Cologne8 zero-shot evaluation ==="
echo "Net : nets/cologne8/cologne8.net.xml"
echo "Time: 25200s-28800s (7am-8am, 3600s peak hour)"
echo ""

python evaluate_cologne8.py \
    --models_dir ./exp_grid \
    --save_dir   ./results_cologne8 \
    --workers    4 \
    --resume

echo ""
echo "Ket qua luu tai: $WORK_DIR/results_cologne8/"
