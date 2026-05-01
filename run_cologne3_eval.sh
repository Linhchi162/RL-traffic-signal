#!/bin/bash
# run_cologne3_eval.sh — Danh gia zero-shot tren Cologne3 (3 nut giao that, CHLB Duc)
#
# Usage: cd /workspace/rl-traffic && bash run_cologne3_eval.sh

set -e

WORK_DIR="/workspace/rl-traffic"
cd "$WORK_DIR"

source .venv/bin/activate

export SUMO_HOME="/usr/share/sumo"
export LIBSUMO_AS_TRACI=1

NET="$WORK_DIR/nets/cologne3/cologne3.net.xml"
ROUTE="$WORK_DIR/nets/cologne3/cologne3.rou.xml"

# Download neu chua co
if [ ! -f "$NET" ]; then
    echo "Downloading cologne3..."
    RESCO="https://raw.githubusercontent.com/Pi-Star-Lab/RESCO/main/resco_benchmark/environments/cologne3"
    mkdir -p "$WORK_DIR/nets/cologne3"
    wget -q "$RESCO/cologne3.net.xml" -O "$NET"
    wget -q "$RESCO/cologne3.rou.xml" -O "$ROUTE"
    echo "    OK"
fi

NCPU=$(nproc)
WORKERS=$(( NCPU < 16 ? NCPU : 16 ))
SAVE_DIR="$WORK_DIR/results_cologne3"
LOG="$SAVE_DIR/run.log"

echo "=== Cologne3 zero-shot evaluation ==="
echo "Net    : $NET"
echo "Workers: $WORKERS / $NCPU CPU"
echo "Log    : $LOG"
echo ""

python evaluate_cologne8.py \
    --models_dir ./exp_grid \
    --save_dir   "$SAVE_DIR" \
    --net_file   "$NET" \
    --route_file "$ROUTE" \
    --workers    "$WORKERS" \
    --resume 2>&1 | tee "$LOG"

echo ""
echo "Ket qua: $SAVE_DIR"
