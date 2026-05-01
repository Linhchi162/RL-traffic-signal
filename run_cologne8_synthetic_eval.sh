#!/bin/bash
# run_cologne8_synthetic_eval.sh
# Danh gia tren Cologne8 voi luong xe trung binh cua 6 kich ban train (~1179 veh/hr)
# Dung de kiem tra zero-shot transfer trong dieu kien demand tuong tu training.
#
# Usage: cd /workspace/rl-traffic && bash run_cologne8_synthetic_eval.sh

set -e

WORK_DIR="/workspace/rl-traffic"
cd "$WORK_DIR"

source .venv/bin/activate

export SUMO_HOME="/usr/share/sumo"
export LIBSUMO_AS_TRACI=1

SYN_ROUTE="$WORK_DIR/nets/cologne8/cologne8_synthetic.rou.xml"

echo "=== Cologne8 synthetic eval (demand ~ training distribution) ==="
echo "Demand: ~1179 veh/hr (trung binh 6 kich ban train)"
echo ""

# Buoc 1: Sinh route file neu chua co
if [ ! -f "$SYN_ROUTE" ]; then
    echo "[1/2] Sinh synthetic route file..."
    python generate_cologne8_flows.py --vph 1179 --seed 42
    echo ""
else
    echo "[1/2] Route file da ton tai: $SYN_ROUTE"
fi

# Buoc 2: Chay eval
NCPU=$(nproc)
WORKERS=$(( NCPU < 16 ? NCPU : 16 ))
SAVE_DIR="$WORK_DIR/results_cologne8_synthetic"
LOG="$SAVE_DIR/run.log"

echo "[2/2] Chay eval voi $WORKERS workers..."
echo "Save dir: $SAVE_DIR"
echo "Log     : $LOG"
echo ""

python evaluate_cologne8.py \
    --models_dir ./exp_grid \
    --save_dir   "$SAVE_DIR" \
    --route_file "$SYN_ROUTE" \
    --workers    "$WORKERS" \
    --resume 2>&1 | tee "$LOG"

echo ""
echo "Ket qua: $SAVE_DIR"
