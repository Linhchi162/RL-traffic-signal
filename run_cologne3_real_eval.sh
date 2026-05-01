#!/bin/bash
# run_cologne3_real_eval.sh
# Danh gia DQN + DDQN + PPO tren Cologne3 luu luong that
# So sanh voi Random / Fixed-time / Webster baseline
#
# Chay sau khi instance_cologne3_real_all.sh hoan thanh.
# Usage: cd /workspace/rl-traffic && bash run_cologne3_real_eval.sh

set -e
cd /workspace/rl-traffic
source .venv/bin/activate

export SUMO_HOME="/usr/share/sumo"
export LIBSUMO_AS_TRACI=1

NCPU=$(nproc)
WORKERS=$(( NCPU < 16 ? NCPU : 16 ))
SAVE_DIR="./results_cologne3_real"

echo "=== Cologne3 real-traffic evaluation ==="
echo "Workers : $WORKERS / $NCPU CPU"
echo "Models  : ./exp_cologne3_real"
echo "Results : $SAVE_DIR"
echo ""

python evaluate_cologne3_real.py \
    --models_dir ./exp_cologne3_real \
    --save_dir   "$SAVE_DIR" \
    --workers    "$WORKERS" \
    --resume 2>&1 | tee "$SAVE_DIR/run.log"

echo ""
echo "Ket qua: $SAVE_DIR/summary.csv"
