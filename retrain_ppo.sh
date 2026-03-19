#!/bin/bash
# retrain_ppo.sh — Retrain 12 PPO jobs (3 rewards x 4 seeds)
# Dung sau khi da co DQN/DDQN tu run_vast.sh

set -euo pipefail
cd "$(dirname "$0")"

export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
export LIBSUMO_AS_TRACI="1"
export OPENBLAS_NUM_THREADS="1"
export OMP_NUM_THREADS="1"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

STEPS=200000
EXP_DIR="./experiments"
LOG_DIR="./logs_vast"
PPO_REWARDS=("queue" "pressure" "wait-clip")
SEEDS=(42 123 777 999)

mkdir -p "$LOG_DIR"

# Xoa PPO cu (tranh checkpoint resume sai)
echo ">>> Xoa PPO experiments cu..."
rm -rf "$EXP_DIR"/ppo_*
echo "    Done."

echo ""
echo ">>> Launch 12 PPO jobs..."
printf "    %-48s  %s\n" "JOB" "PID"
printf "    %-48s  %s\n" "---" "---"

declare -a ALL_PIDS
declare -a ALL_LABELS

for seed in "${SEEDS[@]}"; do
    for reward in "${PPO_REWARDS[@]}"; do
        label="ppo_${reward}_single_s${seed}"
        sd="$EXP_DIR/$label"
        mkdir -p "$sd"
        python train_ppo.py \
            --reward_type "$reward" \
            --obs_mode raw \
            --seed "$seed" \
            --total_steps $STEPS \
            --lr 3e-4 \
            --n_envs 4 \
            --save_dir "$sd" \
            > "$LOG_DIR/${label}.log" 2>&1 &
        pid=$!
        ALL_PIDS+=($pid)
        ALL_LABELS+=("$label")
        printf "    %-48s  %d\n" "$label" $pid
    done
done

echo ""
echo "    12 jobs dang chay (~1h)."
echo "    Theo doi: tail -f $LOG_DIR/ppo_queue_single_s42.log"
echo ""

FAIL=0
for i in "${!ALL_PIDS[@]}"; do
    if wait "${ALL_PIDS[$i]}"; then
        echo "  [OK]   ${ALL_LABELS[$i]}"
    else
        echo "  [FAIL] ${ALL_LABELS[$i]}"
        FAIL=$((FAIL + 1))
    fi
done

OK=$((12 - FAIL))
echo ""
echo "=========================================================="
echo "  PPO retrain xong: $OK/12 OK"
echo ""
echo "  Chay danh gia:"
echo "  python evaluate_parallel.py --jobs 36 --eval_duration 3600 --save_dir ./results_final"
echo "=========================================================="
