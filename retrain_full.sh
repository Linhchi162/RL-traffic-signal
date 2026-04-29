#!/bin/bash
# retrain_full.sh — Retrain toàn bộ experiments
#
# PPO:  8 seeds × 3 rewards = 24 jobs
# DQN:  8 seeds × 3 rewards = 24 jobs
# DDQN: 8 seeds × 3 rewards = 24 jobs
# Tổng: 72 jobs song song, CPU only

set -euo pipefail
cd "$(dirname "$0")"

export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
export LIBSUMO_AS_TRACI="1"
export OPENBLAS_NUM_THREADS="1"
export OMP_NUM_THREADS="1"
export CUDA_VISIBLE_DEVICES=""

if [ -f ".venv/bin/activate" ]; then source .venv/bin/activate; fi

STEPS=200000
SIM_DUR=7200
EXP_DIR="./experiments"
LOG_DIR="./logs_full"

SEEDS=(42 123 777 999 314 2025 2718 9999)
PPO_REWARDS=("queue" "pressure" "wait-clip")
DQN_REWARDS=("queue" "pressure" "wait-clip")

mkdir -p "$LOG_DIR"

declare -a ALL_PIDS ALL_LABELS

# -----------------------------------------------------------------------
# PPO: 8 seeds × 3 rewards = 24 jobs
# -----------------------------------------------------------------------
echo ""
echo ">>> Xoa PPO experiments cu (retrain voi sim_duration=$SIM_DUR)..."
for seed in "${SEEDS[@]}"; do
    for reward in "${PPO_REWARDS[@]}"; do
        rm -rf "$EXP_DIR/ppo_${reward}_single_s${seed}"
    done
done

echo ""
echo ">>> Launch 24 PPO jobs..."
for seed in "${SEEDS[@]}"; do
    for reward in "${PPO_REWARDS[@]}"; do
        label="ppo_${reward}_single_s${seed}"
        sd="$EXP_DIR/$label"
        mkdir -p "$sd"
        python train_ppo.py \
            --reward_type "$reward" \
            --obs_mode baseline \
            --seed "$seed" \
            --total_steps $STEPS \
            --sim_duration $SIM_DUR \
            --lr 3e-4 \
            --n_envs 2 \
            --save_dir "$sd" \
            > "$LOG_DIR/${label}.log" 2>&1 &
        pid=$!
        ALL_PIDS+=($pid)
        ALL_LABELS+=("$label")
        printf "  %-50s  PID %d\n" "$label" $pid
    done
done

# -----------------------------------------------------------------------
# DQN: 8 seeds × 3 rewards = 24 jobs
# -----------------------------------------------------------------------
echo ""
echo ">>> Launch 24 DQN jobs..."
for seed in "${SEEDS[@]}"; do
    for reward in "${DQN_REWARDS[@]}"; do
        label="dqn_${reward}_s${seed}"
        sd="$EXP_DIR/$label"
        mkdir -p "$sd"
        python train_dqn.py \
            --algo dqn \
            --reward_type "$reward" \
            --seed "$seed" \
            --total_steps $STEPS \
            --n_envs 2 \
            --buffer_size 500000 \
            --save_dir "$sd" \
            > "$LOG_DIR/${label}.log" 2>&1 &
        pid=$!
        ALL_PIDS+=($pid)
        ALL_LABELS+=("$label")
        printf "  %-50s  PID %d\n" "$label" $pid
    done
done

# -----------------------------------------------------------------------
# DDQN: 8 seeds × 3 rewards = 24 jobs
# -----------------------------------------------------------------------
echo ""
echo ">>> Launch 24 DDQN jobs..."
for seed in "${SEEDS[@]}"; do
    for reward in "${DQN_REWARDS[@]}"; do
        label="ddqn_${reward}_s${seed}"
        sd="$EXP_DIR/$label"
        mkdir -p "$sd"
        python train_dqn.py \
            --algo ddqn \
            --reward_type "$reward" \
            --seed "$seed" \
            --total_steps $STEPS \
            --n_envs 2 \
            --buffer_size 500000 \
            --save_dir "$sd" \
            > "$LOG_DIR/${label}.log" 2>&1 &
        pid=$!
        ALL_PIDS+=($pid)
        ALL_LABELS+=("$label")
        printf "  %-50s  PID %d\n" "$label" $pid
    done
done

TOTAL=${#ALL_PIDS[@]}
echo ""
echo "  $TOTAL jobs dang chay. Log: $LOG_DIR/"
echo ""

declare -a DONE_FLAGS
for i in "${!ALL_PIDS[@]}"; do DONE_FLAGS[$i]=0; done
FAIL=0

while true; do
    DONE_COUNT=0
    for i in "${!ALL_PIDS[@]}"; do
        if [ "${DONE_FLAGS[$i]}" -eq 1 ]; then
            DONE_COUNT=$((DONE_COUNT + 1))
        elif ! kill -0 "${ALL_PIDS[$i]}" 2>/dev/null; then
            DONE_FLAGS[$i]=1
            DONE_COUNT=$((DONE_COUNT + 1))
            if wait "${ALL_PIDS[$i]}"; then
                echo "  OK  ${ALL_LABELS[$i]}"
            else
                echo "  ERR ${ALL_LABELS[$i]}"
                FAIL=$((FAIL + 1))
            fi
        fi
    done
    PCT=$(( DONE_COUNT * 100 / TOTAL ))
    printf "\r  [%s] %d/%d (%d%%)   " "$(date '+%H:%M:%S')" $DONE_COUNT $TOTAL $PCT
    [ $DONE_COUNT -eq $TOTAL ] && break
    sleep 15
done

echo ""
OK=$((TOTAL - FAIL))
echo ""
echo "=========================================================="
echo "  Hoan tat: $OK/$TOTAL OK"
echo ""
echo "  Buoc tiep theo:"
echo "  1. python evaluate_all.py --models_dir ./experiments \\"
echo "          --save_dir ./results_final --skip_ae"
echo "  2. python analyze_results.py --results_dir ./results_final"
echo "=========================================================="
