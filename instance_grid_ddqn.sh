#!/bin/bash
# instance_grid_ddqn.sh — Train 24 grid-DDQN jobs (8 seeds × 3 rewards) on one Vast.ai instance
# Multi-agent IPPO-style DDQN: 4 junctions, shared weights, MultiAgentVecEnv

set -euo pipefail
cd "$(dirname "$0")"

export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
export LIBSUMO_AS_TRACI="1"
export OPENBLAS_NUM_THREADS="1"
export OMP_NUM_THREADS="1"
export CUDA_VISIBLE_DEVICES=""

if [ -f ".venv/bin/activate" ]; then source .venv/bin/activate; fi

STEPS=200000
SIM_DUR=100000
EXP_DIR="./exp_grid"
LOG_DIR="./logs_grid_ddqn"
SEEDS=(42 123 777 999 314 2025 2718 9999)
REWARDS=("queue" "pressure" "wait-clip")

mkdir -p "$LOG_DIR"

declare -a ALL_PIDS ALL_LABELS

echo ">>> Launch 24 grid-DDQN jobs..."
for seed in "${SEEDS[@]}"; do
    for reward in "${REWARDS[@]}"; do
        label="ddqn_${reward}_s${seed}"
        sd="$EXP_DIR/$label"
        rm -rf "$sd"
        mkdir -p "$sd"
        python train_grid_dqn.py \
            --algo ddqn \
            --reward_type "$reward" \
            --seed "$seed" \
            --total_steps $STEPS \
            --sim_duration $SIM_DUR \
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
echo "  $TOTAL grid-DDQN jobs running. Logs: $LOG_DIR/"
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
echo "=========================================="
echo "  grid-DDQN done: $OK/$TOTAL OK"
echo "=========================================="
