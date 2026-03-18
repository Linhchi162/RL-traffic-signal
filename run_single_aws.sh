#!/bin/bash
# run_single_aws.sh — Train 27 single-intersection experiments tren GCP
#
# DQN  x {queue, pressure, wait-clip} x {42, 123, 777}  =  9 runs
# DDQN x {queue, pressure, wait-clip} x {42, 123, 777}  =  9 runs
# PPO 7D (baseline) x {queue, pressure, wait-clip} x {42, 123, 777} = 9 runs
# Tong: 27 runs

set -e
cd "$(dirname "$0")"

export SUMO_HOME="/usr/share/sumo"
export LIBSUMO_AS_TRACI="1"
export OPENBLAS_NUM_THREADS="1"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"

source .venv/bin/activate

JOBS=9   # 27 models / 9 = 3 batch; tang len neu may co nhieu vCPU hon
STEPS=200000
EXP_DIR="./experiments"
LOG_DIR="./logs_single"
mkdir -p "$EXP_DIR" "$LOG_DIR"

SEEDS=(42 123 777)
REWARDS=("queue" "pressure" "wait-clip")

CMDS=()

for seed in "${SEEDS[@]}"; do
    for reward in "${REWARDS[@]}"; do
        # DQN single
        sd="$EXP_DIR/dqn_${reward}_single_s${seed}"
        if [ ! -f "$sd/dqn_final_model.zip" ]; then
            CMDS+=("python train_dqn.py --algo dqn --reward_type $reward --mode single --seed $seed --total_steps $STEPS --save_dir $sd > $LOG_DIR/dqn_${reward}_single_s${seed}.log 2>&1")
        else
            echo "[SKIP] dqn_${reward}_single_s${seed}"
        fi

        # DDQN single
        sd="$EXP_DIR/ddqn_${reward}_single_s${seed}"
        if [ ! -f "$sd/dqn_final_model.zip" ]; then
            CMDS+=("python train_dqn.py --algo ddqn --reward_type $reward --mode single --seed $seed --total_steps $STEPS --save_dir $sd > $LOG_DIR/ddqn_${reward}_single_s${seed}.log 2>&1")
        else
            echo "[SKIP] ddqn_${reward}_single_s${seed}"
        fi

        # PPO 7D baseline single
        sd="$EXP_DIR/ppo_${reward}_baseline_single_s${seed}"
        if [ ! -f "$sd/ppo_final_model.zip" ]; then
            CMDS+=("python train_ppo.py --reward_type $reward --obs_mode baseline --mode single --seed $seed --total_steps $STEPS --lr 3e-4 --save_dir $sd > $LOG_DIR/ppo_${reward}_baseline_single_s${seed}.log 2>&1")
        else
            echo "[SKIP] ppo_${reward}_baseline_single_s${seed}"
        fi
    done
done

TOTAL=${#CMDS[@]}
if [ $TOTAL -eq 0 ]; then
    echo "Tat ca models da train xong."
    exit 0
fi

echo "============================================"
echo "  Train $TOTAL models | $JOBS jobs song song"
echo "  Mode: single intersection"
echo "  LIBSUMO: $LIBSUMO_AS_TRACI"
echo "============================================"

if command -v parallel &>/dev/null; then
    printf '%s\n' "${CMDS[@]}" | parallel -j $JOBS --bar
else
    RUNNING=0
    PIDS=()
    LABELS=()
    IDX=0

    for cmd in "${CMDS[@]}"; do
        IDX=$((IDX + 1))
        label=$(echo "$cmd" | grep -oP '(?<=save_dir )[^ ]+' | xargs basename)
        echo "[$IDX/$TOTAL] Start: $label"
        eval "$cmd" &
        PIDS+=($!)
        LABELS+=("$label")
        RUNNING=$((RUNNING + 1))

        if [ $RUNNING -ge $JOBS ]; then
            wait "${PIDS[0]}" && echo "[DONE] ${LABELS[0]}" || echo "[FAIL] ${LABELS[0]}"
            PIDS=("${PIDS[@]:1}")
            LABELS=("${LABELS[@]:1}")
            RUNNING=$((RUNNING - 1))
        fi
    done

    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}" && echo "[DONE] ${LABELS[$i]}" || echo "[FAIL] ${LABELS[$i]}"
    done
fi

echo ""
echo "============================================"
echo "  HOAN TAT! Chay evaluate:"
echo "  python evaluate_parallel.py --jobs 4 --scope single --save_dir ./results_single"
echo "============================================"
