#!/bin/bash
# retrain_full.sh — Retrain toàn bộ với seed mở rộng + PPO fix
#
# PPO: 8 seeds × 3 rewards = 24 jobs (sim_duration=7200 — fix episode length)
# DQN: 4 seeds mới × wait-clip = 4 jobs  (giữ 4 seeds cũ, chỉ thêm mới)
# DDQN: 4 seeds mới × wait-clip = 4 jobs
# Tổng: 32 jobs song song, ~64 CPU cores, xong trong ~1h

set -euo pipefail
cd "$(dirname "$0")"

export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
export LIBSUMO_AS_TRACI="1"
export OPENBLAS_NUM_THREADS="1"
export OMP_NUM_THREADS="1"

if [ -f ".venv/bin/activate" ]; then source .venv/bin/activate; fi

# ---- Config ----
STEPS=200000
SIM_DUR=7200     # fix: 7200s/ep → ~139 eps (cu: 205000s/ep → 5 eps)
EXP_DIR="./experiments"
LOG_DIR="./logs_full"

OLD_SEEDS=(42 123 777 999)
NEW_SEEDS=(314 2025 2718 9999)
ALL_SEEDS=("${OLD_SEEDS[@]}" "${NEW_SEEDS[@]}")

PPO_REWARDS=("queue" "pressure" "wait-clip")

mkdir -p "$LOG_DIR"

declare -a ALL_PIDS ALL_LABELS

# -----------------------------------------------------------------------
# PPO: 8 seeds × 3 rewards = 24 jobs
# Xoa cu de tranh resume checkpoint sai (sim_duration cu khac)
# -----------------------------------------------------------------------
echo ""
echo ">>> Xoa PPO experiments cu (se retrain voi sim_duration=$SIM_DUR)..."
for seed in "${ALL_SEEDS[@]}"; do
    for reward in "${PPO_REWARDS[@]}"; do
        rm -rf "$EXP_DIR/ppo_${reward}_single_s${seed}"
    done
done
echo "    Done."

echo ""
echo ">>> Launch 24 PPO jobs (n_envs=2, sim_duration=$SIM_DUR)..."
printf "    %-50s  %s\n" "JOB" "PID"
printf "    %-50s  %s\n" "---" "---"

for seed in "${ALL_SEEDS[@]}"; do
    for reward in "${PPO_REWARDS[@]}"; do
        label="ppo_${reward}_single_s${seed}"
        sd="$EXP_DIR/$label"
        mkdir -p "$sd"
        python train_ppo.py \
            --reward_type "$reward" \
            --obs_mode raw \
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
        printf "    %-50s  %d\n" "$label" $pid
    done
done

# -----------------------------------------------------------------------
# DQN wait-clip: 4 seeds mới (giữ nguyên 4 seeds cũ)
# -----------------------------------------------------------------------
echo ""
echo ">>> Launch 4 DQN wait-clip jobs (seeds moi)..."

for seed in "${NEW_SEEDS[@]}"; do
    label="dqn_wait-clip_s${seed}"
    sd="$EXP_DIR/$label"
    mkdir -p "$sd"
    python train_dqn.py \
        --algo dqn \
        --reward_type wait-clip \
        --seed "$seed" \
        --total_steps $STEPS \
        --n_envs 2 \
        --buffer_size 500000 \
        --save_dir "$sd" \
        > "$LOG_DIR/${label}.log" 2>&1 &
    pid=$!
    ALL_PIDS+=($pid)
    ALL_LABELS+=("$label")
    printf "    %-50s  %d\n" "$label" $pid
done

# -----------------------------------------------------------------------
# DDQN wait-clip: 4 seeds mới
# -----------------------------------------------------------------------
echo ""
echo ">>> Launch 4 DDQN wait-clip jobs (seeds moi)..."

for seed in "${NEW_SEEDS[@]}"; do
    label="ddqn_wait-clip_s${seed}"
    sd="$EXP_DIR/$label"
    mkdir -p "$sd"
    python train_dqn.py \
        --algo ddqn \
        --reward_type wait-clip \
        --seed "$seed" \
        --total_steps $STEPS \
        --n_envs 2 \
        --buffer_size 500000 \
        --save_dir "$sd" \
        > "$LOG_DIR/${label}.log" 2>&1 &
    pid=$!
    ALL_PIDS+=($pid)
    ALL_LABELS+=("$label")
    printf "    %-50s  %d\n" "$label" $pid
done

echo ""
echo "    32 jobs dang chay (~1h)."
echo "    Theo doi PPO : tail -f $LOG_DIR/ppo_queue_single_s42.log"
echo "    Theo doi DQN : tail -f $LOG_DIR/dqn_wait-clip_s314.log"
echo ""

# -----------------------------------------------------------------------
# Wait tất cả
# -----------------------------------------------------------------------
TOTAL=${#ALL_PIDS[@]}
FAIL=0

for i in "${!ALL_PIDS[@]}"; do
    if wait "${ALL_PIDS[$i]}"; then
        echo "  [OK]   ${ALL_LABELS[$i]}"
    else
        echo "  [FAIL] ${ALL_LABELS[$i]}"
        FAIL=$((FAIL + 1))
    fi
done

OK=$((TOTAL - FAIL))
echo ""
echo "=========================================================="
echo "  Hoan tat: $OK/$TOTAL OK"
echo ""
echo "  Buoc tiep theo:"
echo "  1. Eval PPO best checkpoints:"
echo "     python eval_checkpoints.py --all_ppo --eval_duration 1800"
echo "  2. Eval toan bo:"
echo "     python evaluate_all.py --models_dir ./experiments --save_dir ./results_final --skip_ae"
echo "  3. Phan tich:"
echo "     python analyze_results.py --results_dir ./results_final"
echo "=========================================================="
