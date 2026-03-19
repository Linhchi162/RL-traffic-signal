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

TOTAL=${#ALL_PIDS[@]}
echo ""
echo "    $TOTAL jobs dang chay. Log: $LOG_DIR/"
echo ""

# -----------------------------------------------------------------------
# Progress bar — cap nhat moi 15 giay
# -----------------------------------------------------------------------
GREEN="\033[32m"; YELLOW="\033[33m"; RESET="\033[0m"; BOLD="\033[1m"

declare -a DONE_FLAGS
for i in "${!ALL_PIDS[@]}"; do DONE_FLAGS[$i]=0; done

_bar() {
    local done=$1 total=$2 width=30
    local filled=$(( done * width / total ))
    local empty=$(( width - filled ))
    local bar=""
    [ $filled -gt 0 ] && bar+=$(printf '█%.0s' $(seq 1 $filled))
    [ $empty  -gt 0 ] && bar+=$(printf '░%.0s' $(seq 1 $empty))
    echo "$bar"
}

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
                echo -e "  ${GREEN}✓${RESET} ${ALL_LABELS[$i]}"
            else
                echo -e "  ✗ ${ALL_LABELS[$i]}"
                FAIL=$((FAIL + 1))
            fi
        fi
    done

    PCT=$(( DONE_COUNT * 100 / TOTAL ))
    BAR=$(_bar $DONE_COUNT $TOTAL)
    [ $PCT -ge 50 ] && C=$GREEN || C=$YELLOW
    printf "\r${BOLD}[%s]${RESET} ${C}%s${RESET} %d/%d (%d%%)   " \
        "$(date '+%H:%M:%S')" "$BAR" $DONE_COUNT $TOTAL $PCT

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
echo "  1. Eval PPO best checkpoints:"
echo "     python eval_checkpoints.py --all_ppo --eval_duration 1800"
echo "  2. Eval toan bo:"
echo "     python evaluate_all.py --models_dir ./experiments --save_dir ./results_final --skip_ae"
echo "  3. Phan tich:"
echo "     python analyze_results.py --results_dir ./results_final"
echo "=========================================================="
