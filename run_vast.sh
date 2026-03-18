#!/bin/bash
# run_vast.sh — Train 30 models tren Vast.ai (64 CPU)
#
# 30 jobs chay DONG THOI:
#   DQN  : 3 rewards x 3 seeds =  9 jobs  (~1 CPU/job)
#   DDQN : 3 rewards x 3 seeds =  9 jobs  (~1 CPU/job)
#   PPO  : 4 rewards x 3 seeds = 12 jobs  (~2 CPU/job, SubprocVecEnv)
#   Tong CPU su dung: ~9 + 9 + 24 = ~42 CPU (con du 22 CPU cho OS + overhead)
#
# Usage:
#   bash run_vast.sh

set -euo pipefail
cd "$(dirname "$0")"

# ---- Giam conflict BLAS / toi uu hoa CPU cho SUMO ----
export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
export LIBSUMO_AS_TRACI="1"
export OPENBLAS_NUM_THREADS="1"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"

# Kich hoat virtualenv neu co
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

STEPS=200000
EXP_DIR="./experiments"
LOG_DIR="./logs_vast"

DQN_REWARDS=("queue" "pressure" "wait-clip")
PPO_REWARDS=("queue" "diff-waiting-time" "pressure" "average-speed")
SEEDS=(42 123 777)

# ===========================================================
# 1. Xoa experiments cu + tao thu muc
# ===========================================================
echo ""
echo ">>> [1/3] Don dep thu muc..."
rm -rf "$EXP_DIR" "$LOG_DIR"
mkdir -p "$LOG_DIR"
echo "    Done."

# ===========================================================
# 2. Sinh flow files (TCCS 24:2018)
# ===========================================================
echo ""
echo ">>> [2/3] Sinh flow files..."
mkdir -p generated_flows
python generate_train_flows.py --seed 42  --out generated_flows/train_s0.xml
python generate_train_flows.py --seed 123 --out generated_flows/train_s1.xml
python generate_train_flows.py --seed 777 --out generated_flows/train_s2.xml
# Test flow (seed=999, chua tung xuat hien luc train)
python generate_train_flows.py --seed 999 --out generated_flows/test_s999.xml --duration 10000
echo "    Done."

# ===========================================================
# 3. Launch 30 jobs dong thoi
# ===========================================================
echo ""
echo ">>> [3/3] Launch 30 jobs dong thoi..."
printf "    %-48s  %s\n" "JOB" "PID"
printf "    %-48s  %s\n" "---" "---"

declare -a ALL_PIDS
declare -a ALL_LABELS

_launch() {
    local label="$1"; shift
    local sd="$EXP_DIR/$label"
    mkdir -p "$sd"
    "$@" > "$LOG_DIR/${label}.log" 2>&1 &
    local pid=$!
    ALL_PIDS+=($pid)
    ALL_LABELS+=("$label")
    printf "    %-48s  %d\n" "$label" $pid
}

# --- DQN ---
for seed in "${SEEDS[@]}"; do
    for reward in "${DQN_REWARDS[@]}"; do
        _launch "dqn_${reward}_s${seed}" \
            python train_dqn.py \
                --algo dqn \
                --reward_type "$reward" \
                --seed "$seed" \
                --total_steps $STEPS \
                --save_dir "$EXP_DIR/dqn_${reward}_s${seed}"
    done
done

# --- DDQN ---
for seed in "${SEEDS[@]}"; do
    for reward in "${DQN_REWARDS[@]}"; do
        _launch "ddqn_${reward}_s${seed}" \
            python train_dqn.py \
                --algo ddqn \
                --reward_type "$reward" \
                --seed "$seed" \
                --total_steps $STEPS \
                --save_dir "$EXP_DIR/ddqn_${reward}_s${seed}"
    done
done

# --- PPO (n_envs=2: SubprocVecEnv, moi job dung ~2 CPU) ---
for seed in "${SEEDS[@]}"; do
    for reward in "${PPO_REWARDS[@]}"; do
        _launch "ppo_${reward}_single_s${seed}" \
            python train_ppo.py \
                --reward_type "$reward" \
                --obs_mode raw \
                --seed "$seed" \
                --total_steps $STEPS \
                --lr 3e-4 \
                --n_envs 2 \
                --save_dir "$EXP_DIR/ppo_${reward}_single_s${seed}"
    done
done

TOTAL=${#ALL_PIDS[@]}
echo ""
echo "    $TOTAL jobs dang chay."
echo "    Theo doi: tail -f $LOG_DIR/<name>.log"
echo "    Kiem tra tien do:"
echo "    watch -n10 'ls $EXP_DIR/*/ppo_final_model.zip $EXP_DIR/*/dqn_final_model.zip 2>/dev/null | wc -l'"
echo ""

# ===========================================================
# 4. Cho tat ca job hoan thanh
# ===========================================================
START_TIME=$SECONDS
FAIL=0
FAIL_LABELS=()

for i in "${!ALL_PIDS[@]}"; do
    pid=${ALL_PIDS[$i]}
    label=${ALL_LABELS[$i]}
    if wait "$pid"; then
        echo "  [OK]   $label"
    else
        echo "  [FAIL] $label  (xem: $LOG_DIR/${label}.log)"
        FAIL=$((FAIL + 1))
        FAIL_LABELS+=("$label")
    fi
done

ELAPSED=$(( SECONDS - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))
OK=$(( TOTAL - FAIL ))

echo ""
echo "=========================================================="
echo "  HOAN TAT sau ${HOURS}h${MINS}m"
echo "  OK: $OK/$TOTAL models"
if [ $FAIL -gt 0 ]; then
    echo "  FAIL ($FAIL):"
    for lbl in "${FAIL_LABELS[@]}"; do
        echo "    - $lbl"
    done
fi
echo ""
echo "  Zip ket qua de tai ve:"
echo "  tar -czf experiments.tar.gz experiments/ logs_vast/"
echo ""
echo "  Chay danh gia:"
echo "  python evaluate_parallel.py --jobs 8 --save_dir ./results_single"
echo "=========================================================="
