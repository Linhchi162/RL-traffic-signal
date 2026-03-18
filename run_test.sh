#!/bin/bash
# run_test.sh — Smoke test: 9 jobs x 30k steps de kiem tra loi truoc khi chay full
set -euo pipefail
cd "$(dirname "$0")"

export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
export LIBSUMO_AS_TRACI="1"
export OPENBLAS_NUM_THREADS="1"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"

if [ -f ".venv/bin/activate" ]; then source .venv/bin/activate; fi

STEPS=30000
SEED=42
EXP_DIR="./experiments_test"
LOG_DIR="./logs_test"

DQN_REWARDS=("queue" "pressure" "wait-clip")
PPO_REWARDS=("queue" "pressure" "average-speed")

rm -rf "$EXP_DIR" "$LOG_DIR"
mkdir -p "$LOG_DIR"

# Sinh flow file neu chua co
mkdir -p generated_flows
[ -f generated_flows/train_s0.xml ] || python generate_train_flows.py --seed 42 --out generated_flows/train_s0.xml

declare -a ALL_PIDS ALL_LABELS

_launch() {
    local label="$1"; shift
    mkdir -p "$EXP_DIR/$label"
    "$@" > "$LOG_DIR/${label}.log" 2>&1 &
    local pid=$!
    ALL_PIDS+=($pid)
    ALL_LABELS+=("$label")
    printf "  %-50s  PID=%d\n" "$label" $pid
}

echo ""
echo ">>> Launch 9 jobs (30k steps, seed=$SEED)..."

for reward in "${DQN_REWARDS[@]}"; do
    _launch "dqn_${reward}_s${SEED}" \
        python train_dqn.py --algo dqn --reward_type "$reward" --seed $SEED \
        --total_steps $STEPS --sim_duration 5000 --n_envs 8 --log_every 5000 \
        --save_dir "$EXP_DIR/dqn_${reward}_s${SEED}"
done

for reward in "${DQN_REWARDS[@]}"; do
    _launch "ddqn_${reward}_s${SEED}" \
        python train_dqn.py --algo ddqn --reward_type "$reward" --seed $SEED \
        --total_steps $STEPS --sim_duration 5000 --n_envs 8 --log_every 5000 \
        --save_dir "$EXP_DIR/ddqn_${reward}_s${SEED}"
done

for reward in "${PPO_REWARDS[@]}"; do
    _launch "ppo_${reward}_s${SEED}" \
        python train_ppo.py --reward_type "$reward" --obs_mode raw --seed $SEED \
        --total_steps $STEPS --sim_duration 5000 --lr 3e-4 --n_envs 2 --log_every 5000 \
        --save_dir "$EXP_DIR/ppo_${reward}_s${SEED}"
done

echo ""
echo "  ${#ALL_PIDS[@]} jobs dang chay. Logs: $LOG_DIR/"
echo ""

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

echo ""
echo "========================================"
OK=$(( ${#ALL_PIDS[@]} - FAIL ))
echo "  OK: $OK/${#ALL_PIDS[@]}"
if [ $FAIL -gt 0 ]; then
    echo "  FAIL ($FAIL):"
    for lbl in "${FAIL_LABELS[@]}"; do
        echo "    - $lbl"
        echo "    --- tail log ---"
        tail -20 "$LOG_DIR/${lbl}.log" | sed 's/^/    /'
    done
fi
echo "========================================"
