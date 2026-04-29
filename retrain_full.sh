#!/bin/bash
# retrain_full.sh — Train DQN/DDQN các seeds còn thiếu (19 jobs)
#
# PPO đã có đủ 8 seeds × 3 rewards — không cần retrain.
# Thiếu:
#   dqn  pressure:  s777 s999
#   dqn  queue:     s999
#   dqn  wait-clip: s314 s777 s999 s2025 s2718 s9999
#   ddqn pressure:  s777 s999
#   ddqn queue:     s777 s999
#   ddqn wait-clip: s314 s777 s999 s2025 s2718 s9999

set -euo pipefail
cd "$(dirname "$0")"

export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
export LIBSUMO_AS_TRACI="1"
export OPENBLAS_NUM_THREADS="1"
export OMP_NUM_THREADS="1"
export CUDA_VISIBLE_DEVICES=""

if [ -f ".venv/bin/activate" ]; then source .venv/bin/activate; fi

STEPS=200000
EXP_DIR="./experiments"
LOG_DIR="./logs_full"
mkdir -p "$LOG_DIR"

declare -a ALL_PIDS ALL_LABELS

_launch() {
    local algo=$1 reward=$2 seed=$3
    local label="${algo}_${reward}_s${seed}"
    mkdir -p "$EXP_DIR/$label"
    python train_dqn.py \
        --algo "$algo" \
        --reward_type "$reward" \
        --seed "$seed" \
        --total_steps $STEPS \
        --n_envs 2 \
        --buffer_size 500000 \
        --save_dir "$EXP_DIR/$label" \
        > "$LOG_DIR/${label}.log" 2>&1 &
    ALL_PIDS+=($!)
    ALL_LABELS+=("$label")
    printf "  %-45s  PID %d\n" "$label" $!
}

echo ">>> DQN (9 jobs)"
_launch dqn pressure 777
_launch dqn pressure 999
_launch dqn queue    999
for s in 314 777 999 2025 2718 9999; do _launch dqn wait-clip $s; done

echo ""
echo ">>> DDQN (10 jobs)"
_launch ddqn pressure  777
_launch ddqn pressure  999
_launch ddqn queue     777
_launch ddqn queue     999
for s in 314 777 999 2025 2718 9999; do _launch ddqn wait-clip $s; done

TOTAL=${#ALL_PIDS[@]}
echo ""
echo "  $TOTAL jobs dang chay. Log: $LOG_DIR/"

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
    printf "\r  [%s] %d/%d   " "$(date '+%H:%M:%S')" $DONE_COUNT $TOTAL
    [ $DONE_COUNT -eq $TOTAL ] && break
    sleep 15
done

echo ""
echo "  Hoan tat: $((TOTAL - FAIL))/$TOTAL OK"
echo ""
echo "  Buoc tiep theo — copy experiments ve local:"
echo "    rsync -avz --progress <user>@<vast-ip>:/workspace/rl-traffic/experiments/ ./experiments/"
