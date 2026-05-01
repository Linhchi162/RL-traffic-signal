#!/bin/bash
# instance_cologne3_real.sh
# Train DDQN tren Cologne3 voi luu luong THAT tu RESCO benchmark (cologne3.rou.xml)
#
# Route file: du lieu thuc te o Cologne (TAPAS Cologne), 1 gio (0-3600s)
# sim_duration = 3600 (khop voi route that, thay vi 100000s sinh nhan tao)
# total_steps  = 500000 (tang len vi moi episode chi 3600s, ~720 steps/junction)
#
# 8 seeds x 3 reward = 24 jobs chay song song
#
# Usage: cd /workspace/rl-traffic && bash instance_cologne3_real.sh

set -euo pipefail
cd "$(dirname "$0")"

export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
export LIBSUMO_AS_TRACI="1"
export OPENBLAS_NUM_THREADS="1"
export OMP_NUM_THREADS="1"
export CUDA_VISIBLE_DEVICES=""

if [ -f ".venv/bin/activate" ]; then source .venv/bin/activate; fi

NET="./nets/cologne3/cologne3.net.xml"
ROUTE="./nets/cologne3/cologne3.rou.xml"      # route THAT tu RESCO
EXP_DIR="./exp_cologne3_real"
LOG_DIR="./logs_cologne3_real"
STEPS=500000
SIM_DUR=3600                                   # do dai episode (s)
SUMO_BEGIN=25200                               # 7:00 AM tuyet doi trong route that
SEEDS=(42 123 777 999 314 2025 2718 9999)
REWARDS=("queue" "pressure" "wait-clip")

mkdir -p "$LOG_DIR"

# Download Cologne3 neu chua co
RESCO="https://raw.githubusercontent.com/Pi-Star-Lab/RESCO/main/resco_benchmark/environments/cologne3"
if [ ! -f "$NET" ]; then
    echo "Downloading cologne3.net.xml..."
    mkdir -p "./nets/cologne3"
    wget -q "$RESCO/cologne3.net.xml" -O "$NET"
    echo "    cologne3.net.xml OK"
fi
if [ ! -f "$ROUTE" ]; then
    echo "Downloading cologne3.rou.xml (luu luong that)..."
    mkdir -p "./nets/cologne3"
    wget -q "$RESCO/cologne3.rou.xml" -O "$ROUTE"
    echo "    cologne3.rou.xml OK"
fi

echo ""
echo ">>> Net    : $NET"
echo ">>> Route  : $ROUTE  (luu luong that, 3600s)"
echo ">>> Steps  : $STEPS  | SimDur: ${SIM_DUR}s"
echo ">>> Launch ${#SEEDS[@]} seeds x ${#REWARDS[@]} rewards = $((${#SEEDS[@]} * ${#REWARDS[@]})) jobs..."
echo ""

declare -a ALL_PIDS ALL_LABELS

for seed in "${SEEDS[@]}"; do
    for reward in "${REWARDS[@]}"; do
        label="ddqn_${reward}_s${seed}"
        sd="$EXP_DIR/$label"
        rm -rf "$sd"
        mkdir -p "$sd"
        python train_cologne_dqn.py \
            --net_file    "$NET" \
            --route_file  "$ROUTE" \
            --algo        ddqn \
            --reward_type "$reward" \
            --seed        "$seed" \
            --total_steps $STEPS \
            --sim_duration $SIM_DUR \
            --sumo_begin  $SUMO_BEGIN \
            --save_dir    "$sd" \
            > "$LOG_DIR/${label}.log" 2>&1 &
        pid=$!
        ALL_PIDS+=($pid)
        ALL_LABELS+=("$label")
        printf "  %-50s  PID %d\n" "$label" $pid
    done
done

TOTAL=${#ALL_PIDS[@]}
echo ""
echo "  $TOTAL jobs running. Logs: $LOG_DIR/"
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
echo "  cologne3-REAL done: $OK/$TOTAL OK"
echo "  Models: $EXP_DIR/"
echo "=========================================="
