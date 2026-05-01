#!/bin/bash
# instance_cologne3_real_extra.sh
# Chay them 2 seeds (1234, 5678) x 3 rewards x 3 algos = 18 jobs
# Ket qua luu vao exp_cologne3_real/ cung thu muc voi 8 seeds cu
#
# Usage: cd /workspace/rl-traffic && bash instance_cologne3_real_extra.sh

set -euo pipefail
cd "$(dirname "$0")"

export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
export LIBSUMO_AS_TRACI="1"
export OPENBLAS_NUM_THREADS="1"
export OMP_NUM_THREADS="1"
export CUDA_VISIBLE_DEVICES=""

if [ -f ".venv/bin/activate" ]; then source .venv/bin/activate; fi

NET="./nets/cologne3/cologne3.net.xml"
ROUTE="./nets/cologne3/cologne3.rou.xml"
EXP_DIR="./exp_cologne3_real"
LOG_DIR="./logs_cologne3_real_all"
STEPS=500000
SIM_DUR=3600
SUMO_BEGIN=25200
SEEDS=(1234 5678)
REWARDS=("queue" "pressure" "wait-clip")

mkdir -p "$LOG_DIR"

RESCO="https://raw.githubusercontent.com/Pi-Star-Lab/RESCO/main/resco_benchmark/environments/cologne3"
[ ! -f "$NET" ] && wget -q "$RESCO/cologne3.net.xml" -O "$NET"
[ ! -f "$ROUTE" ] && wget -q "$RESCO/cologne3.rou.xml" -O "$ROUTE"

declare -a ALL_PIDS=() ALL_LABELS=()

echo ">>> Cologne3 real — 2 extra seeds: ${SEEDS[*]}"
echo ">>> 18 jobs (DQN+DDQN+PPO x 2 seeds x 3 rewards)"
echo ""

for algo in dqn ddqn; do
    for seed in "${SEEDS[@]}"; do
        for reward in "${REWARDS[@]}"; do
            label="${algo}_${reward}_s${seed}"
            sd="$EXP_DIR/$label"
            rm -rf "$sd" && mkdir -p "$sd"
            python train_cologne_dqn.py \
                --net_file "$NET" --route_file "$ROUTE" \
                --algo "$algo" --reward_type "$reward" \
                --seed "$seed" --total_steps $STEPS \
                --sim_duration $SIM_DUR --sumo_begin $SUMO_BEGIN \
                --save_dir "$sd" > "$LOG_DIR/${label}.log" 2>&1 &
            ALL_PIDS+=($!); ALL_LABELS+=("$label")
            printf "  %-50s PID %d\n" "$label" $!
        done
    done
done

for seed in "${SEEDS[@]}"; do
    for reward in "${REWARDS[@]}"; do
        label="ppo_${reward}_s${seed}"
        sd="$EXP_DIR/$label"
        rm -rf "$sd" && mkdir -p "$sd"
        python train_cologne_ppo.py \
            --net_file "$NET" --route_file "$ROUTE" \
            --reward_type "$reward" \
            --seed "$seed" --total_steps $STEPS \
            --sim_duration $SIM_DUR --sumo_begin $SUMO_BEGIN \
            --save_dir "$sd" > "$LOG_DIR/${label}.log" 2>&1 &
        ALL_PIDS+=($!); ALL_LABELS+=("$label")
        printf "  %-50s PID %d\n" "$label" $!
    done
done

TOTAL=${#ALL_PIDS[@]}
echo ""
echo "  $TOTAL jobs dang chay. Logs: $LOG_DIR/"
echo ""

declare -a DONE_FLAGS
for i in "${!ALL_PIDS[@]}"; do DONE_FLAGS[$i]=0; done
FAIL=0

while true; do
    DONE_COUNT=0
    for i in "${!ALL_PIDS[@]}"; do
        if [ "${DONE_FLAGS[$i]}" -eq 1 ]; then
            DONE_COUNT=$(( DONE_COUNT + 1 ))
        elif ! kill -0 "${ALL_PIDS[$i]}" 2>/dev/null; then
            DONE_FLAGS[$i]=1; DONE_COUNT=$(( DONE_COUNT + 1 ))
            if wait "${ALL_PIDS[$i]}"; then echo "  OK  ${ALL_LABELS[$i]}"
            else echo "  ERR ${ALL_LABELS[$i]}"; FAIL=$(( FAIL + 1 )); fi
        fi
    done
    PCT=$(( DONE_COUNT * 100 / TOTAL ))
    printf "\r  [%s] %d/%d (%d%%)   " "$(date '+%H:%M:%S')" "$DONE_COUNT" "$TOTAL" "$PCT"
    [ "$DONE_COUNT" -eq "$TOTAL" ] && break
    sleep 15
done

echo ""
echo "=========================================="; echo "  Done: $(( TOTAL - FAIL ))/$TOTAL OK"
echo "  Models: $EXP_DIR/"; echo "=========================================="
