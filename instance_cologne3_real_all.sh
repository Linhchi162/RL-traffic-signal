#!/bin/bash
# instance_cologne3_real_all.sh
# Setup + Train DQN + DDQN + PPO tren Cologne3 voi luu luong that (cologne3.rou.xml)
#
# Lam luon:
#   1. Cai SUMO + Python deps (neu chua co)
#   2. Git pull ban moi nhat
#   3. Download net/route Cologne3 (neu chua co)
#   4. Chay 72 jobs (3 algo x 8 seeds x 3 rewards) voi worker pool
#
# Usage (chay ngay sau khi SSH vao Vast):
#   bash <(curl -s https://raw.githubusercontent.com/Linhchi162/RL-traffic-signal/main/instance_cologne3_real_all.sh)
# hoac:
#   git clone https://github.com/Linhchi162/RL-traffic-signal.git /workspace/rl-traffic
#   cd /workspace/rl-traffic && bash instance_cologne3_real_all.sh

set -euo pipefail

REPO_URL="https://github.com/Linhchi162/RL-traffic-signal.git"
WORK_DIR="/workspace/rl-traffic"
MAX_WORKERS=${MAX_WORKERS:-24}   # so jobs chay song song (giam xuong 16 neu RAM thieu)

# ── 1. Cai SUMO ──────────────────────────────────────────────────────────────
if ! command -v sumo &>/dev/null; then
    echo "=== [1/4] Cai SUMO ==="
    add-apt-repository ppa:sumo/stable -y 2>/dev/null || true
    apt-get update -q
    apt-get install -y sumo sumo-tools python3-pip python3-venv git curl wget
else
    echo "=== [1/4] SUMO da co ($(sumo --version 2>&1 | head -1)) ==="
fi

export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
export LIBSUMO_AS_TRACI="1"
export OPENBLAS_NUM_THREADS="1"
export OMP_NUM_THREADS="1"
export CUDA_VISIBLE_DEVICES=""

# ── 2. Clone / pull repo ─────────────────────────────────────────────────────
echo "=== [2/4] Clone / pull repo ==="
if [ -d "$WORK_DIR/.git" ]; then
    echo "    git pull..."
    git -C "$WORK_DIR" pull
else
    echo "    git clone..."
    git clone "$REPO_URL" "$WORK_DIR"
fi
cd "$WORK_DIR"

# ── 3. Cai Python deps ───────────────────────────────────────────────────────
echo "=== [3/4] Cai Python deps ==="
if [ ! -f ".venv/bin/activate" ]; then
    python3 -m venv --system-site-packages .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip

pip_retry() {
    for attempt in 1 2 3; do
        pip install --quiet --no-cache-dir --timeout 120 "$@" && return 0
        echo "    [pip attempt $attempt/3] $* that bai, thu lai..."
        sleep 5
    done
    echo "[FATAL] pip install that bai: $*" >&2; return 1
}
pip_retry sumolib traci gymnasium stable-baselines3 libsumo pandas

# Kiem tra nhanh
python3 -c "import libsumo; print('  libsumo OK:', libsumo.__version__)"

# ── 4. Download Cologne3 net + route ─────────────────────────────────────────
echo "=== [4/4] Download Cologne3 (neu chua co) ==="
RESCO="https://raw.githubusercontent.com/Pi-Star-Lab/RESCO/main/resco_benchmark/environments/cologne3"
NET="$WORK_DIR/nets/cologne3/cologne3.net.xml"
ROUTE="$WORK_DIR/nets/cologne3/cologne3.rou.xml"
mkdir -p "$WORK_DIR/nets/cologne3"

if [ ! -f "$NET" ]; then
    wget -q "$RESCO/cologne3.net.xml" -O "$NET" && echo "    cologne3.net.xml OK"
else
    echo "    cologne3.net.xml da co"
fi
if [ ! -f "$ROUTE" ]; then
    wget -q "$RESCO/cologne3.rou.xml" -O "$ROUTE" && echo "    cologne3.rou.xml OK"
else
    echo "    cologne3.rou.xml da co"
fi

# ── 5. Tao danh sach 72 jobs ─────────────────────────────────────────────────
EXP_DIR="$WORK_DIR/exp_cologne3_real"
LOG_DIR="$WORK_DIR/logs_cologne3_real_all"
STEPS=500000
SIM_DUR=3600
SEEDS=(42 123 777 999 314 2025 2718 9999)
REWARDS=("queue" "pressure" "wait-clip")

mkdir -p "$LOG_DIR"

declare -a JOB_CMDS JOB_LABELS

for algo in dqn ddqn; do
    for seed in "${SEEDS[@]}"; do
        for reward in "${REWARDS[@]}"; do
            label="${algo}_${reward}_s${seed}"
            sd="$EXP_DIR/$label"
            JOB_LABELS+=("$label")
            JOB_CMDS+=("rm -rf '$sd' && mkdir -p '$sd' && python train_cologne_dqn.py \
                --net_file '$NET' --route_file '$ROUTE' \
                --algo $algo --reward_type '$reward' \
                --seed $seed --total_steps $STEPS --sim_duration $SIM_DUR \
                --save_dir '$sd' > '$LOG_DIR/${label}.log' 2>&1")
        done
    done
done

for seed in "${SEEDS[@]}"; do
    for reward in "${REWARDS[@]}"; do
        label="ppo_${reward}_s${seed}"
        sd="$EXP_DIR/$label"
        JOB_LABELS+=("$label")
        JOB_CMDS+=("rm -rf '$sd' && mkdir -p '$sd' && python train_cologne_ppo.py \
            --net_file '$NET' --route_file '$ROUTE' \
            --reward_type '$reward' \
            --seed $seed --total_steps $STEPS --sim_duration $SIM_DUR \
            --save_dir '$sd' > '$LOG_DIR/${label}.log' 2>&1")
    done
done

TOTAL=${#JOB_CMDS[@]}
echo ""
echo ">>> Net    : $NET"
echo ">>> Route  : $ROUTE  (luu luong that, 3600s)"
echo ">>> Steps  : $STEPS  | SimDur: ${SIM_DUR}s"
echo ">>> $TOTAL jobs tong cong (DQN x24 + DDQN x24 + PPO x24), chay ${MAX_WORKERS} song song"
echo ">>> Logs: $LOG_DIR/"
echo ""

# ── 6. Worker pool ───────────────────────────────────────────────────────────
declare -a RUNNING_PIDS=() RUNNING_LABELS=()
FAIL=0
DONE_COUNT=0
JOB_IDX=0

while [ "$DONE_COUNT" -lt "$TOTAL" ]; do
    # Nap them job vao pool neu con cho trong
    while [ "${#RUNNING_PIDS[@]}" -lt "$MAX_WORKERS" ] && [ "$JOB_IDX" -lt "$TOTAL" ]; do
        label="${JOB_LABELS[$JOB_IDX]}"
        eval "${JOB_CMDS[$JOB_IDX]}" &
        pid=$!
        RUNNING_PIDS+=("$pid")
        RUNNING_LABELS+=("$label")
        printf "  START %-52s PID %d\n" "$label" "$pid"
        JOB_IDX=$(( JOB_IDX + 1 ))
    done

    # Reap cac job da xong
    new_pids=()
    new_labels=()
    for i in "${!RUNNING_PIDS[@]}"; do
        if ! kill -0 "${RUNNING_PIDS[$i]}" 2>/dev/null; then
            if wait "${RUNNING_PIDS[$i]}"; then
                echo "  OK  ${RUNNING_LABELS[$i]}"
            else
                echo "  ERR ${RUNNING_LABELS[$i]}"
                FAIL=$(( FAIL + 1 ))
            fi
            DONE_COUNT=$(( DONE_COUNT + 1 ))
        else
            new_pids+=("${RUNNING_PIDS[$i]}")
            new_labels+=("${RUNNING_LABELS[$i]}")
        fi
    done
    RUNNING_PIDS=("${new_pids[@]+"${new_pids[@]}"}")
    RUNNING_LABELS=("${new_labels[@]+"${new_labels[@]}"}")

    PCT=$(( DONE_COUNT * 100 / TOTAL ))
    printf "\r  [%s] %d/%d (%d%%)  dang chay: %d   " \
        "$(date '+%H:%M:%S')" "$DONE_COUNT" "$TOTAL" "$PCT" "${#RUNNING_PIDS[@]}"

    sleep 10
done

echo ""
OK=$(( TOTAL - FAIL ))
echo "=========================================="
echo "  cologne3-real ALL done: $OK/$TOTAL OK"
echo "  Models: $EXP_DIR/"
echo "=========================================="
