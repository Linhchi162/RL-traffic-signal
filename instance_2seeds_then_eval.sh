#!/bin/bash
# instance_2seeds_then_eval.sh
# Train them 2 seed moi (1111, 5678) cho C3 va C8, sau do eval toan bo 10 seed.
#
# Usage:
#   bash <(curl -s https://raw.githubusercontent.com/Linhchi162/RL-traffic-signal/main/instance_2seeds_then_eval.sh)
# hoac (neu da co repo):
#   cd /workspace/rl-traffic && git pull && bash instance_2seeds_then_eval.sh

set -euo pipefail

REPO_URL="https://github.com/Linhchi162/RL-traffic-signal.git"
WORK_DIR="/workspace/rl-traffic"

# ── 1. SUMO ──────────────────────────────────────────────────────────────────
if ! command -v sumo &>/dev/null; then
    echo "=== [1/5] Cai SUMO ==="
    add-apt-repository ppa:sumo/stable -y 2>/dev/null || true
    apt-get update -q
    apt-get install -y sumo sumo-tools python3-pip python3-venv git curl wget
else
    echo "=== [1/5] SUMO da co ($(sumo --version 2>&1 | head -1)) ==="
fi

export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
export LIBSUMO_AS_TRACI="1"
export OPENBLAS_NUM_THREADS="1"
export OMP_NUM_THREADS="1"
export CUDA_VISIBLE_DEVICES=""

# ── 2. Clone / pull ──────────────────────────────────────────────────────────
echo "=== [2/5] Clone / pull repo ==="
if [ -d "$WORK_DIR/.git" ]; then
    git -C "$WORK_DIR" pull
else
    git clone "$REPO_URL" "$WORK_DIR"
fi
cd "$WORK_DIR"

# ── 3. Python deps ───────────────────────────────────────────────────────────
echo "=== [3/5] Cai Python deps ==="
if [ ! -f ".venv/bin/activate" ]; then
    python3 -m venv --system-site-packages .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip

pip_retry() {
    for attempt in 1 2 3; do
        pip install --quiet --no-cache-dir --timeout 120 "$@" && return 0
        echo "    [pip attempt $attempt/3] thu lai..."
        sleep 5
    done
    echo "[FATAL] pip install that bai: $*" >&2; return 1
}
pip_retry sumolib traci gymnasium stable-baselines3 libsumo pandas

# ── 4. Kiem tra net + route files ────────────────────────────────────────────
echo "=== [4/5] Kiem tra net + route files ==="
NET3="$WORK_DIR/nets/cologne3/cologne3.net.xml"
ROU3="$WORK_DIR/nets/cologne3/cologne3.rou.xml"
NET8="$WORK_DIR/nets/cologne8/cologne8.net.xml"
ROU8="$WORK_DIR/nets/cologne8/cologne8.rou.xml"

RESCO3="https://raw.githubusercontent.com/Pi-Star-Lab/RESCO/main/resco_benchmark/environments/cologne3"
RESCO8="https://raw.githubusercontent.com/Pi-Star-Lab/RESCO/main/resco_benchmark/environments/cologne8"
mkdir -p "$WORK_DIR/nets/cologne3" "$WORK_DIR/nets/cologne8"

[ -f "$NET3" ] || { wget -q "$RESCO3/cologne3.net.xml" -O "$NET3" && echo "    cologne3.net.xml OK"; }
[ -f "$ROU3" ] || { wget -q "$RESCO3/cologne3.rou.xml" -O "$ROU3" && echo "    cologne3.rou.xml OK"; }
[ -f "$NET8" ] || { wget -q "$RESCO8/cologne8.net.xml" -O "$NET8" && echo "    cologne8.net.xml OK"; }
[ -f "$ROU8" ] || { wget -q "$RESCO8/cologne8.rou.xml" -O "$ROU8" && echo "    cologne8.rou.xml OK"; }

# Kiem tra random route files da co chua
for pct in 25 50 75 100; do
    [ -f "$WORK_DIR/nets/cologne3/cologne3_rand_${pct}pct.rou.xml" ] \
        || { echo "    [WARN] C3 rand ${pct}pct missing — chay make_random_routes.py"; \
             python make_random_routes.py --net "$NET3" --rou "$ROU3" --seed 0; break; }
    [ -f "$WORK_DIR/nets/cologne8/cologne8_rand_${pct}pct.rou.xml" ] \
        || { echo "    [WARN] C8 rand ${pct}pct missing — chay make_random_routes.py"; \
             python make_random_routes.py --net "$NET8" --rou "$ROU8" --seed 0; break; }
done
echo "    Random route files OK"

# ── 5. Train 2 seed moi ──────────────────────────────────────────────────────
echo "=== [5/5] Train 2 seed moi (1111, 5678) ==="

EXP3="$WORK_DIR/exp_cologne3_rand_curr"
EXP8="$WORK_DIR/exp_cologne8_rand_curr"
LOG3="$WORK_DIR/logs_cologne3_rand_curr"
LOG8="$WORK_DIR/logs_cologne8_rand_curr"
STEPS=500000
SIM_DUR=3600
SUMO_BEGIN=25200
NEW_SEEDS=(1111 5678)
REWARDS=("queue" "pressure" "wait-clip")

mkdir -p "$LOG3" "$LOG8"

declare -a JOB_CMDS=() JOB_LABELS=()

for algo in dqn ddqn ppo; do
    for seed in "${NEW_SEEDS[@]}"; do
        for reward in "${REWARDS[@]}"; do
            # C3
            label3="c3_${algo}_${reward}_s${seed}"
            sd3="$EXP3/${algo}_${reward}_s${seed}"
            JOB_LABELS+=("$label3")
            JOB_CMDS+=("rm -rf '$sd3' && mkdir -p '$sd3' && \
                python train_cologne_curriculum.py \
                    --net_file '$NET3' --route_file '$ROU3' \
                    --algo $algo --reward_type '$reward' \
                    --seed $seed --total_steps $STEPS \
                    --sim_duration $SIM_DUR --sumo_begin $SUMO_BEGIN \
                    --save_dir '$sd3' --rand \
                    > '$LOG3/${algo}_${reward}_s${seed}.log' 2>&1")
            # C8
            label8="c8_${algo}_${reward}_s${seed}"
            sd8="$EXP8/${algo}_${reward}_s${seed}"
            JOB_LABELS+=("$label8")
            JOB_CMDS+=("rm -rf '$sd8' && mkdir -p '$sd8' && \
                python train_cologne_curriculum.py \
                    --net_file '$NET8' --route_file '$ROU8' \
                    --algo $algo --reward_type '$reward' \
                    --seed $seed --total_steps $STEPS \
                    --sim_duration $SIM_DUR --sumo_begin $SUMO_BEGIN \
                    --save_dir '$sd8' --rand \
                    > '$LOG8/${algo}_${reward}_s${seed}.log' 2>&1")
        done
    done
done

TOTAL=${#JOB_CMDS[@]}
echo ">>> Launch $TOTAL training jobs (C3+C8, 2 seed moi) song song..."

declare -a ALL_PIDS=() ALL_LABELS=()
for i in "${!JOB_CMDS[@]}"; do
    eval "${JOB_CMDS[$i]}" &
    pid=$!
    ALL_PIDS+=("$pid")
    ALL_LABELS+=("${JOB_LABELS[$i]}")
    printf "  %-50s PID %d\n" "${JOB_LABELS[$i]}" "$pid"
done

echo ""
echo "  $TOTAL training jobs dang chay..."

declare -a DONE_FLAGS
for i in "${!ALL_PIDS[@]}"; do DONE_FLAGS[$i]=0; done
FAIL=0

while true; do
    DONE_COUNT=0
    for i in "${!ALL_PIDS[@]}"; do
        if [ "${DONE_FLAGS[$i]}" -eq 1 ]; then
            DONE_COUNT=$(( DONE_COUNT + 1 ))
        elif ! kill -0 "${ALL_PIDS[$i]}" 2>/dev/null; then
            DONE_FLAGS[$i]=1
            DONE_COUNT=$(( DONE_COUNT + 1 ))
            if wait "${ALL_PIDS[$i]}"; then
                echo "  OK  ${ALL_LABELS[$i]}"
            else
                echo "  ERR ${ALL_LABELS[$i]}"
                FAIL=$(( FAIL + 1 ))
            fi
        fi
    done
    PCT=$(( DONE_COUNT * 100 / TOTAL ))
    printf "\r  [%s] %d/%d (%d%%)   " "$(date '+%H:%M:%S')" "$DONE_COUNT" "$TOTAL" "$PCT"
    [ "$DONE_COUNT" -eq "$TOTAL" ] && break
    sleep 15
done

echo ""
OK=$(( TOTAL - FAIL ))
echo "========================================================"
echo "  Training done: $OK/$TOTAL OK"
[ "$FAIL" -gt 0 ] && echo "  [WARN] $FAIL job that bai — kiem tra log truoc khi eval"
echo "========================================================"

if [ "$FAIL" -gt 0 ]; then
    echo "[WARN] Co $FAIL job that bai. Tiep tuc eval voi cac model da co..."
fi

# ── 6. Eval toan bo 10 seed ───────────────────────────────────────────────────
WORKERS=$(nproc)
echo ""
echo "========================================================"
echo "  Bat dau eval toan bo models ($WORKERS workers)..."
echo "========================================================"

RES3="$WORK_DIR/results_cologne3_rand_curr"
RES8="$WORK_DIR/results_cologne8_rand_curr"

echo ""
echo "--- Cologne3 eval ---"
python evaluate_cologne3_real.py \
    --models_dir "$EXP3" \
    --save_dir   "$RES3" \
    --workers    "$WORKERS" \
    --resume

echo ""
echo "--- Cologne8 eval ---"
python evaluate_cologne8.py \
    --models_dir "$EXP8" \
    --save_dir   "$RES8" \
    --workers    "$WORKERS" \
    --resume

echo ""
echo "========================================================"
echo "  XONG!"
echo "  C3 results : $RES3/summary.csv"
echo "  C8 results : $RES8/summary.csv"
echo "========================================================"
