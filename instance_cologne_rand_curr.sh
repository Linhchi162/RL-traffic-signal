#!/bin/bash
# instance_cologne_rand_curr.sh
# Curriculum training tren SYNTHETIC random routes, eval tren route that.
# Chay cho CA HAI C3 va C8: 144 jobs tong cong.
#
# Thiet ke train/test:
#   Train : 4 stage (25->50->75->100%) voi random OD routes (KHONG PHAI du lieu that)
#   Eval  : cologne{3,8}.rou.xml that (chua bao gio xuat hien trong training)
#
# Usage:
#   bash <(curl -s https://raw.githubusercontent.com/Linhchi162/RL-traffic-signal/main/instance_cologne_rand_curr.sh)
# hoac:
#   git clone https://github.com/Linhchi162/RL-traffic-signal.git /workspace/rl-traffic
#   cd /workspace/rl-traffic && bash instance_cologne_rand_curr.sh

set -euo pipefail

REPO_URL="https://github.com/Linhchi162/RL-traffic-signal.git"
WORK_DIR="/workspace/rl-traffic"

# ── 1. Cai SUMO ──────────────────────────────────────────────────────────────
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

# ── 2. Clone / pull repo ─────────────────────────────────────────────────────
echo "=== [2/5] Clone / pull repo ==="
if [ -d "$WORK_DIR/.git" ]; then
    git -C "$WORK_DIR" pull
else
    git clone "$REPO_URL" "$WORK_DIR"
fi
cd "$WORK_DIR"

# ── 3. Cai Python deps ───────────────────────────────────────────────────────
echo "=== [3/5] Cai Python deps ==="
if [ ! -f ".venv/bin/activate" ]; then
    python3 -m venv --system-site-packages .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip

pip_retry() {
    for attempt in 1 2 3; do
        pip install --quiet --no-cache-dir --timeout 120 "$@" && return 0
        echo "    [pip attempt $attempt/3] that bai, thu lai..."
        sleep 5
    done
    echo "[FATAL] pip install that bai: $*" >&2; return 1
}
pip_retry sumolib traci gymnasium stable-baselines3 libsumo pandas
python3 -c "import libsumo; print('  libsumo OK:', libsumo.__version__)"

# ── 4. Download Cologne3 + Cologne8 ──────────────────────────────────────────
echo "=== [4/5] Download Cologne net + route files ==="
RESCO3="https://raw.githubusercontent.com/Pi-Star-Lab/RESCO/main/resco_benchmark/environments/cologne3"
RESCO8="https://raw.githubusercontent.com/Pi-Star-Lab/RESCO/main/resco_benchmark/environments/cologne8"

mkdir -p "$WORK_DIR/nets/cologne3" "$WORK_DIR/nets/cologne8"

NET3="$WORK_DIR/nets/cologne3/cologne3.net.xml"
ROU3="$WORK_DIR/nets/cologne3/cologne3.rou.xml"
NET8="$WORK_DIR/nets/cologne8/cologne8.net.xml"
ROU8="$WORK_DIR/nets/cologne8/cologne8.rou.xml"

[ -f "$NET3" ] && echo "    cologne3.net.xml da co" || { wget -q "$RESCO3/cologne3.net.xml" -O "$NET3" && echo "    cologne3.net.xml OK"; }
[ -f "$ROU3" ] && echo "    cologne3.rou.xml da co" || { wget -q "$RESCO3/cologne3.rou.xml" -O "$ROU3" && echo "    cologne3.rou.xml OK"; }
[ -f "$NET8" ] && echo "    cologne8.net.xml da co" || { wget -q "$RESCO8/cologne8.net.xml" -O "$NET8" && echo "    cologne8.net.xml OK"; }
[ -f "$ROU8" ] && echo "    cologne8.rou.xml da co" || { wget -q "$RESCO8/cologne8.rou.xml" -O "$ROU8" && echo "    cologne8.rou.xml OK"; }

# ── 5. Tao random route files (neu chua co) ───────────────────────────────────
echo "=== [5/5] Tao synthetic random route files ==="
python make_random_routes.py --net "$NET3" --rou "$ROU3" --seed 0
python make_random_routes.py --net "$NET8" --rou "$ROU8" --seed 0

# Kiem tra
for pct in 25 50 75 100; do
    [ -f "$WORK_DIR/nets/cologne3/cologne3_rand_${pct}pct.rou.xml" ] \
        && echo "    C3 rand ${pct}pct OK" \
        || echo "    [WARN] C3 rand ${pct}pct MISSING"
    [ -f "$WORK_DIR/nets/cologne8/cologne8_rand_${pct}pct.rou.xml" ] \
        && echo "    C8 rand ${pct}pct OK" \
        || echo "    [WARN] C8 rand ${pct}pct MISSING"
done

# ── 6. Tao danh sach 144 jobs ─────────────────────────────────────────────────
EXP3="$WORK_DIR/exp_cologne3_rand_curr"
EXP8="$WORK_DIR/exp_cologne8_rand_curr"
LOG3="$WORK_DIR/logs_cologne3_rand_curr"
LOG8="$WORK_DIR/logs_cologne8_rand_curr"
STEPS=500000
SIM_DUR=3600
SUMO_BEGIN=25200
SEEDS=(42 123 777 999 314 2025 2718 9999 1111 5678)
REWARDS=("queue" "pressure" "wait-clip")

mkdir -p "$LOG3" "$LOG8"

declare -a JOB_CMDS JOB_LABELS

# Cologne3 — DQN + DDQN
for algo in dqn ddqn; do
    for seed in "${SEEDS[@]}"; do
        for reward in "${REWARDS[@]}"; do
            label="c3_${algo}_${reward}_s${seed}"
            sd="$EXP3/${algo}_${reward}_s${seed}"
            JOB_LABELS+=("$label")
            JOB_CMDS+=("rm -rf '$sd' && mkdir -p '$sd' && \
                python train_cologne_curriculum.py \
                    --net_file '$NET3' --route_file '$ROU3' \
                    --algo $algo --reward_type '$reward' \
                    --seed $seed --total_steps $STEPS \
                    --sim_duration $SIM_DUR --sumo_begin $SUMO_BEGIN \
                    --save_dir '$sd' --rand \
                    > '$LOG3/${algo}_${reward}_s${seed}.log' 2>&1")
        done
    done
done

# Cologne3 — PPO
for seed in "${SEEDS[@]}"; do
    for reward in "${REWARDS[@]}"; do
        label="c3_ppo_${reward}_s${seed}"
        sd="$EXP3/ppo_${reward}_s${seed}"
        JOB_LABELS+=("$label")
        JOB_CMDS+=("rm -rf '$sd' && mkdir -p '$sd' && \
            python train_cologne_curriculum.py \
                --net_file '$NET3' --route_file '$ROU3' \
                --algo ppo --reward_type '$reward' \
                --seed $seed --total_steps $STEPS \
                --sim_duration $SIM_DUR --sumo_begin $SUMO_BEGIN \
                --save_dir '$sd' --rand \
                > '$LOG3/ppo_${reward}_s${seed}.log' 2>&1")
    done
done

# Cologne8 — DQN + DDQN
for algo in dqn ddqn; do
    for seed in "${SEEDS[@]}"; do
        for reward in "${REWARDS[@]}"; do
            label="c8_${algo}_${reward}_s${seed}"
            sd="$EXP8/${algo}_${reward}_s${seed}"
            JOB_LABELS+=("$label")
            JOB_CMDS+=("rm -rf '$sd' && mkdir -p '$sd' && \
                python train_cologne_curriculum.py \
                    --net_file '$NET8' --route_file '$ROU8' \
                    --algo $algo --reward_type '$reward' \
                    --seed $seed --total_steps $STEPS \
                    --sim_duration $SIM_DUR --sumo_begin $SUMO_BEGIN \
                    --save_dir '$sd' --rand \
                    > '$LOG8/${algo}_${reward}_s${seed}.log' 2>&1")
        done
    done
done

# Cologne8 — PPO
for seed in "${SEEDS[@]}"; do
    for reward in "${REWARDS[@]}"; do
        label="c8_ppo_${reward}_s${seed}"
        sd="$EXP8/ppo_${reward}_s${seed}"
        JOB_LABELS+=("$label")
        JOB_CMDS+=("rm -rf '$sd' && mkdir -p '$sd' && \
            python train_cologne_curriculum.py \
                --net_file '$NET8' --route_file '$ROU8' \
                --algo ppo --reward_type '$reward' \
                --seed $seed --total_steps $STEPS \
                --sim_duration $SIM_DUR --sumo_begin $SUMO_BEGIN \
                --save_dir '$sd' --rand \
                > '$LOG8/ppo_${reward}_s${seed}.log' 2>&1")
    done
done

TOTAL=${#JOB_CMDS[@]}
echo ""
echo ">>> C3 net : $NET3  |  rand routes: cologne3_rand_*pct.rou.xml"
echo ">>> C8 net : $NET8  |  rand routes: cologne8_rand_*pct.rou.xml"
echo ">>> Eval   : route that (cologne{3,8}.rou.xml) — chua thay trong training"
echo ">>> Steps  : $STEPS  |  SimDur: ${SIM_DUR}s  |  SumoBegin: ${SUMO_BEGIN}s"
echo ">>> Launch $TOTAL jobs (C3: 90 + C8: 90) song song..."
echo ""

# ── 7. Launch ────────────────────────────────────────────────────────────────
declare -a ALL_PIDS=() ALL_LABELS=()

for i in "${!JOB_CMDS[@]}"; do
    eval "${JOB_CMDS[$i]}" &
    pid=$!
    ALL_PIDS+=("$pid")
    ALL_LABELS+=("${JOB_LABELS[$i]}")
    printf "  %-55s PID %d\n" "${JOB_LABELS[$i]}" "$pid"
done

echo ""
echo "  $TOTAL jobs dang chay. Logs: $LOG3/ va $LOG8/"
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
echo "  rand-curriculum ALL done: $OK/$TOTAL OK"
echo "  C3 models : $EXP3/"
echo "  C8 models : $EXP8/"
echo ""
echo "  Buoc tiep theo — Eval tren route THAT:"
echo "    python evaluate_cologne3_real.py --models_dir $EXP3 \\"
echo "        --save_dir results_cologne3_rand_curr --workers 8"
echo "    python evaluate_cologne8_real.py --models_dir $EXP8 \\"
echo "        --save_dir results_cologne8_rand_curr --workers 8"
echo ""
echo "  Ve training curves:"
echo "    python plot_learning_curves.py \\"
echo "        --c3_logs $LOG3 --c8_logs $LOG8 --out figures"
echo "========================================================"
