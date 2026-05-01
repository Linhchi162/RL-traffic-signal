#!/bin/bash
# setup_vast.sh — Cai moi truong tren Vast.ai instance (chay 1 lan)
#
# Usage (sau khi SSH vao instance):
#   bash setup_vast.sh
#
# Vast.ai thuong dung Docker image san (PyTorch/CUDA), work dir la /workspace.
# Script nay:
#   1. Cai SUMO
#   2. Clone repo (hoac pull neu da co)
#   3. Cai Python deps (trong .venv)
#   4. Kiem tra SUMO + libsumo

set -e

REPO_URL="https://github.com/Linhchi162/RL-traffic-signal.git"
WORK_DIR="/workspace/rl-traffic"

echo "=== [1/4] Cai SUMO ==="
add-apt-repository ppa:sumo/stable -y 2>/dev/null || true
apt-get update -q
apt-get install -y sumo sumo-tools python3-pip python3-venv git curl

export SUMO_HOME="/usr/share/sumo"
echo "export SUMO_HOME=/usr/share/sumo" >> ~/.bashrc
echo "export LIBSUMO_AS_TRACI=1"        >> ~/.bashrc

echo "=== [2/4] Clone / update repo ==="
if [ -d "$WORK_DIR/.git" ]; then
    echo "    Repo da ton tai, git pull..."
    cd "$WORK_DIR"
    git pull
else
    echo "    Clone moi..."
    git clone "$REPO_URL" "$WORK_DIR"
    cd "$WORK_DIR"
fi

echo "=== [3/4] Cai Python deps ==="
# --system-site-packages: ke thua torch/numpy tu image Vast (tranh download ~500MB)
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --quiet --upgrade pip

# Cai tung goi mot, moi goi nho nen it bi broken pipe
pip_retry() {
    local attempt
    for attempt in 1 2 3 4 5; do
        pip install --quiet --no-cache-dir --timeout 120 "$@" && return 0
        echo "    [pip attempt $attempt/5] $* that bai, thu lai sau 5s..."
        sleep 5
    done
    echo "[FATAL] pip install that bai sau 5 lan: $*" >&2
    return 1
}

pip_retry sumolib
pip_retry traci
pip_retry gymnasium
pip_retry stable-baselines3
pip_retry libsumo
pip_retry pandas

echo "=== [4/5] Download Cologne networks (RESCO benchmark) ==="
RESCO_BASE="https://raw.githubusercontent.com/Pi-Star-Lab/RESCO/main/resco_benchmark/environments"

mkdir -p "$WORK_DIR/nets/cologne8"
wget -q "$RESCO_BASE/cologne8/cologne8.net.xml" -O "$WORK_DIR/nets/cologne8/cologne8.net.xml"
wget -q "$RESCO_BASE/cologne8/cologne8.rou.xml" -O "$WORK_DIR/nets/cologne8/cologne8.rou.xml"
echo "    cologne8 OK"

mkdir -p "$WORK_DIR/nets/cologne3"
wget -q "$RESCO_BASE/cologne3/cologne3.net.xml" -O "$WORK_DIR/nets/cologne3/cologne3.net.xml"
wget -q "$RESCO_BASE/cologne3/cologne3.rou.xml" -O "$WORK_DIR/nets/cologne3/cologne3.rou.xml"
echo "    cologne3 OK"

echo "=== [5/5] Kiem tra ==="
python3 -c "import libsumo; print('libsumo OK:', libsumo.__version__)"
python3 -c "
import os, sys
os.environ['SUMO_HOME'] = '/usr/share/sumo'
sys.path += ['/usr/share/sumo/tools']
import sumolib, traci
print('traci OK')
"

echo ""
echo "================================================"
echo "  Setup xong!"
echo "  Work dir : $WORK_DIR"
echo ""
echo "  De chay training lan dau:"
echo "    cd $WORK_DIR && bash run_vast.sh"
echo ""
echo "  De resume (neu da co data cu trong experiments/):"
echo "    cd $WORK_DIR && bash run_vast_resume.sh"
echo "================================================"
