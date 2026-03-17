#!/bin/bash
# aws_setup.sh — Chạy 1 lần trên AWS instance sau khi SSH vào
# Cài SUMO, Python deps, clone repo, rồi train

set -e

REPO_URL="https://github.com/Linhchi162/RL-traffic-signal.git"  # đổi nếu khác
WORK_DIR="$HOME/rl-traffic"

echo "=== [1/4] Cai SUMO ==="
sudo add-apt-repository ppa:sumo/stable -y
sudo apt-get update -q
sudo apt-get install -y sumo sumo-tools python3-pip python3-venv git

export SUMO_HOME="/usr/share/sumo"
echo "export SUMO_HOME=/usr/share/sumo" >> ~/.bashrc

echo "=== [2/4] Clone repo ==="
git clone "$REPO_URL" "$WORK_DIR"
cd "$WORK_DIR"

echo "=== [3/4] Cai Python deps ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --quiet -r requirements.txt
pip install --quiet libsumo

echo "=== [4/4] Kiem tra SUMO + libsumo ==="
python3 -c "import libsumo; print('libsumo OK:', libsumo.__version__)"
python3 -c "import sumolib, traci; print('traci OK')"

echo ""
echo "Setup xong! Chay training:"
echo "  cd $WORK_DIR && bash run_training_aws.sh"
