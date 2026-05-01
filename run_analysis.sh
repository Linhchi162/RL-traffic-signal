#!/bin/bash
# run_analysis.sh
# Chay toan bo pipeline phan tich sau khi co ket qua eval.
#
# Thu tu:
#   1. Cai matplotlib (neu chua co)
#   2. Ve bieu do tu CSV (plot_cologne_results.py)
#   3. Ve training curves tu log files (plot_training_curves.py)
#   4. (Optional) Per-junction analysis Cologne8 (evaluate_per_junction.py)
#
# Usage:
#   cd /workspace/rl-traffic && bash run_analysis.sh
#   bash run_analysis.sh --skip_junction   # bo qua buoc 4

set -euo pipefail
cd "$(dirname "$0")"

export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
export LIBSUMO_AS_TRACI="1"
export OPENBLAS_NUM_THREADS="1"
export OMP_NUM_THREADS="1"

if [ -f ".venv/bin/activate" ]; then source .venv/bin/activate; fi

SKIP_JUNCTION=false
for arg in "$@"; do
    [ "$arg" = "--skip_junction" ] && SKIP_JUNCTION=true
done

C3_DIR="./results_cologne3_real"
C8_DIR="./results_cologne8_real"
C3_LOGS="./logs_cologne3_real_all"
C8_LOGS="./logs_cologne8_real_all"
FIG_DIR="./figures"

echo "================================================="
echo "  RL Traffic Analysis Pipeline"
echo "  C3 results : $C3_DIR"
echo "  C8 results : $C8_DIR"
echo "  Output     : $FIG_DIR/"
echo "================================================="
echo ""

# ── 1. Cai dependencies ──────────────────────────────────────────────────────
echo "[1/4] Kiem tra / cai matplotlib, pandas..."
pip install --quiet --upgrade matplotlib pandas 2>/dev/null || \
    echo "  [WARN] pip install that bai, hy vong da co san"

# ── 2. Kiem tra eval results ton tai ─────────────────────────────────────────
echo ""
echo "[2/4] Kiem tra file eval results..."

MISSING=false
for f in "$C3_DIR/all_results.csv" "$C8_DIR/all_results.csv"; do
    if [ -f "$f" ]; then
        echo "  OK: $f ($(wc -l < "$f") rows)"
    else
        echo "  MISSING: $f"
        MISSING=true
    fi
done

if [ "$MISSING" = "true" ]; then
    echo ""
    echo "  [WARN] Mot so file results chua co."
    echo "  Chay eval truoc:"
    echo "    bash run_cologne3_real_eval.sh"
    echo "    bash run_cologne8_real_eval.sh"
    echo "  Tiep tuc voi du lieu hien co..."
fi

# ── 3 & 4. Ve bieu do SONG SONG ──────────────────────────────────────────────
echo ""
echo "[3+4] Ve bieu do va training curves song song..."
mkdir -p "$FIG_DIR"

python plot_cologne_results.py \
    --c3 "$C3_DIR" \
    --c8 "$C8_DIR" \
    --out "$FIG_DIR" > "$FIG_DIR/plot_results.log" 2>&1 &
PID_PLOT=$!
echo "  plot_cologne_results.py   PID=$PID_PLOT"

if [ -d "$C3_LOGS" ] || [ -d "$C8_LOGS" ]; then
    python plot_training_curves.py \
        --c3_logs "$C3_LOGS" \
        --c8_logs "$C8_LOGS" \
        --out "$FIG_DIR" > "$FIG_DIR/plot_curves.log" 2>&1 &
    PID_CURVES=$!
    echo "  plot_training_curves.py   PID=$PID_CURVES"
else
    PID_CURVES=""
    echo "  [WARN] Khong tim thay log dirs, bo qua training curves."
fi

# Wait for both to finish
wait $PID_PLOT
echo "  plot_cologne_results.py   DONE (exit $?)"
[ -n "$PID_CURVES" ] && wait $PID_CURVES && echo "  plot_training_curves.py   DONE (exit $?)"

# Show output
echo ""
echo "  --- plot_results.log ---"
tail -20 "$FIG_DIR/plot_results.log"
[ -f "$FIG_DIR/plot_curves.log" ] && echo "" && echo "  --- plot_curves.log ---" && tail -10 "$FIG_DIR/plot_curves.log"

# ── 5. (Optional) Per-junction analysis ──────────────────────────────────────
if [ "$SKIP_JUNCTION" = "false" ]; then
    echo ""
    echo "[5/5] Per-junction analysis (evaluate_per_junction.py)..."
    echo "  Chay eval SUMO moi cho Cologne8 — mat ~20-40 phut..."
    python evaluate_per_junction.py \
        --models_dir ./exp_cologne8_real \
        --save_dir   "$C8_DIR" \
        --out        "$FIG_DIR"
else
    echo ""
    echo "[5/5] Per-junction analysis: DA BO QUA (--skip_junction)"
fi

# ── Ket qua ──────────────────────────────────────────────────────────────────
echo ""
echo "================================================="
echo "  HOAN TAT"
echo "  Bieu do luu tai: $FIG_DIR/"
echo "================================================="
echo ""
ls "$FIG_DIR/"*.png 2>/dev/null | while read f; do
    printf "  %s\n" "$(basename "$f")"
done
echo ""
