#!/bin/bash
# watch_training.sh — Thanh tien do cho training dang chay
#
# Su dung:
#   bash watch_training.sh              # tu detect tat ca experiments dang cho
#   bash watch_training.sh 27           # biet truoc tong so job
#   bash watch_training.sh 27 10        # cap nhat moi 10 giay (mac dinh: 15)

EXP_DIR="./experiments"
TOTAL="${1:-0}"       # 0 = tu dem tu experiments/
INTERVAL="${2:-15}"   # giay

# Mau ANSI
GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; RESET="\033[0m"; BOLD="\033[1m"

_count_done() {
    # Model hoan tat = co file final_model.zip
    ls "$EXP_DIR"/*/ppo_final_model.zip \
       "$EXP_DIR"/*/dqn_final_model.zip 2>/dev/null | wc -l
}

_count_running() {
    pgrep -c -f "python train_" 2>/dev/null || echo 0
}

_bar() {
    local done=$1 total=$2 width=30
    local filled=$(( done * width / (total > 0 ? total : 1) ))
    local empty=$(( width - filled ))
    local bar=""
    [ $filled -gt 0 ] && bar+=$(printf '█%.0s' $(seq 1 $filled))
    [ $empty  -gt 0 ] && bar+=$(printf '░%.0s' $(seq 1 $empty))
    echo "$bar"
}

# Auto-detect total nếu không truyền
if [ "$TOTAL" -eq 0 ] 2>/dev/null; then
    TOTAL=$(ls "$EXP_DIR" 2>/dev/null | wc -l)
fi

echo -e "${BOLD}=== Training Monitor ===${RESET}  (Ctrl+C de thoat)"
echo -e "Exp dir : $EXP_DIR"
echo -e "Interval: ${INTERVAL}s\n"

while true; do
    DONE=$(_count_done)
    RUNNING=$(_count_running)
    PCT=$(( DONE * 100 / (TOTAL > 0 ? TOTAL : 1) ))
    BAR=$(_bar $DONE $TOTAL)
    NOW=$(date '+%H:%M:%S')

    # Mau tuy % xong
    if   [ $PCT -ge 100 ]; then COLOR=$GREEN
    elif [ $PCT -ge 50  ]; then COLOR=$YELLOW
    else                        COLOR=$RESET
    fi

    # In progress bar (ghi de dong cu)
    printf "\r${BOLD}[%s]${RESET}  ${COLOR}%s${RESET}  %d/%d (%-3d%%)  |  running: %d  |  %s   " \
        "$NOW" "$BAR" "$DONE" "$TOTAL" "$PCT" "$RUNNING" ""

    # In chi tiet cac model vua hoan tat (dong moi)
    if [ "${PREV_DONE:-0}" -ne "$DONE" ] 2>/dev/null; then
        echo ""   # xuong dong moi khi co model moi xong
        # Tim model moi nhat vua hoan tat
        NEWEST=$(ls -t "$EXP_DIR"/*/ppo_final_model.zip \
                        "$EXP_DIR"/*/dqn_final_model.zip 2>/dev/null | head -1)
        if [ -n "$NEWEST" ]; then
            LABEL=$(echo "$NEWEST" | awk -F'/' '{print $(NF-1)}')
            echo -e "  ${GREEN}✓${RESET} $LABEL"
        fi
    fi
    PREV_DONE=$DONE

    if [ "$DONE" -ge "$TOTAL" ] && [ "$TOTAL" -gt 0 ]; then
        echo ""
        echo -e "\n${GREEN}${BOLD}Hoan tat! $DONE/$TOTAL models.${RESET}"
        break
    fi

    sleep "$INTERVAL"
done
