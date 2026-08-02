#!/bin/bash
# EC2 cron用エントリポイント。
# 土日祝はスキップし、前日分が万一残っていれば多重起動を防止した上で run_daily.py を実行する。
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

LOCK_FILE="/tmp/stock_run_daily.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "$(date '+%F %T'): 既にrun_daily.pyが実行中のためスキップ"
    exit 0
fi

if ! venv/bin/python -c "
import sys
import jpholiday
from datetime import date
today = date.today()
sys.exit(0 if (today.weekday() < 5 and not jpholiday.is_holiday(today)) else 1)
"; then
    echo "$(date '+%F %T'): 休日のためrun_daily.pyをスキップ"
    exit 0
fi

mkdir -p logs
LOG="logs/run_daily_$(date +%Y%m%d).log"
echo "$(date '+%F %T'): run_daily.py 開始" >> "$LOG"
venv/bin/python run_daily.py >> "$LOG" 2>&1
echo "$(date '+%F %T'): run_daily.py 終了（exit $?）" >> "$LOG"
