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
# jpholidayは国民の祝日のみ判定するため、東証の年末年始休場(祝日ではない慣例休場)は別途除外する
tse_year_end_new_year = (today.month == 12 and today.day == 31) or (today.month == 1 and today.day in (2, 3))
is_trading_day = today.weekday() < 5 and not jpholiday.is_holiday(today) and not tse_year_end_new_year
sys.exit(0 if is_trading_day else 1)
"; then
    echo "$(date '+%F %T'): 休日（祝日 or 年末年始休場）のためrun_daily.pyをスキップ"
    exit 0
fi

mkdir -p logs
LOG="logs/run_daily_$(date +%Y%m%d).log"
echo "$(date '+%F %T'): run_daily.py 開始" >> "$LOG"
venv/bin/python run_daily.py >> "$LOG" 2>&1
echo "$(date '+%F %T'): run_daily.py 終了（exit $?）" >> "$LOG"
