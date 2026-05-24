# -*- coding: utf-8 -*-
"""
run_daily.py - 朝の一括起動スクリプト

実行するだけで:
  1. scan_daily.py --date 前営業日  （データ取得・スコアリング）
  2. scan_morning.py                （地合い予測・自動発注開始）

使い方:
    python run_daily.py
"""

import sys
import os
import subprocess
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding="utf-8")


def get_prev_business_day():
    """前営業日を返す（土日・祝日をスキップ）。"""
    prev = datetime.now() - timedelta(days=1)

    try:
        import jpholiday
        while prev.weekday() >= 5 or jpholiday.is_holiday(prev):
            prev -= timedelta(days=1)
    except ImportError:
        # jpholiday 未インストールの場合は土日のみスキップ
        while prev.weekday() >= 5:
            prev -= timedelta(days=1)

    return prev.strftime("%Y%m%d")


def main():
    prev_day = get_prev_business_day()
    prev_day_fmt = f"{prev_day[:4]}-{prev_day[4:6]}-{prev_day[6:]}"
    today = datetime.now().strftime("%Y-%m-%d")

    print(f"{'='*60}")
    print(f"  朝スキャン一括起動（{today}）")
    print(f"  前営業日: {prev_day_fmt}")
    print(f"{'='*60}\n")

    # ── Step 1: scan_daily.py ──────────────────────────────
    print(f"【Step 1/2】scan_daily.py --date {prev_day} 起動中...")
    print(f"  前営業日（{prev_day_fmt}）のデータを取得・スコアリングします\n")

    result = subprocess.run(
        [sys.executable, "scan_daily.py", "--date", prev_day],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    if result.returncode != 0:
        print(f"\n{'='*60}")
        print(f"  ❌ scan_daily.py がエラーで終了しました（コード: {result.returncode}）")
        print(f"  scan_morning.py の起動をスキップします。")
        print(f"  → 問題を確認後、python scan_morning.py を手動で実行してください。")
        print(f"{'='*60}")
        return

    print(f"\n  ✅ scan_daily.py 完了\n")

    # ── Step 2: scan_morning.py ────────────────────────────
    print(f"【Step 2/2】scan_morning.py 起動中...")
    print(f"  地合い予測・候補表示・自動発注を開始します\n")

    import scan_morning
    scan_morning.main()


if __name__ == "__main__":
    main()
