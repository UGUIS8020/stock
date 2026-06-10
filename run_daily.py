# -*- coding: utf-8 -*-
"""
run_daily.py - 1日の全スクリプトを順番に起動

実行するだけで:
  1. download_db.py        S3から最新DBをダウンロード
  2. tachibana_login.py    電話認証 + セッション取得
  3. scan_daily.py         前営業日データ取得・スコアリング
  4. scan_morning.py       地合い予測・候補選定
  5. market_watch.py       9:00〜9:30 リアルタイム監視・発注（戦略A）
  6. position_monitor.py   9:30〜15:28 TP/SL自動売り（バックグラウンド）
  7. closing_watch.py      14:55〜15:15 引け前スキャン・発注（戦略B）

使い方:
    python run_daily.py           # 通常実行
    python run_daily.py --manual  # closing_watch を手動確認モードで起動
"""

import sys
import os
import subprocess
import argparse
from datetime import datetime, timedelta, timezone

sys.stdout.reconfigure(encoding="utf-8")

JST = timezone(timedelta(hours=9))


def get_prev_business_day():
    """前営業日を返す（土日・祝日をスキップ）。"""
    prev = datetime.now() - timedelta(days=1)
    try:
        import jpholiday
        while prev.weekday() >= 5 or jpholiday.is_holiday(prev):
            prev -= timedelta(days=1)
    except ImportError:
        while prev.weekday() >= 5:
            prev -= timedelta(days=1)
    return prev.strftime("%Y%m%d")


def run_step(label, cmd, cwd):
    """サブプロセスを実行してリターンコードを返す。"""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="1日の全スクリプト順次起動")
    parser.add_argument("--manual", action="store_true",
                        help="closing_watch を手動確認モードで起動")
    args = parser.parse_args()

    cwd   = os.path.dirname(os.path.abspath(__file__))
    py    = sys.executable
    today = datetime.now(JST).strftime("%Y-%m-%d")

    print(f"{'='*60}")
    print(f"  run_daily.py 起動（{today}）")
    print(f"{'='*60}")

    # ── Step 1: S3から最新DBをダウンロード ───────────────
    rc = run_step("【Step 1/7】download_db.py（S3から最新DBをダウンロード）",
                  [py, "download_db.py", "--db-only"], cwd)
    if rc != 0:
        print("⚠️  DBダウンロード失敗。ローカルDBで続行します。")

    # ── Step 2: 電話認証 + ログイン ───────────────────────
    rc = run_step("【Step 2/7】Tachibana 電話認証 + ログイン",
                  [py, "tachibana_login.py"], cwd)
    if rc != 0:
        print("❌ ログイン失敗。中断します。")
        return

    # ── Step 3: scan_daily.py ─────────────────────────────
    prev_day = get_prev_business_day()
    prev_fmt = f"{prev_day[:4]}-{prev_day[4:6]}-{prev_day[6:]}"
    rc = run_step(f"【Step 3/7】scan_daily.py --date {prev_fmt}（前営業日データ取得）",
                  [py, "scan_daily.py", "--date", prev_day], cwd)
    if rc != 0:
        print("❌ scan_daily.py エラー。scan_morning 以降をスキップします。")
        return

    # ── Step 4: scan_morning.py ───────────────────────────
    # scan_morning は今日のログイン済みセッションを使うので電話認証不要
    rc = run_step("【Step 4/7】scan_morning.py（地合い予測・候補選定）",
                  [py, "scan_morning.py"], cwd)
    if rc != 0:
        print("❌ scan_morning.py エラー。market_watch 以降をスキップします。")
        return

    # ── Step 5: market_watch.py（9:00〜9:30）─────────────
    rc = run_step("【Step 5/7】market_watch.py（9:00〜9:30 リアルタイム監視・戦略A発注）",
                  [py, "market_watch.py"], cwd)
    # market_watch は時間切れでも rc=0 なので中断しない

    # ── Step 6: position_monitor.py（バックグラウンド起動）──
    print(f"\n{'='*60}")
    print(f"  【Step 6/7】position_monitor.py（TP/SL自動売り・バックグラウンド）")
    print(f"{'='*60}\n")
    pm_proc = subprocess.Popen([py, "position_monitor.py"], cwd=cwd)
    print(f"  ✅ position_monitor をバックグラウンドで起動しました（PID: {pm_proc.pid}）")
    print(f"     TP/SL到達で自動売り、15:28に自動終了します")

    # ── Step 7: closing_watch.py（14:55〜15:15）──────────
    closing_cmd = [py, "closing_watch.py"]
    if args.manual:
        closing_cmd.append("--manual")
    run_step("【Step 7/7】closing_watch.py（14:55〜 引け前スキャン・戦略B発注）",
             closing_cmd, cwd)

    # position_monitor の終了を待つ（〜15:28）
    print(f"\n  ⏳ position_monitor の終了を待っています...")
    pm_proc.wait()
    print(f"  ✅ position_monitor 終了")

    print(f"\n{'='*60}")
    print(f"  本日の全ステップ完了（{datetime.now(JST).strftime('%H:%M:%S')}）")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
