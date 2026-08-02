# -*- coding: utf-8 -*-
"""
run_daily.py - 1日の全スクリプトを順番に起動

実行するだけで:
  1. download_db.py        S3から最新DBをダウンロード
  2. tachibana_login.py    ログイン + セッション取得
  3. scan_daily.py         前営業日データ取得・スコアリング（失敗時は警告して続行）
  4. scan_morning.py       地合い予測・候補選定
                           ↓ scan_morning.py が内部で以下を全起動:
                           - position_monitor.py  9:00〜15:28 TP/SL自動売り（BG）
                           - daytime.py           9:30〜14:30 日中シグナル監視（BG）
                           - market_watch.py      9:00〜9:30 戦略A発注
                           - closing_watch.py     14:55〜15:15 戦略B発注
                           - scan_daily.py        16:45 本日データ自動取得

使い方:
    python run_daily.py           # 通常実行
    python run_daily.py --manual  # closing_watch を手動確認モードで起動（未実装）
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


def is_trading_day(d=None):
    """土日・祝日・東証の年末年始休場（12/31・1/2・1/3、祝日ではない取引所慣例）を除く営業日か判定する。
    jpholidayは国民の祝日のみ判定するため、年末年始休場は別途明示的に除外する。"""
    d = d or datetime.now(JST).date()
    if d.weekday() >= 5:
        return False
    if (d.month == 12 and d.day == 31) or (d.month == 1 and d.day in (2, 3)):
        return False
    try:
        import jpholiday
        if jpholiday.is_holiday(d):
            return False
    except ImportError:
        pass
    return True


def run_step(label, cmd, cwd):
    """サブプロセスを実行してリターンコードを返す。Ctrl+C は graceful に処理。"""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    try:
        result = subprocess.run(cmd, cwd=cwd)
        return result.returncode
    except KeyboardInterrupt:
        print(f"\n  ⚠️  Ctrl+C を受信 → {label} を中断しました")
        return 1


def main():
    parser = argparse.ArgumentParser(description="1日の全スクリプト順次起動")
    parser.add_argument("--manual", action="store_true",
                        help="closing_watch を手動確認モードで起動（scan_morning経由では未対応）")
    args = parser.parse_args()

    cwd   = os.path.dirname(os.path.abspath(__file__))
    py    = sys.executable
    today = datetime.now(JST).strftime("%Y-%m-%d")

    print(f"{'='*60}")
    print(f"  run_daily.py 起動（{today}）")
    print(f"{'='*60}")

    if not is_trading_day():
        print(f"\n  ⏸️  本日（{today}）は休日（土日・祝日・年末年始休場）のため実行しません")
        return

    # ── Step 1: S3から最新DBをダウンロード ───────────────
    rc = run_step("【Step 1/4】download_db.py（S3から最新DBをダウンロード）",
                  [py, "download_db.py", "--db-only"], cwd)
    if rc != 0:
        print("⚠️  DBダウンロード失敗。ローカルDBで続行します。")

    # ── Step 2: ログイン（scan_morning自動起動なし）─────────
    print(f"\n{'='*60}")
    print(f"  【Step 2/4】Tachibana ログイン（sAuthId認証）")
    print(f"{'='*60}\n")
    sys.path.insert(0, cwd)
    from tachibana_login import do_login
    if not do_login():
        print("❌ ログイン失敗。中断します。")
        return
    print("✅ ログイン完了")

    # ── Step 3: 前営業日データ取得 ────────────────────────
    # scan_morning.py が今日のスコアリングに使う前日データを補完する。
    # 前日の _run_scan_daily_at_1640() が失敗した場合のリカバリー。
    # 失敗しても続行（scan_morning が最新 DB データで動作する）。
    prev_day = get_prev_business_day()
    prev_fmt = f"{prev_day[:4]}-{prev_day[4:6]}-{prev_day[6:]}"
    rc = run_step(f"【Step 3/4】scan_daily.py --date {prev_fmt}（前営業日データ補完）",
                  [py, "scan_daily.py", "--date", prev_day], cwd)
    if rc != 0:
        print(f"⚠️  scan_daily.py エラー（{prev_fmt} のデータ取得失敗）。")
        print("   前日までのDBデータで scan_morning を続行します。")
        print("   ※ 16:45以降に python scan_daily.py を手動実行してデータを補完してください。")

    # ── Step 4: scan_morning.py（今日のメイン処理を全て内包）──
    # scan_morning.py は以下を順に起動して 17:00頃まで動作し続ける:
    #   - position_monitor / daytime（バックグラウンド）
    #   - market_watch（戦略A）→ closing_watch（戦略B）
    #   - scan_daily（16:45 本日データ自動取得）
    # Steps 5〜7 は scan_morning.py 内で実行されるため run_daily.py 側では不要。
    run_step("【Step 4/4】scan_morning.py（地合い判定・全戦略起動・16:45 scan_daily）",
             [py, "scan_morning.py"], cwd)

    print(f"\n{'='*60}")
    print(f"  本日の全ステップ完了（{datetime.now(JST).strftime('%H:%M:%S')}）")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  ⚠️  run_daily.py を中断しました（Ctrl+C）")
        sys.exit(1)
