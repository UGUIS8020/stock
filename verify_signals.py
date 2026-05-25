# -*- coding: utf-8 -*-
"""
verify_signals.py - daytime.py シグナルの事後検証

使い方:
    python verify_signals.py            # 直近シグナル日を自動選択
    python verify_signals.py 2026-05-25 # 日付指定
"""

import sys
import db
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding="utf-8")


def verify(target_date=None):
    db.init_db()
    conn = db.get_conn()

    # 日付未指定なら直近シグナル日を自動選択
    if not target_date:
        r = conn.execute(
            "SELECT DISTINCT date FROM daytime_signals ORDER BY date DESC LIMIT 1"
        ).fetchone()
        if not r:
            print("daytime_signals にデータがありません。")
            conn.close()
            return
        target_date = r[0]

    print(f"\n{'='*65}")
    print(f"  シグナル検証: {target_date}")
    print(f"{'='*65}")

    # シグナル取得
    signals = conn.execute(
        """SELECT code, name, time, price, tp_price, sl_price,
                  breakout, volume_surge, momentum, conditions_met
           FROM daytime_signals WHERE date=? ORDER BY time""",
        (target_date,)
    ).fetchall()

    if not signals:
        print("  該当日のシグナルデータなし")
        conn.close()
        return

    # 当日の日足データ取得
    codes = [s[0] for s in signals]
    ohlcv = {}
    for code in codes:
        r = conn.execute(
            "SELECT High, Low, Close FROM daily_prices WHERE code=? AND Date=?",
            (code, target_date)
        ).fetchone()
        if r:
            ohlcv[code] = {"high": r[0], "low": r[1], "close": r[2]}

    conn.close()

    # 集計
    tp_count = sl_count = both_count = hold_count = no_data = 0

    print(f"\n  {'時刻':6} {'コード':6} {'銘柄名':18} {'シグナル値':>8} {'高値':>8} {'安値':>8} {'終値':>8}  結果")
    print("  " + "─" * 80)

    for s in signals:
        code, name, sig_time, sig_price, tp_price, sl_price, bo, vs, mo, conds = s
        flags = []
        if bo: flags.append("BO")
        if vs: flags.append("出来高")
        if mo: flags.append("MO")
        flag_str = "/".join(flags)

        if code not in ohlcv:
            no_data += 1
            print(f"  {sig_time[:5]:6} {code:6} {str(name)[:16]:<18} "
                  f"{sig_price:>8,.0f}                            データなし")
            continue

        d = ohlcv[code]
        high  = d["high"]
        low   = d["low"]
        close = d["close"]

        hit_tp = high  >= tp_price
        hit_sl = low   <= sl_price
        close_chg = (close - sig_price) / sig_price * 100

        if hit_tp and hit_sl:
            result = "⚠️  TP/SL両到達（順序不明）"
            both_count += 1
        elif hit_tp:
            result = f"✅ TP達成 (+{(tp_price/sig_price-1)*100:.1f}%)"
            tp_count += 1
        elif hit_sl:
            result = f"🛑 SL到達 ({(sl_price/sig_price-1)*100:.1f}%)"
            sl_count += 1
        else:
            arrow = "▲" if close_chg >= 0 else "▼"
            result = f"  保有終了 ({arrow}{abs(close_chg):.1f}%)"
            hold_count += 1

        print(f"  {sig_time[:5]:6} {code:6} {str(name)[:16]:<18} "
              f"{sig_price:>8,.0f} {high:>8,.0f} {low:>8,.0f} {close:>8,.0f}  {result}  [{flag_str}]")

    total = len(signals)
    print(f"\n  {'─'*65}")
    print(f"  【集計】 {target_date}  シグナル計{total}件")
    print(f"  ✅ TP達成    : {tp_count}件 ({tp_count/total*100:.0f}%)")
    print(f"  🛑 SL到達    : {sl_count}件 ({sl_count/total*100:.0f}%)")
    print(f"  ⚠️  TP/SL両方 : {both_count}件 ({both_count/total*100:.0f}%)")
    print(f"     保有終了  : {hold_count}件 ({hold_count/total*100:.0f}%)")
    if no_data:
        print(f"     データなし: {no_data}件")

    if tp_count + sl_count > 0:
        win_rate = tp_count / (tp_count + sl_count) * 100
        print(f"\n  勝率 (TP/SL決着分): {win_rate:.1f}%  ({tp_count}勝{sl_count}敗)")

    print(f"{'='*65}\n")
    print("  ※ TP/SL両到達はどちらが先かintraday_pricesで要確認")


if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    verify(date_arg)
