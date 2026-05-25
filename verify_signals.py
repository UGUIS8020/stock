# -*- coding: utf-8 -*-
"""
verify_signals.py - daytime.py シグナルの事後検証

使い方:
    python verify_signals.py            # 直近シグナル日を自動選択（DB終値）
    python verify_signals.py 2026-05-25 # 日付指定（DB終値）
    python verify_signals.py --live     # 当日リアルタイム価格で即時検証
"""

import sys
import json
import time
import urllib3
import db
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

JST = timezone(timedelta(hours=9))


def fetch_live_prices(codes):
    """yfinanceから当日の高値・安値・終値を取得する（16:30以降推奨）。"""
    import yfinance as yf
    result = {}
    print(f"  yfinance から {len(codes)}銘柄を取得中...")
    for code in codes:
        try:
            ticker = yf.Ticker(f"{code}.T")
            hist = ticker.history(period="2d")
            if not hist.empty:
                row = hist.iloc[-1]
                result[code] = {
                    "close": float(row["Close"]),
                    "high":  float(row["High"]),
                    "low":   float(row["Low"]),
                }
        except Exception:
            pass
    return result


def _analyze(records, target_date):
    """シグナル要因別の勝敗分析を出力する。"""
    if not records:
        return

    def win_rate_str(wins, losses):
        total = wins + losses
        if total == 0:
            return "─"
        return f"{wins/(total)*100:.0f}%  ({wins}勝{losses}敗/{total}件)"

    def outcome_counts(subset):
        tp = sum(1 for r in subset if r["outcome"] == "tp")
        sl = sum(1 for r in subset if r["outcome"] == "sl")
        bo = sum(1 for r in subset if r["outcome"] == "both")
        ho = sum(1 for r in subset if r["outcome"] == "hold")
        return tp, sl, bo, ho

    print(f"\n  {'─'*65}")
    print(f"  【要因分析】{target_date}")
    print(f"  {'─'*65}")

    # ── 条件フラグ別 ──────────────────────────────────────
    print(f"\n  ▍条件フラグ別")
    for flag, key in [("ブレイクアウト(BO)", "bo"), ("出来高急増", "vs"), ("モメンタム(MO)", "mo")]:
        with_flag    = [r for r in records if r[key]]
        without_flag = [r for r in records if not r[key]]
        tp_w, sl_w, _, _ = outcome_counts(with_flag)
        tp_n, sl_n, _, _ = outcome_counts(without_flag)
        print(f"  {flag:16} あり: {win_rate_str(tp_w, sl_w):<30} なし: {win_rate_str(tp_n, sl_n)}")

    # ── 条件数別 ──────────────────────────────────────────
    print(f"\n  ▍条件数別")
    for cond_count in sorted({r["conds"] for r in records}):
        subset = [r for r in records if r["conds"] == cond_count]
        tp, sl, both, hold = outcome_counts(subset)
        label = f"{cond_count}条件"
        print(f"  {label:8} {len(subset):2}件  TP:{tp} SL:{sl} both:{both} hold:{hold}  "
              f"勝率: {win_rate_str(tp, sl)}")

    # ── 時間帯別 ──────────────────────────────────────────
    print(f"\n  ▍時間帯別")
    bands = [("午前前半", "09:00", "10:30"),
             ("午前後半", "10:30", "11:30"),
             ("午後前半", "12:30", "13:30"),
             ("午後後半", "13:30", "15:00")]
    for label, t_from, t_to in bands:
        subset = [r for r in records if t_from <= r["time"] < t_to]
        if not subset:
            continue
        tp, sl, both, hold = outcome_counts(subset)
        print(f"  {label}({t_from}〜{t_to})  {len(subset):2}件  "
              f"TP:{tp} SL:{sl} hold:{hold}  勝率: {win_rate_str(tp, sl)}")

    # ── 終値変化率の分布 ──────────────────────────────────
    print(f"\n  ▍終値変化率（シグナル価格比）")
    chg_vals = [r["close_chg"] for r in records]
    avg_chg  = sum(chg_vals) / len(chg_vals)
    pos_cnt  = sum(1 for c in chg_vals if c >= 0)
    neg_cnt  = sum(1 for c in chg_vals if c < 0)
    max_chg  = max(chg_vals)
    min_chg  = min(chg_vals)
    print(f"  平均: {avg_chg:+.2f}%  "
          f"最大: {max_chg:+.2f}%  最小: {min_chg:+.2f}%  "
          f"プラス: {pos_cnt}件  マイナス: {neg_cnt}件")

    # TP達成 vs SL到達の終値変化率比較
    tp_chgs   = [r["close_chg"] for r in records if r["outcome"] == "tp"]
    sl_chgs   = [r["close_chg"] for r in records if r["outcome"] == "sl"]
    hold_chgs = [r["close_chg"] for r in records if r["outcome"] == "hold"]
    if tp_chgs:
        print(f"  TP達成グループ終値平均: {sum(tp_chgs)/len(tp_chgs):+.2f}%")
    if sl_chgs:
        print(f"  SL到達グループ終値平均: {sum(sl_chgs)/len(sl_chgs):+.2f}%")
    if hold_chgs:
        print(f"  保有終了グループ終値平均: {sum(hold_chgs)/len(hold_chgs):+.2f}%")

    # ── SL到達銘柄の詳細 ──────────────────────────────────
    sl_records = [r for r in records if r["outcome"] in ("sl", "both")]
    if sl_records:
        print(f"\n  ▍SL到達銘柄 ({len(sl_records)}件) — 下落要因確認用")
        print(f"  {'時刻':6} {'コード':6} {'銘柄名':16} {'終値変化':>8}  フラグ")
        for r in sorted(sl_records, key=lambda x: x["close_chg"]):
            print(f"  {r['time']:6} {r['code']:6} {str(r['name'])[:14]:<16} "
                  f"{r['close_chg']:>+7.2f}%  [{r['flags']}]  conds:{r['conds']}")

    # ── TP達成銘柄の共通点 ────────────────────────────────
    tp_records = [r for r in records if r["outcome"] == "tp"]
    if tp_records:
        print(f"\n  ▍TP達成銘柄 ({len(tp_records)}件) — 成功要因")
        bo_rate = sum(1 for r in tp_records if r["bo"]) / len(tp_records) * 100
        vs_rate = sum(1 for r in tp_records if r["vs"]) / len(tp_records) * 100
        mo_rate = sum(1 for r in tp_records if r["mo"]) / len(tp_records) * 100
        avg_conds = sum(r["conds"] for r in tp_records) / len(tp_records)
        print(f"  BO保有率: {bo_rate:.0f}%  出来高急増保有率: {vs_rate:.0f}%  "
              f"MO保有率: {mo_rate:.0f}%  平均条件数: {avg_conds:.1f}")


def verify(target_date=None, live=False):
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

    mode_str = "yfinance（当日終値）" if live else "DB終値"
    print(f"\n{'='*65}")
    print(f"  シグナル検証: {target_date}  [{mode_str}]")
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

    # 価格取得（リアルタイム or DB）
    codes = [s[0] for s in signals]
    ohlcv = {}

    if live:
        print(f"  Tachibana API から現在値を取得中...")
        ohlcv = fetch_live_prices(codes)
        print(f"  → {len(ohlcv)}/{len(codes)}銘柄 取得完了\n")
    else:
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
    records = []

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
            outcome = "both"
            result = "⚠️  TP/SL両到達（順序不明）"
            both_count += 1
        elif hit_tp:
            outcome = "tp"
            result = f"✅ TP達成 (+{(tp_price/sig_price-1)*100:.1f}%)"
            tp_count += 1
        elif hit_sl:
            outcome = "sl"
            result = f"🛑 SL到達 ({(sl_price/sig_price-1)*100:.1f}%)"
            sl_count += 1
        else:
            outcome = "hold"
            arrow = "▲" if close_chg >= 0 else "▼"
            result = f"  保有終了 ({arrow}{abs(close_chg):.1f}%)"
            hold_count += 1

        records.append({
            "code": code, "name": name, "time": sig_time[:5],
            "sig_price": sig_price, "high": high, "low": low, "close": close,
            "close_chg": close_chg, "outcome": outcome,
            "bo": bool(bo), "vs": bool(vs), "mo": bool(mo),
            "conds": int(conds), "flags": flag_str,
        })

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

    # ── 要因分析 ──────────────────────────────────────────
    _analyze(records, target_date)

    print(f"{'='*65}\n")
    print("  ※ TP/SL両到達はどちらが先かintraday_pricesで要確認")


if __name__ == "__main__":
    args = sys.argv[1:]
    live = "--live" in args
    date_arg = next((a for a in args if a != "--live"), None)
    verify(date_arg, live=live)
