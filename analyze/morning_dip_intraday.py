# -*- coding: utf-8 -*-
"""
morning_dip_intraday.py - 「朝下落→上昇開始」当日パフォーマンス検証

【想定シナリオ】
  朝に下落して安値をつけた後、上昇し始めたタイミングで買う
  → 終値まで保有した場合の損益を分析

【使う指標】
  lower_shadow_pct = (Open - Low) / Open * 100   朝の下落幅
  close_position   = (Close - Low) / (High - Low) レンジ内終値位置
  open_return      = (Close - Open) / Open * 100  始値比終値騰落率
  low_to_close     = (Close - Low) / Low * 100    安値から終値の上昇率（安値で買えた場合）
"""
import sys
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = str(Path(__file__).parent.parent / "out" / "stock.db")

MIN_VOLUME = 50_000
MIN_PRICE  = 200
MAX_PRICE  = 50_000

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT code, Date, Open, High, Low, Close, Volume
        FROM daily_prices
        WHERE Open > 0 AND Close > 0 AND Low > 0 AND High > 0
          AND High > Low
          AND Volume >= ?
          AND Close BETWEEN ? AND ?
        ORDER BY code, Date
    """, conn, params=(MIN_VOLUME, MIN_PRICE, MAX_PRICE))
    conn.close()
    df["Date"] = pd.to_datetime(df["Date"])
    df["lower_shadow_pct"] = (df["Open"] - df["Low"])  / df["Open"] * 100
    df["close_position"]   = (df["Close"] - df["Low"]) / (df["High"] - df["Low"])
    df["open_return"]      = (df["Close"] - df["Open"]) / df["Open"] * 100   # 始値→終値
    df["low_to_close"]     = (df["Close"] - df["Low"])  / df["Low"]  * 100   # 安値→終値（理想的な買い）
    return df

def print_stats(label, sr, extra=None):
    n       = len(sr)
    avg     = sr.mean()
    med     = sr.median()
    win     = (sr > 0).sum()
    win_pct = win / n * 100 if n > 0 else 0
    p25     = sr.quantile(0.25)
    p75     = sr.quantile(0.75)
    line = (f"  {label:<30} N={n:>7,}件  "
            f"平均={avg:>+6.2f}%  中央値={med:>+6.2f}%  "
            f"勝率={win_pct:>5.1f}%  "
            f"[25%={p25:>+5.1f}% / 75%={p75:>+5.1f}%]")
    if extra:
        line += f"  {extra}"
    print(line)

def main():
    print("=" * 90)
    print("  「朝下落→上昇開始」 当日パフォーマンス検証")
    print("=" * 90)

    print("\nデータ読み込み中...")
    df = load_data()
    print(f"  {df['code'].nunique():,}銘柄 × {len(df):,}日分のデータ")

    # ── セクション1: 朝の下落幅別 始値→終値リターン ──
    print(f"\n\n【1】朝の下落幅別 始値→終値リターン（open_return）")
    print("    ※ 始値で買って終値で売った場合の損益")
    print("-" * 90)
    bins   = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 99]
    labels = ["0〜0.5%", "0.5〜1.0%", "1.0〜1.5%", "1.5〜2.0%", "2.0〜3.0%", "3.0〜5.0%", "5.0%以上"]
    df["shadow_bin"] = pd.cut(df["lower_shadow_pct"], bins=bins, labels=labels, right=False)
    for lbl, grp in df.groupby("shadow_bin", observed=True):
        if len(grp) > 100:
            print_stats(str(lbl), grp["open_return"])

    # ── セクション2: 朝の下落幅別 安値→終値リターン ──
    print(f"\n\n【2】朝の下落幅別 安値→終値リターン（low_to_close）")
    print("    ※ 安値付近で買えた場合の理想的な損益")
    print("-" * 90)
    for lbl, grp in df.groupby("shadow_bin", observed=True):
        if len(grp) > 100:
            print_stats(str(lbl), grp["low_to_close"])

    # ── セクション3: close_position 分布 ──
    print(f"\n\n【3】朝に-1%以上下落した銘柄の close_position 分布")
    print("    ※ その日どのくらい回復して引けたか（0=安値、1=高値）")
    print("-" * 90)
    dipped = df[df["lower_shadow_pct"] >= 1.0]
    cp_bins   = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.01]
    cp_labels = ["0〜0.2(ほぼ安値)", "0.2〜0.4", "0.4〜0.6(中央)", "0.6〜0.7", "0.7〜0.8", "0.8〜0.9", "0.9〜1.0(ほぼ高値)"]
    dipped = dipped.copy()
    dipped["cp_bin"] = pd.cut(dipped["close_position"], bins=cp_bins, labels=cp_labels, right=False)
    total = len(dipped)
    print(f"  朝-1%以上下落した銘柄 合計: {total:,}件")
    print()
    for lbl, grp in dipped.groupby("cp_bin", observed=True):
        pct = len(grp) / total * 100
        print_stats(f"{lbl}", grp["open_return"], extra=f"({pct:.1f}%)")

    # ── セクション4: 「回復開始」が確認できた場合の成績 ──
    print(f"\n\n【4】朝-1%以上下落 かつ close_position別の open_return")
    print("    ※ 「朝下落→回復して引けた」パターンの当日成績")
    print("-" * 90)
    for cp_min, label in [(0.5, "cp≥0.5(中央以上)"), (0.6, "cp≥0.6"), (0.7, "cp≥0.7(上位30%)"), (0.8, "cp≥0.8(上位20%)")]:
        sub = dipped[dipped["close_position"] >= cp_min]
        print_stats(label, sub["open_return"])

    print()
    print("  ＜比較＞朝下落なし（lower_shadow < 0.5%）")
    nodip = df[df["lower_shadow_pct"] < 0.5]
    print_stats("下落なし", nodip["open_return"])

    # ── セクション5: 深い下落ほど回復率は？ ──
    print(f"\n\n【5】「朝の下落幅」と「当日中に始値まで回復する確率」")
    print("-" * 90)
    print(f"  {'下落幅':<15} {'件数':>8}  {'始値まで回復(Close≥Open)':>20}  {'安値から+1%以上回復':>20}")
    for lbl, grp in df.groupby("shadow_bin", observed=True):
        if len(grp) < 100:
            continue
        n = len(grp)
        recover_open  = (grp["open_return"] >= 0).sum()
        recover_1pct  = (grp["low_to_close"] >= 1.0).sum()
        print(f"  {str(lbl):<15} {n:>8,}件  "
              f"  {recover_open/n*100:>6.1f}% ({recover_open:,}件)"
              f"          {recover_1pct/n*100:>6.1f}% ({recover_1pct:,}件)")

    print(f"\n{'=' * 90}")
    print("完了")

if __name__ == "__main__":
    main()
