# -*- coding: utf-8 -*-
"""
v_recovery_analysis.py - V字回復パターンの翌日騰落率検証

【定義】
  V字回復日 = 当日安値が始値比 -X% 以下に下落 かつ 終値が始値より上昇
  非V字回復 = 上記以外（始値比で下落したまま or そもそもあまり下げなかった）

【検証内容】
  V字回復した銘柄の翌日始値→終値騰落率 vs 非V字回復銘柄の翌日騰落率を比較
"""
import sys
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = str(Path(__file__).parent.parent / "out" / "stock.db")

# ── パラメータ ──────────────────────────────────────────
DIP_THRESHOLDS = [-0.5, -1.0, -1.5, -2.0]  # 安値が始値比で何%以下に下落したか
MIN_VOLUME     = 50_000       # 最低出来高（薄商い除外）
MIN_PRICE      = 200          # 最低株価（超低位株除外）
MAX_PRICE      = 50_000       # 最高株価（異常値除外）
NEXT_DAY_CAP   = 20.0         # 翌日騰落率の外れ値除外（±20%超）

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT code, Date, Open, High, Low, Close, Volume, prev_close
        FROM daily_prices
        WHERE Open > 0 AND Close > 0 AND Low > 0
          AND Volume >= ?
          AND Close BETWEEN ? AND ?
        ORDER BY code, Date
    """, conn, params=(MIN_VOLUME, MIN_PRICE, MAX_PRICE))
    conn.close()
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def analyze(df, dip_threshold):
    """
    dip_threshold: 始値比の安値下落率（例: -1.0 = 始値より1%以上下落した）
    """
    records = []

    for code, grp in df.groupby("code"):
        grp = grp.sort_values("Date").reset_index(drop=True)
        for i in range(len(grp) - 1):
            row      = grp.iloc[i]
            next_row = grp.iloc[i + 1]

            # 翌日が連続した営業日かチェック（最大4日以内）
            days_diff = (next_row["Date"] - row["Date"]).days
            if days_diff > 4:
                continue

            o     = row["Open"]
            lo    = row["Low"]
            c     = row["Close"]

            # 安値の始値比下落率
            dip_pct   = (lo - o) / o * 100
            # 終値の始値比上昇率
            close_pct = (c  - o) / o * 100

            # V字回復条件
            is_v = (dip_pct <= dip_threshold) and (close_pct > 0)

            # 翌日騰落率（始値→終値）
            n_o = next_row["Open"]
            n_c = next_row["Close"]
            if n_o <= 0:
                continue
            next_ret = (n_c - n_o) / n_o * 100

            # 外れ値除外
            if abs(next_ret) > NEXT_DAY_CAP:
                continue

            records.append({
                "is_v":    is_v,
                "dip_pct": round(dip_pct, 2),
                "next_ret": round(next_ret, 2),
            })

    result = pd.DataFrame(records)
    return result

def print_stats(label, sr):
    n       = len(sr)
    avg     = sr.mean()
    med     = sr.median()
    win     = (sr > 0).sum()
    win_pct = win / n * 100 if n > 0 else 0
    p25     = sr.quantile(0.25)
    p75     = sr.quantile(0.75)
    print(f"  {label:<20} N={n:>6,}件  "
          f"平均={avg:>+6.2f}%  中央値={med:>+6.2f}%  "
          f"勝率={win_pct:>5.1f}%  "
          f"[25%={p25:>+5.1f}% / 75%={p75:>+5.1f}%]")

def main():
    print("=" * 70)
    print("  V字回復パターン 翌日騰落率検証")
    print("=" * 70)

    print("\nデータ読み込み中...")
    df = load_data()
    print(f"  {df['code'].nunique():,}銘柄 × {len(df):,}日分のデータ")

    for dip in DIP_THRESHOLDS:
        print(f"\n【安値が始値比 {dip:+.1f}% 以下に下落 → 終値が始値比プラス = V字回復】")
        print("-" * 70)

        result = analyze(df, dip)
        v     = result[result["is_v"]]["next_ret"]
        non_v = result[~result["is_v"]]["next_ret"]

        print_stats("V字回復あり", v)
        print_stats("V字回復なし", non_v)

        # 差の有意性（簡易）
        if len(v) > 30 and len(non_v) > 30:
            diff = v.mean() - non_v.mean()
            print(f"\n  差（V字 - 非V字）: {diff:+.3f}%")
            if diff > 0.1:
                print(f"  → V字回復銘柄の翌日が {diff:+.3f}% 高い傾向")
            elif diff < -0.1:
                print(f"  → V字回復銘柄の翌日が {diff:+.3f}% 低い傾向（逆効果）")
            else:
                print(f"  → 有意差なし（差が小さい）")

    # ── ベストDIPの深さ別詳細分析 ──
    print(f"\n\n【安値下落率の深さ別 翌日成績（dip < -0.5%のV字回復のみ）】")
    print("-" * 70)
    result_all = analyze(df, -0.5)
    v_all = result_all[result_all["is_v"]].copy()

    bins  = [-99, -3.0, -2.0, -1.5, -1.0, -0.5, 0]
    labels = ["-3%以下", "-3〜-2%", "-2〜-1.5%", "-1.5〜-1%", "-1〜-0.5%", "-0.5%未満"]
    v_all["dip_bin"] = pd.cut(v_all["dip_pct"], bins=bins, labels=labels)
    for label, grp in v_all.groupby("dip_bin", observed=True):
        if len(grp) > 10:
            print_stats(str(label), grp["next_ret"])

    print(f"\n{'=' * 70}")
    print("完了")

if __name__ == "__main__":
    main()
