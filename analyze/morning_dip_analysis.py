# -*- coding: utf-8 -*-
"""
morning_dip_analysis.py - 「朝下落→上昇開始」パターンの翌日騰落率検証

【定義】
  朝下落後上昇 = 安値が始値比 -X% 以下 かつ 終値がレンジ上位 Y% 以上
                 (close_position = (Close-Low)/(High-Low) >= Y)

  V字回復との違い：終値が始値を超えなくてもよい（レンジ内で上方回復を評価）

【close_positionの意味】
  0.0 = 終値が安値と同じ（下落したまま）
  0.5 = 終値がレンジ中央
  0.7 = 終値がレンジ上位30%（上昇開始と判断できる水準）
  1.0 = 終値が高値と同じ（最強）
"""
import sys
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = str(Path(__file__).parent.parent / "out" / "stock.db")

# ── パラメータ ──────────────────────────────────────────
DIP_THRESHOLDS    = [-0.5, -1.0, -1.5, -2.0]   # 安値の始値比下落率
CLOSE_POS_LEVELS  = [0.5, 0.6, 0.7, 0.8]        # 終値レンジ位置の下限
MIN_VOLUME        = 50_000
MIN_PRICE         = 200
MAX_PRICE         = 50_000
NEXT_DAY_CAP      = 20.0

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
    return df

def compute_features(df):
    df = df.copy()
    df["lower_shadow_pct"] = (df["Open"] - df["Low"]) / df["Open"] * 100   # 始値からの下落幅%
    df["close_position"]   = (df["Close"] - df["Low"]) / (df["High"] - df["Low"])  # 0〜1
    df["range_pct"]        = (df["High"] - df["Low"]) / df["Open"] * 100    # 日中レンジ幅%
    return df

def analyze(df, dip_threshold, close_pos_min):
    records = []
    for code, grp in df.groupby("code"):
        grp = grp.sort_values("Date").reset_index(drop=True)
        for i in range(len(grp) - 1):
            row      = grp.iloc[i]
            next_row = grp.iloc[i + 1]

            days_diff = (next_row["Date"] - row["Date"]).days
            if days_diff > 4:
                continue

            n_o = next_row["Open"]
            n_c = next_row["Close"]
            if n_o <= 0:
                continue
            next_ret = (n_c - n_o) / n_o * 100
            if abs(next_ret) > NEXT_DAY_CAP:
                continue

            # 朝下落後上昇条件
            is_pattern = (
                row["lower_shadow_pct"] >= abs(dip_threshold) and
                row["close_position"]   >= close_pos_min
            )

            records.append({
                "is_pattern":       is_pattern,
                "lower_shadow_pct": round(row["lower_shadow_pct"], 2),
                "close_position":   round(row["close_position"], 3),
                "next_ret":         round(next_ret, 2),
            })

    return pd.DataFrame(records)

def print_stats(label, sr):
    n       = len(sr)
    avg     = sr.mean()
    med     = sr.median()
    win     = (sr > 0).sum()
    win_pct = win / n * 100 if n > 0 else 0
    p25     = sr.quantile(0.25)
    p75     = sr.quantile(0.75)
    print(f"  {label:<26} N={n:>7,}件  "
          f"平均={avg:>+6.2f}%  中央値={med:>+6.2f}%  "
          f"勝率={win_pct:>5.1f}%  "
          f"[25%={p25:>+5.1f}% / 75%={p75:>+5.1f}%]")

def main():
    print("=" * 80)
    print("  「朝下落→上昇開始」パターン 翌日騰落率検証")
    print("  (lower_shadow: 始値比下落幅  /  close_position: レンジ内終値位置)")
    print("=" * 80)

    print("\nデータ読み込み中...")
    df = load_data()
    df = compute_features(df)
    print(f"  {df['code'].nunique():,}銘柄 × {len(df):,}日分のデータ")

    # ── メイン比較：dip別 × close_position別 ──
    for dip in DIP_THRESHOLDS:
        print(f"\n【安値が始値比 {dip:+.1f}% 以下に下落した場合】")
        print("-" * 80)
        for cp in CLOSE_POS_LEVELS:
            result = analyze(df, dip, cp)
            pat     = result[result["is_pattern"]]["next_ret"]
            non_pat = result[~result["is_pattern"]]["next_ret"]

            if len(pat) < 30:
                print(f"  close_pos≥{cp:.1f}: データ不足 (N={len(pat)})")
                continue

            diff = pat.mean() - non_pat.mean()
            label = f"dip{dip:+.1f}% / cp≥{cp:.1f}"
            print_stats(label, pat)

        # 非パターンを1回表示
        result_base = analyze(df, dip, 0.5)
        non_all = result_base[~result_base["is_pattern"]]["next_ret"]
        print_stats(f"  ↑ 非パターン(dip{dip:+.1f}%)", non_all)

    # ── close_position の分布別詳細（dip=-1.0%固定）──
    print(f"\n\n【close_position の区間別 翌日成績（安値≤始値比-1.0% かつ始値下落あり）】")
    print("-" * 80)
    target = df[df["lower_shadow_pct"] >= 1.0].copy()
    target_next = []
    for code, grp in target.groupby("code"):
        grp = grp.sort_values("Date").reset_index(drop=True)
        for i in range(len(grp) - 1):
            row = grp.iloc[i]
            next_row = grp.iloc[i + 1]
            if (next_row["Date"] - row["Date"]).days > 4:
                continue
            n_o = next_row["Open"]
            n_c = next_row["Close"]
            if n_o <= 0:
                continue
            next_ret = (n_c - n_o) / n_o * 100
            if abs(next_ret) > NEXT_DAY_CAP:
                continue
            target_next.append({
                "cp_bin": row["close_position"],
                "next_ret": next_ret,
            })

    if target_next:
        tn = pd.DataFrame(target_next)
        bins   = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.01]
        labels = ["0〜0.2", "0.2〜0.4", "0.4〜0.6", "0.6〜0.7", "0.7〜0.8", "0.8〜0.9", "0.9〜1.0"]
        tn["bin"] = pd.cut(tn["cp_bin"], bins=bins, labels=labels, right=False)
        for lbl, grp in tn.groupby("bin", observed=True):
            if len(grp) > 10:
                print_stats(str(lbl), grp["next_ret"])

    print(f"\n{'=' * 80}")
    print("完了")

if __name__ == "__main__":
    main()
