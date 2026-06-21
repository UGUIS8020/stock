#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grid_search_a.py - 戦略A の TP/SL をグリッドサーチで最適化

エントリー近似:
  - モメンタム確認（High >= prev_close * 1.003）した銘柄のみ対象
  - 発注価格 = max(Open, prev_close * 1.003)
  - PANIC日は全見送り

IS: 2024-06-01 ~ 2025-09-30
OOS: 2025-10-01 ~

出力: TP/SL ヒートマップ（Sharpe）+ OOS上位10組み合わせ
"""

import sys
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))
import db

# ── フィルタ条件（戦略Aの選定条件を近似）─────────────────────
MOMENTUM_PCT  = 0.3    # モメンタム確認閾値（%）
PREV_RISE_MIN = 0.0    # 前日騰落最低値（%）
MA5_DEV_MAX   = 5.0    # MA5乖離率上限（%）
MA25_DEV_MAX  = 8.0    # MA25乖離率上限（%）
MIN_VOLUME    = 50_000 # 最低出来高
MIN_PRICE     = 200    # 最低価格

PANIC_AD  = 0.20
STRONG_AD = 0.60

# ── グリッドサーチ範囲 ──────────────────────────────────────
TP_VALUES = [round(v, 1) for v in np.arange(1.0, 8.5, 0.5)]   # 1.0%〜8.0%
SL_VALUES = [round(v, 1) for v in np.arange(2.0, 9.0, 1.0)]   # 2.0%〜8.0%

IS_END    = "2025-09-30"
OOS_START = "2025-10-01"


def load_data():
    conn = sqlite3.connect(db.DB_PATH)
    print("  daily_prices 読み込み中...")
    df = pd.read_sql(
        "SELECT code, Date, Open, High, Low, Close, Volume, market_code "
        "FROM daily_prices WHERE Date >= '2024-06-01' ORDER BY code, Date",
        conn
    )
    conn.close()

    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    # ETF除外
    df = df[df["market_code"] != "0109"]

    df = df.sort_values(["code", "Date"]).reset_index(drop=True)
    df["prev_close"]  = df.groupby("code")["Close"].shift(1)
    df["prev_close2"] = df.groupby("code")["Close"].shift(2)
    df["prev_rise"]   = (df["prev_close"] - df["prev_close2"]) / df["prev_close2"] * 100
    df["ma5"]  = df.groupby("code")["Close"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["ma25"] = df.groupby("code")["Close"].transform(lambda x: x.shift(1).rolling(25).mean())
    df["ma5_dev"]  = (df["prev_close"] / df["ma5"]  - 1) * 100
    df["ma25_dev"] = (df["prev_close"] / df["ma25"] - 1) * 100
    df = df.dropna(subset=["prev_close", "prev_rise", "ma5_dev", "ma25_dev"])
    df = df[df["prev_close"] > 0]

    # 地合い計算
    df["up"] = df["Close"] > df["prev_close"]
    daily_ad = (df.groupby("Date")
                  .agg(up_count=("up", "sum"), total=("up", "count"))
                  .reset_index())
    daily_ad["ad_ratio"]  = daily_ad["up_count"] / daily_ad["total"]
    daily_ad["condition"] = daily_ad["ad_ratio"].apply(lambda r:
        "PANIC"  if r <= PANIC_AD  else
        "STRONG" if r >= STRONG_AD else "OTHER")

    df = df.merge(daily_ad[["Date", "condition"]], on="Date", how="left")
    df["Date_str"] = df["Date"].dt.strftime("%Y-%m-%d")

    print(f"  {len(df):,}行 / {df['Date'].nunique()}日 / {df['code'].nunique()}銘柄（ETF除外済み）")
    return df


def prefilter(df, date_start=None, date_end=None):
    """固定フィルタを適用してエントリー候補を返す。"""
    mask = (
        (df["condition"] != "PANIC") &
        (df["prev_rise"]  >= PREV_RISE_MIN) &
        (df["ma5_dev"]    <= MA5_DEV_MAX) &
        (df["ma25_dev"]   <= MA25_DEV_MAX) &
        (df["Volume"]     >= MIN_VOLUME) &
        (df["prev_close"] >= MIN_PRICE)
    )
    if date_start:
        mask &= df["Date_str"] >= date_start
    if date_end:
        mask &= df["Date_str"] <= date_end

    sub = df[mask].copy()

    # モメンタム確認: 当日 High が前日終値+0.3%以上に到達
    sub["trigger"] = sub["prev_close"] * (1 + MOMENTUM_PCT / 100)
    sub = sub[sub["High"] >= sub["trigger"]].copy()

    # 発注価格 = max(寄り付き, トリガー価格)
    sub["entry"] = sub[["Open", "trigger"]].max(axis=1)

    print(f"  エントリー候補: {len(sub):,}件 ({date_start or '全期間'}〜{date_end or ''})")
    return sub[["Date_str", "code", "entry", "High", "Low", "Close", "condition"]].reset_index(drop=True)


def simulate(pre, tp_pct, sl_pct):
    """(TP%, SL%) のリターン配列を返す。"""
    ep = pre["entry"].values
    hi = pre["High"].values
    lo = pre["Low"].values
    cl = pre["Close"].values
    tp = ep * (1 + tp_pct / 100)
    sl = ep * (1 - sl_pct / 100)

    # SL優先（lo <= sl なら損切り、次に hi >= tp なら利確、それ以外は引け決済）
    rets = np.where(lo <= sl,  (sl - ep) / ep * 100,
           np.where(hi >= tp,  (tp - ep) / ep * 100,
                               (cl - ep) / ep * 100))
    return rets


def calc_sharpe(rets, min_n=30):
    if len(rets) < min_n:
        return np.nan, len(rets), np.nan, np.nan
    r   = np.array(rets)
    wr  = (r > 0).mean() * 100
    avg = r.mean()
    std = r.std(ddof=1)
    sh  = avg / std * np.sqrt(252) if std > 0 else 0.0
    return sh, len(r), wr, avg


def run_grid(pre_is, pre_oos):
    results = []
    total = len(TP_VALUES) * len(SL_VALUES)
    done  = 0

    for tp, sl in product(TP_VALUES, SL_VALUES):
        rets_is  = simulate(pre_is,  tp, sl)
        rets_oos = simulate(pre_oos, tp, sl)
        sh_is,  n_is,  wr_is,  avg_is  = calc_sharpe(rets_is)
        sh_oos, n_oos, wr_oos, avg_oos = calc_sharpe(rets_oos)
        results.append({
            "TP": tp, "SL": sl,
            "IS_Sharpe": sh_is,  "IS_N": n_is,  "IS_WR": wr_is,  "IS_avg": avg_is,
            "OOS_Sharpe": sh_oos, "OOS_N": n_oos, "OOS_WR": wr_oos, "OOS_avg": avg_oos,
        })
        done += 1
        if done % 20 == 0 or done == total:
            print(f"  {done}/{total} 完了...", end="\r")

    print()
    return pd.DataFrame(results)


def print_heatmap(results, metric="OOS_Sharpe"):
    pivot = results.pivot(index="SL", columns="TP", values=metric)
    pivot = pivot.sort_index(ascending=False)

    print(f"\n  ── {metric} ヒートマップ ──")
    print(f"  SL\\TP  " + "  ".join(f"{tp:>5.1f}%" for tp in TP_VALUES))
    print(f"  {'─'*90}")
    for sl_val, row in pivot.iterrows():
        vals = []
        for tp_val in TP_VALUES:
            v = row.get(tp_val, np.nan)
            if np.isnan(v):
                vals.append("   nan")
            elif v == pivot.values[~np.isnan(pivot.values)].max():
                vals.append(f"\033[1;32m{v:+5.2f}\033[0m")  # 最大値を緑太字
            else:
                vals.append(f"{v:+5.2f}")
        print(f"  SL{sl_val:.0f}%  " + "  ".join(vals))


def main():
    print("=" * 64)
    print("  グリッドサーチ: 戦略A TP/SL 最適化")
    print(f"  TP: {TP_VALUES[0]}%〜{TP_VALUES[-1]}%（0.5%刻み）× "
          f"SL: {SL_VALUES[0]}%〜{SL_VALUES[-1]}%（1%刻み）")
    print(f"  IS: 2024-06-01〜{IS_END}  /  OOS: {OOS_START}〜")
    print("=" * 64)

    df = load_data()

    print("\n  IS データ絞り込み...")
    pre_is  = prefilter(df, "2024-06-01", IS_END)
    print("  OOS データ絞り込み...")
    pre_oos = prefilter(df, OOS_START)

    print(f"\n  グリッドサーチ実行中（{len(TP_VALUES)}×{len(SL_VALUES)}={len(TP_VALUES)*len(SL_VALUES)}通り）...")
    results = run_grid(pre_is, pre_oos)

    # OOS Sharpe ヒートマップ
    print_heatmap(results, "OOS_Sharpe")

    # IS Sharpe ヒートマップ
    print_heatmap(results, "IS_Sharpe")

    # OOS上位10
    top = results.dropna(subset=["OOS_Sharpe"]).sort_values("OOS_Sharpe", ascending=False).head(10)
    print(f"\n  ── OOS Sharpe 上位10組み合わせ ──")
    print(f"  {'TP':>5}  {'SL':>5}  {'IS_Sh':>7}  {'OOS_Sh':>7}  {'OOS_N':>6}  {'OOS_WR':>7}  {'OOS_avg':>8}")
    print(f"  {'─'*58}")
    for _, r in top.iterrows():
        marker = " ← 現行(TP3/SL5)" if r["TP"] == 3.0 and r["SL"] == 5.0 else ""
        print(f"  {r['TP']:>4.1f}%  {r['SL']:>4.1f}%  {r['IS_Sharpe']:>+7.3f}  "
              f"{r['OOS_Sharpe']:>+7.3f}  {int(r['OOS_N']):>6}  "
              f"{r['OOS_WR']:>6.1f}%  {r['OOS_avg']:>+7.3f}%{marker}")

    # 現行設定（TP3/SL5）の成績
    cur = results[(results["TP"] == 3.0) & (results["SL"] == 5.0)]
    if not cur.empty:
        r = cur.iloc[0]
        print(f"\n  ── 現行設定 TP+3% / SL-5% ──")
        print(f"  IS:  N={int(r['IS_N'])}  WR={r['IS_WR']:.1f}%  avg={r['IS_avg']:+.3f}%  Sharpe={r['IS_Sharpe']:+.3f}")
        print(f"  OOS: N={int(r['OOS_N'])}  WR={r['OOS_WR']:.1f}%  avg={r['OOS_avg']:+.3f}%  Sharpe={r['OOS_Sharpe']:+.3f}")

    # 推奨
    best = results.dropna(subset=["OOS_Sharpe"]).loc[results["OOS_Sharpe"].idxmax()]
    print(f"\n{'='*64}")
    print(f"  【推奨】TP={best['TP']:.1f}%  SL={best['SL']:.1f}%")
    print(f"   IS Sharpe={best['IS_Sharpe']:+.3f}  OOS Sharpe={best['OOS_Sharpe']:+.3f}")
    print(f"   OOS: N={int(best['OOS_N'])}  WR={best['OOS_WR']:.1f}%  avg={best['OOS_avg']:+.3f}%")
    print(f"{'='*64}")

    # CSV保存
    out_path = Path(__file__).parent / "out" / "grid_search_a.csv"
    results.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n  結果保存: {out_path}")


if __name__ == "__main__":
    main()
