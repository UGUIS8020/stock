#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ga_monday_normal.py - 月曜×NORMAL日の戦略DパラメータをGA最適化

A案: 月曜×NORMAL → 全見送り
B案: 月曜×NORMAL → 独自gap範囲（GAで最適化）

IS: 2024-06-01~2025-09-30
OOS: 2025-10-01~
"""

import sys, random
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db

TP_PCT        = 0.03
SL_PCT        = 0.05
MOMENTUM_PCT  = 0.3
PREV_RISE_MIN = 0.0
MA5_DEV_MAX   = 3.0
MA25_DEV_MAX  = 6.0
MIN_VOLUME    = 50_000
MIN_PRICE     = 200

PANIC_AD  = 0.20
WEAK_AD   = 0.35
STRONG_AD = 0.60

IS_END    = "2025-09-30"
OOS_START = "2025-10-01"

GA_POP   = 300
GA_GENS  = 100
GA_ELITE = 0.20
GA_MUT   = 0.30

random.seed(42)
np.random.seed(42)


def load_and_prepare():
    conn = sqlite3.connect(db.DB_PATH)
    print("  daily_prices 読み込み中...")
    df = pd.read_sql(
        "SELECT code, Date, Open, High, Low, Close, Volume "
        "FROM daily_prices WHERE Date >= '2024-06-01' ORDER BY code, Date",
        conn
    )
    conn.close()

    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    print(f"  {len(df):,}行 / {df['Date'].nunique()}日 / {df['code'].nunique()}銘柄")

    df = df.sort_values(["code", "Date"]).reset_index(drop=True)
    df["prev_close"]  = df.groupby("code")["Close"].shift(1)
    df["prev_close2"] = df.groupby("code")["Close"].shift(2)
    df["prev_rise"]   = (df["prev_close"] - df["prev_close2"]) / df["prev_close2"] * 100
    df["ma5"]  = df.groupby("code")["Close"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["ma25"] = df.groupby("code")["Close"].transform(lambda x: x.shift(1).rolling(25).mean())
    df["ma5_dev"]  = (df["prev_close"] / df["ma5"]  - 1) * 100
    df["ma25_dev"] = (df["prev_close"] / df["ma25"] - 1) * 100
    df["gap_pct"] = (df["Open"] - df["prev_close"]) / df["prev_close"] * 100
    df["weekday"]   = df["Date"].dt.weekday
    df["is_monday"] = df["weekday"] == 0
    df = df.dropna(subset=["prev_close", "prev_rise", "gap_pct", "ma5_dev", "ma25_dev"])
    df = df[df["prev_close"] > 0]

    # AD比率・地合い
    df["up"] = df["Close"] > df["prev_close"]
    daily_ad = (df.groupby("Date")
                  .agg(up_count=("up","sum"), total=("up","count"))
                  .reset_index())
    daily_ad["ad_ratio"]  = daily_ad["up_count"] / daily_ad["total"]
    daily_ad["condition"] = daily_ad["ad_ratio"].apply(lambda r:
        "PANIC"  if r <= PANIC_AD  else
        "STRONG" if r >= STRONG_AD else
        "WEAK"   if r <= WEAK_AD   else "NORMAL")
    daily_ad["is_monday"] = daily_ad["Date"].dt.weekday == 0

    df = df.merge(daily_ad[["Date", "condition"]], on="Date", how="left")
    df["Date_str"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df, daily_ad


def prefilter(df, condition, is_monday, date_start=None, date_end=None):
    """gap_pct以外の固定フィルタを適用済みのサブセットを返す（高速化）"""
    mask = (
        (df["condition"]  == condition) &
        (df["is_monday"]  == is_monday) &
        (df["prev_rise"]  >= PREV_RISE_MIN) &
        (df["ma5_dev"]    <= MA5_DEV_MAX) &
        (df["ma25_dev"]   <= MA25_DEV_MAX) &
        (df["Volume"]     >= MIN_VOLUME) &
        (df["Open"]       >= MIN_PRICE)
    )
    if date_start:
        mask &= df["Date_str"] >= date_start
    if date_end:
        mask &= df["Date_str"] <= date_end

    sub = df[mask].copy()
    # モメンタム到達確認
    sub["trigger"] = sub["prev_close"] * (1 + MOMENTUM_PCT / 100)
    sub = sub[sub["High"] >= sub["trigger"]].copy()
    sub["entry"] = sub[["Open", "trigger"]].max(axis=1)
    sub["tp"]    = sub["entry"] * (1 + TP_PCT)
    sub["sl"]    = sub["entry"] * (1 - SL_PCT)

    # リターン計算
    lo, hi, cl = sub["Low"].values, sub["High"].values, sub["Close"].values
    ep, tp, sl = sub["entry"].values, sub["tp"].values, sub["sl"].values
    rets = np.where(lo <= sl,  (sl - ep) / ep * 100,
           np.where(hi >= tp,  (tp - ep) / ep * 100,
                               (cl - ep) / ep * 100))
    sub["ret"] = rets
    return sub[["Date_str", "gap_pct", "ret"]].reset_index(drop=True)


def simulate_from_pre(pre, gap_min, gap_max):
    sub = pre[(pre["gap_pct"] >= gap_min) & (pre["gap_pct"] <= gap_max)]
    return sub["ret"].values


def sharpe(rets, min_n=10):
    if len(rets) < min_n:
        return -99.0, 0, 0.0, 0.0
    r   = np.array(rets)
    wr  = (r > 0).mean() * 100
    avg = r.mean()
    std = r.std(ddof=1)
    sh  = avg / std * np.sqrt(252) if std > 0 else 0.0
    return sh, len(r), wr, avg


def ga_run(pre_is, label=""):
    """prefiltered IS データに対してgap_min/gap_maxをGA最適化"""

    def fit(ind):
        rets = simulate_from_pre(pre_is, ind[0], ind[1])
        return sharpe(rets, min_n=5)[0]

    def rand_ind():
        lo = round(random.uniform(-8.0, 0.5), 1)
        hi = round(random.uniform(max(-5.0, lo + 0.5), 4.0), 1)
        return [lo, hi]

    def mutate(ind):
        i = random.randint(0, 1)
        new = ind.copy()
        new[i] = round(new[i] + random.uniform(-1.5, 1.5), 1)
        new[0] = max(-8.0, min(0.5, new[0]))
        new[1] = max(-5.0, min(4.0, new[1]))
        if new[1] <= new[0]:
            new[1] = round(new[0] + 0.5, 1)
        return new

    def cross(a, b):
        return [a[0], b[1]] if random.random() < 0.5 else [b[0], a[1]]

    pop = [rand_ind() for _ in range(GA_POP)]
    best_ind, best_sh = None, -99.0
    elite_n = max(1, int(GA_POP * GA_ELITE))

    for gen in range(GA_GENS):
        scored = sorted([(fit(ind), ind) for ind in pop], key=lambda x: -x[0])
        if scored[0][0] > best_sh:
            best_sh  = scored[0][0]
            best_ind = scored[0][1]

        if (gen + 1) % 25 == 0 or gen == GA_GENS - 1:
            valid = sum(1 for s, _ in scored if s > -90)
            print(f"    Gen{gen+1:3d}: Sharpe={best_sh:+.3f}  gap=[{best_ind[0]},{best_ind[1]}]  valid={valid}/{GA_POP}")

        elite    = [ind for _, ind in scored[:elite_n]]
        children = []
        while len(children) < GA_POP - elite_n:
            a = random.choice(elite)
            b = random.choice(scored[:GA_POP // 2])[1]
            c = cross(a, b)
            if random.random() < GA_MUT:
                c = mutate(c)
            children.append(c)
        pop = elite + children

    return best_ind, best_sh


def print_stats(label, rets, dates=None):
    sh, n, wr, avg = sharpe(rets)
    print(f"  {label}: N={n:5}  WR={wr:.1f}%  avg={avg:+.3f}%  Sharpe={sh:+.3f}")
    return sh, n


def main():
    print("=" * 64)
    print("  GA最適化: 月曜×NORMAL日の戦略Dギャップパラメータ")
    print(f"  IS: 2024-06-01~{IS_END}  /  OOS: {OOS_START}~")
    print(f"  GA: {GA_POP}個体 × {GA_GENS}世代")
    print("=" * 64)

    df, daily_ad = load_and_prepare()

    dist = daily_ad.groupby(["condition", "is_monday"]).size().unstack(fill_value=0)
    print(f"\n  地合い分布（日数）:\n{dist.to_string()}\n")

    # ── ベースライン ──────────────────────────────────────
    print("【ベースライン】")
    # 非月曜 NORMAL gap=-5~0%
    pre = prefilter(df, "NORMAL", False, "2024-06-01", IS_END)
    print_stats("非月曜×NORMAL IS  gap=-5~0%", simulate_from_pre(pre, -5.0, 0.0))
    pre_oos = prefilter(df, "NORMAL", False, OOS_START)
    print_stats("非月曜×NORMAL OOS gap=-5~0%", simulate_from_pre(pre_oos, -5.0, 0.0))

    # 月曜 STRONG gap=0~+2%
    pre2 = prefilter(df, "STRONG", True, "2024-06-01", IS_END)
    print_stats("月曜×STRONG   IS  gap=0~2%", simulate_from_pre(pre2, 0.0, 2.0))
    pre2o = prefilter(df, "STRONG", True, OOS_START)
    print_stats("月曜×STRONG   OOS gap=0~2%", simulate_from_pre(pre2o, 0.0, 2.0))

    # ── A案: 全見送り ──────────────────────────────────────
    print("\n【A案】月曜×NORMAL: 全見送り  Sharpe=0.000")

    # ── B案候補1: NORMALのgap条件をそのまま適用 ─────────────
    print("\n【B案候補1: NORMALのgap条件 -5~0% をそのまま適用】")
    pre3   = prefilter(df, "NORMAL", True, "2024-06-01", IS_END)
    pre3_o = prefilter(df, "NORMAL", True, OOS_START)
    sh_b1_is, _  = print_stats("月曜×NORMAL IS  gap=-5~0%", simulate_from_pre(pre3, -5.0, 0.0))
    sh_b1_oos, _ = print_stats("月曜×NORMAL OOS gap=-5~0%", simulate_from_pre(pre3_o, -5.0, 0.0))

    # gap分布確認
    all_pre = prefilter(df, "NORMAL", True)
    print(f"\n  月曜×NORMAL モメンタム到達件数: {len(all_pre)}件 (IS+OOS合計)")
    gd = pd.cut(all_pre["gap_pct"], bins=[-8,-4,-3,-2,-1,0,1,2,4], right=True)
    print(f"  gap分布:\n{all_pre.groupby(gd)['ret'].agg(['count','mean']).to_string()}")

    # ── GA最適化 ──────────────────────────────────────────
    print("\n【GA最適化: 月曜×NORMAL の gap範囲を探索】")
    best_gap, best_sh_is = ga_run(pre3)

    print(f"\n  GA最適解: gap=[{best_gap[0]}, {best_gap[1]}]  IS Sharpe={best_sh_is:+.3f}")
    sh_ga_is,  _ = print_stats(f"月曜×NORMAL IS  gap={best_gap}", simulate_from_pre(pre3,   best_gap[0], best_gap[1]))
    sh_ga_oos, n_ga_oos = print_stats(f"月曜×NORMAL OOS gap={best_gap}", simulate_from_pre(pre3_o, best_gap[0], best_gap[1]))

    # OOS月次成績
    oos_sub = pre3_o[(pre3_o["gap_pct"] >= best_gap[0]) & (pre3_o["gap_pct"] <= best_gap[1])].copy()
    if len(oos_sub) > 0:
        print("\n  【OOS月次成績 (GA最適)】")
        oos_sub["month"] = oos_sub["Date_str"].str[:7]
        m = oos_sub.groupby("month")["ret"].agg(N="count", avg="mean",
            WR=lambda x: (x > 0).mean() * 100).reset_index()
        for _, row in m.iterrows():
            print(f"    {row['month']}  N={int(row['N']):3}  WR={row['WR']:.0f}%  avg={row['avg']:+.3f}%")

    # ── 結論 ──────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  【結論】")
    print(f"  A案 (全見送り)       : Sharpe= 0.000")
    print(f"  B案1 (gap=-5~0%)    : IS={sh_b1_is:+.3f} / OOS={sh_b1_oos:+.3f}")
    print(f"  B案2 (GA最適)       : IS={sh_ga_is:+.3f} / OOS={sh_ga_oos:+.3f}  gap={best_gap}  N_oos={n_ga_oos}")

    oos_threshold = 0.3
    if sh_ga_oos > oos_threshold and n_ga_oos >= 10:
        print(f"\n  → 推奨: B案2 (GA最適 gap={best_gap}) を採用")
        print(f"         OOS Sharpe={sh_ga_oos:+.3f} > {oos_threshold}  N={n_ga_oos}件")
    elif sh_b1_oos > oos_threshold:
        print(f"\n  → 推奨: B案1 (gap=-5~0%) を採用")
        print(f"         OOS Sharpe={sh_b1_oos:+.3f} > {oos_threshold}")
    else:
        print(f"\n  → 推奨: A案（全見送り）を採用")
        print(f"         月曜×NORMALはOOS Sharpe={sh_ga_oos:+.3f} ≤ {oos_threshold} または N<10")
    print("=" * 64)


if __name__ == "__main__":
    main()
