# -*- coding: utf-8 -*-
"""
simulate_ratio.py - NORMAL日における ratio 閾値のパフォーマンス分析

sim_precise_trades.csv（2年分・約57万件）を使って
戦略A judge_entry_a の ratio 条件を最適化する。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from pathlib import Path

CSV = Path(__file__).parent.parent / "out" / "sim_precise_trades.csv"

print("データ読み込み中...")
df = pd.read_csv(CSV, low_memory=False)
print(f"総件数: {len(df):,}件  期間: {df['trade_date'].min()} 〜 {df['trade_date'].max()}")

# 勝利フラグ（始値→終値で+1%以上）
df["win"] = df["ret_oc"] > 0

# NORMAL日 × 戦略Aの対象フィルタ
# judge_entry_a の基本フィルタを適用:
#   - score >= 5.0
#   - ratio < 10.0（過熱除外: 実際のコードでPASS）
#   - today_rise: -3〜+2%（有効範囲）
#   - gap_pct: -5〜+2%（daytime.pyのフィルタ）
df_normal = df[
    (df["trade_cond"] == "NORMAL") &
    (df["score"] >= 5.0) &
    (df["today_rise"] >= -3.0) &
    (df["today_rise"] <= 2.0) &
    (df["gap_pct"] >= -5.0) &
    (df["gap_pct"] <= 2.0)
].copy()

print(f"NORMAL日フィルタ後: {len(df_normal):,}件\n")

# ── 1. ratio帯別パフォーマンス ────────────────────────
ratio_bins = [0, 1, 2, 3, 4, 5, 7, 10, 15, 999]
labels     = ["0〜1", "1〜2", "2〜3", "3〜4", "4〜5", "5〜7", "7〜10", "10〜15", "15以上"]

df_normal["ratio_band"] = pd.cut(df_normal["ratio"], bins=ratio_bins, labels=labels, right=False)

print(f"【1】ratio帯別パフォーマンス（NORMAL日、score>=5、前日-3〜+2%）")
print(f"  {'ratio帯':>8}  {'件数':>7}  {'勝率':>6}  {'平均%':>7}  {'中央値%':>8}  {'Sharpe':>8}")
print("  " + "─" * 60)

best_band = None
best_sharpe = -999

for band in labels:
    grp = df_normal[df_normal["ratio_band"] == band]
    if len(grp) < 30:
        print(f"  {band:>8}  {len(grp):>7,}  {'(少)':>6}  {'---':>7}  {'---':>8}  {'---':>8}")
        continue
    wr     = grp["win"].mean() * 100
    avg    = grp["ret_oc"].mean()
    med    = grp["ret_oc"].median()
    std    = grp["ret_oc"].std()
    sharpe = avg / std * np.sqrt(252) if std > 0 else 0
    mark   = " ★" if wr >= 52 and avg > 0.1 else ""
    if wr >= 52 and avg > 0.1 and sharpe > best_sharpe:
        best_sharpe = sharpe
        best_band   = band
    print(f"  {band:>8}  {len(grp):>7,}  {wr:>5.1f}%  {avg:>+7.3f}%  {med:>+8.3f}%  {sharpe:>8.2f}{mark}")

# ── 2. ratio閾値の累積効果 ────────────────────────────
print(f"\n【2】ratio閾値別累積パフォーマンス（ratio >= X かつ ratio < 10）")
print(f"  {'ratio>=':>8}  {'件数':>7}  {'勝率':>6}  {'平均%':>7}  {'Sharpe':>8}")
print("  " + "─" * 50)

best_thr = None
best_thr_sharpe = -999

for thr in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
    grp = df_normal[(df_normal["ratio"] >= thr) & (df_normal["ratio"] < 10)]
    if len(grp) < 30:
        break
    wr     = grp["win"].mean() * 100
    avg    = grp["ret_oc"].mean()
    std    = grp["ret_oc"].std()
    sharpe = avg / std * np.sqrt(252) if std > 0 else 0
    mark   = " ◀ 現在" if thr == 3 else ""
    if wr >= 52 and avg > 0.1 and sharpe > best_thr_sharpe:
        best_thr_sharpe = sharpe
        best_thr        = thr
    star   = " ★" if wr >= 52 and avg > 0.1 else ""
    print(f"  {thr:>8}  {len(grp):>7,}  {wr:>5.1f}%  {avg:>+7.3f}%  {sharpe:>8.2f}{mark}{star}")

# ── 3. score × ratio の組み合わせ ─────────────────────
print(f"\n【3】score × ratio 組み合わせ（NORMAL日、件数100件以上）")
print(f"  {'score帯':>10}  {'ratio帯':>8}  {'件数':>7}  {'勝率':>6}  {'平均%':>7}  {'Sharpe':>8}")
print("  " + "─" * 65)

score_bins   = [(5, 7), (7, 9), (9, 999)]
score_labels = ["5〜7", "7〜9", "9以上"]

for (s_lo, s_hi), sl in zip(score_bins, score_labels):
    for band in labels:
        grp = df_normal[
            (df_normal["score"] >= s_lo) &
            (df_normal["score"] <  s_hi) &
            (df_normal["ratio_band"] == band)
        ]
        if len(grp) < 100:
            continue
        wr     = grp["win"].mean() * 100
        avg    = grp["ret_oc"].mean()
        std    = grp["ret_oc"].std()
        sharpe = avg / std * np.sqrt(252) if std > 0 else 0
        mark   = " ★" if wr >= 52 and avg > 0.1 else ""
        print(f"  {sl:>10}  {band:>8}  {len(grp):>7,}  {wr:>5.1f}%  {avg:>+7.3f}%  {sharpe:>8.2f}{mark}")

# ── 4. 月次推移（ratio 3〜10） ─────────────────────────
print(f"\n【4】月次推移（ratio 3〜10 × NORMAL日）")
print(f"  {'年月':>8}  {'件数':>6}  {'勝率':>6}  {'平均%':>7}")
print("  " + "─" * 38)

df_range = df_normal[(df_normal["ratio"] >= 3) & (df_normal["ratio"] < 10)].copy()
df_range["ym"] = df_range["trade_date"].str[:7]
for ym, grp in df_range.groupby("ym"):
    if len(grp) < 10:
        continue
    wr  = grp["win"].mean() * 100
    avg = grp["ret_oc"].mean()
    print(f"  {ym:>8}  {len(grp):>6}  {wr:>5.1f}%  {avg:>+7.3f}%")

# ── まとめ ────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"【まとめ】")
print(f"  現在のコード: 3.0 <= ratio < 10.0 でBUYゾーン判定")
if best_thr is not None:
    print(f"  ★ 最良閾値(Sharpe最大): ratio >= {best_thr}")
else:
    print(f"  ★ 有意な最良閾値なし（NORMAL日BUYゾーン廃止を検討）")
