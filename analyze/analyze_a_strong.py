# -*- coding: utf-8 -*-
"""
analyze_a_strong.py - 戦略A STRONG日 特化分析

STRONG日の判定条件（score/ratio/today_rise の閾値）を2年データで検証。
sim_precise_trades.csv（2年分・約57万件）を使用。

実行方法:
    cd stock/
    python analyze/analyze_a_strong.py

分析内容:
    1. STRONG日 基本統計（全体勝率・月次推移）
    2. 特徴量別パフォーマンス（score / ratio / gap / 前日騰落）
    3. 現在のPASS条件の妥当性検証
       （score<5 / ratio>=5 / today_rise>=2 が本当に除外すべきか）
    4. 多変量条件探索（最適な条件の組み合わせを発見）
    5. OOS検証（前半60%で発見 → 後半40%で汎化確認）
    6. 有望条件まとめ → scan_morning_strategy_a_strong.py への反映候補
"""

import sys
import os
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TRADES_CSV = "out/sim_precise_trades.csv"
REPORT_TXT = "out/analyze_a_strong_report.txt"

MIN_N       = 30
MIN_WR      = 55.0   # STRONG日は基準を高めに（全体勝率65%のため）
P_THRESHOLD = 0.05
OOS_SPLIT   = 0.6

# ── ユーティリティ ────────────────────────────────────

def _erf(x):
    sign = np.sign(x); x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (0.254829592*t - 0.284496736*t**2 + 1.421413741*t**3
               - 1.453152027*t**4 + 1.061405429*t**5) * np.exp(-x*x)
    return sign * y

def _norm_cdf(z):
    return 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))

def binomial_pvalue(wins, n, p0=0.5):
    if n < 5: return 1.0
    z = (wins - n * p0) / np.sqrt(n * p0 * (1 - p0))
    return float(1.0 - _norm_cdf(z))

def stats_row(series_win, label=""):
    n    = len(series_win)
    wins = series_win.sum()
    wr   = wins / n * 100 if n > 0 else 0.0
    pval = binomial_pvalue(wins, n)
    sig  = "★" if (pval < P_THRESHOLD and wr >= MIN_WR and n >= MIN_N) else ""
    return {"label": label, "n": n, "wins": int(wins), "wr": round(wr,1),
            "pval": round(pval,4), "sig": sig}

def sharpe(series_ret):
    avg = series_ret.mean()
    std = series_ret.std()
    return round(avg / std * np.sqrt(252), 2) if std > 0 else 0.0

# ── データ読み込み ─────────────────────────────────────

def load_strong_data():
    print("データ読み込み中...")
    df = pd.read_csv(TRADES_CSV, low_memory=False)
    df["win"] = (df["ret_oc"] > 0).astype(int)
    df["trade_date"] = df["trade_date"].astype(str)
    df["ym"] = df["trade_date"].str[:7]

    df_s = df[df["trade_cond"] == "STRONG"].copy()
    print(f"STRONG日: {len(df_s):,}件  期間: {df_s['trade_date'].min()} 〜 {df_s['trade_date'].max()}")
    print(f"全体勝率: {df_s['win'].mean()*100:.1f}%  平均: {df_s['ret_oc'].mean():+.3f}%\n")
    return df_s

# ── 1. 月次推移 ────────────────────────────────────────

def analyze_monthly(df):
    lines = ["\n【1】月次推移（STRONG日 全体）"]
    lines.append(f"  {'年月':>8}  {'件数':>7}  {'勝率':>6}  {'平均%':>8}  {'Sharpe':>8}")
    lines.append("  " + "─" * 48)
    for ym, g in df.groupby("ym"):
        if len(g) < 10: continue
        wr  = g["win"].mean() * 100
        avg = g["ret_oc"].mean()
        sp  = sharpe(g["ret_oc"])
        mark = " ★" if wr >= 60 else ""
        lines.append(f"  {ym:>8}  {len(g):>7,}  {wr:>5.1f}%  {avg:>+8.3f}%  {sp:>8.2f}{mark}")
    return lines

# ── 2. 現在のPASS条件の妥当性検証 ─────────────────────

def verify_pass_conditions(df):
    lines = ["\n【2】現在のPASS条件の妥当性検証"]
    lines.append("  現在のコード: score<5 / ratio>=5 / today_rise>=2 でPASS")
    lines.append("  これらを除外することは正しいか？除外しなかった場合のパフォーマンスを確認\n")

    # score < 5.0 の場合
    g = df[df["score"] < 5.0]
    wr  = g["win"].mean() * 100
    avg = g["ret_oc"].mean()
    mark = "→ PASS正しい ✅" if wr < 55 else "→ PASS不要かも ⚠️"
    lines.append(f"  score < 5.0:       {len(g):>7,}件  勝率{wr:>5.1f}%  平均{avg:>+8.3f}%  {mark}")

    # ratio >= 5.0 の場合
    g = df[df["ratio"] >= 5.0]
    wr  = g["win"].mean() * 100
    avg = g["ret_oc"].mean()
    mark = "→ PASS正しい ✅" if wr < 55 else "→ PASS不要かも ⚠️"
    lines.append(f"  ratio >= 5.0:      {len(g):>7,}件  勝率{wr:>5.1f}%  平均{avg:>+8.3f}%  {mark}")

    # today_rise >= 2.0 の場合
    g = df[df["today_rise"] >= 2.0]
    wr  = g["win"].mean() * 100
    avg = g["ret_oc"].mean()
    mark = "→ PASS正しい ✅" if wr < 55 else "→ PASS不要かも ⚠️"
    lines.append(f"  today_rise >= 2.0: {len(g):>7,}件  勝率{wr:>5.1f}%  平均{avg:>+8.3f}%  {mark}")

    return lines

# ── 3. 特徴量別パフォーマンス ──────────────────────────

def analyze_features(df):
    lines = ["\n【3】特徴量別パフォーマンス（STRONG日）"]

    def band_analysis(col, bins, labels, title):
        sub = [f"\n  ── {title} ──"]
        sub.append(f"  {'帯':>10}  {'件数':>7}  {'勝率':>6}  {'平均%':>8}  {'Sharpe':>8}")
        sub.append("  " + "─" * 52)
        df2 = df.copy()
        df2["_band"] = pd.cut(df2[col], bins=bins, labels=labels, right=False)
        for label in labels:
            g = df2[df2["_band"] == label]
            if len(g) < MIN_N:
                sub.append(f"  {label:>10}  {len(g):>7,}  {'(少)':>6}  {'---':>8}  {'---':>8}")
                continue
            wr  = g["win"].mean() * 100
            avg = g["ret_oc"].mean()
            sp  = sharpe(g["ret_oc"])
            mark = " ★" if wr >= MIN_WR and avg > 0.3 else ""
            sub.append(f"  {label:>10}  {len(g):>7,}  {wr:>5.1f}%  {avg:>+8.3f}%  {sp:>8.2f}{mark}")
        return sub

    lines += band_analysis("score",
        [0, 3, 5, 7, 9, 999],
        ["0〜3", "3〜5", "5〜7", "7〜9", "9以上"],
        "score帯")

    lines += band_analysis("ratio",
        [0, 2, 4, 6, 8, 10, 999],
        ["0〜2", "2〜4", "4〜6", "6〜8", "8〜10", "10以上"],
        "ratio帯（出来高比率）")

    lines += band_analysis("gap_pct",
        [-10, -3, -1, 0, 1, 3, 10],
        ["-10〜-3", "-3〜-1", "-1〜0", "0〜+1", "+1〜+3", "+3〜+10"],
        "gap_pct帯（寄り付きギャップ）")

    lines += band_analysis("today_rise",
        [-20, -5, -3, -1, 0, 1, 2, 3, 20],
        ["-20〜-5", "-5〜-3", "-3〜-1", "-1〜0", "0〜+1", "+1〜+2", "+2〜+3", "+3〜+20"],
        "today_rise帯（前日騰落）")

    return lines

# ── 4. 多変量条件探索 ──────────────────────────────────

def analyze_multivariate(df):
    lines  = ["\n【4】多変量条件探索（STRONG日、有意条件のみ表示）"]
    lines.append(f"  基準: 件数>={MIN_N} & 勝率>={MIN_WR}% & p<{P_THRESHOLD}")
    lines.append(f"  {'条件':52}  {'件数':>6}  {'勝率':>6}  {'平均%':>7}  {'Sharpe':>8}")
    lines.append("  " + "─" * 85)

    score_ranges = [(3, 5), (5, 7), (7, 9), (9, 999)]
    ratio_ranges = [(0, 3), (0, 5), (2, 5), (0, 8)]
    rise_ranges  = [(-5, 0), (-3, 1), (-2, 2), (-1, 2), (0, 2)]
    gap_ranges   = [(-5, 0), (-3, 1), (-2, 2), (0, 3), (-5, 3)]

    found = []
    for (s_lo, s_hi), (r_lo, r_hi), (ri_lo, ri_hi), (g_lo, g_hi) in product(
            score_ranges, ratio_ranges, rise_ranges, gap_ranges):

        mask = (
            (df["score"]      >= s_lo) & (df["score"]      <  s_hi) &
            (df["ratio"]      >= r_lo) & (df["ratio"]      <  r_hi) &
            (df["today_rise"] >= ri_lo)& (df["today_rise"] <  ri_hi) &
            (df["gap_pct"]    >= g_lo) & (df["gap_pct"]    <  g_hi)
        )
        g = df[mask]
        if len(g) < MIN_N:
            continue
        r = stats_row(g["win"])
        if r["sig"] == "★":
            sp    = sharpe(g["ret_oc"])
            avg   = g["ret_oc"].mean()
            label = (f"score{s_lo}〜{s_hi} ratio{r_lo}〜{r_hi} "
                     f"前日{ri_lo:+}〜{ri_hi:+}% gap{g_lo:+}〜{g_hi:+}%")
            lines.append(f"  ★ {label:<50}  {r['n']:>6,}  {r['wr']:>5.1f}%"
                         f"  {avg:>+7.3f}%  {sp:>8.2f}")
            found.append({"label": label, "n": r["n"], "wr": r["wr"],
                          "avg": avg, "sharpe": sp,
                          "s_lo": s_lo, "s_hi": s_hi,
                          "r_lo": r_lo, "r_hi": r_hi,
                          "ri_lo": ri_lo, "ri_hi": ri_hi,
                          "g_lo": g_lo, "g_hi": g_hi})

    if not found:
        lines.append("  （有意な条件は見つかりませんでした）")
    else:
        lines.append(f"\n  合計 {len(found)} 件の有意条件を発見")

    return lines, found

# ── 5. OOS検証 ────────────────────────────────────────

def analyze_oos(df, found):
    if not found:
        return ["\n【5】OOS検証: 有意条件なし → スキップ"], []

    lines = ["\n【5】OOS検証（訓練60% → 検証40%）"]
    lines.append("  過学習チェック: 訓練期間で発見した条件が未知期間でも有効か確認")

    dates  = sorted(df["trade_date"].unique())
    split  = dates[int(len(dates) * OOS_SPLIT)]
    df_trn = df[df["trade_date"] <  split]
    df_val = df[df["trade_date"] >= split]
    lines.append(f"  訓練: 〜{split}（{len(df_trn):,}件）  検証: {split}〜（{len(df_val):,}件）\n")
    lines.append(f"  {'条件':42}  {'訓練WR':>7}  {'検証WR':>7}  {'訓練avg':>8}  {'検証avg':>8}  判定")
    lines.append("  " + "─" * 85)

    passed = []
    for f in sorted(found, key=lambda x: -x["sharpe"])[:20]:
        def mask_df(d):
            return (
                (d["score"]      >= f["s_lo"])  & (d["score"]      <  f["s_hi"]) &
                (d["ratio"]      >= f["r_lo"])  & (d["ratio"]      <  f["r_hi"]) &
                (d["today_rise"] >= f["ri_lo"]) & (d["today_rise"] <  f["ri_hi"]) &
                (d["gap_pct"]    >= f["g_lo"])  & (d["gap_pct"]    <  f["g_hi"])
            )
        g_trn = df_trn[mask_df(df_trn)]
        g_val = df_val[mask_df(df_val)]
        if len(g_trn) < MIN_N or len(g_val) < MIN_N:
            continue
        wr_t  = g_trn["win"].mean() * 100
        wr_v  = g_val["win"].mean() * 100
        avg_t = g_trn["ret_oc"].mean()
        avg_v = g_val["ret_oc"].mean()
        ok    = "✅ OOS通過" if wr_v >= MIN_WR and avg_v > 0 else "❌ 過学習"
        short = f["label"][:40]
        lines.append(f"  {short:<42}  {wr_t:>6.1f}%  {wr_v:>6.1f}%  "
                     f"{avg_t:>+8.3f}%  {avg_v:>+8.3f}%  {ok}")
        if wr_v >= MIN_WR and avg_v > 0:
            passed.append({**f, "wr_val": wr_v, "avg_val": avg_v})

    return lines, passed

# ── 6. まとめ ──────────────────────────────────────────

def summarize(passed):
    lines = ["\n【6】まとめ・scan_morning_strategy_a_strong.py への反映候補"]
    lines.append("  OOS検証を通過した条件（汎化性能あり）:")
    lines.append("  " + "─" * 70)
    if not passed:
        lines.append("  OOS検証を通過した有意条件はありませんでした。")
        lines.append("  → 現在のSTRONG日「全CAUTION」戦略を継続推奨")
    else:
        for f in sorted(passed, key=lambda x: -x["avg_val"]):
            lines.append(f"  ★ {f['label']}")
            lines.append(f"     検証期間: 勝率{f['wr_val']:.1f}%  平均{f['avg_val']:+.3f}%")
        lines.append("\n  上記条件を scan_morning_strategy_a_strong.py に反映を検討してください。")
    return lines

# ── メイン ────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  analyze_a_strong.py - 戦略A STRONG日 特化分析")
    print(f"  実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    df = load_strong_data()
    all_lines = []

    print("【1】月次推移を計算中...")
    all_lines += analyze_monthly(df)

    print("【2】現在のPASS条件を検証中...")
    all_lines += verify_pass_conditions(df)

    print("【3】特徴量別パフォーマンスを計算中...")
    all_lines += analyze_features(df)

    print("【4】多変量条件探索中（しばらくかかります）...")
    mv_lines, found = analyze_multivariate(df)
    all_lines += mv_lines
    print(f"     → {len(found)}件の有意条件を発見")

    print("【5】OOS検証中...")
    oos_lines, passed = analyze_oos(df, found)
    all_lines += oos_lines

    all_lines += summarize(passed)

    report = "\n".join(all_lines)
    print(report)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✅ レポート保存: {REPORT_TXT}")


if __name__ == "__main__":
    main()
