# -*- coding: utf-8 -*-
"""
optimize_f.py - 戦略F（銀行×金利マクロ連動）バックテストデータ生成

米10年債利回り(^TNX)の前日比が閾値を超えて上昇した日、
銀行セクター5銘柄（calc_score_d採用・scan_morning_macro_scan.pyと同一ロジック）が
その日どう動いたかを検証する。

【実行方法】
    python optimize_f.py

【出力ファイル】
    out/opt_f_trades.csv   - 全トレードデータ（rate_chg・score・当日リターン）
    out/opt_f_report.txt   - 現行設定（閾値+1.0% / score>=7.5）でのバックテスト結果
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import db
import yfinance as yf
from scan_morning_macro_scan import calc_score_d
from scan_morning_market_data import RATE_TICKER, MACRO_SECTORS

OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "out")
MIN_HIST  = 26   # calc_score_d に必要な最低日数
MIN_SAMPLE = 20  # レポート表示に必要な最小サンプル数

BANK_SECTOR = next(s for s in MACRO_SECTORS if s["trigger_key"] == "rate")
BANK_STOCKS = BANK_SECTOR["stocks"]

# 現行のライブ設定（scan_morning_market_data.py / scan_morning_macro_scan.py と同一）
LIVE_RATE_THRESHOLD = BANK_SECTOR["threshold"]   # 1.0
LIVE_SCORE_MIN       = 7.5                       # WEAK日「⚠️ 少額検討」の判定ライン


# ══════════════════════════════════════════════
# データ準備
# ══════════════════════════════════════════════
def load_rate_series():
    """^TNXの日次終値・前日比(%)をDataFrameで返す"""
    data = yf.Ticker(RATE_TICKER).history(period="max")
    data = data.reset_index()
    data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None).dt.normalize()
    data = data[["Date", "Close"]].sort_values("Date").reset_index(drop=True)
    data["rate_chg"] = data["Close"].pct_change() * 100
    return data


def build_trades():
    """5銘柄×全営業日について、当日朝時点で入手可能な直近rate_chgと
    calc_score_dスコア、当日の寄り→高値/安値/引けリターンを集計する。
    """
    rate_df = load_rate_series()
    rows = []

    for code, name in BANK_STOCKS:
        hist = db.get_stock_history(code)
        if hist.empty:
            print(f"  ⚠️  [{code}] {name}: 価格データなし")
            continue
        hist = hist.sort_values("Date").reset_index(drop=True)
        hist["Date_dt"] = pd.to_datetime(hist["Date"])

        n_added = 0
        for i in range(MIN_HIST, len(hist)):
            day      = hist.iloc[i]
            date_dt  = day["Date_dt"]

            prior_rate = rate_df[rate_df["Date"] < date_dt]
            if prior_rate.empty:
                continue
            rate_chg = prior_rate.iloc[-1]["rate_chg"]
            if pd.isna(rate_chg):
                continue

            past_hist = hist.iloc[:i]   # 当日を含まない（D-1以前のみ）
            s = calc_score_d(past_hist)
            if s is None:
                continue

            o, h, l, c = (float(day["Open"]), float(day["High"]),
                          float(day["Low"]),  float(day["Close"]))
            if o <= 0:
                continue

            rows.append({
                "date":     day["Date"],
                "code":     code,
                "name":     name,
                "rate_chg": round(float(rate_chg), 3),
                "score":    s["score"],
                "ratio":    s["ratio"],
                "ret_oc":   round((c - o) / o * 100, 3),
                "ret_max":  round((h - o) / o * 100, 3),
                "ret_low":  round((l - o) / o * 100, 3),
            })
            n_added += 1
        print(f"  [{code}] {name}: {n_added}件")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════
# シミュレーション・指標計算
# ══════════════════════════════════════════════
def sim_tp_sl(sub, tp, sl):
    """TP/SL適用後リターン配列を返す。tp=None なら引け決済。
    同日中にTP/SL両方到達した場合は順序が不明なため引け値（ret_oc）で近似する。
    """
    if len(sub) == 0:
        return np.array([])
    if tp is None:
        return sub["ret_oc"].values
    hi, lo, oc = sub["ret_max"].values, sub["ret_low"].values, sub["ret_oc"].values
    return np.where(
        (hi >= tp) & (lo > sl),  tp,
        np.where(
            (lo <= sl) & (hi < tp), sl,
            oc   # 両方到達 or どちらも未到達 → 引け値で近似
        )
    )


def calc_metrics(rets):
    if len(rets) < MIN_SAMPLE:
        return None
    wr     = (rets > 0).mean() * 100
    avg    = rets.mean()
    std    = rets.std()
    sharpe = avg / std * np.sqrt(252) if std > 0 else 0.0
    return {"n": len(rets), "wr": round(wr, 1), "avg": round(avg, 3),
            "sharpe": round(sharpe, 2), "cum": round(rets.sum(), 1)}


# ══════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════
def main():
    print("=" * 70)
    print(f"  戦略F バックテストデータ生成  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  対象: {', '.join(f'{c}({n})' for c, n in BANK_STOCKS)}")
    print("=" * 70)

    print("\n【データ収集】")
    df = build_trades()
    if df.empty:
        print("❌ トレードデータが1件も生成できませんでした。")
        sys.exit(1)

    trades_path = os.path.join(OUT_DIR, "opt_f_trades.csv")
    df.to_csv(trades_path, index=False, encoding="utf-8-sig")
    print(f"\n  合計 {len(df):,}件（{df['date'].min()} 〜 {df['date'].max()}）")
    print(f"  保存: {trades_path}")

    # ── 現行ライブ設定でのベースライン検証（引け決済）──
    print(f"\n{'='*70}")
    print(f"【ベースライン】現行ライブ設定  rate_chg>={LIVE_RATE_THRESHOLD}% × score>={LIVE_SCORE_MIN}")
    print(f"{'='*70}")
    sub = df[(df["rate_chg"] >= LIVE_RATE_THRESHOLD) & (df["score"] >= LIVE_SCORE_MIN)]
    m = calc_metrics(sim_tp_sl(sub, None, None))
    if m is None:
        print(f"  ⚠️  該当トレードが{MIN_SAMPLE}件未満のため信頼できる検証ができません"
              f"（該当{len(sub)}件）。閾値そのものの検証が必要です。")
    else:
        print(f"  N={m['n']}  勝率={m['wr']}%  平均={m['avg']:+.3f}%  "
              f"Sharpe={m['sharpe']:+.2f}  累計={m['cum']:+.1f}%")

    # ── 閾値なし（rate条件を外した場合）との比較 ──
    print(f"\n【比較】rate条件なし（score>={LIVE_SCORE_MIN}のみ）")
    sub_norate = df[df["score"] >= LIVE_SCORE_MIN]
    m2 = calc_metrics(sim_tp_sl(sub_norate, None, None))
    if m2:
        print(f"  N={m2['n']}  勝率={m2['wr']}%  平均={m2['avg']:+.3f}%  Sharpe={m2['sharpe']:+.2f}")
        if m and m['avg'] <= m2['avg']:
            print(f"  ⚠️  rate条件を付けても平均リターンが改善していません（現行閾値の妥当性に疑問）")
    else:
        print(f"  （サンプル不足）")

    # ── 簡易グリッド（rate閾値 × score閾値）──
    print(f"\n{'='*70}")
    print(f"【簡易グリッドサーチ】rate閾値 × score閾値（引け決済・参考値）")
    print(f"{'='*70}")
    print(f"  {'rate>=':>8} {'score>=':>8} {'N':>6} {'勝率':>7} {'平均':>8} {'Sharpe':>8}")
    print("  " + "─" * 50)
    grid_results = []
    for rt in [0.0, 0.5, 1.0, 1.5, 2.0]:
        for sm in [0.0, 5.0, 6.0, 7.0, 7.5, 8.0]:
            sub_g = df[(df["rate_chg"] >= rt) & (df["score"] >= sm)]
            mg = calc_metrics(sim_tp_sl(sub_g, None, None))
            if mg is None:
                continue
            grid_results.append({"rate_min": rt, "score_min": sm, **mg})
            print(f"  {rt:>7.1f}% {sm:>7.1f}  {mg['n']:>6} {mg['wr']:>6.1f}% "
                  f"{mg['avg']:>+7.3f}% {mg['sharpe']:>+7.2f}")

    grid_df = pd.DataFrame(grid_results).sort_values("sharpe", ascending=False)
    grid_path = os.path.join(OUT_DIR, "opt_f_grid.csv")
    grid_df.to_csv(grid_path, index=False, encoding="utf-8-sig")

    # ── レポート ──
    report_lines = [
        "戦略F（銀行×金利マクロ連動）バックテストレポート",
        f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"対象銘柄: {', '.join(f'{c}{n}' for c, n in BANK_STOCKS)}",
        f"データ期間: {df['date'].min()} 〜 {df['date'].max()}  (全{len(df):,}件)",
        "",
        f"【現行ライブ設定】rate_chg>={LIVE_RATE_THRESHOLD}% × score>={LIVE_SCORE_MIN}",
    ]
    if m:
        report_lines.append(f"  N={m['n']} 勝率={m['wr']}% 平均={m['avg']:+.3f}% Sharpe={m['sharpe']:+.2f} 累計={m['cum']:+.1f}%")
    else:
        report_lines.append(f"  サンプル不足（{len(sub)}件 < {MIN_SAMPLE}件）— 統計的に検証不能")
    report_lines += [
        "",
        "【グリッドサーチ Sharpe上位5】",
    ]
    for _, r in grid_df.head(5).iterrows():
        report_lines.append(
            f"  rate>={r['rate_min']:.1f}% score>={r['score_min']:.1f} | "
            f"N={r['n']} WR={r['wr']}% avg={r['avg']:+.3f}% Sharpe={r['sharpe']:+.2f}"
        )
    report_lines.append("\n次ステップ: python optimize/evolve_f.py でTP/SLを含めたGA最適化を実行")
    report_txt = "\n".join(report_lines)

    rep_path = os.path.join(OUT_DIR, "opt_f_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(report_txt)
    print(f"\n保存完了: {trades_path}")
    print(f"          {grid_path}")
    print(f"          {rep_path}")
    print(f"\n{report_txt}")


if __name__ == "__main__":
    main()
