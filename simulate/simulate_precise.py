"""
simulate_precise.py - 精密バックテストシミュレーション

改善点（通常シミュレーションとの違い）:
  1. 地合い判定を1年分全日付に適用（PANIC日は除外）
  2. ギャップフィルター（前日終値→翌日寄付き）
  3. TP/SL到達順序を分足データで校正
  4. 複数TP/SL組み合わせを検証
  5. 市場フィルター付き月次推移

出力:
  out/sim_precise_trades.csv
  out/sim_precise_report.txt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from datetime import datetime
import db

OUT_DIR    = "out"

MIN_VOLUME   = 50_000
MIN_TURNOVER = 50_000_000

PANIC_NIKKEI  = -2.0
PANIC_AD      = 0.20
WEAK_NIKKEI   = -1.0
WEAK_AD       = 0.35
STRONG_NIKKEI = 0.5
STRONG_AD     = 0.60

SL_CALIBRATION = {
    -1.0: {"tp_prob": 0.101, "sl_prob": 0.810, "neither_prob": 0.090},
    -2.0: {"tp_prob": 0.155, "sl_prob": 0.684, "neither_prob": 0.162},
    -3.0: {"tp_prob": 0.183, "sl_prob": 0.586, "neither_prob": 0.231},
}


# ══════════════════════════════════════════════
# 1. 全営業日の地合いを計算
# ══════════════════════════════════════════════
def build_market_conditions():
    """SQLiteから全営業日の地合い判定を計算"""
    print("地合い判定を計算中...")

    conn = db.get_conn()
    df = pd.read_sql(
        "SELECT Date, Close, prev_close FROM daily_prices WHERE prev_close > 0 AND Close > 0",
        conn
    )
    conn.close()

    df["Date"]       = pd.to_datetime(df["Date"])
    df["change_pct"] = (df["Close"] - df["prev_close"]) / df["prev_close"] * 100

    date_stats = []
    for date, grp in df.groupby("Date"):
        up    = (grp["change_pct"] > 0).sum()
        down  = (grp["change_pct"] < 0).sum()
        total = len(grp)
        ad    = up / total if total > 0 else 0.5
        nk    = float(grp["change_pct"].median())

        if nk <= PANIC_NIKKEI or ad <= PANIC_AD:
            cond = "PANIC"
        elif nk <= WEAK_NIKKEI or ad <= WEAK_AD:
            cond = "WEAK"
        elif nk >= STRONG_NIKKEI and ad >= STRONG_AD:
            cond = "STRONG"
        else:
            cond = "NORMAL"

        date_stats.append({
            "date": date.strftime("%Y-%m-%d"), "condition": cond,
            "ad_ratio": round(ad, 4), "nikkei_est": round(nk, 2),
            "up": up, "down": down,
        })

    df_mc = pd.DataFrame(date_stats).set_index("date")
    print(f"  地合い判定完了: {len(df_mc)}営業日")
    print(f"  分布: {df_mc['condition'].value_counts().to_dict()}")
    return df_mc


# ══════════════════════════════════════════════
# 2. スコア計算
# ══════════════════════════════════════════════
def calc_score_a(hist):
    if len(hist) < 22:
        return None
    vol = pd.to_numeric(hist["Volume"], errors="coerce").fillna(0)
    vol_20d = vol.iloc[-20:].values
    vol_5d  = vol.iloc[-5:].values
    v1      = float(vol.iloc[-1])
    v3      = float(vol.iloc[-3])
    avg20   = vol_20d.mean()
    avg5    = vol_5d.mean()
    if avg20 == 0 or avg5 == 0 or v3 == 0:
        return None
    trend = np.polyfit(range(5), vol_5d, 1)[0] / avg5 * 100
    accel = (v1 - v3) / v3 * 100
    ratio = v1 / avg20
    total = min(trend / 10, 3.0) + min(accel / 100, 3.0) + min(ratio / 3, 4.0)
    return {"score": round(total, 2), "ratio": round(ratio, 2)}


# ══════════════════════════════════════════════
# 3. 1銘柄分のシミュレーション
# ══════════════════════════════════════════════
def simulate_one(code, market_cond):
    trades = []
    try:
        df = db.get_stock_history(code)
        if df.empty:
            return trades

        for col in ["Volume", "Close", "Open", "High", "Low", "Va"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Volume"] = df["Volume"].fillna(0)
        df["Date"]   = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        for i in range(22, len(df) - 1):
            hist     = df.iloc[:i+1]
            next_row = df.iloc[i+1]
            today    = hist.iloc[-1]
            scan_dt  = today["Date"].strftime("%Y-%m-%d")
            trade_dt = next_row["Date"].strftime("%Y-%m-%d")

            trade_cond = market_cond.get(trade_dt, {}).get("condition", "UNKNOWN")
            scan_cond  = market_cond.get(scan_dt,  {}).get("condition", "UNKNOWN")

            if scan_cond == "PANIC" or trade_cond == "PANIC":
                continue
            if pd.isna(next_row["Open"])  or next_row["Open"]  <= 0:
                continue
            if pd.isna(next_row["Close"]) or next_row["Close"] <= 0:
                continue

            vol_today = float(today["Volume"]) if not pd.isna(today["Volume"]) else 0
            va_today  = float(today["Va"]) if ("Va" in today and not pd.isna(today["Va"])) else vol_today * float(today["Close"])

            if vol_today < MIN_VOLUME or va_today < MIN_TURNOVER:
                continue

            sa = calc_score_a(hist)
            if sa is None or sa["score"] < 3.0:
                continue

            open_p  = float(next_row["Open"])
            high_p  = float(next_row["High"]) if not pd.isna(next_row["High"]) else np.nan
            low_p   = float(next_row["Low"])  if not pd.isna(next_row["Low"])  else np.nan
            close_p = float(next_row["Close"])

            ret_oc  = (close_p - open_p) / open_p * 100
            ret_max = (high_p  - open_p) / open_p * 100 if not pd.isna(high_p) else np.nan
            ret_low = (low_p   - open_p) / open_p * 100 if not pd.isna(low_p)  else np.nan
            gap_pct = (open_p  - float(today["Close"])) / float(today["Close"]) * 100 if float(today["Close"]) > 0 else np.nan

            today_rise = np.nan
            if i > 0:
                prev_close = float(df.iloc[i - 1]["Close"])
                if not pd.isna(prev_close) and prev_close > 0:
                    today_rise = (float(today["Close"]) - prev_close) / prev_close * 100

            trades.append({
                "code":        code,
                "scan_date":   scan_dt,
                "trade_date":  trade_dt,
                "scan_cond":   scan_cond,
                "trade_cond":  trade_cond,
                "score":       sa["score"],
                "ratio":       sa["ratio"],
                "open_price":  open_p,
                "close_price": close_p,
                "high_price":  high_p,
                "low_price":   low_p,
                "ret_oc":      round(ret_oc, 2),
                "ret_max":     round(ret_max, 2) if not pd.isna(ret_max) else np.nan,
                "ret_low":     round(ret_low, 2) if not pd.isna(ret_low) else np.nan,
                "gap_pct":     round(gap_pct, 2) if not pd.isna(gap_pct) else np.nan,
                "today_rise":  round(today_rise, 2) if not pd.isna(today_rise) else np.nan,
            })
    except Exception:
        pass
    return trades


# ══════════════════════════════════════════════
# 4. TP/SL シミュレーション（分足校正版）
# ══════════════════════════════════════════════
def sim_tp_sl(sub, tp, sl):
    cal = SL_CALIBRATION.get(sl, {"tp_prob": 0.15, "sl_prob": 0.70, "neither_prob": 0.15})
    results = []
    for _, r in sub.iterrows():
        hi = r["ret_max"]
        lo = r["ret_low"]
        if pd.isna(hi) or pd.isna(lo):
            results.append(r["ret_oc"])
            continue

        if hi >= tp and lo > sl:
            results.append(tp)
        elif lo <= sl and hi < tp:
            results.append(sl)
        elif hi >= tp and lo <= sl:
            total_p = cal["tp_prob"] + cal["sl_prob"]
            tp_w = cal["tp_prob"] / total_p if total_p > 0 else 0.5
            sl_w = cal["sl_prob"] / total_p if total_p > 0 else 0.5
            results.append(tp * tp_w + sl * sl_w)
        else:
            results.append(r["ret_oc"])
    return np.array(results)


# ══════════════════════════════════════════════
# 5. レポート生成
# ══════════════════════════════════════════════
def make_report(df, lines):
    lines.append(f"\n{'─'*70}")
    lines.append("【PANIC除外】地合い別パフォーマンス（スコア>=5）")
    lines.append(f"{'─'*70}")

    sub5 = df[df["score"] >= 5]
    for cond in ["NORMAL", "WEAK", "STRONG", "ALL"]:
        sub = sub5 if cond == "ALL" else sub5[(sub5["scan_cond"] == cond) | (sub5["trade_cond"] == cond)]
        if len(sub) < 10:
            continue
        wr     = (sub["ret_oc"] > 0).mean() * 100
        avg    = sub["ret_oc"].mean()
        avg_hi = sub["ret_max"].mean()
        lines.append(f"  {cond:<8}: {len(sub):>6}件  勝率:{wr:.1f}%  終値平均:{avg:+.2f}%  高値平均:{avg_hi:+.2f}%")

    lines.append(f"\n{'─'*70}")
    lines.append("【TP/SL最適化】スコア・ギャップ・地合いフィルター込み")
    lines.append("  ※ 分足データ校正済み（TP/SL両到達時の確率を考慮）")
    lines.append(f"{'─'*70}")

    combos = [
        (1.5, -1.0), (2.0, -1.0), (3.0, -1.0),
        (2.0, -2.0), (3.0, -2.0), (5.0, -2.0),
        (3.0, -3.0), (5.0, -3.0),
    ]

    for score_thr in [5.0, 7.0, 9.0]:
        lines.append(f"\n  [スコア >= {score_thr:.0f}]")
        lines.append(f"  {'TP':>5} {'SL':>5}  {'件数':>6}  {'勝率':>6}  {'平均%':>7}  {'期待値/日':>9}  {'判定':>6}")
        lines.append("  " + "─" * 55)

        base = df[df["score"] >= score_thr]
        for tp, sl in combos:
            sub = base[base["today_rise"].abs() < 30]
            sub = sub[sub["gap_pct"] < 5]
            if len(sub) < 20:
                continue

            rets   = sim_tp_sl(sub, tp, sl)
            n      = len(rets)
            wr     = (rets > 0).mean() * 100
            avg    = rets.mean()
            n_days = sub["trade_date"].nunique()
            daily  = avg * n / n_days if n_days > 0 else 0
            mark   = "★" if avg > 0 else ("△" if avg > -0.1 else " ")
            lines.append(
                f"  {tp:>+5.1f}% {sl:>+5.1f}%  {n:>6}  {wr:>5.1f}%  {avg:>+6.3f}%  "
                f"{daily:>+8.3f}%  {mark:>5}"
            )

    lines.append(f"\n{'─'*70}")
    lines.append("【最良条件詳細】TP+3% / SL-2% / スコア>=7 / ギャップ<5%")
    lines.append(f"{'─'*70}")

    best      = df[(df["score"] >= 7) & (df["gap_pct"] < 5) & (df["today_rise"].abs() < 30)]
    rets_best = sim_tp_sl(best, 3.0, -2.0)

    lines.append(f"  件数: {len(best):,}")
    lines.append(f"  勝率: {(rets_best>0).mean()*100:.1f}%")
    lines.append(f"  平均リターン: {rets_best.mean():+.3f}%/回")
    lines.append(f"  累計: {rets_best.sum():+.1f}%")

    lines.append(f"\n  [月別推移]")
    best2          = best.copy()
    best2["sim_ret"] = rets_best
    best2["month"]   = best2["trade_date"].str[:7]
    mg = best2.groupby("month").agg(
        n=("sim_ret", "count"),
        wr=("sim_ret", lambda x: (x > 0).mean() * 100),
        avg=("sim_ret", "mean"),
        total=("sim_ret", "sum"),
    )
    lines.append(f"  {'月':>8}  {'件数':>5}  {'勝率':>6}  {'平均%':>7}  {'月次累計%':>9}")
    for idx, r in mg.iterrows():
        lines.append(f"  {idx:>8}  {int(r['n']):>5}  {r['wr']:>5.1f}%  {r['avg']:>+6.3f}%  {r['total']:>+8.2f}%")

    lines.append(f"\n  [スコア別詳細]")
    best2["score_bin"] = pd.cut(best2["score"], bins=[7, 8, 9, 10, 11],
                                labels=["7-8", "8-9", "9-10", "10+"], right=False)
    sg = best2.groupby("score_bin", observed=True).agg(
        n=("sim_ret", "count"),
        wr=("sim_ret", lambda x: (x > 0).mean() * 100),
        avg=("sim_ret", "mean"),
        hi_avg=("ret_max", "mean"),
    )
    lines.append(f"  {'スコア':>6}  {'件数':>5}  {'勝率':>6}  {'平均%':>7}  {'高値平均%':>9}")
    for idx, r in sg.iterrows():
        lines.append(f"  {str(idx):>6}  {int(r['n']):>5}  {r['wr']:>5.1f}%  {r['avg']:>+6.3f}%  {r['hi_avg']:>+8.2f}%")

    lines.append(f"\n{'─'*70}")
    lines.append("【ギャップフィルター効果】TP+3% / SL-2% / スコア>=7")
    lines.append(f"{'─'*70}")
    lines.append(f"  {'ギャップ条件':>14}  {'件数':>6}  {'勝率':>6}  {'平均%':>7}")
    base7 = df[df["score"] >= 7]
    for lo, hi, lab in [
        (-99, -3, "ギャップ<-3%"),
        (-3,   0, "ギャップ-3~0%"),
        ( 0,   2, "ギャップ0~+2%"),
        ( 2,   5, "ギャップ+2~+5%"),
        ( 5,  10, "ギャップ+5~+10%"),
        (10, 999, "ギャップ+10%超"),
        (-99,  5, "ギャップ<+5%（推奨）"),
    ]:
        sub  = base7[(base7["gap_pct"] >= lo) & (base7["gap_pct"] < hi)]
        if len(sub) < 10:
            continue
        rets = sim_tp_sl(sub, 3.0, -2.0)
        wr   = (rets > 0).mean() * 100
        avg  = rets.mean()
        mark = "★" if avg > 0 else ""
        lines.append(f"  {lab:>14}: {len(sub):>6}  {wr:>5.1f}%  {avg:>+6.3f}%  {mark}")


# ══════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════
def main():
    db.init_db()

    market_cond = build_market_conditions()
    market_dict = market_cond.to_dict(orient="index")

    codes = db.get_all_codes()
    print(f"\nシミュレーション開始: {len(codes)}銘柄")

    all_trades = []
    for i, code in enumerate(codes):
        if i % 500 == 0:
            print(f"  {i}/{len(codes)}...")
        all_trades.extend(simulate_one(code, market_dict))

    if not all_trades:
        print("トレードデータなし")
        return

    df = pd.DataFrame(all_trades)
    df = df.sort_values(["trade_date", "code"]).reset_index(drop=True)
    df.to_csv("out/sim_precise_trades.csv", index=False, encoding="utf-8-sig")
    print(f"\n  シグナル数（PANIC除外済み）: {len(df):,}")

    lines = []
    lines.append("=" * 70)
    lines.append("  精密バックテストレポート（PANIC除外・ギャップフィルター・分足校正）")
    lines.append(f"  対象期間: {df['scan_date'].min()} ~ {df['scan_date'].max()}")
    lines.append(f"  生成日  : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  シグナル数: {len(df):,}  (PANIC日除外済み)")
    lines.append("=" * 70)

    make_report(df, lines)

    lines.append(f"\n{'='*70}")
    lines.append("  ※ 分足校正: TP/SL両到達時の確率を389件の分足データから推定")
    lines.append("  ※ PANIC日（前後）除外: スキャン日・トレード日ともに除外")
    lines.append("  ※ ギャップフィルター: 前日終値→翌日寄付き +5%超は除外")
    lines.append("  ※ 手数料・スリッページは未考慮（実際は各0.1〜0.3%差引）")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print("\n" + report)
    with open("out/sim_precise_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n保存: out/sim_precise_report.txt")
    print(f"保存: out/sim_precise_trades.csv")


if __name__ == "__main__":
    main()
