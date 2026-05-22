"""
simulate_normal.py - NORMAL日専用パラメータ探索

scan_morning.py の戦略A判定をNORMAL日に限定してグリッドサーチする。
スコア帯 / ratio帯 / 前日騰落帯 の全組み合わせを検証し、
有効な買い条件を探す。

出力:
  out/sim_normal_report.txt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from datetime import datetime
import db

MIN_VOLUME   = 50_000
MIN_TURNOVER = 50_000_000
MIN_SAMPLES  = 50   # 統計的に信頼できる最小件数

PANIC_NIKKEI  = -2.0
PANIC_AD      = 0.20
WEAK_NIKKEI   = -1.0
WEAK_AD       = 0.35
STRONG_NIKKEI = 0.5
STRONG_AD     = 0.60


def build_market_conditions():
    print("地合い判定を計算中...")
    conn = db.get_conn()
    df = pd.read_sql(
        "SELECT Date, Close, prev_close FROM daily_prices WHERE prev_close > 0 AND Close > 0",
        conn
    )
    conn.close()
    df["Date"]       = pd.to_datetime(df["Date"])
    df["change_pct"] = (df["Close"] - df["prev_close"]) / df["prev_close"] * 100

    rows = []
    for date, grp in df.groupby("Date"):
        up    = (grp["change_pct"] > 0).sum()
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
        rows.append({"date": date.strftime("%Y-%m-%d"), "condition": cond})

    df_mc = pd.DataFrame(rows).set_index("date")
    print(f"  完了: {len(df_mc)}営業日  分布: {df_mc['condition'].value_counts().to_dict()}")
    return df_mc


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


def collect_trades(market_cond):
    print("全銘柄のトレードデータを収集中...")
    codes = db.get_all_codes()
    all_trades = []
    n = len(codes)

    for i, code in enumerate(codes):
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n}... ({len(all_trades)}件)")
        try:
            df = db.get_stock_history(code)
            if df is None or df.empty:
                continue
            for col in ["Volume", "Close", "Open", "High", "Low", "Va"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df["Volume"] = df["Volume"].fillna(0)
            df["Date"]   = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)

            for idx in range(22, len(df) - 1):
                hist     = df.iloc[:idx+1]
                today    = hist.iloc[-1]
                next_row = df.iloc[idx + 1]
                scan_dt  = today["Date"].strftime("%Y-%m-%d")
                trade_dt = next_row["Date"].strftime("%Y-%m-%d")

                # スキャン日はNORMAL、取引日もNORMAL以上を対象
                scan_cond  = market_cond.get(scan_dt,  {}).get("condition", "UNKNOWN")
                trade_cond = market_cond.get(trade_dt, {}).get("condition", "UNKNOWN")
                if scan_cond != "NORMAL":
                    continue
                if trade_cond in ("PANIC",):
                    continue

                vol_today = float(today["Volume"]) if not pd.isna(today["Volume"]) else 0
                va_today  = float(today["Va"]) if ("Va" in today and not pd.isna(today["Va"])) else vol_today * float(today["Close"])
                if vol_today < MIN_VOLUME or va_today < MIN_TURNOVER:
                    continue

                sa = calc_score_a(hist)
                if sa is None or sa["score"] < 3.0:
                    continue

                open_p  = float(next_row["Open"])
                close_p = float(next_row["Close"])
                high_p  = float(next_row["High"]) if not pd.isna(next_row["High"]) else np.nan
                low_p   = float(next_row["Low"])  if not pd.isna(next_row["Low"])  else np.nan
                if open_p <= 0 or close_p <= 0:
                    continue

                ret_oc  = (close_p - open_p) / open_p * 100
                ret_max = (high_p  - open_p) / open_p * 100 if not pd.isna(high_p) else np.nan
                ret_low = (low_p   - open_p) / open_p * 100 if not pd.isna(low_p)  else np.nan
                gap_pct = (open_p  - float(today["Close"])) / float(today["Close"]) * 100 if float(today["Close"]) > 0 else np.nan

                today_rise = np.nan
                if idx > 0:
                    prev_close = float(df.iloc[idx - 1]["Close"])
                    if not pd.isna(prev_close) and prev_close > 0:
                        today_rise = (float(today["Close"]) - prev_close) / prev_close * 100

                all_trades.append({
                    "code":       code,
                    "trade_date": trade_dt,
                    "score":      sa["score"],
                    "ratio":      sa["ratio"],
                    "ret_oc":     round(ret_oc, 2),
                    "ret_max":    round(ret_max, 2) if not pd.isna(ret_max) else np.nan,
                    "ret_low":    round(ret_low, 2) if not pd.isna(ret_low) else np.nan,
                    "gap_pct":    round(gap_pct, 2) if not pd.isna(gap_pct) else np.nan,
                    "today_rise": round(today_rise, 2) if not pd.isna(today_rise) else np.nan,
                    "trade_cond": trade_cond,
                })
        except Exception:
            continue

    print(f"  収集完了: {len(all_trades)}件")
    return pd.DataFrame(all_trades)


def sharpe(returns):
    arr = np.array(returns)
    if len(arr) < 5 or arr.std() == 0:
        return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(252))


def sim_tp_sl(sub, tp, sl):
    results = []
    for _, r in sub.iterrows():
        hi = r["ret_max"]
        lo = r["ret_low"]
        if pd.isna(hi) or pd.isna(lo):
            results.append(r["ret_oc"])
        elif hi >= tp and lo > sl:
            results.append(tp)
        elif lo <= sl and hi < tp:
            results.append(sl)
        elif hi >= tp and lo <= sl:
            results.append(tp * 0.4 + sl * 0.6)
        else:
            results.append(r["ret_oc"])
    return np.array(results)


def grid_search(df):
    print("\nグリッドサーチ中...")

    score_bins = [
        (3.0, 5.0, "3〜5"),
        (5.0, 7.0, "5〜7"),
        (7.0, 9.0, "7〜9"),
        (9.0, 99., "9+"),
        (5.0, 9.0, "5〜9"),
        (3.0, 99., "全体"),
    ]
    ratio_bins = [
        (0.0, 3.0,  "〜3x"),
        (3.0, 6.0,  "3〜6x"),
        (6.0, 10.0, "6〜10x"),
        (0.0, 6.0,  "〜6x"),
        (3.0, 10.0, "3〜10x"),
        (0.0, 99.0, "全体"),
    ]
    rise_bins = [
        (-99., -2.0, "〜-2%"),
        (-2.0,  0.0, "-2〜0%"),
        ( 0.0,  2.0, "0〜+2%"),
        ( 2.0, 99.0, "+2%〜"),
        (-2.0,  2.0, "-2〜+2%"),
        (-99., 99.0, "全体"),
    ]
    exits = [
        ("OC",   None, None),
        ("TP3SL3", 3.0, -3.0),
        ("TP3SL5", 3.0, -5.0),
        ("TP5SL3", 5.0, -3.0),
        ("TP5SL5", 5.0, -5.0),
    ]

    results = []
    total = len(score_bins) * len(ratio_bins) * len(rise_bins) * len(exits)
    cnt = 0

    for s_lo, s_hi, s_label in score_bins:
        for r_lo, r_hi, r_label in ratio_bins:
            for ri_lo, ri_hi, ri_label in rise_bins:
                sub = df[
                    (df["score"]      >= s_lo) & (df["score"]      < s_hi) &
                    (df["ratio"]      >= r_lo) & (df["ratio"]      < r_hi) &
                    (df["today_rise"] >= ri_lo) & (df["today_rise"] < ri_hi)
                ].dropna(subset=["ret_oc"])

                for ex_label, tp, sl in exits:
                    cnt += 1
                    if len(sub) < MIN_SAMPLES:
                        continue
                    if tp is not None:
                        rets = sim_tp_sl(sub, tp, sl)
                    else:
                        rets = sub["ret_oc"].values
                    avg = float(rets.mean())
                    wr  = float((rets > 0).mean() * 100)
                    sh  = sharpe(rets)
                    results.append({
                        "score":  s_label,
                        "ratio":  r_label,
                        "rise":   ri_label,
                        "exit":   ex_label,
                        "N":      len(sub),
                        "avg":    round(avg, 3),
                        "WR":     round(wr, 1),
                        "sharpe": round(sh, 3),
                        "cumsum": round(float(rets.sum()), 1),
                    })

    print(f"  完了: {len(results)}/{total}件の有効組み合わせ")
    return pd.DataFrame(results)


def main():
    db.init_db()
    print("=" * 70)
    print("  NORMAL日専用 戦略A パラメータ探索シミュレーション")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    market_cond = build_market_conditions()
    market_cond_dict = market_cond.to_dict("index")

    df = collect_trades(market_cond_dict)
    if df.empty:
        print("データなし")
        return

    print(f"\nデータ概要: {len(df)}件 / NORMAL日スキャン銘柄")
    print(f"  期間: {df['trade_date'].min()} 〜 {df['trade_date'].max()}")
    print(f"  スコア: mean={df['score'].mean():.2f}  ratio: mean={df['ratio'].mean():.2f}")
    print(f"  ret_oc全体: avg={df['ret_oc'].mean():>+.3f}%  WR={(df['ret_oc']>0).mean()*100:.1f}%")

    results = grid_search(df)
    if results.empty:
        print("有効な組み合わせなし")
        return

    # ── Sharpe 上位表示 ──
    top = results.sort_values("sharpe", ascending=False).head(30)

    print(f"\n{'='*70}")
    print("【Sharpe上位30件】")
    print(f"{'='*70}")
    print(f"  {'スコア':<8} {'ratio':<8} {'前日騰落':<8} {'出口':<9} {'N':>4}  {'avg':>7}  {'WR':>5}  {'Sharpe':>7}  {'累計%':>7}")
    print("  " + "─" * 68)
    for _, r in top.iterrows():
        print(f"  {r['score']:<8} {r['ratio']:<8} {r['rise']:<8} {r['exit']:<9} "
              f"{r['N']:>4}  {r['avg']:>+7.3f}%  {r['WR']:>4.1f}%  {r['sharpe']:>+7.3f}  {r['cumsum']:>+7.1f}%")

    # ── avg > 0 の条件のみ ──
    positive = results[results["avg"] > 0].sort_values("sharpe", ascending=False)
    print(f"\n{'='*70}")
    print(f"【avg > 0 の条件 上位20件】（{len(positive)}件中）")
    print(f"{'='*70}")
    print(f"  {'スコア':<8} {'ratio':<8} {'前日騰落':<8} {'出口':<9} {'N':>4}  {'avg':>7}  {'WR':>5}  {'Sharpe':>7}")
    print("  " + "─" * 64)
    for _, r in positive.head(20).iterrows():
        print(f"  {r['score']:<8} {r['ratio']:<8} {r['rise']:<8} {r['exit']:<9} "
              f"{r['N']:>4}  {r['avg']:>+7.3f}%  {r['WR']:>4.1f}%  {r['sharpe']:>+7.3f}")

    # ── 比較: 現行ロジック相当 ──
    print(f"\n{'='*70}")
    print("【現行ロジック相当（score<7 × ratio<3 × 前日-2〜0%）との比較】")
    print(f"{'='*70}")
    current = df[
        (df["score"] < 7) &
        (df["ratio"] < 3) &
        (df["today_rise"] >= -2) & (df["today_rise"] < 0)
    ].dropna(subset=["ret_oc"])
    if len(current) >= 5:
        print(f"  件数: {len(current)}  avg: {current['ret_oc'].mean():>+.3f}%  "
              f"WR: {(current['ret_oc']>0).mean()*100:.1f}%  Sharpe: {sharpe(current['ret_oc'].values):>+.3f}")
    else:
        print(f"  件数不足: {len(current)}件")

    # ── レポート保存 ──
    report_path = os.path.join("out", "sim_normal_report.txt")
    lines = [
        "NORMAL日専用 戦略A パラメータ探索レポート",
        f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"総トレード数: {len(df)}件",
        "",
        "【Sharpe上位10件】",
    ]
    for _, r in top.head(10).iterrows():
        lines.append(f"  score{r['score']} / ratio{r['ratio']} / 前日{r['rise']} / {r['exit']}"
                     f"  | N={r['N']} avg={r['avg']:>+.3f}% WR={r['WR']:.1f}% Sharpe={r['sharpe']:>+.3f}")
    lines.append("")
    lines.append("【avg>0 上位10件】")
    for _, r in positive.head(10).iterrows():
        lines.append(f"  score{r['score']} / ratio{r['ratio']} / 前日{r['rise']} / {r['exit']}"
                     f"  | N={r['N']} avg={r['avg']:>+.3f}% WR={r['WR']:.1f}% Sharpe={r['sharpe']:>+.3f}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nレポート保存: {report_path}")


if __name__ == "__main__":
    main()
