"""
simulate.py - 株価予想スコアのバックテストシミュレーション
キャッシュデータ（out/cache/*.csv）を使って戦略A・Bのパフォーマンスを検証

出力:
  out/sim_trades.csv   - 全トレード詳細
  out/sim_report.txt   - 集計レポート
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, timedelta

CACHE_DIR  = "out/cache"
OUT_DIR    = "out"
TRADES_CSV = "out/sim_trades.csv"
REPORT_TXT = "out/sim_report.txt"

MIN_VOLUME   = 50_000
MIN_TURNOVER = 50_000_000

# ── スコア計算（戦略A: 出来高急増） ──────────────────────────────────────
def calc_score_a(hist):
    """histはその日までのDataFrame（最終行が当日）"""
    if len(hist) < 22:
        return None
    vol_20d = hist["Volume"].iloc[-20:].values
    vol_5d  = hist["Volume"].iloc[-5:].values
    v1 = float(hist["Volume"].iloc[-1])
    v3 = float(hist["Volume"].iloc[-3])
    avg20 = vol_20d.mean()
    avg5  = vol_5d.mean()
    if avg20 == 0 or avg5 == 0 or v3 == 0:
        return None
    trend = np.polyfit(range(5), vol_5d, 1)[0] / avg5 * 100
    accel = (v1 - v3) / v3 * 100
    ratio = v1 / avg20
    total = min(trend / 10, 3.0) + min(accel / 100, 3.0) + min(ratio / 3, 4.0)
    return {"trend": round(trend, 2), "accel": round(accel, 2),
            "ratio": round(ratio, 2), "score": round(total, 2)}


# ── RSI計算（戦略B用） ──────────────────────────────────────────────────
def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    deltas = prices.diff().dropna()
    gain = deltas.clip(lower=0).rolling(period).mean()
    loss = (-deltas.clip(upper=0)).rolling(period).mean()
    rs   = gain / loss.replace(0, 1e-9)
    rsi  = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 1)


# ── スコア計算（戦略B: リバウンド） ─────────────────────────────────────
def calc_score_b(hist):
    """当日が-5%以下の下落銘柄のリバウンドスコア（0〜10点）"""
    if len(hist) < 26:
        return None

    score   = 0
    reasons = []

    # RSI（0〜4点）
    rsi = calc_rsi(hist["Close"], 14)
    if rsi is not None:
        if rsi < 20:
            score += 4; reasons.append(f"RSI極度売られ過ぎ({rsi:.0f})")
        elif rsi < 30:
            score += 3; reasons.append(f"RSI売られ過ぎ({rsi:.0f})")
        elif rsi < 40:
            score += 2; reasons.append(f"RSIやや低め({rsi:.0f})")
        elif rsi < 50:
            score += 1; reasons.append(f"RSI中立({rsi:.0f})")

    # 出来高倍率（0〜3点）
    if len(hist) >= 21:
        avg20     = hist["Volume"].iloc[-21:-1].mean()
        vol_today = hist["Volume"].iloc[-1]
        if avg20 > 0:
            vol_ratio = vol_today / avg20
            if vol_ratio >= 3.0:
                score += 3; reasons.append(f"出来高急増({vol_ratio:.1f}倍)")
            elif vol_ratio >= 2.0:
                score += 2; reasons.append(f"出来高増加({vol_ratio:.1f}倍)")
            elif vol_ratio >= 1.5:
                score += 1; reasons.append(f"出来高やや増({vol_ratio:.1f}倍)")

    # MA25乖離（0〜3点）
    if len(hist) >= 25:
        ma25      = hist["Close"].iloc[-25:].mean()
        deviation = (hist["Close"].iloc[-1] / ma25 - 1) * 100
        if deviation < -15:
            score += 3; reasons.append(f"MA25大幅下離れ({deviation:.1f}%)")
        elif deviation < -10:
            score += 2; reasons.append(f"MA25下離れ({deviation:.1f}%)")
        elif deviation < -5:
            score += 1; reasons.append(f"MA25やや下離れ({deviation:.1f}%)")

    return {"score": score, "reasons": "; ".join(reasons)}


# ── 1銘柄分のシミュレーション ───────────────────────────────────────────
def simulate_one(filepath):
    trades = []
    try:
        df = pd.read_csv(filepath)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
        df["Close"]  = pd.to_numeric(df["Close"], errors="coerce")
        df["Open"]   = pd.to_numeric(df["Open"], errors="coerce")
        df["High"]   = pd.to_numeric(df["High"], errors="coerce")
        df["Low"]    = pd.to_numeric(df["Low"], errors="coerce")
        df["Va"]     = pd.to_numeric(df.get("Va", pd.Series([np.nan]*len(df))), errors="coerce")

        code = os.path.basename(filepath).replace(".csv", "")

        for i in range(22, len(df) - 1):
            hist     = df.iloc[:i+1]   # スキャン日まで
            next_row = df.iloc[i+1]    # 翌営業日（トレード日）

            today    = hist.iloc[-1]
            scan_dt  = today["Date"]

            # 翌日データの確認
            if pd.isna(next_row["Open"]) or pd.isna(next_row["Close"]):
                continue
            if next_row["Open"] <= 0 or next_row["Close"] <= 0:
                continue

            vol_today     = today["Volume"] if not pd.isna(today["Volume"]) else 0
            turnover_today = today.get("Va", np.nan)
            if pd.isna(turnover_today):
                turnover_today = vol_today * today["Close"] if not pd.isna(today["Close"]) else 0

            # ── 戦略A: 出来高急増 ────────────────
            if (vol_today >= MIN_VOLUME and
                    turnover_today >= MIN_TURNOVER):
                sa = calc_score_a(hist)
                if sa is not None and sa["score"] >= 3.0:
                    ret_oc  = (next_row["Close"] - next_row["Open"]) / next_row["Open"] * 100
                    ret_max = (next_row["High"] - next_row["Open"]) / next_row["Open"] * 100 if not pd.isna(next_row["High"]) else np.nan
                    ret_low = (next_row["Low"]  - next_row["Open"]) / next_row["Open"] * 100 if not pd.isna(next_row["Low"])  else np.nan
                    # 前日比（スキャン日の騰落）
                    today_rise = (today["Close"] - today["Open"]) / today["Open"] * 100 if today["Open"] > 0 else np.nan

                    trades.append({
                        "strategy":    "A",
                        "code":        code,
                        "scan_date":   scan_dt.strftime("%Y-%m-%d"),
                        "trade_date":  next_row["Date"].strftime("%Y-%m-%d") if hasattr(next_row["Date"], "strftime") else str(next_row["Date"])[:10],
                        "score":       sa["score"],
                        "ratio":       sa["ratio"],
                        "open_price":  next_row["Open"],
                        "close_price": next_row["Close"],
                        "high_price":  next_row["High"],
                        "low_price":   next_row["Low"],
                        "ret_oc":      round(ret_oc, 2),
                        "ret_max":     round(ret_max, 2) if not pd.isna(ret_max) else np.nan,
                        "ret_low":     round(ret_low, 2) if not pd.isna(ret_low) else np.nan,
                        "today_rise":  round(today_rise, 2) if not pd.isna(today_rise) else np.nan,
                    })

            # ── 戦略B: リバウンド ─────────────────
            # 当日-5%以上の下落が条件
            if today["Open"] > 0 and not pd.isna(today["Close"]):
                today_pct = (today["Close"] - today["Open"]) / today["Open"] * 100
                if today_pct <= -5.0 and vol_today >= MIN_VOLUME:
                    sb = calc_score_b(hist)
                    if sb is not None and sb["score"] >= 3:
                        ret_oc  = (next_row["Close"] - next_row["Open"]) / next_row["Open"] * 100
                        ret_max = (next_row["High"] - next_row["Open"]) / next_row["Open"] * 100 if not pd.isna(next_row["High"]) else np.nan
                        ret_low = (next_row["Low"]  - next_row["Open"]) / next_row["Open"] * 100 if not pd.isna(next_row["Low"])  else np.nan

                        trades.append({
                            "strategy":    "B",
                            "code":        code,
                            "scan_date":   scan_dt.strftime("%Y-%m-%d"),
                            "trade_date":  next_row["Date"].strftime("%Y-%m-%d") if hasattr(next_row["Date"], "strftime") else str(next_row["Date"])[:10],
                            "score":       sb["score"],
                            "ratio":       np.nan,
                            "open_price":  next_row["Open"],
                            "close_price": next_row["Close"],
                            "high_price":  next_row["High"],
                            "low_price":   next_row["Low"],
                            "ret_oc":      round(ret_oc, 2),
                            "ret_max":     round(ret_max, 2) if not pd.isna(ret_max) else np.nan,
                            "ret_low":     round(ret_low, 2) if not pd.isna(ret_low) else np.nan,
                            "today_rise":  round(today_pct, 2),
                        })
    except Exception as e:
        pass

    return trades


# ── メイン処理 ──────────────────────────────────────────────────────────
def main():
    files = sorted(glob.glob(f"{CACHE_DIR}/*.csv"))
    print(f"対象ファイル数: {len(files)}")

    all_trades = []
    for i, f in enumerate(files):
        if i % 500 == 0:
            print(f"  処理中... {i}/{len(files)}")
        all_trades.extend(simulate_one(f))

    if not all_trades:
        print("トレードデータなし")
        return

    df = pd.DataFrame(all_trades)
    df = df.sort_values(["scan_date", "code"]).reset_index(drop=True)
    df.to_csv(TRADES_CSV, index=False, encoding="utf-8-sig")
    print(f"\n全トレード数: {len(df)}")
    print(f"保存: {TRADES_CSV}")

    # ── レポート生成 ─────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("  株価予想スコア バックテスト レポート")
    lines.append(f"  対象期間: {df['scan_date'].min()} ~ {df['scan_date'].max()}")
    lines.append(f"  生成日  : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)

    for strategy in ["A", "B"]:
        ds = df[df["strategy"] == strategy].copy()
        if ds.empty:
            continue

        lines.append(f"\n{'─'*70}")
        if strategy == "A":
            lines.append("【戦略A】出来高急増スコア（翌日寄付き→引け）")
        else:
            lines.append("【戦略B】リバウンドスコア（当日-5%以下 → 翌日寄付き→引け）")
        lines.append(f"{'─'*70}")

        total = len(ds)
        wins  = (ds["ret_oc"] > 0).sum()
        flat  = (ds["ret_oc"] == 0).sum()
        loss  = (ds["ret_oc"] < 0).sum()
        win_rate = wins / total * 100

        lines.append(f"  総シグナル数  : {total:,}")
        lines.append(f"  勝率          : {win_rate:.1f}%  (勝:{wins} 引:{flat} 負:{loss})")
        lines.append(f"  平均リターン  : {ds['ret_oc'].mean():.2f}%")
        lines.append(f"  中央値        : {ds['ret_oc'].median():.2f}%")
        lines.append(f"  標準偏差      : {ds['ret_oc'].std():.2f}%")
        lines.append(f"  最大利益      : {ds['ret_oc'].max():.2f}%")
        lines.append(f"  最大損失      : {ds['ret_oc'].min():.2f}%")
        if ds["ret_oc"].std() > 0:
            sharpe = ds["ret_oc"].mean() / ds["ret_oc"].std() * np.sqrt(252)
            lines.append(f"  Sharpe比(年換算): {sharpe:.2f}")
        lines.append(f"  最大高値利益  : {ds['ret_max'].max():.2f}%  (高値での最大)")
        lines.append(f"  平均高値リターン: {ds['ret_max'].mean():.2f}%")

        # ── スコア別集計 ─────────────────────────────────────────────
        lines.append(f"\n  [スコア別パフォーマンス]")
        if strategy == "A":
            bins   = [3, 5, 6, 7, 8, 9, 10, 11]
            labels = ["3-5", "5-6", "6-7", "7-8", "8-9", "9-10", "10+"]
        else:
            bins   = [3, 4, 5, 6, 7, 8, 9, 10, 11]
            labels = ["3", "4", "5", "6", "7", "8", "9", "10"]

        ds["score_bin"] = pd.cut(ds["score"], bins=bins, labels=labels, right=False)
        score_grp = ds.groupby("score_bin", observed=True).agg(
            count  = ("ret_oc", "count"),
            win_r  = ("ret_oc", lambda x: (x > 0).mean() * 100),
            avg_r  = ("ret_oc", "mean"),
            med_r  = ("ret_oc", "median"),
            max_r  = ("ret_oc", "max"),
        )
        lines.append(f"  {'スコア':>6}  {'件数':>6}  {'勝率':>6}  {'平均%':>7}  {'中央値%':>8}  {'最大%':>7}")
        lines.append("  " + "─" * 55)
        for idx, row in score_grp.iterrows():
            lines.append(
                f"  {str(idx):>6}  {int(row['count']):>6}  "
                f"{row['win_r']:>5.1f}%  {row['avg_r']:>+6.2f}%  "
                f"{row['med_r']:>+7.2f}%  {row['max_r']:>+6.2f}%"
            )

        # ── 月別集計 ─────────────────────────────────────────────────
        ds["month"] = ds["scan_date"].str[:7]
        lines.append(f"\n  [月別パフォーマンス]")
        month_grp = ds.groupby("month").agg(
            count  = ("ret_oc", "count"),
            win_r  = ("ret_oc", lambda x: (x > 0).mean() * 100),
            avg_r  = ("ret_oc", "mean"),
        )
        lines.append(f"  {'月':>8}  {'件数':>6}  {'勝率':>6}  {'平均%':>7}")
        lines.append("  " + "─" * 35)
        for idx, row in month_grp.iterrows():
            lines.append(
                f"  {idx:>8}  {int(row['count']):>6}  "
                f"{row['win_r']:>5.1f}%  {row['avg_r']:>+6.2f}%"
            )

    # ── 戦略A: スコア閾値別の予想力 ─────────────────────────────────────
    da = df[df["strategy"] == "A"].copy()
    if not da.empty:
        lines.append(f"\n{'─'*70}")
        lines.append("【予測力指数】戦略A スコア閾値別シミュレーション")
        lines.append("  （閾値以上のシグナルのみ採用した場合）")
        lines.append(f"{'─'*70}")
        lines.append(f"  {'閾値':>5}  {'件数':>6}  {'勝率':>6}  {'平均%':>7}  {'期待値':>7}  {'予測力指数':>10}")
        lines.append("  " + "─" * 55)

        for thr in [3.0, 4.0, 5.0, 6.0, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]:
            sub = da[da["score"] >= thr]
            if len(sub) < 10:
                continue
            count   = len(sub)
            wr      = (sub["ret_oc"] > 0).mean() * 100
            avg_r   = sub["ret_oc"].mean()
            # 予測力指数 = 勝率 × 平均利益 / 平均損失（修正勝率ベース）
            wins_r  = sub[sub["ret_oc"] > 0]["ret_oc"].mean() if (sub["ret_oc"] > 0).any() else 0
            loss_r  = abs(sub[sub["ret_oc"] < 0]["ret_oc"].mean()) if (sub["ret_oc"] < 0).any() else 1
            pred_idx = (wr / 100) * wins_r / loss_r if loss_r > 0 else 0
            lines.append(
                f"  {thr:>5.1f}  {count:>6}  {wr:>5.1f}%  {avg_r:>+6.2f}%  "
                f"{avg_r:>+6.2f}%  {pred_idx:>9.3f}"
            )

    lines.append(f"\n{'='*70}")
    lines.append(f"  ※ 「翌日寄付き買い・引け売り」の単純シミュレーション")
    lines.append(f"  ※ 手数料・スプレッド・スリッページは未考慮")
    lines.append(f"  ※ 地合い（PANIC/WEAK）での非推奨フィルタは未適用")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print("\n" + report)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n保存: {REPORT_TXT}")


if __name__ == "__main__":
    main()
