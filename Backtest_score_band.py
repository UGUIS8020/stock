"""
スコア帯別の翌日TOP20命中率を検証
当日データ含む（<=）で計算
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

CACHE_DIR   = "out/cache"
MONTHS_BACK = 3
MIN_VOLUME  = 50_000
MIN_TURNOVER= 50_000_000

def get_trading_days(months):
    today = datetime.now()
    start = today - relativedelta(months=months)
    days, cur = [], start
    while cur <= today - timedelta(days=1):
        if cur.weekday() < 5:
            days.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return days

def calc_score(hist):
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
    total = min(trend/10, 3.0) + min(accel/100, 3.0) + min(ratio/3, 4.0)
    return {"trend": round(trend,2), "accel": round(accel,2),
            "ratio": round(ratio,2), "score": round(total,2)}

def get_top20(date_str, all_cache):
    top = []
    for code4, hist in all_cache.items():
        row = hist[hist["Date"] == pd.Timestamp(date_str)]
        if row.empty or row.iloc[0]["Open"] <= 0:
            continue
        r = row.iloc[0]
        surge = (r["Close"] - r["Open"]) / r["Open"] * 100
        top.append((code4, surge))
    if not top:
        return set()
    top.sort(key=lambda x: x[1], reverse=True)
    return set(c for c, _ in top[:20])

def main():
    print("=== キャッシュ読み込み中... ===")
    all_cache = {}
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".csv")]
    for i, fname in enumerate(files):
        code4 = fname.replace(".csv", "")
        hist  = pd.read_csv(f"{CACHE_DIR}/{fname}")
        hist["Date"] = pd.to_datetime(hist["Date"])
        all_cache[code4] = hist.sort_values("Date").reset_index(drop=True)
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(files)} 読み込み済み...")
    print(f"  {len(all_cache)}銘柄完了\n")

    days = get_trading_days(MONTHS_BACK)
    all_results = []

    for i in range(len(days) - 1):
        scan_date = days[i]
        next_date = days[i + 1]
        next_top20 = get_top20(next_date, all_cache)
        if not next_top20:
            continue

        for code4, hist in all_cache.items():
            # 当日データ
            today_row = hist[hist["Date"] == pd.Timestamp(scan_date)]
            if today_row.empty:
                continue
            t = today_row.iloc[0]
            if t["Open"] <= 0:
                continue

            vol      = float(t["Volume"])
            turnover = vol * float(t["Close"])
            surge    = (t["Close"] - t["Open"]) / t["Open"] * 100

            # 流動性フィルター
            if vol < MIN_VOLUME or turnover < MIN_TURNOVER:
                continue

            # スコア計算（当日含む）
            ht = hist[hist["Date"] <= pd.Timestamp(scan_date)]
            s  = calc_score(ht)
            if s is None or s["score"] < 3.0:
                continue

            all_results.append({
                "scan_date": scan_date,
                "code":      code4,
                "score":     s["score"],
                "ratio":     s["ratio"],
                "accel":     s["accel"],
                "surge":     round(surge, 2),
                "hit":       code4 in next_top20,
            })

    df = pd.DataFrame(all_results)
    print(f"=== 📊 スコア帯別・翌日TOP20命中率 ===")
    print(f"  総データ数: {len(df)}件  日数: {df['scan_date'].nunique()}日\n")

    # スコア帯を細かく分割
    bins   = [3, 4, 5, 6, 7, 8, 8.5, 9, 9.5, 10.01]
    labels = ["3-4", "4-5", "5-6", "6-7", "7-8", "8-8.5", "8.5-9", "9-9.5", "9.5-10"]
    df["band"] = pd.cut(df["score"], bins=bins, labels=labels, right=False)

    grp = df.groupby("band", observed=True).agg(
        件数=("hit","count"),
        命中=("hit","sum"),
        平均surge=("surge","mean"),
        平均ratio=("ratio","mean"),
    ).reset_index()
    grp["命中率"] = grp["命中"] / grp["件数"] * 100
    grp["ランダム比"] = grp["命中率"] / (20/4422*100)

    print(f"  {'スコア帯':<10} {'件数':>5} {'命中':>5} {'命中率':>7} {'ランダム比':>8}  グラフ")
    print("  " + "─" * 60)
    for _, r in grp.iterrows():
        bar = "█" * int(r["命中率"] * 3)
        rand_str = f"{r['ランダム比']:.1f}x"
        print(f"  {r['band']:<10} {int(r['件数']):>5}件 {int(r['命中']):>4}件 "
              f"{r['命中率']:>6.1f}%  {rand_str:>7}  {bar}")

    # 当日上昇率帯別
    print(f"\n【当日上昇率帯別・翌日TOP20命中率（スコア3以上全体）】")
    df["surge_band"] = pd.cut(df["surge"],
        bins=[-999, 0, 3, 5, 8, 15, 999],
        labels=["0%未満", "0〜3%", "3〜5%", "5〜8%", "8〜15%", "15%超"])
    grp2 = df.groupby("surge_band", observed=True).agg(
        件数=("hit","count"),
        命中=("hit","sum")).reset_index()
    grp2["命中率"] = grp2["命中"] / grp2["件数"] * 100
    print(f"  {'当日上昇率':<10} {'件数':>5} {'命中':>5} {'命中率':>7}  グラフ")
    print("  " + "─" * 50)
    for _, r in grp2.iterrows():
        bar = "█" * int(r["命中率"] * 3)
        print(f"  {r['surge_band']:<10} {int(r['件数']):>5}件 {int(r['命中']):>4}件 "
              f"{r['命中率']:>6.1f}%  {bar}")

    # 最適ゾーン
    best = grp[grp["命中率"] == grp["命中率"].max()].iloc[0]
    print(f"\n🏆 最も命中率が高いスコア帯: {best['band']} ({best['命中率']:.1f}%・ランダムの{best['ランダム比']:.1f}倍)")

if __name__ == "__main__":
    main()