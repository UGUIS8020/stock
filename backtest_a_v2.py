"""
戦略Aの改良版バックテスト
条件: スコア5〜8.5 / 加速度1000%以下 / 当日上昇+8%以下 / 売買代金10億以上 / 株価300円以上
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

CACHE_DIR   = "out/cache"
MONTHS_BACK = 3

# ── 改良条件 ──
SCORE_MIN    = 5.0
SCORE_MAX    = 8.5
ACCEL_MAX    = 1000.0   # 加速度上限
SURGE_MAX    = 8.0      # 当日上昇率上限
TURNOVER_MIN = 1_000_000_000  # 売買代金10億以上
PRICE_MIN    = 300      # 株価300円以上
TOP_N        = 20

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

def get_top20_for_date(date_str, all_cache):
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

    # ── 旧バージョン（スコア3以上）と新バージョンを同時比較 ──
    results_old = []
    results_new = []

    for i in range(len(days) - 1):
        scan_date = days[i]
        next_date = days[i + 1]

        next_top20 = get_top20_for_date(next_date, all_cache)
        if not next_top20:
            continue

        cands_old = []
        cands_new = []

        for code4, hist in all_cache.items():
            # 当日データ
            today_row = hist[hist["Date"] == pd.Timestamp(scan_date)]
            if today_row.empty:
                continue
            t = today_row.iloc[0]
            if t["Open"] <= 0 or t["Close"] < PRICE_MIN:
                continue

            surge    = (t["Close"] - t["Open"]) / t["Open"] * 100
            turnover = t["Volume"] * t["Close"]

            ht = hist[hist["Date"] <= pd.Timestamp(scan_date)]
            s  = calc_score(ht)
            if s is None:
                continue

            # 旧: スコア3以上
            if s["score"] >= 3.0:
                cands_old.append({"code": code4, **s})

            # 新: 改良条件
            if not (SCORE_MIN <= s["score"] <= SCORE_MAX):
                continue
            if s["accel"] > ACCEL_MAX:
                continue
            if surge > SURGE_MAX:
                continue
            if turnover < TURNOVER_MIN:
                continue
            cands_new.append({"code": code4, **s, "surge": surge})

        # 旧TOP20
        if cands_old:
            df_old = pd.DataFrame(cands_old).nlargest(TOP_N, "score")
            for _, r in df_old.iterrows():
                results_old.append({
                    "scan_date": scan_date,
                    "code": r["code"],
                    "score": r["score"],
                    "hit": r["code"] in next_top20
                })

        # 新TOP20
        if cands_new:
            df_new = pd.DataFrame(cands_new).nlargest(TOP_N, "score")
            for _, r in df_new.iterrows():
                results_new.append({
                    "scan_date": scan_date,
                    "code": r["code"],
                    "score": r["score"],
                    "accel": r["accel"],
                    "surge": r["surge"],
                    "hit": r["code"] in next_top20
                })

    # ── 結果比較 ──
    df_o = pd.DataFrame(results_old)
    df_n = pd.DataFrame(results_new)

    print("=== 📊 戦略A バックテスト比較 ===\n")

    for label, df in [("【旧】スコア3以上（現行）", df_o), ("【新】改良版", df_n)]:
        if df.empty:
            print(f"{label}: データなし")
            continue
        hit  = df["hit"].sum()
        total = len(df)
        days_cnt = df["scan_date"].nunique()
        print(f"{label}")
        print(f"  候補総数: {total}件  日数: {days_cnt}日")
        print(f"  TOP20命中: {hit}件  命中率: {hit/total*100:.1f}%")
        print(f"  ランダム比: {(hit/total)/(20/4422)*100:.0f}%")
        print()

    # スコア帯別（新）
    if not df_n.empty:
        print("【新・スコア帯別命中率】")
        df_n["band"] = pd.cut(df_n["score"],
            bins=[5,6,7,8,8.5], labels=["5-6","6-7","7-8","8-8.5"])
        grp = df_n.groupby("band", observed=True).agg(
            件数=("hit","count"),
            命中=("hit","sum")).reset_index()
        grp["率"] = grp["命中"] / grp["件数"] * 100
        for _, row in grp.iterrows():
            print(f"  スコア{row['band']}: {int(row['件数']):>5}件  "
                  f"命中:{int(row['命中']):>3}件({row['率']:>5.1f}%)  "
                  f"{'█'*int(row['率']/2)}")

if __name__ == "__main__":
    main()