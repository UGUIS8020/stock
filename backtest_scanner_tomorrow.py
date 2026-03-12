import jquantsapi
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

load_dotenv()

CACHE_DIR      = "out/cache"
RESULT_CSV     = "out/backtest_results_v6.csv"
MONTHS_BACK    = 3
TOP_N          = 20
SCORE_MIN      = 3.0
SCORE_MAX      = 6.0   # スコア上限（高すぎるパニック売りを除外）
PRICE_DROP_MAX = -10.0  # 当日-10%以下の下落のみ

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

def get_today_rise(hist, scan_date):
    row = hist[hist["Date"] == pd.Timestamp(scan_date)]
    if row.empty or row.iloc[0]["Open"] <= 0:
        return None
    n = row.iloc[0]
    return round((n["Close"] - n["Open"]) / n["Open"] * 100, 2)

def get_next_rise(code4, next_date, all_cache):
    if code4 not in all_cache:
        return None
    hist = all_cache[code4]
    row  = hist[hist["Date"] == pd.Timestamp(next_date)]
    if row.empty or row.iloc[0]["Open"] <= 0:
        return None
    n = row.iloc[0]
    return round((n["Close"] - n["Open"]) / n["Open"] * 100, 2)

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
    print(f"  {len(all_cache)}銘柄のキャッシュ完了\n")

    days = get_trading_days(MONTHS_BACK)
    print(f"=== バックテスト開始 ===")
    print(f"  対象期間: {days[0]} - {days[-1]}")
    print(f"  条件: 当日価格{PRICE_DROP_MAX}%以下 かつ 出来高スコア{SCORE_MIN}〜{SCORE_MAX}\n")

    all_results = []

    for i in range(len(days) - 1):
        scan_date = days[i]
        next_date = days[i + 1]

        candidates = []
        for code4, hist in all_cache.items():
            today_rise = get_today_rise(hist, scan_date)
            if today_rise is None or today_rise > PRICE_DROP_MAX:
                continue

            ht = hist[hist["Date"] <= pd.Timestamp(scan_date)]
            s  = calc_score(ht)
            if s is None:
                continue
            if not (SCORE_MIN <= s["score"] <= SCORE_MAX):
                continue

            candidates.append({
                "code":       code4,
                "today_rise": today_rise,
                **s
            })

        if not candidates:
            continue

        cand_df = pd.DataFrame(candidates).nlargest(TOP_N, "score")

        for _, r in cand_df.iterrows():
            rise = get_next_rise(r["code"], next_date, all_cache)
            all_results.append({
                "scan_date":  scan_date,
                "next_date":  next_date,
                "code":       r["code"],
                "score":      r["score"],
                "ratio":      r["ratio"],
                "accel":      r["accel"],
                "trend":      r["trend"],
                "today_rise": r["today_rise"],
                "next_rise":  rise,
            })

        verified = [x for x in all_results
                    if x["scan_date"] == scan_date and x["next_rise"] is not None]
        hit5 = sum(1 for x in verified if x["next_rise"] >= 5)
        avg  = np.mean([x["next_rise"] for x in verified]) if verified else 0
        print(f"  {scan_date} → {next_date}: "
              f"候補{len(cand_df)}件  "
              f"+5%達成:{hit5}件/{len(verified)}件  "
              f"平均:{avg:+.1f}%")

    df = pd.DataFrame(all_results)
    df = df[df["next_rise"].notna()]
    df.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")

    r = df["next_rise"]
    print(f"\n=== 📊 総合結果 ===")
    print(f"  日数: {df['scan_date'].nunique()}日  候補: {len(df)}件")
    print(f"  翌日平均上昇率: {r.mean():+.2f}%")
    print(f"  翌日中央値    : {r.median():+.2f}%\n")

    print("【翌日上昇率の分布】")
    for label, mask in [
        ("大きく上昇 +10%以上",  r >= 10),
        ("上昇     +7〜10%",    (r >= 7)  & (r < 10)),
        ("上昇     +5〜 7%",    (r >= 5)  & (r < 7)),
        ("小幅上昇  +3〜 5%",   (r >= 3)  & (r < 5)),
        ("微上昇    0〜 3%",    (r >= 0)  & (r < 3)),
        ("小幅下落 -5〜 0%",    (r >= -5) & (r < 0)),
        ("大きく下落 -5%以下",   r < -5),
    ]:
        cnt = mask.sum()
        pct = cnt / len(r) * 100
        avg = r[mask].mean() if cnt > 0 else 0
        print(f"  {label:<22} {cnt:>4}件 {pct:>5.1f}%  平均{avg:>+6.1f}%  {'█'*int(pct/2)}")

    print("\n【閾値別の達成率】")
    for t in [3, 5, 7, 10]:
        hit = (r >= t).sum()
        pct = hit / len(r) * 100
        print(f"  +{t}%以上  {hit:>5}件  {pct:>5.1f}%  {'█'*int(pct/2)}")

    print("\n【当日下落率帯別の翌日結果】")
    df["today_band"] = pd.cut(df["today_rise"],
        bins=[-99, -20, -15, -10],
        labels=["-20%以下", "-20〜-15%", "-15〜-10%"])
    grp = df.groupby("today_band", observed=True).agg(
        件数=("next_rise","count"),
        命中=("next_rise", lambda x: (x >= 5).sum()),
        平均=("next_rise","mean")).reset_index()
    grp["率"] = grp["命中"] / grp["件数"] * 100
    for _, row in grp.iterrows():
        print(f"  当日{row['today_band']}: {int(row['件数']):>5}件  "
              f"+5%達成:{int(row['命中']):>3}件({row['率']:>5.1f}%)  "
              f"平均:{row['平均']:>+5.1f}%  {'█'*int(row['率']/2)}")

    # 期待値計算
    hit5 = (r >= 5).sum()
    lose5 = (r < -5).sum()
    avg_win  = r[r >= 5].mean() if hit5 > 0 else 0
    avg_lose = r[r < -5].mean() if lose5 > 0 else 0
    win_rate = hit5 / len(r)
    lose_rate = lose5 / len(r)
    ev = win_rate * avg_win + lose_rate * avg_lose
    print(f"\n【期待値シミュレーション】")
    print(f"  +5%以上達成率 : {win_rate*100:.1f}%  平均 {avg_win:+.1f}%")
    print(f"  -5%以下下落率 : {lose_rate*100:.1f}%  平均 {avg_lose:+.1f}%")
    print(f"  単純期待値    : {ev:+.2f}%/トレード")

    print(f"\n保存: {RESULT_CSV}")

if __name__ == "__main__":
    main()
