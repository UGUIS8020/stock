import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

CACHE_DIR      = "out/cache"
RESULT_CSV     = "out/backtest_results_v7.csv"
MONTHS_BACK    = 3
TOP_N          = 20
SCORE_MIN      = 3.0
SCORE_MAX      = 6.0
PRICE_DROP_MAX = -10.0

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

def get_today_data(hist, scan_date):
    """当日の始値・終値・騰落率を返す"""
    row = hist[hist["Date"] == pd.Timestamp(scan_date)]
    if row.empty or row.iloc[0]["Open"] <= 0:
        return None
    n = row.iloc[0]
    rise = round((n["Close"] - n["Open"]) / n["Open"] * 100, 2)
    return {"open": n["Open"], "close": n["Close"], "rise": rise}

def get_next_data(code4, next_date, all_cache):
    """翌日の始値・終値を返す"""
    if code4 not in all_cache:
        return None
    hist = all_cache[code4]
    row  = hist[hist["Date"] == pd.Timestamp(next_date)]
    if row.empty or row.iloc[0]["Open"] <= 0:
        return None
    n = row.iloc[0]
    return {"open": n["Open"], "close": n["Close"]}

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
    print(f"=== バックテスト開始（当日終値買いvs翌日始値買い比較）===")
    print(f"  対象期間: {days[0]} - {days[-1]}")
    print(f"  条件: 当日価格{PRICE_DROP_MAX}%以下 かつ スコア{SCORE_MIN}〜{SCORE_MAX}\n")

    all_results = []

    for i in range(len(days) - 1):
        scan_date = days[i]
        next_date = days[i + 1]

        candidates = []
        for code4, hist in all_cache.items():
            today = get_today_data(hist, scan_date)
            if today is None or today["rise"] > PRICE_DROP_MAX:
                continue

            ht = hist[hist["Date"] <= pd.Timestamp(scan_date)]
            s  = calc_score(ht)
            if s is None or not (SCORE_MIN <= s["score"] <= SCORE_MAX):
                continue

            candidates.append({
                "code":        code4,
                "today_rise":  today["rise"],
                "today_close": today["close"],
                **s
            })

        if not candidates:
            continue

        cand_df = pd.DataFrame(candidates).nlargest(TOP_N, "score")

        for _, r in cand_df.iterrows():
            nxt = get_next_data(r["code"], next_date, all_cache)
            if nxt is None:
                continue

            # パターンA: 当日終値で買い → 翌日終値で売り（当日15:00買い）
            rise_a = round((nxt["close"] - r["today_close"]) / r["today_close"] * 100, 2)
            # パターンB: 翌日始値で買い → 翌日終値で売り（翌朝成行買い）
            rise_b = round((nxt["close"] - nxt["open"]) / nxt["open"] * 100, 2) if nxt["open"] > 0 else None
            # パターンC: 当日終値で買い → 翌日始値で売り（翌朝寄付き売り）
            rise_c = round((nxt["open"] - r["today_close"]) / r["today_close"] * 100, 2)

            all_results.append({
                "scan_date":   scan_date,
                "next_date":   next_date,
                "code":        r["code"],
                "score":       r["score"],
                "today_rise":  r["today_rise"],
                "today_close": r["today_close"],
                "next_open":   nxt["open"],
                "next_close":  nxt["close"],
                # 3パターンの損益
                "rise_A":  rise_a,   # 当日終値買い→翌日終値売り
                "rise_B":  rise_b,   # 翌日始値買い→翌日終値売り
                "rise_C":  rise_c,   # 当日終値買い→翌日始値売り
            })

    df = pd.DataFrame(all_results)
    df.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")

    print(f"=== 📊 総合結果（{len(df)}件）===\n")

    for pat, col, desc in [
        ("A", "rise_A", "当日終値買い → 翌日終値売り（15:00買い・翌日引け売り）"),
        ("B", "rise_B", "翌日始値買い → 翌日終値売り（翌朝成行買い・翌日引け売り）"),
        ("C", "rise_C", "当日終値買い → 翌日始値売り（15:00買い・翌朝寄付き売り）"),
    ]:
        r = df[col].dropna()
        hit5  = (r >= 5).sum()
        lose5 = (r < -5).sum()
        avg_win  = r[r >= 5].mean() if hit5 > 0 else 0
        avg_lose = r[r < -5].mean() if lose5 > 0 else 0
        ev = (hit5/len(r)) * avg_win + (lose5/len(r)) * avg_lose

        print(f"【パターン{pat}】{desc}")
        print(f"  平均: {r.mean():+.2f}%  中央値: {r.median():+.2f}%")
        print(f"  +5%達成: {hit5}件({hit5/len(r)*100:.1f}%)  平均{avg_win:+.1f}%")
        print(f"  -5%以下: {lose5}件({lose5/len(r)*100:.1f}%)  平均{avg_lose:+.1f}%")
        print(f"  期待値 : {ev:+.2f}%/トレード")

        for label, mask in [
            ("+10%以上",  r >= 10),
            ("+5〜10%",  (r >= 5) & (r < 10)),
            ("+3〜 5%",  (r >= 3) & (r < 5)),
            ("0〜 3%",   (r >= 0) & (r < 3)),
            ("-5〜 0%",  (r >= -5) & (r < 0)),
            ("-5%以下",  r < -5),
        ]:
            cnt = mask.sum()
            pct = cnt / len(r) * 100
            avg = r[mask].mean() if cnt > 0 else 0
            print(f"  {label:<10} {cnt:>4}件 {pct:>5.1f}%  平均{avg:>+6.1f}%  {'█'*int(pct/3)}")
        print()

    print(f"保存: {RESULT_CSV}")

if __name__ == "__main__":
    main()
