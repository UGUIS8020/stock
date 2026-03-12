"""
scan_today.py
16:30以降に実行して、当日-10%以下かつスコア3〜6の銘柄を表示する
翌朝9:00に成行注文で買い、翌日15:30に売却する
"""
import jquantsapi
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from datetime import datetime
import time

load_dotenv()
cli = jquantsapi.ClientV2(api_key=os.getenv("JQUANTS_API_KEY"))

CACHE_DIR      = "out/cache"
WATCHLIST_CSV  = "out/watchlist.csv"   # 監視リスト（翌日検証用）
SCORE_MIN      = 3.0
SCORE_MAX      = 6.0
PRICE_DROP_MAX = -10.0   # 当日価格-10%以下
TOP_N          = 10      # 表示する候補数

TODAY     = datetime.now().strftime("%Y-%m-%d")
TODAY_STR = datetime.now().strftime("%Y%m%d")

os.makedirs("out", exist_ok=True)

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

def verify_yesterday(today_df):
    """前日の監視リストの結果を検証"""
    if not os.path.exists(WATCHLIST_CSV):
        return
    wl = pd.read_csv(WATCHLIST_CSV, encoding="utf-8-sig")
    yesterday = wl[wl["buy_date"] == wl["buy_date"].max()]
    unverified = yesterday[yesterday["next_rise"].isna()]
    if unverified.empty:
        return

    print(f"=== 📋 前日({yesterday['buy_date'].max()})の結果検証 ===")
    updated = 0
    for idx, row in unverified.iterrows():
        code4 = str(row["code"])
        today_row = today_df[today_df["code4"] == code4]
        if today_row.empty:
            continue
        t = today_row.iloc[0]
        if t["Open"] <= 0:
            continue
        # パターンB: 翌日始値（寄付き）→ 翌日終値
        next_open  = t["Open"]
        next_close = t["Close"]
        if next_open <= 0:
            continue
        rise = round((next_close - next_open) / next_open * 100, 2)
        wl.at[idx, "next_rise"]   = rise
        wl.at[idx, "next_open"]   = next_open
        wl.at[idx, "next_close"]  = next_close
        status = "✅" if rise >= 5 else ("⚠️" if rise >= 0 else "❌")
        name = row.get("name", "")
        print(f"  {status} [{code4}]{name}  翌日: {rise:+.1f}%  "
              f"（寄付き:{next_open:.0f}円 → 終値:{next_close:.0f}円）")
        updated += 1

    if updated > 0:
        wl.to_csv(WATCHLIST_CSV, index=False, encoding="utf-8-sig")

        # 累積成績
        verified = wl[wl["next_rise"].notna()]
        hit = (verified["next_rise"] >= 5).sum()
        avg = verified["next_rise"].mean()
        print(f"\n  累積成績: {len(verified)}件  +5%達成:{hit}件({hit/len(verified)*100:.1f}%)  平均:{avg:+.2f}%")
    print()

def main():
    print(f"=== 🔍 当日スキャナー（{TODAY}）===")
    print(f"  条件: 当日価格{PRICE_DROP_MAX}%以下 かつ スコア{SCORE_MIN}〜{SCORE_MAX}\n")

    # 当日の株価データを取得
    print("  当日データ取得中...")
    try:
        df_today = cli.get_eq_bars_daily(date_yyyymmdd=TODAY_STR)
    except Exception as e:
        print(f"  エラー: {e}")
        return

    if df_today.empty:
        print("  データなし（市場時間外 or 休業日）")
        return

    df_today = df_today.rename(columns={"O":"Open","H":"High","L":"Low","C":"Close","Vo":"Volume"})
    df_today["code4"] = df_today["Code"].astype(str).str[:4]
    df_today = df_today[df_today["Volume"] > 0]
    df_today["today_rise"] = (df_today["Close"] - df_today["Open"]) / df_today["Open"] * 100
    print(f"  {len(df_today)}銘柄のデータ取得完了\n")

    # 前日の結果を検証
    verify_yesterday(df_today)

    # 銘柄名マスタ
    try:
        master = cli.get_eq_master()
        master["code4"] = master["Code"].astype(str).str[:4]
        name_dict = dict(zip(master["code4"], master["CompanyName"]))
    except Exception:
        name_dict = {}

    # 条件①: 当日-10%以下の銘柄を絞り込み
    drop_stocks = df_today[df_today["today_rise"] <= PRICE_DROP_MAX].copy()
    print(f"  当日{PRICE_DROP_MAX}%以下: {len(drop_stocks)}銘柄")

    if drop_stocks.empty:
        print("  該当銘柄なし")
        return

    # 条件②: 出来高スコアを計算
    candidates = []
    for _, row in drop_stocks.iterrows():
        code4 = row["code4"]
        cache_path = f"{CACHE_DIR}/{code4}.csv"
        if not os.path.exists(cache_path):
            continue

        hist = pd.read_csv(cache_path)
        hist["Date"] = pd.to_datetime(hist["Date"])
        hist = hist.sort_values("Date")

        # 前日までのデータでスコア計算
        hist_prev = hist[hist["Date"] < pd.Timestamp(TODAY)]
        s = calc_score(hist_prev)
        if s is None or not (SCORE_MIN <= s["score"] <= SCORE_MAX):
            continue

        name = name_dict.get(code4, "")
        candidates.append({
            "code":        code4,
            "name":        name,
            "today_rise":  round(row["today_rise"], 2),
            "buy_price":   row["Close"],   # 現在値（終値近似）
            "volume":      int(row["Volume"]),
            **s,
        })

    if not candidates:
        print("  スコア条件を満たす銘柄なし")
        return

    cand_df = pd.DataFrame(candidates).sort_values("today_rise")  # 下落率が大きい順

    print(f"  スコア条件クリア: {len(cand_df)}銘柄\n")
    print(f"=== 🎯 本日の買い候補（翌朝9:00に成行注文）===\n")
    print(f"  {'コード':<6} {'銘柄名':<16} {'当日騰落':>8} {'現在値':>8} {'スコア':>6} {'出来高比':>8}")
    print("  " + "─" * 60)

    display_df = cand_df.head(TOP_N)
    for _, r in display_df.iterrows():
        drop_icon = "💥" if r["today_rise"] <= -20 else ("🔻" if r["today_rise"] <= -15 else "↘️")
        print(f"  {r['code']:<6} {r['name']:<16} "
              f"{drop_icon}{r['today_rise']:>+6.1f}% "
              f"{r['buy_price']:>8.0f}円 "
              f"{r['score']:>6.2f} "
              f"{r['ratio']:>6.1f}倍")

    print(f"\n  💡 下落が深いほど反発期待値が高い傾向があります")
    print(f"  💡 翌朝9:00に成行注文（寄付き買い）→ 翌日15:30に売却\n")

    # 監視リストに保存
    save_rows = []
    for _, r in display_df.iterrows():
        save_rows.append({
            "buy_date":   TODAY,
            "code":       r["code"],
            "name":       r["name"],
            "today_rise": r["today_rise"],
            "buy_price":  r["buy_price"],
            "score":      r["score"],
            "ratio":      r["ratio"],
            "next_rise":  None,
            "next_open":  None,
            "next_close": None,
        })

    new_df = pd.DataFrame(save_rows)
    if os.path.exists(WATCHLIST_CSV):
        existing = pd.read_csv(WATCHLIST_CSV, encoding="utf-8-sig")
        existing = existing[existing["buy_date"] != TODAY]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(WATCHLIST_CSV, index=False, encoding="utf-8-sig")
    print(f"  ✅ 監視リスト保存: {WATCHLIST_CSV}")
    print(f"     ※翌日15:00以降に再実行すると結果が自動検証されます")

if __name__ == "__main__":
    main()