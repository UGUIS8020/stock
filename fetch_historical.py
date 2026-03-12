import jquantsapi
from dotenv import load_dotenv
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import sys
sys.stdout.reconfigure(encoding='utf-8')


load_dotenv()
cli = jquantsapi.ClientV2(api_key=os.getenv("JQUANTS_API_KEY"))

# ── 設定 ──
MONTHS_BACK     = 6       # 履歴取得期間
HISTORY_MONTHS  = 3       # 遡る期間（3ヶ月）
TOP_N           = 20
SURGE_THRESHOLD = 10.0
CACHE_DIR       = "out/cache"   # 銘柄ごとの履歴キャッシュ
ACCUM_CSV       = "out/accumulated_surge.csv"

SIGNAL1_RATIO_MIN = 2.0
SIGNAL2_TREND_MIN = 5.0
SIGNAL2_ACCEL_MIN = 100.0
SIGNAL2_SCORE_MIN = 6.0

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("out", exist_ok=True)


def get_trading_days(start_date: datetime, end_date: datetime) -> list[str]:
    """土日祝を除いた営業日リストを返す（簡易版：土日のみ除外）"""
    days = []
    cur = start_date
    while cur <= end_date:
        if cur.weekday() < 5:  # 月〜金
            days.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return days


def get_stop_limit(price: float) -> float:
    limits = [
        (100,30),(200,50),(500,80),(700,100),(1000,150),(1500,300),
        (2000,400),(3000,500),(5000,700),(7000,1000),(10000,1500),
        (15000,3000),(20000,4000),(30000,5000),(50000,7000),
        (70000,10000),(100000,15000),
    ]
    for threshold, limit in limits:
        if price < threshold:
            return limit
    return price * 0.3


def is_stop_high(row):
    prev = row.get("prev_close")
    if prev is None or pd.isna(prev) or prev <= 0:
        return False
    return abs(row["Close"] - (prev + get_stop_limit(prev))) <= 1


def fetch_top20_for_date(date_str: str) -> pd.DataFrame:
    """指定日のTOP20銘柄を取得（429エラー時は自動リトライ）"""
    for attempt in range(4):
        try:
            if attempt > 0:
                wait = 60 * attempt
                print(f"\n    ⏳ 429エラー: {wait}秒待機してリトライ({attempt}/3)...", end=" ", flush=True)
                time.sleep(wait)
            df = cli.get_eq_bars_daily(date_yyyymmdd=date_str)
            if df.empty:
                return pd.DataFrame()
            df = df.rename(columns={"O":"Open","H":"High","L":"Low","C":"Close","Vo":"Volume"})
            df = df[df["Open"] > 0].copy()
            df["surge_%"] = (df["Close"] - df["Open"]) / df["Open"] * 100
            df["code4"] = df["Code"].astype(str).str[:4]
            df = df[df["code4"].str.match(r"^\d{4}$") | df["code4"].str.match(r"^\d{3}[A-Z]$")]
            df = df[(df["Volume"] > 0) & (df["Close"] >= 50)]
            df = df[df["surge_%"] <= 300]
            top20 = df.nlargest(TOP_N, "surge_%")[["Code","code4","Close","Open","Volume","surge_%"]].copy()
            return top20
        except Exception as e:
            if attempt == 3:
                print(f"    取得失敗（3回リトライ後）: {e}")
                return pd.DataFrame()
    return pd.DataFrame()


def fetch_stock_history_cached(code4: str, from_date: str, to_date: str) -> pd.DataFrame | None:
    """キャッシュ付きで6ヶ月履歴を取得"""
    cache_path = f"{CACHE_DIR}/{code4}.csv"

    # キャッシュが存在し最新なら読み込む
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        cached["Date"] = pd.to_datetime(cached["Date"])
        latest = cached["Date"].max().strftime("%Y%m%d")
        if latest >= to_date:
            return cached

    # APIから取得
    try:
        code5 = code4 + "0" if len(code4) == 4 and code4.isdigit() else code4
        df = cli.get_eq_bars_daily(code=code5, from_yyyymmdd=from_date, to_yyyymmdd=to_date)
        if df.empty:
            return None
        df = df.rename(columns={"O":"Open","H":"High","L":"Low","C":"Close","Vo":"Volume"})
        df["Date"]       = pd.to_datetime(df["Date"])
        df               = df.sort_values("Date").reset_index(drop=True)
        df["prev_close"] = df["Close"].shift(1)
        df["stop_high"]  = df.apply(is_stop_high, axis=1)
        df.to_csv(cache_path, index=False)
        return df
    except Exception:
        return None


def calc_scores(before: pd.DataFrame) -> dict:
    if len(before) < 22:
        return {}
    vol_20d   = before["Volume"].iloc[-20:].values
    vol_5d    = before["Volume"].iloc[-5:].values
    vol_prev1 = float(before["Volume"].iloc[-1])
    vol_prev3 = float(before["Volume"].iloc[-3])
    avg_20d   = vol_20d.mean()
    avg_5d    = vol_5d.mean()

    trend = np.polyfit(range(5), vol_5d, 1)[0] / avg_5d * 100 if avg_5d > 0 else 0
    accel = (vol_prev1 - vol_prev3) / vol_prev3 * 100 if vol_prev3 > 0 else 0
    ratio = vol_prev1 / avg_20d if avg_20d > 0 else 0
    total = min(trend/10, 3.0) + min(accel/100, 3.0) + min(ratio/3, 4.0)

    return {"trend":round(trend,2), "accel":round(accel,2),
            "ratio":round(ratio,2), "total":round(total,2)}


def judge_signals(s: dict) -> tuple[bool, bool]:
    day1 = (s["ratio"] >= SIGNAL1_RATIO_MIN and s["total"] > 0)
    day2 = (s["trend"] >= SIGNAL2_TREND_MIN and
            s["accel"] >= SIGNAL2_ACCEL_MIN  and
            s["total"] >= SIGNAL2_SCORE_MIN)
    return day1, day2


def print_stats(df: pd.DataFrame):
    total_days   = df["signal_date"].nunique()
    total_stocks = len(df)
    # 翌日データが確定している行のみで統計
    d1 = df[(df["day1"] == True) & (df["next_surge_%"].notna())]
    d2 = df[(df["day2"] == True) & (df["next_surge_%"].notna())]

    print(f"\n=== 📊 累積学習データ統計（翌日上昇率ベース）===")
    print(f"  蓄積日数   : {total_days}日")
    print(f"  総サンプル : {total_stocks}銘柄")

    if len(d1) > 0:
        hit = (d1["next_surge_%"] >= SURGE_THRESHOLD).sum()
        print(f"\n  🌱 1日目シグナル")
        print(f"    発火回数    : {len(d1)}回")
        print(f"    急騰成功率  : {hit}/{len(d1)}回 ({hit/len(d1)*100:.1f}%)")
        print(f"    翌日平均    : {d1['next_surge_%'].mean():.2f}%")
        print(f"    翌日中央値  : {d1['next_surge_%'].median():.2f}%")

    if len(d2) > 0:
        hit = (d2["next_surge_%"] >= SURGE_THRESHOLD).sum()
        print(f"\n  🚀 2日目シグナル")
        print(f"    発火回数    : {len(d2)}回")
        print(f"    急騰成功率  : {hit}/{len(d2)}回 ({hit/len(d2)*100:.1f}%)")
        print(f"    翌日平均    : {d2['next_surge_%'].mean():.2f}%")
        print(f"    翌日中央値  : {d2['next_surge_%'].median():.2f}%")


def main():
    today     = datetime.now()
    start     = today - relativedelta(months=HISTORY_MONTHS)
    from_date = (today - relativedelta(months=MONTHS_BACK)).strftime("%Y%m%d")
    to_date   = today.strftime("%Y%m%d")

    trading_days = get_trading_days(start, today - timedelta(days=1))  # 今日は除く
    print(f"=== 過去{HISTORY_MONTHS}ヶ月分の履歴データ取得 ===")
    print(f"  対象期間: {trading_days[0]} 〜 {trading_days[-1]}")
    print(f"  営業日数: {len(trading_days)}日")
    print(f"  想定銘柄数: 最大{len(trading_days) * TOP_N}件\n")

    # 既存の蓄積データを読み込む
    if os.path.exists(ACCUM_CSV):
        existing = pd.read_csv(ACCUM_CSV, encoding="utf-8-sig")
        done_dates = set(existing["signal_date"].unique())
        print(f"  既存データ: {len(existing)}件（{len(done_dates)}日分）\n")
    else:
        existing = pd.DataFrame()
        done_dates = set()

    all_rows = []
    api_calls = 0

    for day_i, date_str in enumerate(trading_days):
        date_disp = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

        # 既に取得済みの日はスキップ
        if date_disp in done_dates:
            print(f"  [{day_i+1:>3}/{len(trading_days)}] {date_disp} スキップ（取得済み）")
            continue

        print(f"  [{day_i+1:>3}/{len(trading_days)}] {date_disp} TOP{TOP_N}取得中...", end=" ", flush=True)

        # TOP20取得
        top20 = fetch_top20_for_date(date_str)
        api_calls += 1
        time.sleep(0.5)

        if top20.empty:
            print("データなし（祝日？）")
            continue

        surge_range = f"{top20['surge_%'].min():.1f}%〜{top20['surge_%'].max():.1f}%"
        print(f"上昇率 {surge_range}")

        # 各銘柄のスコアリング
        day_rows = []
        for row in top20.itertuples():
            code4 = row.code4

            df = fetch_stock_history_cached(code4, from_date, to_date)
            if df is None:
                continue
            api_calls += 1

            # その日より前のデータでスコア計算
            before = df[df["Date"] < pd.Timestamp(date_disp)]
            s = calc_scores(before)
            if not s:
                continue

            day1, day2 = judge_signals(s)

            # 翌日の上昇率を記録（シグナルの予測精度を測る）
            today_rows_df  = df[df["Date"] == pd.Timestamp(date_disp)]
            future_rows    = df[df["Date"] > pd.Timestamp(date_disp)]
            
            # 当日ストップ高判定
            stop = bool(today_rows_df.iloc[0]["stop_high"]) if not today_rows_df.empty else False
            
            # 翌日上昇率（翌営業日の終値-始値/始値）
            if not future_rows.empty:
                n = future_rows.iloc[0]
                next_surge = round((n["Close"] - n["Open"]) / n["Open"] * 100, 2) if n["Open"] > 0 else None
                next_date  = n["Date"].strftime("%Y-%m-%d")
            else:
                next_surge = None
                next_date  = None

            day_rows.append({
                "signal_date":    date_disp,
                "next_date":      next_date,
                "code":           code4,
                "score":          s["total"],
                "trend":          s["trend"],
                "accel":          s["accel"],
                "ratio":          s["ratio"],
                "day1":           day1,
                "day2":           day2,
                "next_surge_%":   next_surge,
                "stop_high":      stop,
            })
            time.sleep(1.0)

        all_rows.extend(day_rows)
        print(f"    → {len(day_rows)}銘柄スコアリング完了（API呼び出し累計: {api_calls}回）")

    # 蓄積データに保存
    if all_rows:
        new_df   = pd.DataFrame(all_rows)
        combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
        combined = combined.drop_duplicates(subset=["signal_date","code"], keep="last")
        combined = combined.sort_values("signal_date").reset_index(drop=True)
        combined.to_csv(ACCUM_CSV, index=False, encoding="utf-8-sig")
        print(f"\n✅ {len(all_rows)}件追加（累積: {len(combined)}件 / {combined['date'].nunique()}日分）")
        print_stats(combined)
    else:
        print("\n新規データなし")
        if not existing.empty:
            print_stats(existing)


if __name__ == "__main__":
    main()
