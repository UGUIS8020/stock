import jquantsapi
from dotenv import load_dotenv
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

load_dotenv()
cli = jquantsapi.ClientV2(api_key=os.getenv("JQUANTS_API_KEY"))

MONTHS_BACK     = 6
TOP_N           = 20       # 値上がりランキング上位何位まで取得するか
SURGE_THRESHOLD = 10.0     # 急騰とみなす上昇率（%）
ACCUM_CSV       = "out/accumulated_surge.csv"  # 蓄積データ保存先

# シグナル閾値
SIGNAL1_RATIO_MIN = 2.0
SIGNAL2_TREND_MIN = 5.0
SIGNAL2_ACCEL_MIN = 100.0
SIGNAL2_SCORE_MIN = 6.0

FROM_DATE  = (datetime.now() - relativedelta(months=MONTHS_BACK)).strftime("%Y%m%d")
TO_DATE    = datetime.now().strftime("%Y%m%d")
TODAY      = datetime.now().strftime("%Y-%m-%d")


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


def is_stop_high(row: pd.Series) -> bool:
    prev = row.get("prev_close")
    if prev is None or pd.isna(prev) or prev <= 0:
        return False
    return abs(row["Close"] - (prev + get_stop_limit(prev))) <= 1


def fetch_top20_codes(date_str: str) -> pd.DataFrame:
    """
    指定日の全銘柄株価を取得し、上昇率TOP20を返す
    date_str: YYYYMMDD形式
    """
    print(f"  全銘柄の当日株価を取得中（{date_str}）...")
    df = cli.get_eq_bars_daily(date_yyyymmdd=date_str)
    if df.empty:
        print("  データなし")
        return pd.DataFrame()

    df = df.rename(columns={"O":"Open","H":"High","L":"Low","C":"Close","Vo":"Volume"})

    # 上昇率を計算
    df = df[df["Open"] > 0].copy()
    df["surge_%"] = (df["Close"] - df["Open"]) / df["Open"] * 100

    # ETF・REITを除外（コードが5桁で末尾が1,2以外は除外）
    df["code4"] = df["Code"].astype(str).str[:4]
    df = df[df["code4"].str.match(r"^\d{4}$") | df["code4"].str.match(r"^\d{3}[A-Z]$")]

    # 上昇率TOP20を抽出
    top20 = df.nlargest(TOP_N, "surge_%")[["Code","Close","Open","Volume","surge_%"]].copy()
    top20["code4"] = top20["Code"].astype(str).str[:4]
    print(f"  TOP{TOP_N}銘柄を取得: 上昇率 {top20['surge_%'].min():.1f}%〜{top20['surge_%'].max():.1f}%")
    return top20


def fetch_stock_history(code4: str) -> pd.DataFrame | None:
    """6ヶ月分の株価履歴を取得"""
    try:
        code5 = code4 + "0" if len(code4) == 4 and code4.isdigit() else code4
        df = cli.get_eq_bars_daily(code=code5, from_yyyymmdd=FROM_DATE, to_yyyymmdd=TO_DATE)
        if df.empty:
            return None
        df = df.rename(columns={"O":"Open","H":"High","L":"Low","C":"Close","Vo":"Volume"})
        df["Date"]       = pd.to_datetime(df["Date"])
        df               = df.sort_values("Date").reset_index(drop=True)
        df["prev_close"] = df["Close"].shift(1)
        df["stop_high"]  = df.apply(is_stop_high, axis=1)
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


def load_accumulated() -> pd.DataFrame:
    """蓄積データを読み込む"""
    if os.path.exists(ACCUM_CSV):
        return pd.read_csv(ACCUM_CSV, encoding="utf-8-sig")
    return pd.DataFrame()


def save_accumulated(new_rows: list, today: str) -> pd.DataFrame:
    """新しいデータを蓄積CSVに追記保存"""
    existing = load_accumulated()

    new_df = pd.DataFrame(new_rows)

    if not existing.empty:
        # 同じ日付のデータは上書き（再実行対応）
        existing = existing[existing["date"] != today]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    os.makedirs("out", exist_ok=True)
    combined.to_csv(ACCUM_CSV, index=False, encoding="utf-8-sig")
    return combined


def print_accumulated_stats(df: pd.DataFrame):
    """蓄積データの統計を表示"""
    if df.empty:
        return

    total_days  = df["date"].nunique()
    total_stocks = len(df)

    d1 = df[df["day1"] == True]
    d2 = df[df["day2"] == True]

    print(f"\n=== 📊 累積学習データ統計 ===")
    print(f"  蓄積日数   : {total_days}日")
    print(f"  総サンプル : {total_stocks}銘柄")

    if len(d1) > 0:
        hit1 = (d1["actual_surge_%"] >= SURGE_THRESHOLD).sum()
        print(f"\n  🌱 1日目シグナル")
        print(f"    発火回数: {len(d1)}回 / 急騰成功: {hit1}回 ({hit1/len(d1)*100:.1f}%)")
        print(f"    翌日平均上昇率: {d1['actual_surge_%'].mean():.2f}%")
        print(f"    翌日中央値:     {d1['actual_surge_%'].median():.2f}%")

    if len(d2) > 0:
        hit2 = (d2["actual_surge_%"] >= SURGE_THRESHOLD).sum()
        print(f"\n  🚀 2日目シグナル")
        print(f"    発火回数: {len(d2)}回 / 急騰成功: {hit2}回 ({hit2/len(d2)*100:.1f}%)")
        print(f"    翌日平均上昇率: {d2['actual_surge_%'].mean():.2f}%")
        print(f"    翌日中央値:     {d2['actual_surge_%'].median():.2f}%")



def fetch_name_master() -> dict:
    """銘柄コード→会社名の辞書を返す"""
    try:
        df = cli.get_eq_master()
        df["code4"] = df["Code"].astype(str).str[:4]
        return dict(zip(df["code4"], df["CompanyName"]))
    except Exception:
        return {}

def main():
    today_str = datetime.now().strftime("%Y%m%d")
    print(f"=== 本日のTOP{TOP_N}自動取得＆スコアリング（{TODAY}）===\n")
    # 銘柄名マスタ取得
    name_master = fetch_name_master()
    def get_name(code): return name_master.get(code, "")

    # ── STEP1: 当日TOP20を自動取得 ──
    top20 = fetch_top20_codes(today_str)
    if top20.empty:
        print("TOP20取得失敗。終了します。")
        return

    print(f"\n{'順位':>4} {'コード':>6} {'終値':>8} {'上昇率':>8}  銘柄名")
    print("-" * 55)
    for i, row in enumerate(top20.itertuples(), 1):
        name = name_master.get(row.code4, "")
        print(f"  {i:>2}位  {row.code4:>6}  {row.Close:>8.0f}円  {row._5:>7.2f}%  {name}")

    # ── STEP2: 各銘柄の6ヶ月データ取得＆スコアリング ──
    print(f"\n=== 6ヶ月データ取得＆スコアリング ===\n")
    today_rows = []

    for row in top20.itertuples():
        code4 = row.code4
        df = fetch_stock_history(code4)
        if df is None:
            continue

        before = df[df["Date"] < pd.Timestamp(TODAY)]
        s = calc_scores(before)
        if not s:
            continue

        day1, day2 = judge_signals(s)

        # 当日の実際の上昇率
        surge_rows = df[df["Date"] >= pd.Timestamp(TODAY)]
        actual = round((surge_rows.iloc[0]["Close"] - surge_rows.iloc[0]["Open"]) /
                       surge_rows.iloc[0]["Open"] * 100, 2) if not surge_rows.empty else None
        stop = bool(surge_rows.iloc[0]["stop_high"]) if not surge_rows.empty else False

        today_rows.append({
            "date":          TODAY,
            "code":          code4,
            "score":         s["total"],
            "trend":         s["trend"],
            "accel":         s["accel"],
            "ratio":         s["ratio"],
            "day1":          day1,
            "day2":          day2,
            "actual_surge_%": actual,
            "stop_high":     stop,
        })

        d1 = "✅" if day1 else "❌"
        d2 = "✅" if day2 else "❌"
        act = f"{actual:+.1f}%" if actual is not None else "未取得"
        stp = "🔴ストップ高" if stop else ""
        name = get_name(code4)
        print(f"  [{code4}]{name:　<10} スコア:{s['total']:>5.2f}  1日目:{d1} 2日目:{d2}  上昇率:{act:>8}  {stp}")

        time.sleep(1.0)

    # ── STEP3: 蓄積データに保存 ──
    accumulated = save_accumulated(today_rows, TODAY)
    print(f"\n✅ 本日分 {len(today_rows)}銘柄を保存（累積: {len(accumulated)}件）")
    print(f"   保存先: {ACCUM_CSV}")

    # ── STEP4: 累積統計を表示 ──
    print_accumulated_stats(accumulated)

    # ── STEP5: 本日のシグナル銘柄を表示 ──
    signals = [r for r in today_rows if r["day1"] or r["day2"]]
    if signals:
        print(f"\n=== 🎯 本日のシグナル銘柄（明日の候補）===")
        for r in sorted(signals, key=lambda x: x["score"], reverse=True):
            d1 = "🌱1日目" if r["day1"] else ""
            d2 = "🚀2日目" if r["day2"] else ""
            tags = " ".join(filter(None, [d1, d2]))
            name = get_name(r["code"])
            print(f"  [{r['code']}] {name}  スコア:{r['score']:>5.2f}  {tags}")
    else:
        print("\n本日はシグナル銘柄なし")


if __name__ == "__main__":
    main()
