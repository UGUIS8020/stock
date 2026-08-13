# -*- coding: utf-8 -*-
"""
simulate_d.py - 戦略D(daytime.py)専用の正確なバックテストデータ生成

evolve_d.pyがsim_precise_trades.csvを流用していたが、実際にはgap_pctと
地合いだけで判定しており、daytime.pyの実際の事前フィルタ(PREV_RISE_MIN・
MA5_DEV_MAX・MA25_DEV_MAX・MIN_VOLUME)を反映していなかった。これにより
2026-08-13の株価帯別検証で件数が48万件(市場のほぼ全体)という桁違いの
規模になり、信頼できない結果になることが判明した。

本スクリプトはdaily_pricesから直接、daytime.pyの実際のStage1事前フィルタ
(scan_morning_strategy_d_candidates.pyと同一ロジック)を再現し、その上で
gap_pct(寄り付きギャップ)・TP+3%/SL-5%・同日完結(D/A/AN/ASと同じ強制決済
ルール)を適用する。ザラ場内のMOMENTUM_PCT(0.3%)トリガーの正確な時刻までは
日足データでは再現できない点に注意（evolve_d.py同様の制約）。

出力: out/sim_d_trades.csv

実行方法:
    python simulate/simulate_d.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np

import db

OUT_CSV = "out/sim_d_trades.csv"

PREV_RISE_MIN = -0.5
MA5_DEV_MAX   = 3.0
MA25_DEV_MAX  = 6.0
MIN_VOLUME    = 50_000
GAP_MIN_PCT   = -5.0
GAP_MAX_PCT   = 2.0


def load_daily_prices():
    conn = db.get_conn()
    codes = set(db.get_all_codes(exclude_etf=True))
    df = pd.read_sql(
        "SELECT code, Date, Open, High, Low, Close, Volume, prev_close "
        "FROM daily_prices ORDER BY code, Date",
        conn,
    )
    df = df[df["code"].isin(codes)].reset_index(drop=True)
    return df


def build_ad_ratio(df):
    valid = df[df["prev_close"] > 0].copy()
    valid["chg"] = (valid["Close"] - valid["prev_close"]) / valid["prev_close"] * 100
    valid["is_up"] = valid["chg"] > 0
    ad_ratio = valid.groupby("Date")["is_up"].mean()

    def classify(r):
        if r <= 0.20: return "PANIC"
        if r >= 0.60: return "STRONG"
        if r <= 0.35: return "WEAK"
        return "NORMAL"
    return ad_ratio.apply(classify)


def main():
    print("daily_prices読み込み中...")
    df = load_daily_prices()
    by_code = {code: g.reset_index(drop=True) for code, g in df.groupby("code")}
    dates_all = sorted(df["Date"].unique())
    condition_by_date = build_ad_ratio(df)
    print(f"  {len(df):,}行 / {len(dates_all)}日分")

    rows = []
    for i, scan_date in enumerate(dates_all[:-1]):
        cond = condition_by_date.get(scan_date)
        if cond != "STRONG":  # REQUIRE_STRONG=Trueのため実発注はSTRONG日のみ
            continue

        day_df = df[df["Date"] == scan_date]

        for _, r in day_df.iterrows():
            code = r["code"]
            hist_all = by_code.get(code)
            if hist_all is None:
                continue
            idx = hist_all.index[hist_all["Date"] == scan_date]
            if len(idx) == 0 or idx[0] < 25:
                continue
            idx = idx[0]

            if r["Volume"] < MIN_VOLUME or r["prev_close"] <= 0:
                continue
            prev_close = r["prev_close"]
            hist = hist_all.iloc[max(0, idx-25):idx+1]
            closes = hist["Close"].values
            if len(closes) < 26:
                continue

            prev2_close = closes[-2]
            if prev2_close <= 0:
                continue
            prev_rise = (closes[-1] - prev2_close) / prev2_close * 100
            if prev_rise < PREV_RISE_MIN:
                continue

            ma5 = closes[-5:].mean()
            ma25 = closes[-25:].mean()
            ma5_dev = (closes[-1] / ma5 - 1) * 100 if ma5 > 0 else 0
            ma25_dev = (closes[-1] / ma25 - 1) * 100 if ma25 > 0 else 0
            if ma5_dev > MA5_DEV_MAX or ma25_dev > MA25_DEV_MAX:
                continue

            if idx + 1 >= len(hist_all):
                continue
            trade_row = hist_all.iloc[idx + 1]
            entry_open = trade_row["Open"]
            if not entry_open or entry_open <= 0:
                continue
            gap_pct = (entry_open - r["Close"]) / r["Close"] * 100
            if gap_pct < GAP_MIN_PCT or gap_pct > GAP_MAX_PCT:
                continue

            day1_high, day1_low, day1_close = trade_row["High"], trade_row["Low"], trade_row["Close"]
            if any(pd.isna(v) for v in [day1_high, day1_low, day1_close]):
                continue

            ret_max = (day1_high - entry_open) / entry_open * 100
            ret_low = (day1_low - entry_open) / entry_open * 100
            ret_oc  = (day1_close - entry_open) / entry_open * 100

            rows.append({
                "code": code, "scan_date": scan_date, "trade_date": trade_row["Date"],
                "prev_rise": round(prev_rise, 2), "gap_pct": round(gap_pct, 2),
                "ma5_dev": round(ma5_dev, 2), "ma25_dev": round(ma25_dev, 2),
                "ret_oc": round(ret_oc, 3), "ret_max": round(ret_max, 3), "ret_low": round(ret_low, 3),
            })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(dates_all)}日処理済み... 候補{len(rows)}件")

    os.makedirs("out", exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n完了: {len(out_df):,}件 → {OUT_CSV}")


if __name__ == "__main__":
    main()
