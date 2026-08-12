# -*- coding: utf-8 -*-
"""
simulate_b_2night.py - 戦略B(2営業日保有)専用のTP/SLシミュレーションデータ生成

既存のopt_b_trades.csv(optimize/optimize_b.py生成)は、ret_max/ret_low/ret_ocが
「買った当日1日分」のOpen/High/Low/Closeのみで計算されており、戦略Bの実際の
運用(2営業日保有)を反映していなかった。2026-08-12、この不一致が原因で
バックテストがライブ実績(勝率56.7%・平均+1.97%)を大きく下回る結果になる
ことが判明したため、2営業日分の値動きを正しく捉えるデータを新たに生成する。

candidates_log等の日次候補ログには依存せず、closing_watch.pyの実際の候補生成
ロジック(RB_MIN・EXCESS_DECLINE_MIN・3日連続下落・today_rise<0)をdaily_prices
から独立に再現し、その上で2営業日分のOpen/High/Low/Closeを結合する。

出力: out/sim_b_2night_trades.csv
    code, scan_date, trade_date, exit_date, rb_score, today_rise, excess_decline,
    consec_drop, ret_oc(2日目終値ベース), ret_max(2日間の最大値), ret_low(2日間の最小値)

実行方法:
    python simulate/simulate_b_2night.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np

import db
from closing_watch import (calc_rebound_score, RB_MIN, EXCESS_DECLINE_MIN, MIN_VOLUME,
                            PANIC_AD, WEAK_AD, STRONG_AD)

OUT_CSV = "out/sim_b_2night_trades.csv"
DROP_LO, DROP_HI = -5.0, -3.1  # closing_watch.py DROP_LO/DROP_HIと同一（2026-08-12: 値を修正）


def load_daily_prices():
    conn = db.get_conn()
    codes = set(db.get_all_codes(exclude_etf=True))
    df = pd.read_sql(
        "SELECT code, Date, Open, High, Low, Close, Volume, prev_close "
        "FROM daily_prices ORDER BY code, Date",
        conn,
    )
    df = df[df["code"].isin(codes)].reset_index(drop=True)
    df["chg"] = (df["Close"] - df["prev_close"]) / df["prev_close"] * 100
    return df


def build_market_avg(df):
    valid = df[df["prev_close"] > 0]
    return valid.groupby("Date")["chg"].mean()


def build_ad_ratio(df):
    """日経225データがdaily_pricesに無いため、AD比率のみでNORMAL/STRONG/WEAK/PANICを
    分類する（closing_watch._classify_conditionのnikkei_change=Noneフォールバックと同一）。"""
    valid = df[df["prev_close"] > 0].copy()
    valid["is_up"] = valid["chg"] > 0
    ad_ratio = valid.groupby("Date")["is_up"].mean()

    def classify(r):
        if r <= PANIC_AD:
            return "PANIC"
        if r >= STRONG_AD:
            return "STRONG"
        if r <= WEAK_AD:
            return "WEAK"
        return "NORMAL"

    return ad_ratio.apply(classify)


def main():
    print("daily_prices読み込み中...")
    df = load_daily_prices()
    dates_all = sorted(df["Date"].unique())
    market_avg = build_market_avg(df)
    condition_by_date = build_ad_ratio(df)
    n_normal_strong = (condition_by_date.isin(["NORMAL", "STRONG"])).sum()
    print(f"  {len(df):,}行 / {len(dates_all)}日分（うちNORMAL/STRONG判定: {n_normal_strong}日）")

    # コード別にDate→行インデックスの辞書を作り、翌営業日参照を高速化
    by_code = {code: g.reset_index(drop=True) for code, g in df.groupby("code")}

    rows = []
    n_candidates_before_excess = 0

    for i, scan_date in enumerate(dates_all[:-2]):  # 直後2営業日が必要なため末尾2日は除外
        if condition_by_date.get(scan_date) not in ("NORMAL", "STRONG"):
            continue  # closing_watch.pyはWEAK/PANIC日は起動しない
        day_df = df[df["Date"] == scan_date]
        m_avg = market_avg.get(scan_date, 0.0)

        # dropping候補: DROP_LO〜DROP_HI%の下落 かつ 出来高条件 かつ 株価200円以上・-15%超
        cand = day_df[(day_df["chg"] >= DROP_LO) & (day_df["chg"] <= DROP_HI)
                      & (day_df["Volume"] >= MIN_VOLUME) & (day_df["prev_close"] > 0)
                      & (day_df["chg"] > -15) & (day_df["Close"] >= 200)]

        for _, r in cand.iterrows():
            code = r["code"]
            hist_all = by_code.get(code)
            if hist_all is None:
                continue
            idx = hist_all.index[hist_all["Date"] == scan_date]
            if len(idx) == 0 or idx[0] < 25:
                continue
            idx = idx[0]
            hist = hist_all.iloc[max(0, idx - 25):idx + 1]  # calc_rebound_score用（当日まで）

            rb_score = calc_rebound_score(hist)
            if rb_score < RB_MIN:
                continue

            excess_decline = round(r["chg"] - m_avg, 2)
            if excess_decline < EXCESS_DECLINE_MIN:
                continue

            # 3日連続下落判定（closing_watch.py enrich_with_rbと同一パターン）
            if idx < 2:
                continue
            y_close  = hist_all.iloc[idx]["Close"]
            d2_close = hist_all.iloc[idx - 1]["Close"]
            d3_close = hist_all.iloc[idx - 2]["Close"]
            consec_drop_ok = (y_close < d2_close) and (d2_close < d3_close)
            if not consec_drop_ok:
                continue

            n_candidates_before_excess += 1

            # 翌営業日(trade_date)に買い、その次の営業日(exit_date)まで2営業日保有
            if idx + 2 >= len(hist_all):
                continue
            trade_row = hist_all.iloc[idx + 1]
            exit_row  = hist_all.iloc[idx + 2]
            entry_open = trade_row["Open"]
            if not entry_open or entry_open <= 0:
                continue

            day1_high, day1_low = trade_row["High"], trade_row["Low"]
            day2_high, day2_low, day2_close = exit_row["High"], exit_row["Low"], exit_row["Close"]
            if any(pd.isna(v) for v in [day1_high, day1_low, day2_high, day2_low, day2_close]):
                continue

            ret_max = (max(day1_high, day2_high) - entry_open) / entry_open * 100
            ret_low = (min(day1_low, day2_low) - entry_open) / entry_open * 100
            ret_oc  = (day2_close - entry_open) / entry_open * 100

            rows.append({
                "code": code, "scan_date": scan_date,
                "trade_date": trade_row["Date"], "exit_date": exit_row["Date"],
                "rb_score": rb_score, "today_rise": round(r["chg"], 2),
                "excess_decline": excess_decline, "consec_drop": 3,
                "ret_oc": round(ret_oc, 3), "ret_max": round(ret_max, 3), "ret_low": round(ret_low, 3),
            })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(dates_all)}日処理済み... 候補{len(rows)}件")

    os.makedirs("out", exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n完了: {len(out_df):,}件 → {OUT_CSV}")
    print(f"（超過下落フィルター前の3条件通過候補: {n_candidates_before_excess:,}件）")


if __name__ == "__main__":
    main()
