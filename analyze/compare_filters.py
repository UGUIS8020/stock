# -*- coding: utf-8 -*-
"""
compare_filters.py - フィルタ追加前後の WR/Sharpe 比較
"""
import pandas as pd
import numpy as np
import sqlite3
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_CAL_SL  = np.array([-0.5, -1.0, -2.0, -3.0, -5.0, -7.0])
_CAL_TPC = np.array([0.050, 0.101, 0.155, 0.183, 0.220, 0.250])
_CAL_SLC = np.array([0.930, 0.810, 0.684, 0.586, 0.480, 0.400])

def calc_rets(df_, tp=3.0, sl=-5.0):
    tp_p = float(np.interp(-sl, _CAL_SL[::-1] * -1, _CAL_TPC[::-1]))
    sl_p = float(np.interp(-sl, _CAL_SL[::-1] * -1, _CAL_SLC[::-1]))
    d    = (tp_p + sl_p) if (tp_p + sl_p) > 0 else 1.0
    m    = df_["ret_max"].values.astype(np.float32)
    lo   = df_["ret_low"].values.astype(np.float32)
    o    = df_["ret_oc"].values.astype(np.float32)
    tp_hit = (m >= tp)  & (lo >  sl)
    sl_hit = (lo <= sl) & (m  <  tp)
    both   = (m >= tp)  & (lo <= sl)
    rets   = np.where(tp_hit, tp,
             np.where(sl_hit, sl,
             np.where(both,   tp * tp_p / d + sl * sl_p / d, o)))
    n      = len(rets)
    wr     = float((rets > 0).mean() * 100)
    avg    = float(rets.mean())
    std    = float(rets.std())
    sharpe = avg / std * np.sqrt(252) if std > 1e-9 else 0.0
    return n, round(wr, 1), round(avg, 3), round(sharpe, 3)


print("データ読み込み中...")
df = pd.read_csv("out/sim_precise_trades.csv", encoding="utf-8-sig", low_memory=False)
df = df.dropna(subset=["ret_oc", "score", "ratio", "today_rise", "gap_pct"])
df = df[(df["today_rise"].abs() < 30) & (df["gap_pct"].abs() < 20)]
df["_cs"] = df["code"].astype(str)
df["_ds"] = df["scan_date"].astype(str)

print("上ひげ比率・MA乖離率を計算中...")
conn = sqlite3.connect("out/stock.db")
codes = df["_cs"].unique().tolist()
dates = df["_ds"].unique().tolist()
c_ph  = ",".join(["?"] * len(codes))
d_ph  = ",".join(["?"] * len(dates))

price_df = pd.read_sql(
    f"SELECT code,Date,High,Low,Close FROM daily_prices "
    f"WHERE code IN ({c_ph}) AND Date IN ({d_ph})",
    conn, params=codes + dates)

all_price = pd.read_sql(
    f"SELECT code,Date,Close FROM daily_prices "
    f"WHERE code IN ({c_ph}) ORDER BY code,Date",
    conn, params=codes)
conn.close()

# 上ひげ比率
price_df["code"] = price_df["code"].astype(str)
price_df["_r"]   = price_df["High"] - price_df["Low"]
price_df["usr"]  = np.where(
    price_df["_r"] > 0,
    (price_df["High"] - price_df["Close"]) / price_df["_r"], np.nan)

# MA乖離率
all_price["code"]  = all_price["code"].astype(str)
all_price["Close"] = pd.to_numeric(all_price["Close"], errors="coerce")
grp = all_price.groupby("code")["Close"]
all_price["ma5"]     = grp.transform(lambda x: x.rolling(5,  min_periods=5).mean())
all_price["ma25"]    = grp.transform(lambda x: x.rolling(25, min_periods=25).mean())
all_price["ma5_dev"] = np.where(
    all_price["ma5"]  > 0,
    (all_price["Close"] - all_price["ma5"])  / all_price["ma5"]  * 100, np.nan)
all_price["ma25_dev"] = np.where(
    all_price["ma25"] > 0,
    (all_price["Close"] - all_price["ma25"]) / all_price["ma25"] * 100, np.nan)

df = df.merge(price_df[["code", "Date", "usr"]],
              left_on=["_cs", "_ds"], right_on=["code", "Date"],
              how="left", suffixes=("", "_p"))
df = df.merge(all_price[["code", "Date", "ma5_dev", "ma25_dev"]],
              left_on=["_cs", "_ds"], right_on=["code", "Date"],
              how="left", suffixes=("", "_m"))

df["_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
df["_dow"]  = df["_date"].dt.dayofweek   # 0=月

# --- 条件マスク ---
base = (
    (df["trade_cond"] == "STRONG") &
    (df["gap_pct"]    >= -5.0) &
    (df["gap_pct"]    <=  2.0) &
    (df["today_rise"] >=  0.0)
)
ma_ok = (
    (df["ma5_dev"].isna()  | (df["ma5_dev"]  <= 3.0)) &
    (df["ma25_dev"].isna() | (df["ma25_dev"] <= 6.0))
)
mon_strict = ~(
    (df["_dow"] == 0) &
    ((df["gap_pct"] < 0.0) | (df["score"] >= 3))
)

conditions = [
    ("修正前: 基本条件のみ（Section 6まで）",         base),
    ("+ MAフィルタ（5日+3% / 25日+6% 超を除外）",    base & ma_ok),
    ("+ MAフィルタ + 月曜フィルタ",                   base & ma_ok & mon_strict),
]

print()
header = f"{'条件':<44}  {'件数':>6}  {'勝率':>6}  {'平均%':>7}  {'Sharpe':>8}  変化"
print(header)
print("-" * 82)

base_n, base_wr, base_avg, base_sh = None, None, None, None
for label, mask in conditions:
    n, wr, avg, sh = calc_rets(df[mask])
    if base_wr is None:
        base_wr, base_avg, base_sh = wr, avg, sh
        diff = ""
    else:
        dwr = wr  - base_wr
        dsh = sh  - base_sh
        diff = f"WR{dwr:+.1f}%  Sharpe{dsh:+.3f}"
    print(f"{label:<44}  {n:>6,}  {wr:>5.1f}%  {avg:>+6.3f}%  {sh:>8.3f}  {diff}")

print()
print("【adj_score ランキング効果（参考）】")
print("  属性重み・上ひげ重みは「同一ポーリングで複数シグナルが出た場合の発注優先順位」")
print("  に作用するため、シミュレーションデータでは直接計測できません。")
print("  MAX_POSITIONS=3 の制限下で上位 adj_score 銘柄を優先するほど WR が向上します。")
