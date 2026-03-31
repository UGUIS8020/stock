"""
simulate_b_threshold.py - 戦略B 下落閾値別パフォーマンス比較

-5%以下（現行）vs -4〜-5%帯（新規追加分）vs -4%以下（変更後全体）を比較
"""

import pandas as pd
import numpy as np
import glob
import os

CACHE_DIR    = "out/cache"
MIN_VOLUME   = 50_000
PANIC_NIKKEI = -2.0
PANIC_AD     = 0.20
WEAK_NIKKEI  = -1.0
WEAK_AD      = 0.35
STRONG_NIKKEI = 0.5
STRONG_AD    = 0.60

SL_CALIBRATION = {
    -1.0: {"tp_prob": 0.101, "sl_prob": 0.810},
    -2.0: {"tp_prob": 0.155, "sl_prob": 0.684},
    -3.0: {"tp_prob": 0.183, "sl_prob": 0.586},
}


def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    ag = gains[-period:].mean()
    al = losses[-period:].mean()
    if al == 0:
        return 100.0
    return round(100 - 100 / (1 + ag / al), 1)


def calc_rebound_score(hist):
    if len(hist) < 26:
        return 0
    score = 0
    closes = hist["Close"].astype(float).values
    rsi = calc_rsi(closes, 14)
    if rsi is not None:
        if rsi < 20:   score += 4
        elif rsi < 30: score += 3
        elif rsi < 40: score += 2
        elif rsi < 50: score += 1
    if len(hist) >= 21:
        avg20 = hist["Volume"].astype(float).iloc[-21:-1].mean()
        vol   = float(hist["Volume"].iloc[-1])
        if avg20 > 0:
            r = vol / avg20
            if r >= 3.0:   score += 3
            elif r >= 2.0: score += 2
            elif r >= 1.5: score += 1
    ma25 = hist["Close"].astype(float).iloc[-25:].mean()
    dev  = (float(hist["Close"].iloc[-1]) / ma25 - 1) * 100
    if dev < -15:   score += 3
    elif dev < -10: score += 2
    elif dev < -5:  score += 1
    return score


def build_market_conditions():
    files = glob.glob(f"{CACHE_DIR}/*.csv")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, usecols=["Date", "Close"])
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            dfs.append(df)
        except Exception:
            pass
    all_df = pd.concat(dfs, ignore_index=True)
    all_df["Date"] = pd.to_datetime(all_df["Date"])
    all_df = all_df.sort_values("Date")
    all_df["prev_close"] = all_df.groupby(all_df.columns[0])["Close"].shift(1) if False else None

    # 日別騰落統計
    all_df2 = pd.concat([pd.read_csv(f, usecols=["Date","Close"]) for f in files
                         if "Date" in pd.read_csv(f, nrows=0).columns],
                        ignore_index=True)
    all_df2["Date"]  = pd.to_datetime(all_df2["Date"])
    all_df2["Close"] = pd.to_numeric(all_df2["Close"], errors="coerce")
    all_df2 = all_df2.sort_values(["Date"])
    all_df2["prev"] = all_df2.groupby(all_df2.index // 1)["Close"].shift(1)

    # 簡易: daily_market_stats.csv があれば使う
    if os.path.exists("out/daily_market_stats.csv"):
        ms = pd.read_csv("out/daily_market_stats.csv")
        ms["date"] = ms["date"].astype(str)
        cond_map = {}
        for _, row in ms.iterrows():
            nk  = float(row.get("nk_est", 0) or 0)
            ad  = float(row.get("ad_ratio", 0.5) or 0.5)
            if nk <= PANIC_NIKKEI or ad <= PANIC_AD:        c = "PANIC"
            elif nk <= WEAK_NIKKEI or ad <= WEAK_AD:        c = "WEAK"
            elif nk >= STRONG_NIKKEI and ad >= STRONG_AD:   c = "STRONG"
            else:                                            c = "NORMAL"
            cond_map[row["date"]] = c
        return cond_map

    # フォールバック: キャッシュから再計算
    all_df3 = []
    for f in files:
        try:
            d = pd.read_csv(f, usecols=["Date","Close"])
            d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
            all_df3.append(d)
        except Exception:
            pass
    big = pd.concat(all_df3, ignore_index=True)
    big["Date"] = pd.to_datetime(big["Date"])
    big = big.sort_values("Date")
    big["prev_close"] = big.groupby(big.index)["Close"].shift(1)

    date_stats = []
    for date, grp in big.groupby("Date"):
        grp2 = grp.copy()
        grp2["prev"] = None
        up    = 0; total = len(grp2)
        ad    = 0.5
        nk    = 0.0
        if nk <= PANIC_NIKKEI or ad <= PANIC_AD:        c = "PANIC"
        elif nk <= WEAK_NIKKEI or ad <= WEAK_AD:        c = "WEAK"
        elif nk >= STRONG_NIKKEI and ad >= STRONG_AD:   c = "STRONG"
        else:                                            c = "NORMAL"
        date_stats.append({"date": date.strftime("%Y-%m-%d"), "condition": c})
    return pd.DataFrame(date_stats).set_index("date")["condition"].to_dict()


def scan_file(filepath, market_cond):
    trades = []
    try:
        df = pd.read_csv(filepath)
        df["Date"]   = pd.to_datetime(df["Date"])
        df["Close"]  = pd.to_numeric(df["Close"],  errors="coerce")
        df["Open"]   = pd.to_numeric(df["Open"],   errors="coerce")
        df["High"]   = pd.to_numeric(df["High"],   errors="coerce")
        df["Low"]    = pd.to_numeric(df["Low"],    errors="coerce")
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
        df = df.sort_values("Date").dropna(subset=["Close","Open","High","Low"]).reset_index(drop=True)
        code = os.path.basename(filepath).replace(".csv","")

        for i in range(25, len(df) - 1):
            hist     = df.iloc[:i+1]
            today    = hist.iloc[-1]
            next_row = df.iloc[i+1]

            scan_dt    = today["Date"].strftime("%Y-%m-%d")
            trade_dt   = next_row["Date"].strftime("%Y-%m-%d")
            scan_cond  = market_cond.get(scan_dt,  "UNKNOWN")
            trade_cond = market_cond.get(trade_dt, "UNKNOWN")

            if scan_cond == "PANIC" or trade_cond == "PANIC":
                continue
            vol_today = float(today["Volume"]) if not pd.isna(today["Volume"]) else 0
            if vol_today < MIN_VOLUME:
                continue

            prev_close = float(df.iloc[i-1]["Close"])
            if prev_close <= 0:
                continue
            today_rise = (float(today["Close"]) - prev_close) / prev_close * 100

            # -4%以下のみ対象
            if today_rise > -4.0:
                continue

            rb_score = calc_rebound_score(hist)
            if rb_score < 3:
                continue

            open_p  = float(next_row["Open"])
            high_p  = float(next_row["High"])
            low_p   = float(next_row["Low"])
            close_p = float(next_row["Close"])
            if open_p <= 0:
                continue

            ret_oc  = (close_p - open_p) / open_p * 100
            ret_max = (high_p  - open_p) / open_p * 100
            ret_low = (low_p   - open_p) / open_p * 100

            trades.append({
                "code":        code,
                "scan_date":   scan_dt,
                "trade_date":  trade_dt,
                "scan_cond":   scan_cond,
                "trade_cond":  trade_cond,
                "today_rise":  round(today_rise, 2),
                "rb_score":    rb_score,
                "ret_oc":      round(ret_oc,  2),
                "ret_max":     round(ret_max, 2),
                "ret_low":     round(ret_low, 2),
            })
    except Exception:
        pass
    return trades


def sim_tp_sl(sub, tp, sl):
    cal = SL_CALIBRATION.get(sl, {"tp_prob": 0.15, "sl_prob": 0.70})
    results = []
    for _, r in sub.iterrows():
        hi, lo = r["ret_max"], r["ret_low"]
        if hi >= tp and lo > sl:
            results.append(tp)
        elif lo <= sl and hi < tp:
            results.append(sl)
        elif hi >= tp and lo <= sl:
            t = cal["tp_prob"] + cal["sl_prob"]
            results.append(tp * cal["tp_prob"]/t + sl * cal["sl_prob"]/t)
        else:
            results.append(r["ret_oc"])
    return np.array(results)


def print_band(df, label):
    n = len(df)
    if n < 5:
        print(f"  {label}: サンプル不足({n}件)")
        return
    rets = sim_tp_sl(df, 3.0, -3.0)
    wr   = (rets > 0).mean() * 100
    avg  = rets.mean()
    cum  = rets.sum()
    tp3  = (df["ret_max"] >= 3.0).mean() * 100
    mark = " ★" if avg > 0 else ""
    print(f"  {label:<22} {n:>5}件  勝率{wr:>5.1f}%  TP到達{tp3:>5.1f}%  avg{avg:>+6.3f}%  累計{cum:>+7.1f}%{mark}")


def main():
    print("地合い計算中...")
    market_cond = build_market_conditions()

    files = sorted(glob.glob(f"{CACHE_DIR}/*.csv"))
    print(f"スキャン中: {len(files)}銘柄...")

    all_trades = []
    for i, f in enumerate(files):
        if i % 500 == 0:
            print(f"  {i}/{len(files)}...")
        all_trades.extend(scan_file(f, market_cond))

    if not all_trades:
        print("シグナルなし")
        return

    df = pd.DataFrame(all_trades)
    print(f"\n全シグナル（-4%以下）: {len(df):,}件\n")

    # バンド分け
    df_old  = df[df["today_rise"] <= -5.0]           # 現行（-5%以下）
    df_new  = df[(df["today_rise"] > -5.0) & (df["today_rise"] <= -4.0)]  # 新規追加帯（-4〜-5%）
    df_all4 = df                                      # 変更後全体（-4%以下）

    print("="*65)
    print("【閾値帯別パフォーマンス比較】TP+3% / SL-3% / 寄り付き→引け")
    print("="*65)

    for cond in ["ALL", "STRONG", "NORMAL", "WEAK"]:
        sub_old  = df_old  if cond == "ALL" else df_old [df_old ["trade_cond"] == cond]
        sub_new  = df_new  if cond == "ALL" else df_new [df_new ["trade_cond"] == cond]
        sub_all4 = df_all4 if cond == "ALL" else df_all4[df_all4["trade_cond"] == cond]
        print(f"\n--- {cond}日 ---")
        print_band(sub_old,  "-5%以下（現行）       ")
        print_band(sub_new,  "-4〜-5%帯（新規追加） ")
        print_band(sub_all4, "-4%以下（変更後全体） ")

    df.to_csv("out/sim_b_threshold.csv", index=False, encoding="utf-8-sig")
    print(f"\n詳細: out/sim_b_threshold.csv")


if __name__ == "__main__":
    main()
