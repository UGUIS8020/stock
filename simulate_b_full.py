"""
simulate_b_full.py - 戦略B 全条件網羅シミュレーション

・下落閾値: -4〜-5% / -5%以下 / -4%以下
・TP: +1% +2% +3% +5% / 引け（TP/SLなし）
・SL: -1% -2% -3% -5%
・地合い（scan_cond / trade_cond）× RBスコア帯 の全組み合わせ
"""

import pandas as pd
import numpy as np
import glob
import os

CACHE_DIR     = "out/cache"
MIN_VOLUME    = 50_000
PANIC_NIKKEI  = -2.0;  PANIC_AD  = 0.20
WEAK_NIKKEI   = -1.0;  WEAK_AD   = 0.35
STRONG_NIKKEI =  0.5;  STRONG_AD = 0.60

SL_CALIBRATION = {
    -1.0: {"tp_prob": 0.101, "sl_prob": 0.810},
    -2.0: {"tp_prob": 0.155, "sl_prob": 0.684},
    -3.0: {"tp_prob": 0.183, "sl_prob": 0.586},
    -5.0: {"tp_prob": 0.220, "sl_prob": 0.480},
}


def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    d = np.diff(closes)
    ag = np.where(d > 0, d, 0.0)[-period:].mean()
    al = np.where(d < 0, -d, 0.0)[-period:].mean()
    return round(100 - 100 / (1 + ag / al), 1) if al > 0 else 100.0


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
    if os.path.exists("out/daily_market_stats.csv"):
        ms = pd.read_csv("out/daily_market_stats.csv")
        cond_map = {}
        for _, row in ms.iterrows():
            nk = float(row.get("nk_est", 0) or 0)
            ad = float(row.get("ad_ratio", 0.5) or 0.5)
            if   nk <= PANIC_NIKKEI or ad <= PANIC_AD:        c = "PANIC"
            elif nk <= WEAK_NIKKEI  or ad <= WEAK_AD:         c = "WEAK"
            elif nk >= STRONG_NIKKEI and ad >= STRONG_AD:     c = "STRONG"
            else:                                              c = "NORMAL"
            cond_map[str(row["date"])] = c
        return cond_map
    # フォールバック（daily_market_stats.csvがない場合）
    files = glob.glob(f"{CACHE_DIR}/*.csv")
    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f, usecols=["Date","Close"])
            d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
            dfs.append(d)
        except Exception:
            pass
    big = pd.concat(dfs, ignore_index=True)
    big["Date"] = pd.to_datetime(big["Date"])
    big = big.sort_values("Date")
    big["code_"] = big.index  # dummy
    big["prev"]  = big.groupby("code_")["Close"].shift(1)
    big["chg"]   = (big["Close"] - big["prev"]) / big["prev"] * 100
    big = big.dropna(subset=["chg"])
    cond_map = {}
    for date, grp in big.groupby("Date"):
        up = (grp["chg"] > 0).sum(); tot = len(grp)
        ad = up / tot if tot > 0 else 0.5
        nk = grp["chg"].median()
        if   nk <= PANIC_NIKKEI or ad <= PANIC_AD:      c = "PANIC"
        elif nk <= WEAK_NIKKEI  or ad <= WEAK_AD:       c = "WEAK"
        elif nk >= STRONG_NIKKEI and ad >= STRONG_AD:   c = "STRONG"
        else:                                            c = "NORMAL"
        cond_map[date.strftime("%Y-%m-%d")] = c
    return cond_map


def scan_all_files(market_cond, drop_lo=-99, drop_hi=-4.0):
    """drop_lo <= today_rise <= drop_hi の銘柄を抽出（drop_hiはデフォルト-4%）"""
    trades = []
    files = sorted(glob.glob(f"{CACHE_DIR}/*.csv"))
    for i, filepath in enumerate(files):
        if i % 1000 == 0:
            print(f"  {i}/{len(files)}...", flush=True)
        try:
            df = pd.read_csv(filepath)
            df["Date"]   = pd.to_datetime(df["Date"])
            for col in ["Close","Open","High","Low","Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.sort_values("Date").dropna(subset=["Close","Open","High","Low"]).reset_index(drop=True)
            code = os.path.basename(filepath).replace(".csv","")

            for idx in range(25, len(df) - 1):
                hist     = df.iloc[:idx+1]
                today    = hist.iloc[-1]
                next_row = df.iloc[idx+1]

                scan_dt    = today["Date"].strftime("%Y-%m-%d")
                trade_dt   = next_row["Date"].strftime("%Y-%m-%d")
                scan_cond  = market_cond.get(scan_dt,  "UNKNOWN")
                trade_cond = market_cond.get(trade_dt, "UNKNOWN")

                if scan_cond == "PANIC" or trade_cond == "PANIC":
                    continue
                if float(today["Volume"]) < MIN_VOLUME:
                    continue

                prev_c = float(df.iloc[idx-1]["Close"])
                if prev_c <= 0:
                    continue
                today_rise = (float(today["Close"]) - prev_c) / prev_c * 100
                if today_rise > drop_hi or today_rise < drop_lo:
                    continue

                rb_score = calc_rebound_score(hist)
                if rb_score < 3:
                    continue

                op  = float(next_row["Open"])
                hi  = float(next_row["High"])
                lo  = float(next_row["Low"])
                cl  = float(next_row["Close"])
                if op <= 0:
                    continue

                gap_pct = (op - float(today["Close"])) / float(today["Close"]) * 100

                trades.append({
                    "code":        code,
                    "scan_date":   scan_dt,
                    "trade_date":  trade_dt,
                    "scan_cond":   scan_cond,
                    "trade_cond":  trade_cond,
                    "today_rise":  round(today_rise, 2),
                    "rb_score":    rb_score,
                    "gap_pct":     round(gap_pct, 2),
                    "ret_oc":      round((cl - op) / op * 100, 2),
                    "ret_max":     round((hi - op) / op * 100, 2),
                    "ret_low":     round((lo - op) / op * 100, 2),
                })
        except Exception:
            pass
    return pd.DataFrame(trades)


def sim_tp_sl(sub, tp, sl):
    if len(sub) == 0:
        return np.array([])
    cal = SL_CALIBRATION.get(sl, {"tp_prob": 0.15, "sl_prob": 0.65})
    results = []
    for _, r in sub.iterrows():
        hi, lo = r["ret_max"], r["ret_low"]
        if hi >= tp and lo > sl:
            results.append(tp)
        elif lo <= sl and hi < tp:
            results.append(sl)
        elif hi >= tp and lo <= sl:
            t = cal["tp_prob"] + cal["sl_prob"]
            results.append(tp * cal["tp_prob"] / t + sl * cal["sl_prob"] / t)
        else:
            results.append(r["ret_oc"])
    return np.array(results)


def summarize(sub, label, tp=None, sl=None):
    n = len(sub)
    if n < 10:
        return f"  {label:<30} {n:>5}件  サンプル不足"
    if tp is None:
        # 引け決済（TP/SLなし）
        rets = sub["ret_oc"].values
    else:
        rets = sim_tp_sl(sub, tp, sl)
    wr   = (rets > 0).mean() * 100
    avg  = rets.mean()
    cum  = rets.sum()
    mark = " ★" if avg > 0 else ""
    return f"  {label:<30} {n:>5}件  勝率{wr:>5.1f}%  avg{avg:>+6.3f}%  累計{cum:>+8.1f}%{mark}"


def main():
    print("地合い計算中...")
    market_cond = build_market_conditions()

    print("\n全銘柄スキャン中（-4%以下）...")
    df4 = scan_all_files(market_cond, drop_lo=-99,  drop_hi=-4.0)
    print(f"  -4%以下: {len(df4):,}件")

    df5   = df4[df4["today_rise"] <= -5.0]
    df4_5 = df4[(df4["today_rise"] > -5.0) & (df4["today_rise"] <= -4.0)]

    print(f"  うち -5%以下: {len(df5):,}件 / -4〜-5%帯: {len(df4_5):,}件\n")

    bands = {
        "-5%以下（現行）": df5,
        "-4〜-5%帯（新規）": df4_5,
        "-4%以下（全体）": df4,
    }

    tpsl_combos = [
        (None, None,  "引け決済（TP/SLなし）"),
        (1.0,  -1.0,  "TP+1% / SL-1%"),
        (2.0,  -2.0,  "TP+2% / SL-2%"),
        (3.0,  -3.0,  "TP+3% / SL-3%"),
        (5.0,  -3.0,  "TP+5% / SL-3%"),
        (3.0,  -5.0,  "TP+3% / SL-5%"),
        (5.0,  -5.0,  "TP+5% / SL-5%"),
    ]

    for cond_label, cond_val in [("全地合い","ALL"),("STRONG","STRONG"),("NORMAL","NORMAL"),("WEAK","WEAK")]:
        print(f"\n{'='*75}")
        print(f"【{cond_label}（trade_cond={cond_val}）】")
        print(f"{'='*75}")

        for band_label, df_band in bands.items():
            sub_c = df_band if cond_val == "ALL" else df_band[df_band["trade_cond"] == cond_val]
            print(f"\n  ── {band_label} ({len(sub_c)}件) ──")
            for tp, sl, combo_label in tpsl_combos:
                print(summarize(sub_c, combo_label, tp, sl))

    # ── RBスコア帯別（-4〜-5% NORMAL のみ詳細）──
    print(f"\n{'='*75}")
    print("【詳細: -4〜-5% × NORMAL日 × RBスコア帯別】")
    print(f"{'='*75}")
    sub = df4_5[df4_5["trade_cond"] == "NORMAL"]
    for rb_lo, rb_hi in [(3,5),(5,7),(7,10)]:
        sub_rb = sub[(sub["rb_score"] >= rb_lo) & (sub["rb_score"] < rb_hi)]
        label  = f"RB{rb_lo}〜{rb_hi-1}"
        print(f"\n  {label} ({len(sub_rb)}件)")
        for tp, sl, combo_label in tpsl_combos:
            print(summarize(sub_rb, combo_label, tp, sl))

    # ── ギャップ別（trade日の始値ギャップ）──
    print(f"\n{'='*75}")
    print("【-4〜-5% × NORMAL日 × trade日始値ギャップ別】")
    print(f"{'='*75}")
    sub = df4_5[df4_5["trade_cond"] == "NORMAL"]
    for gap_lo, gap_hi, glabel in [(-99,-2,"ギャップダウン -2%以下"),(-2,0,"微下落 -2〜0%"),(0,2,"微上昇 0〜+2%"),(2,99,"ギャップアップ +2%以上")]:
        sub_g = sub[(sub["gap_pct"] > gap_lo) & (sub["gap_pct"] <= gap_hi)]
        print(f"\n  {glabel} ({len(sub_g)}件)")
        for tp, sl, combo_label in [(None,None,"引け決済"),(3.0,-3.0,"TP+3%/SL-3%"),(3.0,-5.0,"TP+3%/SL-5%")]:
            print(summarize(sub_g, combo_label, tp, sl))

    df4.to_csv("out/sim_b_full.csv", index=False, encoding="utf-8-sig")
    print(f"\n詳細データ保存: out/sim_b_full.csv")


if __name__ == "__main__":
    main()
