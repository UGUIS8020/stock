"""
simulate_b.py - 戦略Bバックテスト（逆張り・リバウンド狙い）

条件: 当日-5%以下の下落 + リバウンドスコア3点以上
翌日寄り付きで買い、TP/SLまたは引けで決済

使い方:
    python simulate_b.py
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime

CACHE_DIR = "out/cache"
MIN_VOLUME   = 50_000
MIN_TURNOVER = 50_000_000
PANIC_NIKKEI = -2.0
PANIC_AD     = 0.20
WEAK_NIKKEI  = -1.0
WEAK_AD      = 0.35
STRONG_NIKKEI = 0.5
STRONG_AD     = 0.60

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
    """リバウンドスコア計算（scan_daily.pyと同じロジック）"""
    if len(hist) < 26:
        return 0
    score = 0

    # RSI（0〜4点）
    closes = hist["Close"].astype(float).values
    rsi = calc_rsi(closes, 14)
    if rsi is not None:
        if rsi < 20:   score += 4
        elif rsi < 30: score += 3
        elif rsi < 40: score += 2
        elif rsi < 50: score += 1

    # 出来高倍率（0〜3点）
    if len(hist) >= 21:
        avg20 = hist["Volume"].astype(float).iloc[-21:-1].mean()
        vol   = float(hist["Volume"].iloc[-1])
        if avg20 > 0:
            r = vol / avg20
            if r >= 3.0:   score += 3
            elif r >= 2.0: score += 2
            elif r >= 1.5: score += 1

    # MA25乖離（0〜3点）
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
            df["code"]  = os.path.basename(f).replace(".csv", "")
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            dfs.append(df)
        except Exception:
            pass
    all_df = pd.concat(dfs, ignore_index=True)
    all_df["Date"] = pd.to_datetime(all_df["Date"])
    all_df = all_df.sort_values(["code", "Date"])
    all_df["prev_close"] = all_df.groupby("code")["Close"].shift(1)
    all_df["change_pct"] = (all_df["Close"] - all_df["prev_close"]) / all_df["prev_close"] * 100
    all_df = all_df.dropna(subset=["change_pct"])

    date_stats = []
    for date, grp in all_df.groupby("Date"):
        up    = (grp["change_pct"] > 0).sum()
        total = len(grp)
        ad    = up / total if total > 0 else 0.5
        nk    = float(grp["change_pct"].median())
        if nk <= PANIC_NIKKEI or ad <= PANIC_AD:   cond = "PANIC"
        elif nk <= WEAK_NIKKEI or ad <= WEAK_AD:   cond = "WEAK"
        elif nk >= STRONG_NIKKEI and ad >= STRONG_AD: cond = "STRONG"
        else:                                        cond = "NORMAL"
        date_stats.append({"date": date.strftime("%Y-%m-%d"), "condition": cond})

    return pd.DataFrame(date_stats).set_index("date")["condition"].to_dict()


def simulate_b_one(filepath, market_cond):
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

        code = os.path.basename(filepath).replace(".csv", "")

        for i in range(25, len(df) - 1):
            hist     = df.iloc[:i+1]
            today    = hist.iloc[-1]
            next_row = df.iloc[i+1]

            scan_dt  = today["Date"].strftime("%Y-%m-%d")
            trade_dt = next_row["Date"].strftime("%Y-%m-%d")
            scan_cond  = market_cond.get(scan_dt,  "UNKNOWN")
            trade_cond = market_cond.get(trade_dt, "UNKNOWN")

            # PANIC日除外
            if scan_cond == "PANIC" or trade_cond == "PANIC":
                continue

            # 出来高・売買代金フィルタ
            vol_today = float(today["Volume"]) if not pd.isna(today["Volume"]) else 0
            if vol_today < MIN_VOLUME:
                continue

            # 戦略B条件: 当日-5%以下の下落
            if i == 0:
                continue
            prev_close = float(df.iloc[i-1]["Close"])
            if prev_close <= 0:
                continue
            today_rise = (float(today["Close"]) - prev_close) / prev_close * 100
            if today_rise > -5.0:
                continue

            # リバウンドスコア計算
            rb_score = calc_rebound_score(hist)
            if rb_score < 3:
                continue

            # 翌日指標
            open_p  = float(next_row["Open"])
            high_p  = float(next_row["High"])
            low_p   = float(next_row["Low"])
            close_p = float(next_row["Close"])
            if open_p <= 0:
                continue

            ret_oc  = (close_p - open_p) / open_p * 100
            ret_max = (high_p  - open_p) / open_p * 100
            ret_low = (low_p   - open_p) / open_p * 100
            gap_pct = (open_p  - float(today["Close"])) / float(today["Close"]) * 100

            trades.append({
                "code":        code,
                "scan_date":   scan_dt,
                "trade_date":  trade_dt,
                "scan_cond":   scan_cond,
                "trade_cond":  trade_cond,
                "today_rise":  round(today_rise, 2),
                "rb_score":    rb_score,
                "open_price":  open_p,
                "close_price": close_p,
                "ret_oc":      round(ret_oc, 2),
                "ret_max":     round(ret_max, 2),
                "ret_low":     round(ret_low, 2),
                "gap_pct":     round(gap_pct, 2),
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


def main():
    print("地合い計算中...")
    market_cond = build_market_conditions()

    files = sorted(glob.glob(f"{CACHE_DIR}/*.csv"))
    print(f"戦略Bシミュレーション: {len(files)}銘柄")

    all_trades = []
    for i, f in enumerate(files):
        if i % 500 == 0:
            print(f"  {i}/{len(files)}...")
        all_trades.extend(simulate_b_one(f, market_cond))

    if not all_trades:
        print("シグナルなし")
        return

    df = pd.DataFrame(all_trades).sort_values(["trade_date","code"]).reset_index(drop=True)
    print(f"\nシグナル数: {len(df):,}件 / {df['trade_date'].nunique()}日")
    print(f"期間: {df['scan_date'].min()} 〜 {df['scan_date'].max()}")

    # ── 地合い別・RBスコア別集計 ──
    print(f"\n{'='*65}")
    print("【地合い × RBスコア × TP/SL別パフォーマンス】")
    print(f"{'='*65}")

    for cond in ["ALL", "WEAK", "NORMAL", "STRONG"]:
        sub_c = df if cond == "ALL" else df[df["trade_cond"] == cond]
        if len(sub_c) < 10:
            continue
        print(f"\n--- {cond} ({len(sub_c)}件 / {sub_c['trade_date'].nunique()}日) ---")
        print(f"  {'RBスコア':<10} {'件数':>5}  {'勝率':>6}  {'TP+3%':>6}  {'平均%':>7}  {'累計%':>8}")
        print(f"  {'─'*52}")

        for rb_lo, rb_hi in [(3,5),(5,7),(7,10),(3,10)]:
            sub_rb = sub_c[(sub_c["rb_score"]>=rb_lo) & (sub_c["rb_score"]<rb_hi)]
            if len(sub_rb) < 5:
                continue
            # TP+3%/SL-3%（戦略Bの標準設定）
            rets = sim_tp_sl(sub_rb, 3.0, -3.0)
            wr   = (rets > 0).mean() * 100
            tp3  = (sub_rb["ret_max"] >= 3.0).mean() * 100
            avg  = rets.mean()
            cum  = rets.sum()
            rb_label = f"RB{rb_lo}〜{rb_hi-1}" if rb_hi < 10 else f"RB{rb_lo}〜9（全体）"
            mark = "★" if avg > 0 else ""
            print(f"  {rb_label:<10} {len(sub_rb):>5}  {wr:>5.1f}%  {tp3:>5.1f}%  {avg:>+6.3f}%  {cum:>+7.1f}%  {mark}")

    # ── 最良条件の複利シミュレーション ──
    print(f"\n{'='*65}")
    print("【複利シミュレーション】WEAK日 × RBスコア3〜6 × TP+3%/SL-3%")
    print(f"{'='*65}")

    INITIAL = 1_000_000
    TOP_N   = 5

    for cond_filter, rb_lo, rb_hi, tp, sl, label in [
        ("WEAK",   3, 7, 3.0, -3.0, "WEAK × RB3〜6"),
        ("NORMAL", 3, 7, 3.0, -3.0, "NORMAL × RB3〜6"),
        ("ALL",    3, 7, 3.0, -3.0, "全地合い × RB3〜6"),
    ]:
        sub = df if cond_filter == "ALL" else df[df["trade_cond"] == cond_filter]
        sub = sub[(sub["rb_score"] >= rb_lo) & (sub["rb_score"] < rb_hi)].copy()
        sub["sim_ret"] = sim_tp_sl(sub, tp, sl)
        sub = sub.sort_values(["trade_date", "rb_score"], ascending=[True, False])

        capital = float(INITIAL)
        peak    = capital
        max_dd  = 0.0
        cap_log = []

        for date in sorted(sub["trade_date"].unique()):
            day = sub[sub["trade_date"] == date].head(TOP_N)
            n   = len(day)
            if n == 0:
                continue
            alloc     = capital / n
            day_gain  = sum(alloc * r / 100 for r in day["sim_ret"])
            capital  += day_gain
            if capital > peak:
                peak = capital
            dd = (capital - peak) / peak * 100
            if dd < max_dd:
                max_dd = dd
            cap_log.append({"date": date, "capital": capital, "gain": day_gain})

        if not cap_log:
            continue

        cap_df     = pd.DataFrame(cap_log)
        first_dt   = pd.to_datetime(cap_df["date"].iloc[0])
        last_dt    = pd.to_datetime(cap_df["date"].iloc[-1])
        years      = (last_dt - first_dt).days / 365.0
        annual_ret = ((capital / INITIAL) ** (1 / years) - 1) * 100 if years > 0 else 0
        total_ret  = (capital - INITIAL) / INITIAL * 100

        print(f"\n  [{label}]")
        print(f"  取引日数: {len(cap_log)}日  トレード数: {len(sub[sub['trade_date'].isin(cap_df['date'])])}件")
        print(f"  {INITIAL:,}円 → {capital:,.0f}円  (期間+{total_ret:.1f}%)")
        print(f"  年率: {annual_ret:>+.2f}%  最大DD: {max_dd:>+.2f}%")

    df.to_csv("out/sim_b_trades.csv", index=False, encoding="utf-8-sig")
    print(f"\n保存: out/sim_b_trades.csv")


if __name__ == "__main__":
    main()
