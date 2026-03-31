"""
simulate_panic_tp.py
PANIC日引け買い → 翌日TP+2%/+3%狙い の検証
エントリー: PANIC日終値
決済: 翌日 TP到達 or SL到達 or 引け
"""
import pandas as pd
import numpy as np
import glob
import os

CACHE_DIR = "out/cache"
SL_CAL = {
    -1.0: {"tp_prob": 0.101, "sl_prob": 0.810},
    -2.0: {"tp_prob": 0.155, "sl_prob": 0.684},
    -3.0: {"tp_prob": 0.183, "sl_prob": 0.586},
}


def calc_rsi(closes, period=14):
    if len(closes) < period + 1: return None
    d  = np.diff(closes)
    ag = np.where(d > 0, d, 0.0)[-period:].mean()
    al = np.where(d < 0, -d, 0.0)[-period:].mean()
    return round(100 - 100 / (1 + ag / al), 1) if al > 0 else 100.0


def calc_rb(hist):
    if len(hist) < 26: return 0
    score = 0
    rsi = calc_rsi(hist["Close"].astype(float).values)
    if rsi:
        if rsi < 20:   score += 4
        elif rsi < 30: score += 3
        elif rsi < 40: score += 2
        elif rsi < 50: score += 1
    if len(hist) >= 21:
        avg20 = hist["Volume"].astype(float).iloc[-21:-1].mean()
        v = float(hist["Volume"].iloc[-1])
        if avg20 > 0:
            r = v / avg20
            if r >= 3:     score += 3
            elif r >= 2:   score += 2
            elif r >= 1.5: score += 1
    ma25 = hist["Close"].astype(float).iloc[-25:].mean()
    dev  = (float(hist["Close"].iloc[-1]) / ma25 - 1) * 100
    if dev < -15:   score += 3
    elif dev < -10: score += 2
    elif dev < -5:  score += 1
    return score


def build_mc():
    dfs = []
    for f in glob.glob(f"{CACHE_DIR}/*.csv"):
        try:
            df = pd.read_csv(f, usecols=["Date", "Close"])
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            dfs.append(df)
        except Exception: pass
    all_df = pd.concat(dfs, ignore_index=True)
    all_df["Date"] = pd.to_datetime(all_df["Date"])
    all_df = all_df.sort_values("Date")
    all_df["prev"] = all_df.groupby(all_df.index)["Close"].shift(1)
    # 日別で計算
    all_df2 = pd.concat(dfs, ignore_index=True)
    all_df2["Date"] = pd.to_datetime(all_df2["Date"])
    all_df2 = all_df2.sort_values(["Date"])
    all_df2["prev"] = all_df2.groupby(all_df2.index % len(dfs))["Close"].shift(1)

    # 全銘柄まとめて日別騰落
    dfs2 = []
    for f in glob.glob(f"{CACHE_DIR}/*.csv"):
        try:
            df = pd.read_csv(f, usecols=["Date", "Close"])
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df["code"]  = os.path.basename(f).replace(".csv", "")
            dfs2.append(df)
        except Exception: pass
    big = pd.concat(dfs2, ignore_index=True)
    big["Date"] = pd.to_datetime(big["Date"])
    big = big.sort_values(["code", "Date"])
    big["prev"] = big.groupby("code")["Close"].shift(1)
    big["chg"]  = (big["Close"] - big["prev"]) / big["prev"] * 100
    big = big.dropna(subset=["chg"])
    mc = {}
    for date, grp in big.groupby("Date"):
        up    = (grp["chg"] > 0).sum()
        total = len(grp)
        ad    = up / total if total > 0 else 0.5
        nk    = float(grp["chg"].median())
        if nk <= -2.0 or ad <= 0.20:        cond = "PANIC"
        elif nk <= -1.0 or ad <= 0.35:      cond = "WEAK"
        elif nk >= 0.5  and ad >= 0.60:     cond = "STRONG"
        else:                               cond = "NORMAL"
        mc[date.strftime("%Y-%m-%d")] = cond
    return mc


def build_trades(mc):
    trades = []
    for f in glob.glob(f"{CACHE_DIR}/*.csv"):
        try:
            df = pd.read_csv(f)
            df["Date"] = pd.to_datetime(df["Date"])
            for col in ["Close", "Open", "High", "Low", "Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.sort_values("Date").dropna(
                subset=["Close", "Open", "High", "Low"]
            ).reset_index(drop=True)
            code = os.path.basename(f).replace(".csv", "")

            for i in range(25, len(df) - 1):
                hist  = df.iloc[:i + 1]
                today = hist.iloc[-1]
                nxt   = df.iloc[i + 1]
                sd = today["Date"].strftime("%Y-%m-%d")
                td = nxt["Date"].strftime("%Y-%m-%d")
                if mc.get(sd) != "PANIC": continue
                if mc.get(td) == "PANIC": continue
                if i == 0: continue
                prev_c = float(df.iloc[i - 1]["Close"])
                if prev_c <= 0: continue
                rise = (float(today["Close"]) - prev_c) / prev_c * 100
                if rise > -5.0: continue
                rb = calc_rb(hist)
                if rb < 3: continue

                entry = float(today["Close"])
                if entry <= 0: continue
                hp = float(nxt["High"])
                lp = float(nxt["Low"])
                cp = float(nxt["Close"])

                # 引け価格基準のリターン
                ret_max = (hp - entry) / entry * 100
                ret_low = (lp - entry) / entry * 100
                ret_oc  = (cp - entry) / entry * 100
                gap     = (float(nxt["Open"]) - entry) / entry * 100

                trades.append({
                    "scan_date":  sd,
                    "trade_date": td,
                    "trade_cond": mc.get(td, "?"),
                    "rb_score":   rb,
                    "today_rise": round(rise, 2),
                    "gap":        round(gap, 2),
                    "ret_max":    round(ret_max, 2),
                    "ret_low":    round(ret_low, 2),
                    "ret_oc":     round(ret_oc, 2),
                })
        except Exception:
            pass
    return pd.DataFrame(trades)


def sim_tp_sl(sub, tp, sl):
    cal = SL_CAL.get(sl, {"tp_prob": 0.15, "sl_prob": 0.60})
    rets = []
    for _, r in sub.iterrows():
        hi, lo = r["ret_max"], r["ret_low"]
        if hi >= tp and lo > sl:
            rets.append(tp)
        elif lo <= sl and hi < tp:
            rets.append(sl)
        elif hi >= tp and lo <= sl:
            t  = cal["tp_prob"] + cal["sl_prob"]
            rets.append(tp * cal["tp_prob"] / t + sl * cal["sl_prob"] / t)
        else:
            rets.append(r["ret_oc"])
    return np.array(rets)


def run_compound(df, tp, sl, top_n=5, initial=1_000_000):
    df = df.copy()
    df["sim_ret"] = sim_tp_sl(df, tp, sl)
    df = df.sort_values(["scan_date", "rb_score"], ascending=[True, False])

    capital = float(initial)
    peak    = capital
    max_dd  = 0.0
    cap_log = []

    for date in sorted(df["scan_date"].unique()):
        day = df[df["scan_date"] == date].head(top_n)
        if len(day) == 0: continue
        alloc   = capital / len(day)
        capital += sum(alloc * r / 100 for r in day["sim_ret"])
        if capital > peak: peak = capital
        dd = (capital - peak) / peak * 100
        if dd < max_dd: max_dd = dd
        cap_log.append({"date": date, "capital": capital})

    if not cap_log:
        return None
    cap_df = pd.DataFrame(cap_log)
    years  = (pd.to_datetime(cap_df["date"].iloc[-1]) -
              pd.to_datetime(cap_df["date"].iloc[0])).days / 365.0
    ann    = ((capital / initial) ** (1 / years) - 1) * 100 if years > 0 else 0
    total  = (capital - initial) / initial * 100
    return {"final": capital, "total": total, "annual": ann,
            "max_dd": max_dd, "n_days": len(cap_log)}


def main():
    print("地合い計算中...")
    mc = build_mc()
    print("シグナル計算中...")
    df = build_trades(mc)
    print(f"シグナル: {len(df)}件 / {df['scan_date'].nunique()}日\n")

    # ── TP/SL組み合わせ全検証 ─────────────────────
    print("=" * 62)
    print("【TP/SL別パフォーマンス】PANIC日引け買い・翌日決済")
    print("  ※ リターンはすべて引け価格基準（窓上げ込み）")
    print("=" * 62)
    print(f"  {'TP':>5} {'SL':>5}  {'件数':>5}  {'勝率':>6}  {'平均%':>7}  "
          f"{'年率%':>7}  {'最大DD':>7}")
    print("  " + "─" * 54)

    combos = [
        (99, -99),   # 引け売り（TP/SLなし）
        (2.0, -99),  # TP+2%のみ（SLなし）
        (3.0, -99),  # TP+3%のみ（SLなし）
        (2.0, -1.0),
        (2.0, -2.0),
        (2.0, -3.0),
        (3.0, -1.0),
        (3.0, -2.0),
        (3.0, -3.0),
        (5.0, -3.0),
    ]

    for tp, sl in combos:
        r = run_compound(df, tp, sl)
        if r is None: continue
        rets = sim_tp_sl(df, tp, sl)
        wr   = (rets > 0).mean() * 100
        avg  = rets.mean()
        tp_label = f"+{tp:.0f}%" if tp < 90 else "引け"
        sl_label = f"{sl:.0f}%" if sl > -90 else "なし"
        mark = "★" if r["annual"] > 30 else ""
        print(f"  {tp_label:>5} {sl_label:>5}  {len(rets):>5}  {wr:>5.1f}%  "
              f"{avg:>+6.3f}%  {r['annual']:>+6.1f}%  {r['max_dd']:>+6.1f}%  {mark}")

    # ── 翌日地合い別・最良条件 ───────────────────
    print(f"\n{'=' * 62}")
    print("【翌日地合い別】TP+3% / SLなし")
    print(f"{'=' * 62}")
    for tc in ["STRONG", "NORMAL", "WEAK", "ALL"]:
        sub = df if tc == "ALL" else df[df["trade_cond"] == tc]
        if len(sub) < 5: continue
        rets = sim_tp_sl(sub, 3.0, -99)
        wr   = (rets > 0).mean() * 100
        avg  = rets.mean()
        cum  = rets.sum()
        print(f"  {tc:<8}: {len(sub):>5}件  勝率{wr:>5.1f}%  "
              f"平均{avg:>+6.3f}%  累計{cum:>+7.1f}%")

    # ── 戦略A + PANIC引け買い 複利合算 ──────────
    print(f"\n{'=' * 62}")
    print("【複利シミュレーション】戦略A + PANIC引け買い")
    print(f"{'=' * 62}")
    INITIAL = 1_000_000
    TOP_N   = 5

    a = pd.read_csv("out/sim_precise_trades.csv")
    a = a[(a["trade_cond"] == "STRONG") & (a["gap_pct"] < 5) &
          (a["score"] >= 3) & (a["score"] < 7)].copy()
    a = a.sort_values(["trade_date", "score"])
    SL_A = {-2.0: {"tp_prob": 0.155, "sl_prob": 0.684}}

    def sim_a(r):
        hi, lo, tp, sl = r["ret_max"], r["ret_low"], 3.0, -2.0
        cal = SL_A[sl]
        if hi >= tp and lo > sl: return tp
        if lo <= sl and hi < tp: return sl
        if hi >= tp and lo <= sl:
            t = cal["tp_prob"] + cal["sl_prob"]
            return tp * cal["tp_prob"] / t + sl * cal["sl_prob"] / t
        return r["ret_oc"]
    a["sim_ret"] = a.apply(sim_a, axis=1)

    for tp_p, sl_p, label in [
        (3.0, -99, "TP+3% SLなし"),
        (3.0, -2.0, "TP+3% SL-2%"),
        (2.0, -99, "TP+2% SLなし"),
        (2.0, -2.0, "TP+2% SL-2%"),
    ]:
        df_p = df.copy()
        df_p["sim_ret"] = sim_tp_sl(df_p, tp_p, sl_p)
        df_p_top = df_p.sort_values(
            ["scan_date", "rb_score"], ascending=[True, False]
        )

        all_dates = sorted(
            set(a["trade_date"].unique()) | set(df_p["scan_date"].unique())
        )
        capital = float(INITIAL); peak = capital; max_dd = 0.0; cap_log = []
        a_days = 0; p_days = 0

        for date in all_dates:
            day_a = a[a["trade_date"] == date].head(TOP_N)
            # PANIC引け買いはscan_dateで買い、trade_dateで売る
            day_p = df_p_top[df_p_top["scan_date"] == date].head(TOP_N)
            n = len(day_a) + len(day_p)
            if n == 0: continue
            alloc   = capital / n
            capital += sum(alloc * r / 100 for r in day_a["sim_ret"])
            capital += sum(alloc * r / 100 for r in day_p["sim_ret"])
            if capital > peak: peak = capital
            dd = (capital - peak) / peak * 100
            if dd < max_dd: max_dd = dd
            if len(day_a) > 0: a_days += 1
            if len(day_p) > 0: p_days += 1
            cap_log.append({"date": date, "capital": capital})

        if not cap_log: continue
        cap_df = pd.DataFrame(cap_log)
        years  = (pd.to_datetime(cap_df["date"].iloc[-1]) -
                  pd.to_datetime(cap_df["date"].iloc[0])).days / 365.0
        ann    = ((capital / INITIAL) ** (1 / years) - 1) * 100 if years > 0 else 0
        total  = (capital - INITIAL) / INITIAL * 100
        print(f"\n  PANIC引け[{label}]")
        print(f"  取引日: 戦略A={a_days}日 + PANIC={p_days}日 = 計{len(cap_log)}日")
        print(f"  {INITIAL:,}円 → {capital:,.0f}円  "
              f"期間{total:+.1f}%  年率{ann:+.1f}%  最大DD{max_dd:+.1f}%")


if __name__ == "__main__":
    main()
