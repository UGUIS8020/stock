"""
simulate_r_close.py
戦略R改: -5%下落株をSTRONG/NORMAL日引けで買い → 翌朝寄りで売る（ギャップ取り）
戦略A・戦略Pと合算した場合の複利リターンも計算
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
    if len(closes) < period + 1:
        return None
    d = np.diff(closes)
    ag = np.where(d > 0, d, 0.0)[-period:].mean()
    al = np.where(d < 0, -d, 0.0)[-period:].mean()
    return round(100 - 100 / (1 + ag / al), 1) if al > 0 else 100.0


def calc_rb(hist):
    if len(hist) < 26:
        return 0
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


def build_market_conditions():
    dfs = []
    for f in glob.glob(f"{CACHE_DIR}/*.csv"):
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
    all_df["prev"] = all_df.groupby("code")["Close"].shift(1)
    all_df["chg"]  = (all_df["Close"] - all_df["prev"]) / all_df["prev"] * 100
    all_df = all_df.dropna(subset=["chg"])
    mc = {}
    for date, grp in all_df.groupby("Date"):
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


def build_r_trades(mc, target_conds=("STRONG", "NORMAL")):
    """STRONG/NORMAL日に-5%下落した株の引け買い→翌朝売りシグナル"""
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

                cond_today = mc.get(sd)
                cond_next  = mc.get(td)

                # 指定地合いの日のみ（PANICは除外）
                if cond_today not in target_conds:
                    continue
                # 翌日もPANICは除外
                if cond_next == "PANIC":
                    continue
                if i == 0:
                    continue

                prev_c = float(df.iloc[i - 1]["Close"])
                if prev_c <= 0:
                    continue
                rise = (float(today["Close"]) - prev_c) / prev_c * 100
                # -5%以上の下落
                if rise > -5.0:
                    continue

                rb = calc_rb(hist)
                if rb < 3:
                    continue

                entry    = float(today["Close"])
                nxt_open = float(nxt["Open"])
                nxt_high = float(nxt["High"])
                nxt_low  = float(nxt["Low"])
                nxt_close= float(nxt["Close"])
                if entry <= 0:
                    continue

                gap     = (nxt_open  - entry) / entry * 100
                ret_max = (nxt_high  - entry) / entry * 100
                ret_low = (nxt_low   - entry) / entry * 100
                ret_oc  = (nxt_close - entry) / entry * 100

                trades.append({
                    "scan_date":   sd,
                    "trade_date":  td,
                    "scan_cond":   cond_today,
                    "trade_cond":  cond_next,
                    "rb_score":    rb,
                    "today_rise":  round(rise, 2),
                    "gap":         round(gap, 2),
                    "ret_max":     round(ret_max, 2),
                    "ret_low":     round(ret_low, 2),
                    "ret_oc":      round(ret_oc, 2),
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
            t = cal["tp_prob"] + cal["sl_prob"]
            rets.append(tp * cal["tp_prob"] / t + sl * cal["sl_prob"] / t)
        else:
            rets.append(r["ret_oc"])
    return np.array(rets)


def run_compound_3(picks_a, picks_p, picks_r, label, initial=1_000_000, top_n=5):
    all_dates = sorted(
        set(picks_a["trade_date"].unique()) |
        set(picks_p["trade_date"].unique()) |
        set(picks_r["scan_date"].unique())
    )
    capital = float(initial)
    peak    = capital
    max_dd  = 0.0
    cap_log = []
    a_days = p_days = r_days = 0

    for date in all_dates:
        day_a = picks_a[picks_a["trade_date"] == date].head(top_n)
        day_p = picks_p[picks_p["trade_date"] == date].head(top_n)
        day_r = picks_r[picks_r["scan_date"] == date].head(top_n)
        n = len(day_a) + len(day_p) + len(day_r)
        if n == 0:
            continue
        alloc    = capital / n
        capital += sum(alloc * r / 100 for r in day_a["sim_ret"])
        capital += sum(alloc * r / 100 for r in day_p["gap"])
        capital += sum(alloc * r / 100 for r in day_r["gap"])
        if capital > peak:
            peak = capital
        dd = (capital - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd
        if len(day_a) > 0: a_days += 1
        if len(day_p) > 0: p_days += 1
        if len(day_r) > 0: r_days += 1
        cap_log.append({"date": date, "capital": capital})

    if not cap_log:
        print(f"  [{label}] データなし")
        return

    cap_df = pd.DataFrame(cap_log)
    years  = (pd.to_datetime(cap_df["date"].iloc[-1]) -
              pd.to_datetime(cap_df["date"].iloc[0])).days / 365.0
    ann    = ((capital / initial) ** (1 / years) - 1) * 100 if years > 0 else 0
    total  = (capital - initial) / initial * 100

    print(f"\n【{label}】")
    print(f"  戦略A={a_days}日  戦略P={p_days}日  戦略R={r_days}日  合計={len(cap_log)}日")
    print(f"  {initial:,}円 → {capital:,.0f}円")
    print(f"  期間: {total:+.1f}%  年率: {ann:+.1f}%  最大DD: {max_dd:+.1f}%")


def main():
    print("地合い計算中...")
    mc = build_market_conditions()

    cond_counts = {}
    for c in mc.values():
        cond_counts[c] = cond_counts.get(c, 0) + 1
    for c in ["STRONG", "NORMAL", "WEAK", "PANIC"]:
        print(f"  {c}: {cond_counts.get(c, 0)}日")

    # ── 戦略R（引け買い）シグナル ─────────────────
    print("\n戦略R（引け買い）シグナル計算中...")
    df_r = build_r_trades(mc, target_conds=("STRONG", "NORMAL"))
    df_r = df_r.sort_values(["scan_date", "rb_score"], ascending=[True, False])
    print(f"シグナル数: {len(df_r)}件 / {df_r['scan_date'].nunique()}日")

    # ── ギャップ統計 ─────────────────────────────
    print("\n=== ギャップ統計（引け買い → 翌朝寄り）===")
    gap = df_r["gap"].values
    print(f"全体: {len(gap)}件  勝率{(gap>0).mean()*100:.1f}%  "
          f"平均{gap.mean():+.3f}%  中央値{np.median(gap):+.2f}%")

    print("\n地合い別:")
    for cond in ["STRONG", "NORMAL"]:
        g = df_r[df_r["scan_cond"] == cond]["gap"].values
        if len(g) < 5: continue
        print(f"  {cond}: {len(g)}件  勝率{(g>0).mean()*100:.1f}%  "
              f"平均{g.mean():+.3f}%  累計{g.sum():+.1f}%")

    print("\nRBスコア別:")
    for lo, hi in [(3, 5), (5, 7), (7, 10)]:
        g = df_r[(df_r["rb_score"] >= lo) & (df_r["rb_score"] < hi)]["gap"].values
        if len(g) < 5: continue
        print(f"  RB{lo}〜{hi-1}: {len(g)}件  勝率{(g>0).mean()*100:.1f}%  "
              f"平均{g.mean():+.3f}%  累計{g.sum():+.1f}%")

    print("\n翌日地合い別:")
    for tc in ["STRONG", "NORMAL", "WEAK"]:
        g = df_r[df_r["trade_cond"] == tc]["gap"].values
        if len(g) < 5: continue
        print(f"  翌日{tc}: {len(g)}件  勝率{(g>0).mean()*100:.1f}%  平均{g.mean():+.3f}%")

    # ── 戦略A準備 ─────────────────────────────────
    print("\n戦略A準備中...")
    a = pd.read_csv("out/sim_precise_trades.csv")
    a = a[
        (a["trade_cond"] == "STRONG") &
        (a["gap_pct"] < 5) &
        (a["score"] >= 3) &
        (a["score"] < 7)
    ].copy()
    a = a.sort_values(["trade_date", "score"])
    sl_a = {-2.0: {"tp_prob": 0.155, "sl_prob": 0.684}}

    def sim_a(r):
        hi, lo, tp, sl = r["ret_max"], r["ret_low"], 3.0, -2.0
        cal = sl_a[sl]
        if hi >= tp and lo > sl: return tp
        if lo <= sl and hi < tp: return sl
        if hi >= tp and lo <= sl:
            t = cal["tp_prob"] + cal["sl_prob"]
            return tp * cal["tp_prob"] / t + sl * cal["sl_prob"] / t
        return r["ret_oc"]

    a["sim_ret"] = a.apply(sim_a, axis=1)

    # ── 戦略P準備（PANIC日引け買い→翌朝売り）──────
    print("戦略P準備中...")
    df_p = build_r_trades.__wrapped__(mc) if hasattr(build_r_trades, '__wrapped__') else None
    # simulate_panic_close.pyのbuild_panic_tradesと同じロジックを直接実行
    from simulate_panic_close import build_panic_trades
    df_p = build_panic_trades(mc)
    df_p = df_p.sort_values(["scan_date", "rb_score"], ascending=[True, False])
    print(f"戦略Pシグナル: {len(df_p)}件 / {df_p['scan_date'].nunique()}日")

    empty_a = pd.DataFrame(columns=["trade_date", "sim_ret"])
    empty_p = pd.DataFrame(columns=["trade_date", "gap"])
    empty_r = pd.DataFrame(columns=["scan_date", "gap"])

    # ── 複利シミュレーション比較 ─────────────────
    print(f"\n{'='*60}")
    print("【複利シミュレーション比較】初期資金100万円")
    print(f"{'='*60}")

    run_compound_3(a,       empty_p, empty_r, "戦略Aのみ")
    run_compound_3(empty_a, df_p,    empty_r, "戦略Pのみ（PANIC引け）")
    run_compound_3(empty_a, empty_p, df_r,    "戦略Rのみ（引け買い）")
    run_compound_3(a,       df_p,    empty_r, "戦略A + 戦略P")
    run_compound_3(a,       empty_p, df_r,    "戦略A + 戦略R")
    run_compound_3(empty_a, df_p,    df_r,    "戦略P + 戦略R")
    run_compound_3(a,       df_p,    df_r,    "戦略A + 戦略P + 戦略R（全合算）")

    # STRONGのみのRに絞った場合
    df_r_strong = df_r[df_r["scan_cond"] == "STRONG"]
    run_compound_3(a, df_p, df_r_strong, "戦略A + 戦略P + 戦略R(STRONG日のみ)")

    df_r_rb56 = df_r[df_r["rb_score"] >= 5]
    run_compound_3(a, df_p, df_r_rb56, "戦略A + 戦略P + 戦略R(RB5以上)")


if __name__ == "__main__":
    main()
