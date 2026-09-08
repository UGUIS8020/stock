# -*- coding: utf-8 -*-
"""
simulate_an_gapband.py - 戦略AN エントリーフィルタの正式検証（#2）

simulate_an_walkforward.py で見えた「gap帯の構造」（±1%以内が効く／大きな
ギャップアップと真の横ばいがダメ）を、9分割点ウォークフォワード＋OOSブロック
ブートストラップで検証する。あわせて、live実装が偶然効かせている
「CMEゲート（US前日プラスなら個別株がそれを超えて上げてる時だけ買う）」の
効果も、S&P500前日騰落を代理変数にして測る。

  シグナル   : out/sim_an_signals.csv（simulate_an.py が生成）
  US代理     : out/gspc_chg.csv（^GSPC 日次騰落%。trade_date 直前の値を使う）
  エントリー : 翌営業日 Open / 決済 TP+1.5% or SL-5.9% or 15:00引け（両到達=保守的SL）
  対象       : トレード日 NORMAL のみ

使い方: python simulate/simulate_an_gapband.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import sqlite3

BASE = os.path.join(os.path.dirname(__file__), "..")
DB_PATH = os.path.join(BASE, "out", "stock.db")
SIG_CSV = os.path.join(BASE, "out", "sim_an_signals.csv")
GSPC_CSV = os.path.join(BASE, "out", "gspc_chg.csv")

TP_PCT, SL_PCT = 0.015, 0.059
PANIC_NIKKEI, PANIC_AD = -2.0, 0.20
WEAK_NIKKEI, WEAK_AD = -1.0, 0.35
STRONG_NIKKEI, STRONG_AD = 0.5, 0.60
RNG = np.random.default_rng(42)


def build_regime():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT Date, Close, prev_close FROM daily_prices WHERE prev_close>0 AND Close>0", conn)
    conn.close()
    df["chg"] = (df["Close"] - df["prev_close"]) / df["prev_close"] * 100
    out = {}
    for date, g in df.groupby("Date"):
        ad = (g["chg"] > 0).sum() / len(g)
        nk = float(g["chg"].median())
        if nk <= PANIC_NIKKEI or ad <= PANIC_AD:
            c = "PANIC"
        elif nk <= WEAK_NIKKEI or ad <= WEAK_AD:
            c = "WEAK"
        elif nk >= STRONG_NIKKEI and ad >= STRONG_AD:
            c = "STRONG"
        else:
            c = "NORMAL"
        out[date] = c
    return out


def trade_ret(r):
    o, h, l, c = r["o"], r["h"], r["l"], r["c"]
    tp_p, sl_p = o * (1 + TP_PCT), o * (1 - SL_PCT)
    if h >= tp_p and l > sl_p:
        return TP_PCT * 100
    if l <= sl_p:
        return -SL_PCT * 100
    return (c - o) / o * 100


def stats(d):
    if len(d) == 0:
        return dict(n=0, wr=0, avg=0, sharpe=0, tot=0, ndays=0)
    ret = d["ret"].values
    daily = d.groupby("trade_date")["ret"].mean()
    sh = daily.mean() / daily.std() * np.sqrt(245) if daily.std() > 0 else 0
    return dict(n=len(d), wr=(ret > 0).mean() * 100, avg=ret.mean(),
                sharpe=sh, tot=ret.sum(), ndays=d["trade_date"].nunique())


# ── フィルタ定義（sig の各行 -> bool: 採用するか）──
FILTERS = {
    "R0 ALL（無フィルタ）":            lambda d: pd.Series(True, index=d.index),
    "R1 gap>+1.0% を除外":            lambda d: d["gap"] <= 1.0,
    "R2 gap>+1.0% & 横ばい±0.1% 除外": lambda d: (d["gap"] <= 1.0) & (d["gap"].abs() >= 0.1),
    "R3 gap を [-1.0,+1.0] に限定":    lambda d: d["gap"].between(-1.0, 1.0),
    "R4 gap>+1.5% を除外（緩め）":     lambda d: d["gap"] <= 1.5,
    "G  CMEゲート（US>0なら gap>=US）": lambda d: (d["us_prev"] <= 0) | (d["gap"] >= d["us_prev"]),
    "R2+G 併用":                      lambda d: (d["gap"] <= 1.0) & (d["gap"].abs() >= 0.1)
                                               & ((d["us_prev"] <= 0) | (d["gap"] >= d["us_prev"])),
}


def main():
    print("地合い計算中...", file=sys.stderr)
    regime = build_regime()
    sig = pd.read_csv(SIG_CSV, dtype={"code": str})
    sig["trade_regime"] = sig["trade_date"].map(regime)
    sig = sig[sig["trade_regime"] == "NORMAL"].copy()
    sig["ret"] = sig.apply(trade_ret, axis=1)
    sig["gap"] = (sig["o"] - sig["prev_close"]) / sig["prev_close"] * 100

    gspc = pd.read_csv(GSPC_CSV)
    gspc.columns = ["date", "chg"]
    gspc = gspc.sort_values("date").reset_index(drop=True)
    def us_before(d):
        prev = gspc[gspc["date"] < d]
        return float(prev["chg"].iloc[-1]) if len(prev) else 0.0
    sig["us_prev"] = sig["trade_date"].map(us_before)
    sig = sig.sort_values("trade_date").reset_index(drop=True)

    base = stats(sig)
    print(f"\n対象（NORMAL日）: n={base['n']}  {base['ndays']}日  "
          f"{sig['trade_date'].min()}〜{sig['trade_date'].max()}")
    print(f"ベース: 勝率{base['wr']:.1f}%  平均{base['avg']:+.3f}%  Sharpe{base['sharpe']:+.2f}  累計{base['tot']:+.1f}%")

    # ── 1. 各フィルタ・全期間 ──
    print("\n" + "=" * 104)
    print("  1. フィルタ別・全期間（残る件数 / 落とすスライス単独）")
    print("=" * 104)
    for name, f in FILTERS.items():
        mask = f(sig)
        keep, drop = sig[mask], sig[~mask]
        sk, sd = stats(keep), stats(drop)
        print(f"  {name:<30} 残 n={sk['n']:>4}({sk['n']/base['n']*100:>3.0f}%) "
              f"勝率{sk['wr']:>4.1f}% 平均{sk['avg']:>+6.3f}% Sh{sk['sharpe']:>+5.2f} 累計{sk['tot']:>+7.1f}%"
              f"  │ 落とす分 n={sd['n']:>3} 平均{sd['avg']:>+6.3f}%")

    # ── 2. ウォークフォワード ──
    print("\n" + "=" * 104)
    print("  2. 9分割点ウォークフォワード（train平均% / test平均% / test Sharpe）")
    print("=" * 104)
    dates = np.sort(sig["trade_date"].unique())
    hdr = "  分割  " + "".join(f"{k.split()[0]:>16}" for k in FILTERS)
    print(hdr)
    agg = {k: [] for k in FILTERS}
    for frac in np.arange(0.45, 0.86, 0.05):
        split = dates[int(len(dates) * frac)]
        tr = sig[sig["trade_date"] <= split]
        te = sig[sig["trade_date"] > split]
        cells = []
        for name, f in FILTERS.items():
            s_tr = stats(tr[f(tr)])
            s_te = stats(te[f(te)])
            agg[name].append((s_tr["avg"], s_te["avg"], s_te["sharpe"]))
            cells.append(f"{s_tr['avg']:+.2f}/{s_te['avg']:+.2f}/{s_te['sharpe']:+.1f}")
        print(f"  {frac:>4.0%} " + "".join(f"{c:>16}" for c in cells))
    print("\n  平均（9分割）  train / test / testSharpe:")
    for name in FILTERS:
        a = np.array(agg[name])
        print(f"    {name:<30} {a[:,0].mean():>+6.3f}% / {a[:,1].mean():>+6.3f}% / {a[:,2].mean():>+5.2f}"
              f"   test最小Sharpe {a[:,2].min():>+5.2f}")

    # ── 3. OOSブートストラップ ──
    print("\n" + "=" * 104)
    print("  3. OOS（後半40%日）日別損益ブロックブートストラップ 2000回")
    print("=" * 104)
    oos_split = dates[int(len(dates) * 0.6)]
    oos = sig[sig["trade_date"] > oos_split]
    print(f"  OOS: {oos['trade_date'].min()}〜{oos['trade_date'].max()} ({oos['trade_date'].nunique()}日)")
    for name, f in FILTERS.items():
        d = oos[f(oos)]
        daily = d.groupby("trade_date")["ret"].mean().values
        if len(daily) < 10:
            print(f"  {name}: 日数不足"); continue
        sims = np.array([RNG.choice(daily, len(daily), replace=True).sum() for _ in range(2000)])
        print(f"  {name:<30} 実測{daily.sum():>+7.2f}%  中央{np.median(sims):>+7.2f}%  "
              f"p05{np.percentile(sims,5):>+7.2f}%  p95{np.percentile(sims,95):>+7.2f}%  "
              f"赤字率{(sims<0).mean()*100:>4.1f}%  日数{len(daily)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
