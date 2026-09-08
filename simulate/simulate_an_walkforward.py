# -*- coding: utf-8 -*-
"""
simulate_an_walkforward.py - 戦略AN「上昇率優先」ゲートの正式検証

simulate_an.py が吐いたシグナルキャッシュ(out/sim_an_signals.csv)を使い、
market_watch.py の「寄りで前日比+0.1%以上なら即発注」ゲート（=上昇率優先）が
リターンに寄与しているかを、元のAN検証と同じ3手法で確認する:

  1. 9分割点ウォークフォワード（trade_date の 45〜85% を 5% 刻みで train/test 分割）
  2. 増分単独検証（上昇率優先で「買う」側と「捨てる」側を直接比較）
  3. OOS日別損益のブロックブートストラップ（2000回、最終赤字確率）

エントリー: 翌営業日 Open / 決済: TP+1.5% or SL-5.9% or 15:00引け強制決済（両到達=保守的にSL）
対象: トレード日が NORMAL の日のみ（AN=NORMAL日専用）

使い方: python simulate/simulate_an_walkforward.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "out", "stock.db")
SIG_CSV = os.path.join(os.path.dirname(__file__), "..", "out", "sim_an_signals.csv")

TP_PCT, SL_PCT = 0.015, 0.059
UP_GATE = 0.1  # 前日比 +0.1% 以上で「上昇率優先」発動

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
        return dict(n=0, wr=0, avg=0, daily=0, sharpe=0, tot=0)
    ret = d["ret"].values
    ndays = d["trade_date"].nunique()
    daily = d.groupby("trade_date")["ret"].mean()
    sh = daily.mean() / daily.std() * np.sqrt(245) if daily.std() > 0 else 0
    return dict(n=len(d), wr=(ret > 0).mean() * 100, avg=ret.mean(),
                daily=daily.mean(), sharpe=sh, tot=ret.sum())


def line(tag, s):
    print(f"  {tag:<26} n={s['n']:>4}  勝率{s['wr']:>5.1f}%  平均{s['avg']:>+6.3f}%  "
          f"日次{s['daily']:>+6.3f}%  年率Sharpe{s['sharpe']:>+6.2f}  累計{s['tot']:>+8.1f}%")


def main():
    print("地合い計算中...", file=sys.stderr)
    regime = build_regime()
    sig = pd.read_csv(SIG_CSV, dtype={"code": str})
    sig["trade_regime"] = sig["trade_date"].map(regime)
    sig = sig[sig["trade_regime"] == "NORMAL"].copy()
    sig["ret"] = sig.apply(trade_ret, axis=1)
    sig["gap"] = (sig["o"] - sig["prev_close"]) / sig["prev_close"] * 100
    sig["is_up"] = sig["gap"] >= UP_GATE
    sig = sig.sort_values("trade_date").reset_index(drop=True)

    print(f"\n対象シグナル（NORMAL日）: {len(sig)}  期間 {sig['trade_date'].min()} 〜 {sig['trade_date'].max()}")
    print(f"  うち 寄り+{UP_GATE}%以上（上昇率優先で買う）: {sig['is_up'].sum()}  "
          f"弱寄り（本来スルー対象）: {(~sig['is_up']).sum()}")

    variants = {
        "ALL  （全件・現状ベース）": lambda d: d,
        "UP   （上昇率優先ゲート）": lambda d: d[d["is_up"]],
        "DOWN （弱寄りのみ）":        lambda d: d[~d["is_up"]],
    }

    # ── 1. ウォークフォワード ──
    print("\n" + "=" * 100)
    print("  1. 9分割点ウォークフォワード（train=分割日以前 / test=分割日より後）")
    print("=" * 100)
    dates = np.sort(sig["trade_date"].unique())
    for frac in np.arange(0.45, 0.86, 0.05):
        split = dates[int(len(dates) * frac)]
        tr = sig[sig["trade_date"] <= split]
        te = sig[sig["trade_date"] > split]
        print(f"\n[分割 {frac:.0%}  split={split}  train {tr['trade_date'].nunique()}日 / test {te['trade_date'].nunique()}日]")
        for name, f in variants.items():
            s_tr, s_te = stats(f(tr)), stats(f(te))
            print(f"  {name:<24}  train: 平均{s_tr['avg']:>+6.3f}% 勝率{s_tr['wr']:>4.1f}% Sh{s_tr['sharpe']:>+5.2f}"
                  f"   │ test: 平均{s_te['avg']:>+6.3f}% 勝率{s_te['wr']:>4.1f}% Sh{s_te['sharpe']:>+5.2f}")

    # ── 2. 増分単独 ──
    print("\n" + "=" * 100)
    print("  2. 増分単独検証（上昇率優先が『買う』側 vs『捨てる』側 を直接比較・全期間）")
    print("=" * 100)
    line("UP  （+0.1%以上で買う）", stats(sig[sig["is_up"]]))
    line("DOWN（+0.1%未満＝捨てる）", stats(sig[~sig["is_up"]]))
    line("ALL （合算＝ゲート無し）", stats(sig))
    print("\n  gap帯別:")
    for lo, hi in [(-99, -1), (-1, -0.1), (-0.1, 0.1), (0.1, 1), (1, 3), (3, 99)]:
        d = sig[(sig["gap"] >= lo) & (sig["gap"] < hi)]
        if len(d) >= 10:
            line(f"gap [{lo:+.1f}, {hi:+.1f})%", stats(d))

    # ── 3. ブロックブートストラップ（OOS = 後半40%） ──
    print("\n" + "=" * 100)
    print("  3. OOS日別損益ブロックブートストラップ（OOS=後半40%日・2000回・1日1ブロック）")
    print("=" * 100)
    oos_split = dates[int(len(dates) * 0.6)]
    oos = sig[sig["trade_date"] > oos_split]
    print(f"  OOS期間: {oos['trade_date'].min()} 〜 {oos['trade_date'].max()}  ({oos['trade_date'].nunique()}日)")
    for name, f in variants.items():
        d = f(oos)
        daily = d.groupby("trade_date")["ret"].mean().values
        if len(daily) < 10:
            print(f"  {name}: OOS日数不足"); continue
        sims = np.array([RNG.choice(daily, len(daily), replace=True).sum() for _ in range(2000)])
        obs = daily.sum()
        print(f"  {name:<24}  実測累計{obs:>+7.2f}%  BS中央値{np.median(sims):>+7.2f}%  "
              f"p05{np.percentile(sims,5):>+7.2f}%  p95{np.percentile(sims,95):>+7.2f}%  "
              f"最終赤字確率{(sims<0).mean()*100:>5.1f}%")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
