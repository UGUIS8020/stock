# -*- coding: utf-8 -*-
"""
simulate_an.py - 戦略AN（戦略A・NORMAL日専用）のバックテスト

scan_morning_strategy_an_candidates.py の候補条件をそのまま daily_prices 全期間に
適用し、翌営業日の寄り付きエントリー→当日 TP/SL/引け強制決済 で損益を出す。

本体ロジックとの対応:
  - 候補条件      : scan_morning_strategy_an_candidates.py（score 1.5-3.0 / ratio>=1.4 /
                    寄り引け -2.5〜9.1% / RB>=5.4 / 連続下落>=1 / 価格>=200 / 出来高・売買代金）
  - エントリー    : 翌営業日 Open（market_watch.py、9:03 地合い確定後）
  - 決済          : TP +1.5% / SL -5.9% / 15:00 引け強制決済（position_monitor.py）
  - 地合いゲート  : トレード日が NORMAL のときだけ発注（AN=NORMAL日専用）

TP/SL 両到達時は保守的に SL 約定とみなす（simulate_precise.py と同じ立場）。

使い方:
  python simulate/simulate_an.py                # 全期間・ベースライン
  python simulate/simulate_an.py --entry-gate   # 「上昇率優先」（寄りで+0.1%以上のみ買う）を再現
  python simulate/simulate_an.py --variants     # エントリー timing / ゲートの比較表
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "out", "stock.db")

# ── AN 判定条件（scan_morning_strategy_an_candidates.py と一致）──
SCORE_MIN, SCORE_MAX = 1.5, 3.0
RATIO_MIN = 1.4
RISE_MIN, RISE_MAX = -2.5, 9.1
RB_MIN = 5.4
CD_MIN = 1
MIN_VOLUME, MIN_TURNOVER, MIN_PRICE = 50_000, 50_000_000, 200

TP_PCT, SL_PCT = 0.015, 0.059

# ── 地合い閾値（simulate_precise.py と一致）──
PANIC_NIKKEI, PANIC_AD = -2.0, 0.20
WEAK_NIKKEI,  WEAK_AD   = -1.0, 0.35
STRONG_NIKKEI, STRONG_AD = 0.5, 0.60


def build_regime(conn):
    df = pd.read_sql("SELECT Date, Close, prev_close FROM daily_prices WHERE prev_close>0 AND Close>0", conn)
    df["chg"] = (df["Close"] - df["prev_close"]) / df["prev_close"] * 100
    out = {}
    for date, g in df.groupby("Date"):
        total = len(g)
        ad = (g["chg"] > 0).sum() / total if total else 0.5
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


def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    d = np.diff(closes)
    ag = np.where(d > 0, d, 0.0)[-period:].mean()
    al = np.where(d < 0, -d, 0.0)[-period:].mean()
    return 100 - 100 / (1 + ag / al) if al > 0 else 100.0


def calc_rebound_score(hist):
    if len(hist) < 26:
        return 0
    score = 0
    closes = pd.to_numeric(hist["Close"], errors="coerce").fillna(0).values
    vol = pd.to_numeric(hist["Volume"], errors="coerce").fillna(0)
    rsi = calc_rsi(closes, 14)
    if rsi is not None:
        if rsi < 20: score += 4
        elif rsi < 30: score += 3
        elif rsi < 40: score += 2
        elif rsi < 50: score += 1
    if len(hist) >= 21:
        avg20 = vol.iloc[-21:-1].mean()
        v = float(vol.iloc[-1])
        if avg20 > 0:
            r = v / avg20
            if r >= 3.0: score += 3
            elif r >= 2.0: score += 2
            elif r >= 1.5: score += 1
    ma25 = pd.to_numeric(hist["Close"], errors="coerce").iloc[-25:].mean()
    dev = (float(hist["Close"].iloc[-1]) / ma25 - 1) * 100 if ma25 > 0 else 0
    if dev < -15: score += 3
    elif dev < -10: score += 2
    elif dev < -5: score += 1
    return score


def calc_score_an(hist):
    if len(hist) < 22:
        return None
    vs = pd.to_numeric(hist["Volume"], errors="coerce").fillna(0)
    vol20 = vs.iloc[-20:].values
    vol5 = vs.iloc[-5:].values
    v1, v3 = float(vs.iloc[-1]), float(vs.iloc[-3])
    avg20, avg5 = vol20.mean(), vol5.mean()
    if avg20 == 0 or avg5 == 0 or v3 == 0:
        return None
    trend = np.polyfit(range(5), vol5, 1)[0] / avg5 * 100
    accel = (v1 - v3) / v3 * 100
    ratio = v1 / avg20
    score = min(trend / 10, 3.0) + min(accel / 100, 3.0) + min(ratio / 3, 4.0)
    return round(score, 2), round(ratio, 2)


def calc_consec_drop(hist):
    n = len(hist)
    c = 0
    for back in range(n - 1, max(0, n - 10), -1):
        if back == 0:
            break
        if float(hist.iloc[back]["Close"]) < float(hist.iloc[back - 1]["Close"]):
            c += 1
        else:
            break
    return c


def gen_signals(conn):
    codes = [r[0] for r in conn.execute("SELECT DISTINCT code FROM daily_prices WHERE market_code != '0109'")]
    sigs = []
    for n, code in enumerate(codes):
        if n % 500 == 0:
            print(f"  scan {n}/{len(codes)}", file=sys.stderr)
        df = pd.read_sql(
            "SELECT Date,Open,High,Low,Close,Volume,Va FROM daily_prices WHERE code=? ORDER BY Date", conn, params=(code,))
        if len(df) < 27:
            continue
        for c in ("Open", "High", "Low", "Close", "Volume", "Va"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for i in range(25, len(df) - 1):
            today = df.iloc[i]
            nxt = df.iloc[i + 1]
            close, open_ = today["Close"], today["Open"]
            if not np.isfinite(close) or not np.isfinite(open_) or open_ <= 0:
                continue
            vol = today["Volume"] if np.isfinite(today["Volume"]) else 0
            va = today["Va"] if np.isfinite(today["Va"]) else vol * close
            if close < MIN_PRICE or vol < MIN_VOLUME or va < MIN_TURNOVER:
                continue
            hist = df.iloc[:i + 1]
            sa = calc_score_an(hist)
            if sa is None:
                continue
            score, ratio = sa
            if not (SCORE_MIN <= score < SCORE_MAX) or ratio < RATIO_MIN:
                continue
            rise = (close - open_) / open_ * 100
            if not (RISE_MIN <= rise <= RISE_MAX):
                continue
            if calc_rebound_score(hist) < RB_MIN:
                continue
            if calc_consec_drop(hist) < CD_MIN:
                continue
            if not (np.isfinite(nxt["Open"]) and nxt["Open"] > 0 and np.isfinite(nxt["Close"])):
                continue
            sigs.append({
                "scan_date": today["Date"], "trade_date": nxt["Date"], "code": code,
                "score": score, "ratio": ratio, "scan_rise": round(rise, 2),
                "prev_close": close,
                "o": float(nxt["Open"]), "h": float(nxt["High"]), "l": float(nxt["Low"]), "c": float(nxt["Close"]),
            })
    return pd.DataFrame(sigs)


def sim_exit(o, h, l, c, tp=TP_PCT, sl=SL_PCT):
    """1トレードのリターン%（寄り→決済）。TP/SL両到達は保守的にSL。"""
    tp_p, sl_p = o * (1 + tp), o * (1 - sl)
    hit_tp = h >= tp_p
    hit_sl = l <= sl_p
    if hit_tp and not hit_sl:
        return tp * 100
    if hit_sl:
        return -sl * 100
    return (c - o) / o * 100  # 引け強制決済


def report(sigs, label, regime, gate=None, entry="open", tp=TP_PCT, sl=SL_PCT):
    df = sigs.copy()
    df["trade_regime"] = df["trade_date"].map(regime)
    df["scan_regime"] = df["scan_date"].map(regime)
    df = df[df["trade_regime"] == "NORMAL"]
    df["gap"] = (df["o"] - df["prev_close"]) / df["prev_close"] * 100
    if gate == "up":
        df = df[df["gap"] >= 0.1]
    elif gate == "down":
        df = df[df["gap"] < 0.1]
    if entry == "open":
        df["ret"] = df.apply(lambda r: sim_exit(r["o"], r["h"], r["l"], r["c"], tp, sl), axis=1)
    elif entry == "close":  # 引けで買って翌日は見ない → その日の寄り引けは取れないので c→c=0、近似で寄り引けの逆
        df["ret"] = (df["c"] - df["o"]) / df["o"] * 100 * -1 * 0  # placeholder, not used
    n = len(df)
    if n == 0:
        print(f"{label:<34} n=0")
        return
    wr = (df["ret"] > 0).mean() * 100
    avg = df["ret"].mean()
    tot = df["ret"].sum()
    ndays = df["trade_date"].nunique()
    sharpe = avg / df["ret"].std() * np.sqrt(n / ndays * 245) if df["ret"].std() > 0 else 0
    print(f"{label:<34} n={n:>4}  勝率{wr:>5.1f}%  平均{avg:>+6.3f}%  合計{tot:>+8.1f}%  "
          f"日数{ndays:>3}  年率Sharpe≈{sharpe:>5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry-gate", action="store_true")
    ap.add_argument("--variants", action="store_true")
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(__file__), "..", "out", "sim_an_signals.parquet"))
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    print("地合い判定を計算中...", file=sys.stderr)
    regime = build_regime(conn)
    from collections import Counter
    print("  地合い分布:", dict(Counter(regime.values())), file=sys.stderr)

    if os.path.exists(args.cache):
        print(f"シグナルキャッシュを読み込み: {args.cache}", file=sys.stderr)
        sigs = pd.read_parquet(args.cache)
    else:
        print("シグナル生成中（全銘柄スキャン）...", file=sys.stderr)
        sigs = gen_signals(conn)
        try:
            sigs.to_parquet(args.cache)
        except Exception as e:
            sigs.to_csv(args.cache.replace(".parquet", ".csv"), index=False)
    conn.close()

    print(f"\n生成シグナル総数: {len(sigs)}  期間 {sigs['scan_date'].min()} 〜 {sigs['scan_date'].max()}")
    sigs["trade_regime"] = sigs["trade_date"].map(regime)
    print("トレード日地合い別シグナル数:", sigs["trade_regime"].value_counts().to_dict())

    print("\n" + "=" * 90)
    print("  戦略AN バックテスト（トレード日=NORMAL のみ・翌営業日寄り→当日決済）")
    print("=" * 90)
    report(sigs, "① ベースライン（寄り成entry・全件）", regime)
    report(sigs, "② 上昇率優先ゲート（寄り+0.1%以上のみ）", regime, gate="up")
    report(sigs, "③ 逆ゲート（寄り+0.1%未満＝弱寄りのみ）", regime, gate="down")

    if args.variants:
        print("\n--- TP/SL 感応度（ベースライン・全件）---")
        for tp in (0.010, 0.015, 0.020, 0.030):
            for sl in (0.030, 0.040, 0.059, 0.080):
                report(sigs, f"  TP{tp*100:+.1f}% / SL{-sl*100:+.1f}%", regime, tp=tp, sl=sl)
        print("\n--- 地合いゲートを外した場合（参考）---")
        for rg in ("NORMAL", "WEAK", "STRONG"):
            d = sigs[sigs["trade_date"].map(regime) == rg].copy()
            if len(d) < 10:
                continue
            d["ret"] = d.apply(lambda r: sim_exit(r["o"], r["h"], r["l"], r["c"]), axis=1)
            print(f"  trade={rg:<7} n={len(d):>4}  勝率{(d['ret']>0).mean()*100:>5.1f}%  平均{d['ret'].mean():>+6.3f}%")

        print("\n--- 年月別（ベースライン）---")
        d = sigs[sigs["trade_date"].map(regime) == "NORMAL"].copy()
        d["ret"] = d.apply(lambda r: sim_exit(r["o"], r["h"], r["l"], r["c"]), axis=1)
        d["ym"] = d["trade_date"].str[:7]
        g = d.groupby("ym")["ret"].agg(["count", "mean", "sum"])
        for ym, r in g.iterrows():
            print(f"  {ym}  n={int(r['count']):>3}  平均{r['mean']:>+6.3f}%  合計{r['sum']:>+7.1f}%")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
