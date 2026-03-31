"""
simulate_weak_scalp.py - WEAK日イントラデイスキャルプ戦略シミュレーション

【戦略】
  WEAK日の前日引け後スキャンで score7-9 の銘柄を翌朝寄り付きで買い、
  当日中に TP（利確）または引け決済する。

【データ】
  - キャッシュ全件（4432ファイル・約1年分）を遡及スキャン
  - daily_market_stats.csv から地合い判定を再構築
  - scan_daily.py と同じ calc_score() を使用

【実行】
    python simulate_weak_scalp.py
"""

import os, glob
import pandas as pd
import numpy as np

CACHE_DIR        = "out/cache"
MARKET_STATS_CSV = "out/daily_market_stats.csv"

MIN_VOLUME   = 50_000
MIN_PRICE    = 100

SL_CALIBRATION = {
    -2.0: {"tp_prob": 0.155, "sl_prob": 0.684, "neither_prob": 0.162},
    -3.0: {"tp_prob": 0.183, "sl_prob": 0.586, "neither_prob": 0.231},
    -5.0: {"tp_prob": 0.250, "sl_prob": 0.450, "neither_prob": 0.300},
}

# ──────────────────────────────────────────────
# scan_daily.py と同じスコア計算
# ──────────────────────────────────────────────
def calc_score(hist):
    if len(hist) < 22:
        return None
    vol_20d = hist["Volume"].iloc[-20:].values
    vol_5d  = hist["Volume"].iloc[-5:].values
    v1      = float(hist["Volume"].iloc[-1])
    avg20   = vol_20d.mean()
    avg5    = vol_5d.mean()
    if avg20 == 0 or avg5 == 0:
        return None
    v3    = float(hist["Volume"].iloc[-3])
    if v3 == 0:
        return None
    trend = np.polyfit(range(5), vol_5d, 1)[0] / avg5 * 100
    accel = (v1 - v3) / v3 * 100
    ratio = v1 / avg20
    total = min(trend / 10, 3.0) + min(accel / 100, 3.0) + min(ratio / 3, 4.0)
    return {"score": round(total, 2), "ratio": round(ratio, 2)}

# ──────────────────────────────────────────────
# 地合い判定（daily_market_stats.csv から）
# ──────────────────────────────────────────────
def build_conditions():
    df = pd.read_csv(MARKET_STATS_CSV, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    conds = {}
    for _, r in df.iterrows():
        ad  = float(r["ad_ratio"])
        nk  = float(r["nikkei_est"])
        if nk <= -2.0 or ad <= 0.20:
            c = "PANIC"
        elif nk <= -1.0 or ad <= 0.35:
            c = "WEAK"
        elif nk >= 0.5 and ad >= 0.60:
            c = "STRONG"
        else:
            c = "NORMAL"
        conds[r["date"]] = c
    return conds

# ──────────────────────────────────────────────
# 全キャッシュ読み込み（日付×コード → OHLCV）
# ──────────────────────────────────────────────
def load_all_cache():
    print("キャッシュ読み込み中（しばらくかかります）...")
    dfs = []
    for f in glob.glob(f"{CACHE_DIR}/*.csv"):
        code = os.path.splitext(os.path.basename(f))[0]
        try:
            df = pd.read_csv(f, encoding="utf-8-sig",
                             usecols=["Date","Open","High","Low","Close","Volume"])
            df["code"] = code
            dfs.append(df)
        except Exception:
            pass
    all_df = pd.concat(dfs, ignore_index=True)
    all_df["Date"] = pd.to_datetime(all_df["Date"]).dt.date
    all_df = all_df.sort_values(["code","Date"]).reset_index(drop=True)
    print(f"  読み込み完了: {len(all_df):,}行")
    return all_df

# ──────────────────────────────────────────────
# 遡及スキャン：各WEAK日の前日に score7-9 の銘柄を特定
# ──────────────────────────────────────────────
def retroactive_scan(all_df, conditions):
    all_dates = sorted(all_df["Date"].unique().tolist())
    date_idx  = {d: i for i, d in enumerate(all_dates)}

    # コード別にグループ化（高速化）
    grouped = {code: grp.reset_index(drop=True)
               for code, grp in all_df.groupby("code")}

    signals = []  # {scan_date, entry_date, code, score, ratio}

    weak_dates = [d for d, c in conditions.items() if c == "WEAK" and d in date_idx]
    print(f"WEAK日: {len(weak_dates)}日分を遡及スキャン中...")

    for i, entry_date in enumerate(sorted(weak_dates)):
        ei = date_idx[entry_date]
        if ei == 0:
            continue
        scan_date = all_dates[ei - 1]  # 前日 = スキャン日

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(weak_dates)}日完了...")

        for code, grp in grouped.items():
            # scan_date までのデータを取得
            grp_past = grp[grp["Date"] <= scan_date]
            if len(grp_past) < 22:
                continue

            last = grp_past.iloc[-1]
            if float(last["Volume"]) < MIN_VOLUME:
                continue
            if float(last["Close"]) < MIN_PRICE:
                continue

            sc = calc_score(grp_past)
            if sc is None:
                continue

            signals.append({
                "scan_date":  scan_date,
                "entry_date": entry_date,
                "code":       code,
                "score":      sc["score"],
                "ratio":      sc["ratio"],
            })

    print(f"  シグナル総数: {len(signals):,}件")
    return pd.DataFrame(signals)

# ──────────────────────────────────────────────
# トレードリターン計算
# ──────────────────────────────────────────────
def calc_return(o, h, l, c, tp_pct, sl_pct):
    tp_price = o * (1 + tp_pct / 100)
    tp_hit   = h >= tp_price
    sl_hit   = (sl_pct is not None) and (l <= o * (1 + sl_pct / 100))

    if tp_hit and not sl_hit:
        return tp_pct, "TP"
    elif sl_hit and not tp_hit:
        return sl_pct, "SL"
    elif tp_hit and sl_hit:
        cal = SL_CALIBRATION.get(sl_pct, {"tp_prob":0.2,"sl_prob":0.6,"neither_prob":0.2})
        ret = cal["tp_prob"]*tp_pct + cal["sl_prob"]*sl_pct + cal["neither_prob"]*(c-o)/o*100
        return round(ret, 3), "BOTH"
    else:
        return round((c - o) / o * 100, 3), "CLOSE"

# ──────────────────────────────────────────────
# シミュレーション
# ──────────────────────────────────────────────
def run_sim(signals_df, cache_idx, score_min, score_max,
            tp_pct, sl_pct, top_n):
    df = signals_df[
        (signals_df["score"] >= score_min) &
        (signals_df["score"] <  score_max)
    ].copy()

    if top_n is not None:
        df = df.groupby("scan_date", group_keys=False).apply(
            lambda x: x.nlargest(top_n, "score")
        )

    trades = []
    for _, row in df.iterrows():
        key = (row["code"], row["entry_date"])
        if key not in cache_idx:
            continue
        pr = cache_idx[key]
        o, h, l, c = pr["Open"], pr["High"], pr["Low"], pr["Close"]
        if o <= 0:
            continue
        ret, etype = calc_return(o, h, l, c, tp_pct, sl_pct)
        trades.append({"entry_date": row["entry_date"], "return_pct": ret,
                        "exit_type": etype, "score": row["score"]})

    if not trades:
        return None

    tdf   = pd.DataFrame(trades)
    daily = tdf.groupby("entry_date")["return_pct"].mean()
    rets  = daily.values

    cap = 1.0; peak = 1.0; max_dd = 0.0
    for r in rets:
        cap  *= (1 + r / 100)
        peak  = max(peak, cap)
        max_dd = min(max_dd, (cap - peak) / peak * 100)

    sharpe   = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    win_rate = (tdf["return_pct"] > 0).mean() * 100

    return {
        "n":        len(tdf),
        "days":     len(daily),
        "total":    round((cap - 1) * 100, 2),
        "annual":   round(((cap ** (252 / max(len(daily), 1))) - 1) * 100, 1),
        "sharpe":   round(sharpe, 2),
        "dd":       round(max_dd, 2),
        "win_rate": round(win_rate, 1),
        "avg":      round(tdf["return_pct"].mean(), 3),
    }

# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────
def main():
    conditions = build_conditions()
    all_df     = load_all_cache()

    # エントリー日の価格インデックス（高速化）
    cache_idx = {}
    for _, r in all_df.iterrows():
        cache_idx[(r["code"], r["Date"])] = {
            "Open": float(r["Open"]), "High": float(r["High"]),
            "Low":  float(r["Low"]),  "Close": float(r["Close"]),
        }

    signals_df = retroactive_scan(all_df, conditions)

    score_bands = [
        ("score5-7", 5.0,  7.0),
        ("score7-9", 7.0,  9.0),
        ("score8-9", 8.0,  9.0),
        ("score9+",  9.0, 99.0),
    ]
    tp_list   = [3.0, 5.0]
    sl_list   = [-2.0, -3.0, -5.0, None]
    topn_list = [3, 5, None]

    results = []
    total = len(score_bands) * len(tp_list) * len(sl_list) * len(topn_list)
    print(f"\nパラメータスイープ（{total}組み合わせ）...")

    for band, s_min, s_max in score_bands:
        for tp in tp_list:
            for sl in sl_list:
                for top_n in topn_list:
                    st = run_sim(signals_df, cache_idx, s_min, s_max, tp, sl, top_n)
                    if st:
                        results.append({
                            "スコア帯": band, "TP": f"+{tp}%",
                            "SL": f"{sl}%" if sl else "なし",
                            "上位N": top_n if top_n else "全件", **st,
                        })

    df_r = pd.DataFrame(results).sort_values("sharpe", ascending=False)

    print("\n" + "=" * 95)
    print("【WEAK日スキャルプ戦略 遡及シミュレーション結果】（Sharpe降順 上位20件）")
    print(f"  {'スコア帯':<10} {'TP':>5} {'SL':>6} {'N':>5}  "
          f"{'件数':>5} {'日数':>4}  {'累計':>7} {'年率':>7} {'Sharpe':>7} {'DD':>7} {'勝率':>6} {'avg':>7}")
    print("  " + "-" * 90)
    for _, r in df_r.head(20).iterrows():
        print(f"  {r['スコア帯']:<10} {r['TP']:>5} {r['SL']:>6} {str(r['上位N']):>5}  "
              f"{r['n']:>4}件 {r['days']:>3}日  "
              f"{r['total']:>+6.1f}% {r['annual']:>+6.1f}% {r['sharpe']:>+6.2f}  "
              f"{r['dd']:>+6.1f}% {r['win_rate']:>5.1f}%  {r['avg']:>+6.3f}%")

    print("\n" + "=" * 95)
    print("【score7-9 全組み合わせ】")
    df_79 = df_r[df_r["スコア帯"] == "score7-9"]
    for _, r in df_79.iterrows():
        print(f"  {r['TP']:>5} {r['SL']:>6} {str(r['上位N']):>5}  "
              f"{r['n']:>4}件 {r['days']:>3}日  "
              f"{r['total']:>+6.1f}% {r['annual']:>+6.1f}% {r['sharpe']:>+6.2f}  "
              f"{r['dd']:>+6.1f}% {r['win_rate']:>5.1f}%  {r['avg']:>+6.3f}%")

    out_path = "out/weak_scalp_results.csv"
    df_r.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n結果保存: {out_path}")

if __name__ == "__main__":
    main()
