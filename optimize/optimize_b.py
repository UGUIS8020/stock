"""
optimize_b.py - 戦略B 最適パラメータ探索（定期実行用）

【実行方法】
    python optimize_b.py              # 全Stage実行
    python optimize_b.py --stage 1    # Stage 1のみ（グリッドサーチ）
    python optimize_b.py --stage 2    # Stage 2のみ（ウォークフォワード）
    python optimize_b.py --stage 3    # Stage 3のみ（マクロ連動）
    python optimize_b.py --quick      # 高速モード（上位結果のみ表示）

【出力ファイル】
    out/opt_b_trades.csv       - 全トレードデータ（-3%以下）
    out/opt_b_stage1.csv       - Stage1 グリッドサーチ結果
    out/opt_b_stage2.csv       - Stage2 ウォークフォワード結果
    out/opt_b_stage3.csv       - Stage3 マクロ連動結果
    out/opt_b_report.txt       - サマリーレポート

【定期実行の目安】
    毎月1回（月初）に実行して最新データで最適パラメータを更新
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
import argparse
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

# ══════════════════════════════════════════════
# 定数
# ══════════════════════════════════════════════
CACHE_DIR     = "out/cache"
OUT_DIR       = "out"
MIN_VOLUME    = 50_000
MIN_SAMPLE    = 30          # パラメータ評価に必要な最小サンプル数
WF_SPLIT      = 0.60        # ウォークフォワード: 前60%をin-sample

PANIC_NIKKEI  = -2.0;  PANIC_AD  = 0.20
WEAK_NIKKEI   = -1.0;  WEAK_AD   = 0.35
STRONG_NIKKEI =  0.5;  STRONG_AD = 0.60

SL_CALIBRATION = {
    -1.0: {"tp_prob": 0.101, "sl_prob": 0.810},
    -2.0: {"tp_prob": 0.155, "sl_prob": 0.684},
    -3.0: {"tp_prob": 0.183, "sl_prob": 0.586},
    -5.0: {"tp_prob": 0.220, "sl_prob": 0.480},
}

# マクロトリガー定義（scan_morning.py の SECTOR_TRIGGERS と同じ銘柄群）
MACRO_SECTORS = {
    "自動車":   ["7203","7267","7270","7201","6954"],
    "エネルギー": ["5020","5019","8002","8031","8053"],
    "半導体":   ["8035","6857","6963","4063","6146"],
    "金属・資源": ["5713","5711","5401","5406","5012"],
}


# ══════════════════════════════════════════════
# ユーティリティ
# ══════════════════════════════════════════════
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
    """daily_market_stats.csv から地合いマップを生成"""
    ms_path = os.path.join(OUT_DIR, "daily_market_stats.csv")
    if not os.path.exists(ms_path):
        print("⚠️  daily_market_stats.csv が見つかりません。キャッシュから再計算します...")
        return _build_market_conditions_from_cache()

    ms = pd.read_csv(ms_path)
    cond_map = {}
    for _, row in ms.iterrows():
        nk = float(row.get("nikkei_est", 0) or 0)
        ad = float(row.get("ad_ratio", 0.5) or 0.5)
        if   nk <= PANIC_NIKKEI or ad <= PANIC_AD:      c = "PANIC"
        elif nk <= WEAK_NIKKEI  or ad <= WEAK_AD:       c = "WEAK"
        elif nk >= STRONG_NIKKEI and ad >= STRONG_AD:   c = "STRONG"
        else:                                            c = "NORMAL"
        cond_map[str(row["date"])] = c
    return cond_map


def _build_market_conditions_from_cache():
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
    big = big.dropna(subset=["Close"]).sort_values("Date")
    cond_map = {}
    for date, grp in big.groupby("Date"):
        changes = grp["Close"].pct_change().dropna() * 100
        if len(changes) < 10:
            continue
        ad = (changes > 0).mean()
        nk = changes.median()
        if   nk <= PANIC_NIKKEI or ad <= PANIC_AD:      c = "PANIC"
        elif nk <= WEAK_NIKKEI  or ad <= WEAK_AD:       c = "WEAK"
        elif nk >= STRONG_NIKKEI and ad >= STRONG_AD:   c = "STRONG"
        else:                                            c = "NORMAL"
        cond_map[date.strftime("%Y-%m-%d")] = c
    return cond_map


def sim_tp_sl(sub, tp, sl):
    """TP/SL適用後リターン配列を返す。tp=None なら引け決済。"""
    if len(sub) == 0:
        return np.array([])
    if tp is None:
        return sub["ret_oc"].values
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


def _calc_metrics_small(rets):
    """MIN_SAMPLE未満でも計算（Sharpeは省略）"""
    if len(rets) < 10:
        return None
    wr  = (rets > 0).mean() * 100
    avg = rets.mean()
    return {"n": len(rets), "wr": round(wr,1), "avg": round(avg,3),
            "sharpe": 0.0, "max_dd": 0.0, "cum": round(rets.sum(),1)}


def calc_metrics(rets):
    """勝率・avg・Sharpe・最大DD・累計を返す"""
    if len(rets) < MIN_SAMPLE:
        return None
    wr   = (rets > 0).mean() * 100
    avg  = rets.mean()
    std  = rets.std()
    sharpe = avg / std * np.sqrt(252) if std > 0 else 0.0
    cum  = np.cumsum(rets)
    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak).min()
    return {"n": len(rets), "wr": round(wr,1), "avg": round(avg,3),
            "sharpe": round(sharpe,2), "max_dd": round(dd,1), "cum": round(rets.sum(),1)}


# ══════════════════════════════════════════════
# データ収集（全銘柄スキャン）
# ══════════════════════════════════════════════
def collect_trades(market_cond, drop_threshold=-3.0):
    """
    drop_threshold 以下の下落銘柄を全件収集してトレードリストを返す。
    広めに -3% で収集し、後でフィルタリングする。
    """
    trades = []
    files  = sorted(glob.glob(f"{CACHE_DIR}/*.csv"))
    print(f"  スキャン中: {len(files)}銘柄...", flush=True)

    for i, filepath in enumerate(files):
        if i % 1000 == 0 and i > 0:
            print(f"  {i}/{len(files)}... ({len(trades):,}件)", flush=True)
        try:
            df = pd.read_csv(filepath)
            df["Date"]   = pd.to_datetime(df["Date"])
            for col in ["Close","Open","High","Low","Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.sort_values("Date").dropna(
                subset=["Close","Open","High","Low"]).reset_index(drop=True)
            code = os.path.basename(filepath).replace(".csv","")

            for idx in range(25, len(df) - 1):
                today    = df.iloc[idx]
                next_row = df.iloc[idx + 1]

                scan_dt  = today["Date"].strftime("%Y-%m-%d")
                trade_dt = next_row["Date"].strftime("%Y-%m-%d")
                scan_c   = market_cond.get(scan_dt,  "UNKNOWN")
                trade_c  = market_cond.get(trade_dt, "UNKNOWN")

                if scan_c in ("PANIC","UNKNOWN") or trade_c in ("PANIC","UNKNOWN"):
                    continue
                vol = float(today["Volume"]) if not pd.isna(today["Volume"]) else 0
                if vol < MIN_VOLUME:
                    continue

                prev_c = float(df.iloc[idx - 1]["Close"])
                if prev_c <= 0:
                    continue
                today_rise = (float(today["Close"]) - prev_c) / prev_c * 100
                if today_rise > drop_threshold:
                    continue

                hist     = df.iloc[:idx + 1]
                rb_score = calc_rebound_score(hist)
                if rb_score < 2:
                    continue

                op  = float(next_row["Open"])
                if op <= 0:
                    continue
                hi  = float(next_row["High"])
                lo  = float(next_row["Low"])
                cl  = float(next_row["Close"])
                gap = (op - float(today["Close"])) / float(today["Close"]) * 100

                trades.append({
                    "code":       code,
                    "scan_date":  scan_dt,
                    "trade_date": trade_dt,
                    "scan_cond":  scan_c,
                    "trade_cond": trade_c,
                    "today_rise": round(today_rise, 2),
                    "rb_score":   rb_score,
                    "gap_pct":    round(gap, 2),
                    "ret_oc":     round((cl - op) / op * 100, 2),
                    "ret_max":    round((hi - op) / op * 100, 2),
                    "ret_low":    round((lo - op) / op * 100, 2),
                })
        except Exception:
            pass

    return pd.DataFrame(trades)


# ══════════════════════════════════════════════
# Stage 1: グリッドサーチ
# ══════════════════════════════════════════════
def stage1_grid_search(df):
    print("\n" + "="*70)
    print("【Stage 1】グリッドサーチ — 全パラメータ組み合わせ探索")
    print("="*70)

    # 探索空間
    drop_los   = [-3.0, -4.0, -5.0, -6.0, -7.0, -8.0]  # 下限（これ以上の下落は対象外）
    drop_his   = [-3.0, -4.0, -5.0, -6.0]               # 上限（この%以下の下落が対象）
    rb_mins    = [2, 3, 4, 5, 6]
    top_ns     = [1, 3, 5, 999]                          # 1日に買う上位N件
    gap_maxs   = [999, 2.0, 0.0]                         # ギャップアップ上限
    conditions = ["ALL", "NORMAL", "STRONG", "NORMAL+STRONG"]
    exits      = [
        (None,  None,  "引け決済"),
        (2.0,  -2.0,  "TP+2%/SL-2%"),
        (3.0,  -3.0,  "TP+3%/SL-3%"),
        (5.0,  -3.0,  "TP+5%/SL-3%"),
        (5.0,  -5.0,  "TP+5%/SL-5%"),
    ]

    results = []
    total   = 0
    checked = 0

    for drop_lo in drop_los:
        for drop_hi in drop_his:
            if drop_lo >= drop_hi:
                continue  # 範囲が逆転するケースはスキップ
            for rb_min in rb_mins:
                for gap_max in gap_maxs:
                    for cond_label in conditions:
                        for tp, sl, exit_label in exits:
                            total += 1

    print(f"  総組み合わせ数: {total:,}通り")

    for drop_lo in drop_los:
        for drop_hi in drop_his:
            if drop_lo >= drop_hi:
                continue
            # フィルタ
            sub_drop = df[(df["today_rise"] <= drop_hi) & (df["today_rise"] > drop_lo)]

            for rb_min in rb_mins:
                sub_rb = sub_drop[sub_drop["rb_score"] >= rb_min]

                for gap_max in gap_maxs:
                    sub_gap = sub_rb if gap_max >= 999 else sub_rb[sub_rb["gap_pct"] <= gap_max]

                    for cond_label in conditions:
                        if cond_label == "ALL":
                            sub_c = sub_gap
                        elif cond_label == "NORMAL+STRONG":
                            sub_c = sub_gap[sub_gap["trade_cond"].isin(["NORMAL","STRONG"])]
                        else:
                            sub_c = sub_gap[sub_gap["trade_cond"] == cond_label]

                        for tp, sl, exit_label in exits:
                            checked += 1
                            if checked % 500 == 0:
                                print(f"  {checked}/{total}...", flush=True)

                            rets = sim_tp_sl(sub_c, tp, sl)
                            m    = calc_metrics(rets)
                            if m is None:
                                continue
                            results.append({
                                "drop_lo":    drop_lo,
                                "drop_hi":    drop_hi,
                                "rb_min":     rb_min,
                                "gap_max":    gap_max,
                                "condition":  cond_label,
                                "exit":       exit_label,
                                **m,
                            })

    res_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    out_path = os.path.join(OUT_DIR, "opt_b_stage1.csv")
    res_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n  完了: {len(res_df):,}件の有効組み合わせ")
    print(f"\n  ── Sharpe上位10 ──")
    print(f"  {'drop帯':>12} {'RB':>3} {'gap上限':>7} {'地合い':>14} {'出口':>14} "
          f"{'N':>5} {'勝率':>6} {'avg':>7} {'Sharpe':>7} {'累計%':>8}")
    print("  " + "─"*90)
    for _, r in res_df.head(10).iterrows():
        drop_band = f"{r['drop_lo']}〜{r['drop_hi']}%"
        gap_str   = f"+{r['gap_max']:.0f}%以下" if r["gap_max"] < 999 else "制限なし"
        print(f"  {drop_band:>12} {r['rb_min']:>3} {gap_str:>7} {r['condition']:>14} "
              f"{r['exit']:>14} {r['n']:>5} {r['wr']:>5.1f}% {r['avg']:>+6.3f}% "
              f"{r['sharpe']:>+6.2f} {r['cum']:>+7.1f}%")

    return res_df


# ══════════════════════════════════════════════
# Stage 2: ウォークフォワード検証
# ══════════════════════════════════════════════
def stage2_walkforward(df, stage1_df, top_n=30):
    print("\n" + "="*70)
    print(f"【Stage 2】ウォークフォワード検証 — Stage1上位{top_n}条件")
    print(f"  In-sample: 前{WF_SPLIT*100:.0f}%  Out-of-sample: 後{(1-WF_SPLIT)*100:.0f}%")
    print("="*70)

    all_dates = sorted(df["trade_date"].unique())
    split_idx = int(len(all_dates) * WF_SPLIT)
    split_date = all_dates[split_idx]

    df_in  = df[df["trade_date"] <  split_date]
    df_out = df[df["trade_date"] >= split_date]

    print(f"  分割日: {split_date}  In:{len(df_in):,}件 / Out:{len(df_out):,}件")

    results = []
    for _, row in stage1_df.head(top_n).iterrows():
        drop_lo  = row["drop_lo"]
        drop_hi  = row["drop_hi"]
        rb_min   = int(row["rb_min"])
        gap_max  = row["gap_max"]
        cond     = row["condition"]
        exit_lbl = row["exit"]

        # exit をTP/SLに変換
        tp, sl = _parse_exit(exit_lbl)

        def apply_filters(sub):
            s = sub[(sub["today_rise"] <= drop_hi) & (sub["today_rise"] > drop_lo)]
            s = s[s["rb_score"] >= rb_min]
            if gap_max < 999:
                s = s[s["gap_pct"] <= gap_max]
            if cond == "ALL":
                pass
            elif cond == "NORMAL+STRONG":
                s = s[s["trade_cond"].isin(["NORMAL","STRONG"])]
            else:
                s = s[s["trade_cond"] == cond]
            return s

        sub_in  = apply_filters(df_in)
        sub_out = apply_filters(df_out)

        rets_in  = sim_tp_sl(sub_in,  tp, sl)
        rets_out = sim_tp_sl(sub_out, tp, sl)
        if len(rets_in) < 10 or len(rets_out) < 10:
            continue
        m_in  = calc_metrics(rets_in)  if len(rets_in)  >= MIN_SAMPLE else _calc_metrics_small(rets_in)
        m_out = calc_metrics(rets_out) if len(rets_out) >= MIN_SAMPLE else _calc_metrics_small(rets_out)
        if m_in is None or m_out is None:
            continue

        stable = "★" if m_in["avg"] > 0 and m_out["avg"] > 0 else ""
        results.append({
            "drop_lo": drop_lo, "drop_hi": drop_hi,
            "rb_min": rb_min, "gap_max": gap_max,
            "condition": cond, "exit": exit_lbl,
            "in_n":      m_in["n"],  "in_wr":  m_in["wr"],
            "in_avg":    m_in["avg"], "in_sharpe": m_in["sharpe"],
            "out_n":     m_out["n"], "out_wr": m_out["wr"],
            "out_avg":   m_out["avg"], "out_sharpe": m_out["sharpe"],
            "stable":    stable,
        })

    if not results:
        print("  ※ 有効な条件が見つかりませんでした")
        return pd.DataFrame()
    res_df = pd.DataFrame(results).sort_values("out_avg", ascending=False)
    out_path = os.path.join(OUT_DIR, "opt_b_stage2.csv")
    res_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n  ── In/Out両方プラスの条件（★）──")
    print(f"  {'drop帯':>12} {'RB':>3} {'地合い':>14} {'出口':>14} "
          f"{'In avg':>7} {'Out avg':>7} {'Out N':>6}")
    print("  " + "─"*75)
    stable_df = res_df[res_df["stable"] == "★"]
    for _, r in stable_df.iterrows():
        drop_band = f"{r['drop_lo']}〜{r['drop_hi']}%"
        print(f"  {drop_band:>12} {r['rb_min']:>3} {r['condition']:>14} {r['exit']:>14} "
              f"{r['in_avg']:>+6.3f}% {r['out_avg']:>+6.3f}% {r['out_n']:>6}")

    if stable_df.empty:
        print("  ※ In/Out両方プラスの条件なし（過学習の可能性）")

    return res_df


def _parse_exit(label):
    if label == "引け決済":
        return None, None
    parts = label.replace("TP","").replace("SL","").replace("%","").split("/")
    try:
        tp = float(parts[0].strip())
        sl = float(parts[1].strip())
        return tp, sl
    except Exception:
        return None, None


# ══════════════════════════════════════════════
# Stage 3: マクロ連動フィルター
# ══════════════════════════════════════════════
def stage3_macro(df, market_cond):
    print("\n" + "="*70)
    print("【Stage 3】マクロ連動フィルター — セクター別効果検証")
    print("="*70)

    # セクター別銘柄の前日下落がある場合の翌日パフォーマンスを検証
    results = []

    for sector_name, codes in MACRO_SECTORS.items():
        # そのセクター銘柄が下落した日の翌日パフォーマンス
        sector_trades = df[df["code"].isin(codes)].copy()

        if len(sector_trades) < MIN_SAMPLE:
            continue

        # 同一セクター内で前日に複数銘柄が-4%以下の日（セクター全体売られた日）
        sector_multi = (
            sector_trades[sector_trades["today_rise"] <= -4.0]
            .groupby("scan_date")["code"].count()
        )
        strong_down_dates = sector_multi[sector_multi >= 2].index.tolist()

        sub_strong = sector_trades[
            (sector_trades["today_rise"] <= -4.0) &
            (sector_trades["scan_date"].isin(strong_down_dates))
        ]

        sub_normal = sector_trades[
            (sector_trades["today_rise"] <= -4.0) &
            (~sector_trades["scan_date"].isin(strong_down_dates))
        ]

        for label, sub in [("セクター同時売り", sub_strong), ("個別売り", sub_normal)]:
            for cond in ["ALL","NORMAL","WEAK"]:
                sub_c = sub if cond == "ALL" else sub[sub["trade_cond"] == cond]
                rets  = sim_tp_sl(sub_c, None, None)  # 引け決済
                m     = calc_metrics(rets)
                if m is None:
                    continue
                results.append({
                    "sector":    sector_name,
                    "pattern":   label,
                    "condition": cond,
                    **m,
                })

    if not results:
        print("  ※ セクターデータが不足しています（サンプル30件未満）")
        return pd.DataFrame()
    res_df = pd.DataFrame(results).sort_values("avg", ascending=False)
    out_path = os.path.join(OUT_DIR, "opt_b_stage3.csv")
    res_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n  ── セクター別パフォーマンス（引け決済）──")
    print(f"  {'セクター':>10} {'パターン':>14} {'地合い':>8} "
          f"{'N':>5} {'勝率':>6} {'avg':>7} {'累計%':>8}")
    print("  " + "─"*65)
    for _, r in res_df.iterrows():
        mark = " ★" if r["avg"] > 0 else ""
        print(f"  {r['sector']:>10} {r['pattern']:>14} {r['condition']:>8} "
              f"{r['n']:>5} {r['wr']:>5.1f}% {r['avg']:>+6.3f}% {r['cum']:>+7.1f}%{mark}")

    return res_df


# ══════════════════════════════════════════════
# レポート出力
# ══════════════════════════════════════════════
def write_report(stage1_df, stage2_df, stage3_df, run_date):
    report = []
    report.append(f"戦略B 最適化レポート — {run_date}")
    report.append("=" * 70)

    report.append("\n【Stage 1 最優秀条件 Top5（Sharpe順）】")
    for _, r in stage1_df.head(5).iterrows():
        report.append(
            f"  drop{r['drop_lo']}〜{r['drop_hi']}% / RB>={r['rb_min']:.0f} / "
            f"{r['condition']} / {r['exit']} | "
            f"N={r['n']} WR={r['wr']}% avg={r['avg']:+.3f}% Sharpe={r['sharpe']:+.2f}"
        )

    report.append("\n【Stage 2 In/Out両方プラス条件（推奨パラメータ）】")
    stable = stage2_df[stage2_df["stable"] == "★"] if stage2_df is not None else pd.DataFrame()
    if stable.empty:
        report.append("  ※ 該当なし（全データでの最良条件を暫定使用）")
        report.append(f"  暫定: {stage1_df.iloc[0]['drop_lo']}〜{stage1_df.iloc[0]['drop_hi']}% / "
                      f"RB>={stage1_df.iloc[0]['rb_min']:.0f} / {stage1_df.iloc[0]['condition']} / "
                      f"{stage1_df.iloc[0]['exit']}")
    else:
        for _, r in stable.head(3).iterrows():
            report.append(
                f"  drop{r['drop_lo']}〜{r['drop_hi']}% / RB>={r['rb_min']:.0f} / "
                f"{r['condition']} / {r['exit']} | "
                f"In avg={r['in_avg']:+.3f}% → Out avg={r['out_avg']:+.3f}%"
            )

    report.append("\n【Stage 3 有効なセクター連動パターン】")
    if stage3_df is not None and not stage3_df.empty and "avg" in stage3_df.columns:
        top_macro = stage3_df[stage3_df["avg"] > 0].head(5)
        for _, r in top_macro.iterrows():
            report.append(
                f"  {r['sector']} × {r['pattern']} × {r['condition']} | "
                f"N={r['n']} avg={r['avg']:+.3f}%"
            )
    else:
        report.append("  ※ セクターデータ不足（大型株キャッシュが少ない）")

    report.append(f"\n実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("次回実行推奨: 来月初（データ蓄積後）")

    txt = "\n".join(report)
    path = os.path.join(OUT_DIR, "opt_b_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"\n{'='*70}")
    print(txt)
    print(f"\nレポート保存: {path}")


# ══════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="戦略B 最適パラメータ探索")
    parser.add_argument("--stage", type=int, choices=[1,2,3], help="実行するStageを指定")
    parser.add_argument("--quick", action="store_true", help="高速モード（top5表示のみ）")
    args = parser.parse_args()

    run_date  = datetime.now().strftime("%Y-%m-%d")
    run_all   = args.stage is None
    trades_path = os.path.join(OUT_DIR, "opt_b_trades.csv")

    print(f"{'='*70}")
    print(f"  戦略B 最適化シミュレーション  {run_date}")
    print(f"{'='*70}")

    # ── トレードデータ収集（キャッシュがあれば再利用）──
    if os.path.exists(trades_path):
        print(f"\n既存トレードデータを読み込み中: {trades_path}")
        df = pd.read_csv(trades_path, encoding="utf-8-sig")
        print(f"  {len(df):,}件 読み込み完了")

        # 最新キャッシュより古ければ再収集
        cache_files = glob.glob(f"{CACHE_DIR}/*.csv")
        if cache_files:
            latest_cache = max(os.path.getmtime(f) for f in cache_files)
            trades_mtime = os.path.getmtime(trades_path)
            if latest_cache > trades_mtime:
                print("  ⚠️  キャッシュが更新されています。再収集します...")
                df = None
    else:
        df = None

    if df is None:
        print("\n地合い計算中...")
        market_cond = build_market_conditions()
        print(f"  {len(market_cond)}日分の地合いデータ")
        print("\nトレードデータ収集中（-3%以下・全銘柄）...")
        df = collect_trades(market_cond, drop_threshold=-3.0)
        print(f"  収集完了: {len(df):,}件")
        df.to_csv(trades_path, index=False, encoding="utf-8-sig")
        print(f"  保存: {trades_path}")

    print(f"\nデータ概要: {len(df):,}件 / {df['trade_date'].nunique()}日")
    print(f"  期間: {df['scan_date'].min()} 〜 {df['scan_date'].max()}")
    print(f"  地合い別: " + " / ".join(
        f"{c}:{len(df[df['trade_cond']==c])}件"
        for c in ["STRONG","NORMAL","WEAK"]))

    stage1_df = stage2_df = stage3_df = None

    # ── Stage 1 ──
    if run_all or args.stage == 1:
        stage1_df = stage1_grid_search(df)

    # ── Stage 2 ──
    if run_all or args.stage == 2:
        if stage1_df is None:
            s1_path = os.path.join(OUT_DIR, "opt_b_stage1.csv")
            if os.path.exists(s1_path):
                stage1_df = pd.read_csv(s1_path, encoding="utf-8-sig")
                print(f"\nStage1結果を読み込み: {s1_path}")
            else:
                print("⚠️  Stage1結果が見つかりません。先にStage1を実行してください。")
                return
        stage2_df = stage2_walkforward(df, stage1_df)

    # ── Stage 3 ──
    if run_all or args.stage == 3:
        if not hasattr(build_market_conditions, "_cache"):
            print("\n地合い計算中（Stage3用）...")
        market_cond = build_market_conditions()
        stage3_df = stage3_macro(df, market_cond)

    # ── レポート ──
    if run_all:
        write_report(stage1_df, stage2_df, stage3_df, run_date)
    elif stage1_df is not None:
        print(f"\nStage1結果: {os.path.join(OUT_DIR, 'opt_b_stage1.csv')}")
    elif stage2_df is not None:
        print(f"\nStage2結果: {os.path.join(OUT_DIR, 'opt_b_stage2.csv')}")
    elif stage3_df is not None:
        print(f"\nStage3結果: {os.path.join(OUT_DIR, 'opt_b_stage3.csv')}")


if __name__ == "__main__":
    main()
