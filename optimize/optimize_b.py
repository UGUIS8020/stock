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
import os
import sys
import argparse
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import db

# ══════════════════════════════════════════════
# 定数
# ══════════════════════════════════════════════
OUT_DIR       = "out"
MIN_VOLUME    = 50_000
MIN_PRICE     = 200          # 株価200円未満除外（倒産・上場廃止リスク）
MIN_TURNOVER  = 50_000_000   # 売買代金5,000万円未満除外（流動性不足、scan_daily.pyと統一）
RET_WIN_HI    =  10.0        # 翌日リターン上限（+10%超=M&A・TOBなど特殊イベント）
RET_WIN_LO    = -10.0        # 翌日リターン下限（-10%未満=不祥事・倒産前兆など特殊イベント）
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
    score   = 0
    closes  = pd.to_numeric(hist["Close"], errors="coerce").fillna(0).values
    vol_ser = pd.to_numeric(hist["Volume"], errors="coerce").fillna(0)
    rsi = calc_rsi(closes, 14)
    if rsi is not None:
        if rsi < 20:   score += 4
        elif rsi < 30: score += 3
        elif rsi < 40: score += 2
        elif rsi < 50: score += 1
    if len(hist) >= 21:
        avg20 = vol_ser.iloc[-21:-1].mean()
        vol   = float(vol_ser.iloc[-1])
        if avg20 > 0:
            r = vol / avg20
            if r >= 3.0:   score += 3
            elif r >= 2.0: score += 2
            elif r >= 1.5: score += 1
    ma25 = pd.to_numeric(hist["Close"], errors="coerce").iloc[-25:].mean()
    dev  = (float(hist["Close"].iloc[-1]) / ma25 - 1) * 100 if ma25 > 0 else 0
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
    conn = db.get_conn()
    df = pd.read_sql(
        "SELECT Date, Close, prev_close FROM daily_prices WHERE prev_close > 0 AND Close > 0",
        conn
    )
    conn.close()
    df["Date"]       = pd.to_datetime(df["Date"])
    df["change_pct"] = (df["Close"] - df["prev_close"]) / df["prev_close"] * 100
    cond_map = {}
    for date, grp in df.groupby("Date"):
        up  = (grp["change_pct"] > 0).sum()
        tot = len(grp)
        ad  = up / tot if tot > 0 else 0.5
        nk  = float(grp["change_pct"].median())
        if   nk <= PANIC_NIKKEI or ad <= PANIC_AD:    c = "PANIC"
        elif nk <= WEAK_NIKKEI  or ad <= WEAK_AD:     c = "WEAK"
        elif nk >= STRONG_NIKKEI and ad >= STRONG_AD: c = "STRONG"
        else:                                         c = "NORMAL"
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
def collect_trades(market_cond, drop_threshold=-3.0, cutoff_date=None):
    """
    drop_threshold 以下の下落銘柄を全件収集してトレードリストを返す。
    広めに -3% で収集し、後でフィルタリングする。
    cutoff_date: この日付以降のデータのみ使用（例: 1年前の日付）

    適用フィルタ:
      - 出来高 < MIN_VOLUME (50,000株) → 流動性なし
      - 株価 < MIN_PRICE (200円) → 倒産・上場廃止リスク
      - 売買代金 < MIN_TURNOVER (5,000万円) → 実際に買えない
    """
    trades = []
    skip_price    = 0   # 株価フィルタで除外
    skip_turnover = 0   # 売買代金フィルタで除外
    codes  = db.get_all_codes()
    total  = len(codes)
    if cutoff_date:
        print(f"  スキャン中: {total:,}銘柄（{cutoff_date}以降のデータのみ）...", flush=True)
    else:
        print(f"  スキャン中: {total:,}銘柄...", flush=True)

    for i, code in enumerate(codes):
        if i % 1000 == 0 and i > 0:
            print(f"  {i}/{total}... ({len(trades):,}件)", flush=True)
        try:
            df = db.get_stock_history(code)
            if df.empty:
                continue
            df["Date"] = pd.to_datetime(df["Date"])
            for col in ["Close", "Open", "High", "Low", "Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["Volume"] = df["Volume"].fillna(0)
            df = df.sort_values("Date").dropna(
                subset=["Close", "Open", "High", "Low"]).reset_index(drop=True)

            for idx in range(25, len(df) - 1):
                today    = df.iloc[idx]
                next_row = df.iloc[idx + 1]

                if cutoff_date and today["Date"] < pd.Timestamp(cutoff_date):
                    continue

                scan_dt  = today["Date"].strftime("%Y-%m-%d")
                trade_dt = next_row["Date"].strftime("%Y-%m-%d")
                scan_c   = market_cond.get(scan_dt,  "UNKNOWN")
                trade_c  = market_cond.get(trade_dt, "UNKNOWN")

                if scan_c in ("PANIC", "UNKNOWN") or trade_c in ("PANIC", "UNKNOWN"):
                    continue
                if float(today["Volume"]) < MIN_VOLUME:
                    continue

                # 株価フィルタ（200円未満除外）
                today_close = float(today["Close"])
                if today_close < MIN_PRICE:
                    skip_price += 1
                    continue

                # 売買代金フィルタ（5,000万円未満除外）
                if today_close * float(today["Volume"]) < MIN_TURNOVER:
                    skip_turnover += 1
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

                # ── パターン特徴量 ──
                t_o = float(today["Open"])
                t_h = float(today["High"])
                t_l = float(today["Low"])
                t_c = float(today["Close"])
                day_range = t_h - t_l
                lower_shadow = min(t_o, t_c) - t_l
                lower_shadow_ratio = round(lower_shadow / day_range, 3) if day_range > 0 else 0.0

                # 直近3日の出来高が連続減少（売り枯れ）
                if idx >= 2:
                    v0 = float(df.iloc[idx - 2]["Volume"])
                    v1 = float(df.iloc[idx - 1]["Volume"])
                    v2 = float(today["Volume"])
                    vol_decay_3d = int(v0 > v1 > v2 and v2 > 0)
                else:
                    vol_decay_3d = 0

                # 連続下落日数（終値ベース、当日含む）
                consec_drop = 1  # 当日は today_rise < drop_threshold で確認済み
                for back in range(idx - 1, max(0, idx - 9), -1):
                    if back == 0:
                        break
                    if float(df.iloc[back]["Close"]) < float(df.iloc[back - 1]["Close"]):
                        consec_drop += 1
                    else:
                        break

                trades.append({
                    "code":                code,
                    "scan_date":           scan_dt,
                    "trade_date":          trade_dt,
                    "scan_cond":           scan_c,
                    "trade_cond":          trade_c,
                    "today_rise":          round(today_rise, 2),
                    "rb_score":            rb_score,
                    "gap_pct":             round(gap, 2),
                    "ret_oc":              round((cl - op) / op * 100, 2),
                    "ret_max":             round((hi - op) / op * 100, 2),
                    "ret_low":             round((lo - op) / op * 100, 2),
                    "lower_shadow_ratio":  lower_shadow_ratio,
                    "vol_decay_3d":        vol_decay_3d,
                    "consec_drop":         consec_drop,
                })
        except Exception:
            pass

    print(f"  [フィルタ] 株価<{MIN_PRICE}円除外: {skip_price:,}件 / "
          f"売買代金<{MIN_TURNOVER//10000:,}万円除外: {skip_turnover:,}件", flush=True)
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
    parser.add_argument("--stage",  type=int, choices=[1,2,3], help="実行するStageを指定")
    parser.add_argument("--quick",  action="store_true", help="高速モード（top5表示のみ）")
    parser.add_argument("--months", type=int, default=16, help="使用するデータの期間（月数、デフォルト16）")
    args = parser.parse_args()

    db.init_db()
    run_date    = datetime.now().strftime("%Y-%m-%d")
    run_all     = args.stage is None
    cutoff_date = (datetime.now() - timedelta(days=args.months * 30)).strftime("%Y-%m-%d")
    trades_path = os.path.join(OUT_DIR, "opt_b_trades.csv")
    print(f"  データ期間: 直近{args.months}ヶ月（{cutoff_date}以降）")

    print(f"{'='*70}")
    print(f"  戦略B 最適化シミュレーション  {run_date}")
    print(f"{'='*70}")

    # ── トレードデータ収集（保存済みがあれば再利用）──
    if os.path.exists(trades_path):
        print(f"\n既存トレードデータを読み込み中: {trades_path}")
        df = pd.read_csv(trades_path, encoding="utf-8-sig")
        before = len(df)
        df = df[df["scan_date"] >= cutoff_date]
        print(f"  {before:,}件 → {len(df):,}件（{cutoff_date}以降でフィルタ）")
    else:
        df = None

    filter_log = []   # フィルタ適用記録（後でCSV保存）

    if df is None or df.empty:
        print("\n地合い計算中...")
        market_cond = build_market_conditions()
        print(f"  {len(market_cond)}日分の地合いデータ")
        print("\nトレードデータ収集中（-3%以下・全銘柄）...")
        df = collect_trades(market_cond, drop_threshold=-3.0, cutoff_date=cutoff_date)
        n_raw = len(df)
        print(f"  収集完了（生データ）: {n_raw:,}件")

        # ── ウィンザライゼーション: 翌日リターン ±10% 超を除外 ──
        mask_win  = (df["ret_oc"] >= RET_WIN_LO) & (df["ret_oc"] <= RET_WIN_HI)
        n_removed = (~mask_win).sum()
        df        = df[mask_win].reset_index(drop=True)
        print(f"  [ウィンザライゼーション] ±{RET_WIN_HI:.0f}%超除外: {n_removed:,}件 → {len(df):,}件")
        filter_log.append({
            "run_date":      run_date,
            "filter":        f"ウィンザライゼーション(±{RET_WIN_HI:.0f}%超)",
            "before":        n_raw,
            "removed":       n_removed,
            "after":         len(df),
            "note":          "M&A・TOB・不祥事など再現不能な特殊イベントを除外",
        })

        df.to_csv(trades_path, index=False, encoding="utf-8-sig")
        print(f"  保存: {trades_path}")

    # ── フィルタログ保存 ──
    if filter_log:
        log_path = os.path.join(OUT_DIR, "opt_b_filter_log.csv")
        new_log  = pd.DataFrame(filter_log)
        if os.path.exists(log_path):
            old_log = pd.read_csv(log_path, encoding="utf-8-sig")
            new_log = pd.concat([old_log, new_log], ignore_index=True)
        new_log.to_csv(log_path, index=False, encoding="utf-8-sig")
        print(f"  フィルタログ保存: {log_path}")

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

    # ── S3アップロード（他端末と共有）──
    _upload_optimize_to_s3()


def _upload_optimize_to_s3():
    """最適化ファイルをS3にアップロードして他端末と共有"""
    upload_files = [
        os.path.join(OUT_DIR, "opt_b_trades.csv"),
        os.path.join(OUT_DIR, "opt_b_stage1.csv"),
        os.path.join(OUT_DIR, "opt_b_stage2.csv"),
        os.path.join(OUT_DIR, "opt_b_report.txt"),
    ]
    try:
        import boto3
        s3 = boto3.client("s3")
        bucket = "shibuya8020"
        uploaded = 0
        for path in upload_files:
            if not os.path.exists(path):
                continue
            key = f"stock-optimize/{os.path.basename(path)}"
            s3.upload_file(path, bucket, key)
            uploaded += 1
        if uploaded:
            print(f"  ☁️  S3アップロード完了: {uploaded}ファイル → s3://{bucket}/stock-optimize/")
    except Exception as e:
        print(f"  ⚠️  S3アップロード失敗（ローカル保存は完了）: {e}")


if __name__ == "__main__":
    main()
