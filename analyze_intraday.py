"""
analyze_intraday.py - WEAK日高スコア銘柄の寄り付き後値動きパターン分析

【目的】
WEAK日にCAUTION判定された高スコア銘柄が、
当日寄り付き後にどれだけ動くかを検証する。

【出力指標】
- Open→High : 寄り付きから日中高値までの上昇幅（最大取れる利益）
- Open→Close: 寄り付きから引け値（終日保有した場合）
- Open→Low  : 寄り付きから日中安値（最大リスク）

【実行】
    python analyze_intraday.py
"""

import os
import pandas as pd
import numpy as np

SCAN_CSV  = "out/scan_results.csv"
CACHE_DIR = "out/cache"

def get_next_trading_date(df_all_dates, date):
    """日付の翌取引日を返す"""
    dates = sorted(df_all_dates)
    idx = dates.index(date) if date in dates else -1
    if idx == -1 or idx + 1 >= len(dates):
        return None
    return dates[idx + 1]

def load_cache(code):
    """キャッシュCSVを読み込む"""
    path = os.path.join(CACHE_DIR, f"{code}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        return df
    except Exception:
        return None

def main():
    # ── 1. scan_results 読み込み ──────────────────────
    scan = pd.read_csv(SCAN_CSV, encoding="utf-8-sig")
    scan = scan[scan["market_condition"].notna()].copy()
    scan["scan_date"] = pd.to_datetime(scan["scan_date"]).dt.date
    scan["code"] = scan["code"].astype(str).str.strip()

    # 全取引日リストを作成（翌日計算用）
    all_dates = sorted(scan["scan_date"].unique().tolist())

    # ── 2. 地合い別・スコア帯別に集計 ──────────────────
    conditions  = ["WEAK", "NORMAL", "STRONG", "PANIC"]
    score_bands = [
        ("score5-7",  5.0,  7.0),
        ("score7-9",  7.0,  9.0),
        ("score9+",   9.0, 99.0),
        ("score8+",   8.0, 99.0),
    ]

    results = []

    for cond in conditions:
        df_cond = scan[scan["market_condition"] == cond]

        for band_name, s_min, s_max in score_bands:
            df_band = df_cond[
                (df_cond["score"] >= s_min) & (df_cond["score"] < s_max)
            ]
            if len(df_band) == 0:
                continue

            open_to_high  = []
            open_to_close = []
            open_to_low   = []

            for _, row in df_band.iterrows():
                code      = row["code"]
                scan_date = row["scan_date"]

                # 翌取引日を特定（エントリー日）
                entry_date = get_next_trading_date(all_dates, scan_date)
                if entry_date is None:
                    continue

                # キャッシュから価格取得
                df_cache = load_cache(code)
                if df_cache is None:
                    continue

                day = df_cache[df_cache["Date"] == entry_date]
                if day.empty:
                    continue

                o = float(day["Open"].iloc[0])
                h = float(day["High"].iloc[0])
                l = float(day["Low"].iloc[0])
                c = float(day["Close"].iloc[0])

                if o <= 0:
                    continue

                open_to_high.append((h - o) / o * 100)
                open_to_close.append((c - o) / o * 100)
                open_to_low.append((l - o) / o * 100)

            n = len(open_to_high)
            if n == 0:
                continue

            oh = np.array(open_to_high)
            oc = np.array(open_to_close)
            ol = np.array(open_to_low)

            results.append({
                "地合い":       cond,
                "スコア帯":     band_name,
                "件数":         n,
                "高値_平均":    round(oh.mean(), 2),
                "高値_中央値":  round(np.median(oh), 2),
                "高値_75%ile":  round(np.percentile(oh, 75), 2),
                "高値_95%ile":  round(np.percentile(oh, 95), 2),
                "引け_平均":    round(oc.mean(), 2),
                "引け_中央値":  round(np.median(oc), 2),
                "引け_プラス率": round((oc > 0).mean() * 100, 1),
                "安値_平均":    round(ol.mean(), 2),
                "高値3%超_率":  round((oh >= 3.0).mean() * 100, 1),
                "高値5%超_率":  round((oh >= 5.0).mean() * 100, 1),
                "高値10%超_率": round((oh >= 10.0).mean() * 100, 1),
            })

    df_result = pd.DataFrame(results)

    # ── 3. 表示 ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("【寄り付き後値動きパターン分析】")
    print("  Open→High: その日の最大上昇幅（寄りから高値まで）")
    print("  Open→Close: 引け値での損益（終日保有した場合）")
    print("=" * 80)

    for cond in conditions:
        df_c = df_result[df_result["地合い"] == cond]
        if df_c.empty:
            continue

        print(f"\n▼ {cond}日")
        print(f"  {'スコア帯':<10} {'件数':>4}  "
              f"{'高値avg':>7} {'高値med':>7} {'高値75%':>7} {'高値95%':>7}  "
              f"{'引け平均':>8} {'引け+率':>7}  "
              f"{'安値avg':>7}  "
              f"{'高値3%+':>7} {'高値5%+':>7} {'高値10%+':>8}")
        print("  " + "─" * 100)

        for _, r in df_c.iterrows():
            print(f"  {r['スコア帯']:<10} {r['件数']:>4}件  "
                  f"{r['高値_平均']:>+6.2f}% {r['高値_中央値']:>+6.2f}% "
                  f"{r['高値_75%ile']:>+6.2f}% {r['高値_95%ile']:>+6.2f}%  "
                  f"{r['引け_平均']:>+7.2f}% {r['引け_プラス率']:>6.1f}%  "
                  f"{r['安値_平均']:>+6.2f}%  "
                  f"{r['高値3%超_率']:>6.1f}% {r['高値5%超_率']:>6.1f}% {r['高値10%超_率']:>7.1f}%")

    # ── 4. CSV保存 ──────────────────────────────────
    out_path = "out/intraday_pattern.csv"
    df_result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 結果保存: {out_path}")

    # ── 5. WEAK日の詳細個別リスト ─────────────────────
    print("\n" + "=" * 80)
    print("【WEAK日 score8+ 個別リスト（上位20件）】")
    print("=" * 80)

    df_weak8 = scan[
        (scan["market_condition"] == "WEAK") & (scan["score"] >= 8.0)
    ]
    detail_rows = []
    for _, row in df_weak8.iterrows():
        code       = row["code"]
        scan_date  = row["scan_date"]
        entry_date = get_next_trading_date(all_dates, scan_date)
        if entry_date is None:
            continue
        df_cache = load_cache(code)
        if df_cache is None:
            continue
        day = df_cache[df_cache["Date"] == entry_date]
        if day.empty:
            continue
        o = float(day["Open"].iloc[0])
        h = float(day["High"].iloc[0])
        c = float(day["Close"].iloc[0])
        if o <= 0:
            continue
        detail_rows.append({
            "scan_date":  scan_date,
            "entry_date": entry_date,
            "code":       code,
            "name":       row["name"],
            "score":      row["score"],
            "ratio":      round(float(row.get("ratio", 0)), 1),
            "Open→High":  round((h - o) / o * 100, 2),
            "Open→Close": round((c - o) / o * 100, 2),
        })

    detail_df = pd.DataFrame(detail_rows).sort_values("Open→High", ascending=False)
    print(f"\n  {'日付':<12} {'コード':<6} {'銘柄名':<20} {'スコア':>5} {'倍率':>5}  "
          f"{'高値まで':>8} {'引け値':>8}")
    print("  " + "─" * 80)
    for _, r in detail_df.head(20).iterrows():
        print(f"  {str(r['entry_date']):<12} {r['code']:<6} {str(r['name']):<20} "
              f"{r['score']:>5.1f} {r['ratio']:>4.1f}倍  "
              f"{r['Open→High']:>+7.2f}% {r['Open→Close']:>+7.2f}%")


if __name__ == "__main__":
    main()
