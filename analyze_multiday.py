"""
analyze_multiday.py - 複数日保有バックテスト分析

backtest_log.csv × キャッシュデータを使い、
スキャン翌日の寄り付き買い後、1〜5日目の終値リターンを計算する。

実行方法:
    python analyze_multiday.py
"""

import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

CACHE_DIR       = "out/cache"
BACKTEST_LOG    = "out/backtest_log.csv"
OUTPUT_CSV      = "out/multiday_log.csv"
MAX_DAYS        = 5   # 最大追跡日数


# ══════════════════════════════════════════════
# 翌営業日リストを返す（土日スキップ）
# ══════════════════════════════════════════════
def next_trading_days(start_date_str, n):
    """start_date の翌日から n 営業日分の日付リストを返す"""
    current = pd.Timestamp(start_date_str) + pd.Timedelta(days=1)
    days = []
    while len(days) < n:
        if current.weekday() < 5:   # 月〜金
            days.append(current)
        current += pd.Timedelta(days=1)
    return days


# ══════════════════════════════════════════════
# メイン分析
# ══════════════════════════════════════════════
def build_multiday_log():
    df = pd.read_csv(BACKTEST_LOG, encoding="utf-8-sig")

    # open_price（エントリー価格）がない行はスキップ
    df = df[df["open_price"].notna() & (df["open_price"] > 0)].copy()

    rows = []
    for _, row in df.iterrows():
        code       = str(row["code"])
        scan_date  = row["scan_date"]
        open_price = float(row["open_price"])

        cache_path = f"{CACHE_DIR}/{code}.csv"
        if not os.path.exists(cache_path):
            continue

        hist = pd.read_csv(cache_path)
        hist["Date"] = pd.to_datetime(hist["Date"])
        hist = hist.sort_values("Date").set_index("Date")

        # エントリー翌日から MAX_DAYS 営業日分の日付
        trade_days = next_trading_days(scan_date, MAX_DAYS)

        day_returns = {}
        max_ret     = None
        max_ret_day = None
        best_exit   = None   # +3%利確 or -5%損切り を想定した損益

        for i, day in enumerate(trade_days, start=1):
            # その日以降で最も近いキャッシュデータを取得
            available = hist[hist.index >= day]
            if available.empty:
                break                       # データなし → それ以降も取れない
            close = float(available.iloc[0]["Close"])
            ret   = round((close - open_price) / open_price * 100, 2)
            day_returns[f"day{i}_pct"] = ret

            # 最大リターン追跡
            if max_ret is None or ret > max_ret:
                max_ret     = ret
                max_ret_day = i

            # +3%利確 or -5%損切り シミュレーション（最初に条件を満たした日）
            if best_exit is None:
                if ret >= 3.0:
                    best_exit = ("利確+3%", i, ret)
                elif ret <= -5.0:
                    best_exit = ("損切り-5%", i, ret)

        # 5日後まで利確・損切りに到達しなかった場合は最終日の終値
        if best_exit is None and day_returns:
            last_day = max(day_returns.keys())          # 例: "day3_pct"
            last_day_num = int(last_day.replace("day", "").replace("_pct", ""))
            best_exit = (f"引け(day{last_day_num})", last_day_num, day_returns[last_day])

        result = {
            **row.to_dict(),
            **day_returns,
            "max_ret_5d":    round(max_ret, 2) if max_ret is not None else None,
            "max_ret_day":   max_ret_day,
            "exit_reason":   best_exit[0] if best_exit else None,
            "exit_day":      best_exit[1] if best_exit else None,
            "exit_pct":      round(best_exit[2], 2) if best_exit else None,
        }
        rows.append(result)

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 保存完了: {OUTPUT_CSV}（{len(out)}件）")
    return out


# ══════════════════════════════════════════════
# 集計・表示
# ══════════════════════════════════════════════
def analyze(df):
    df_h = df[df["max_ret_5d"].notna()].copy()

    print(f"\n{'='*60}")
    print(f"【複数日保有バックテスト】{len(df_h)}件")
    print(f"{'='*60}")

    # ── 保有日数別リターン ──
    print("\n=== 保有日数別 平均リターン（全件）===")
    print(f"  {'日数':<6} {'件数':>5} {'avg':>8} {'プラス率':>8} {'+3%超':>7} {'-5%以下':>7}")
    print("  " + "─" * 48)
    for d in range(1, MAX_DAYS + 1):
        col = f"day{d}_pct"
        if col not in df_h.columns:
            break
        s = df_h[col].dropna()
        if s.empty:
            continue
        plus  = (s > 0).sum()
        hit3  = (s >= 3).sum()
        loss5 = (s <= -5).sum()
        print(f"  {d}日後 {len(s):>5}件 {s.mean():>+7.2f}% {plus/len(s)*100:>7.1f}% "
              f"{hit3:>5}件 {loss5:>5}件")

    # ── 朝判定別 ──
    print("\n=== 朝判定別 各日平均リターン ===")
    for jdg in ["BUY", "CAUTION", "PASS", "UNKNOWN"]:
        sub = df_h[df_h["morning_judgment"] == jdg]
        if sub.empty:
            continue
        vals = []
        for d in range(1, MAX_DAYS + 1):
            col = f"day{d}_pct"
            if col in sub.columns:
                v = sub[col].dropna()
                vals.append(f"{v.mean():>+5.1f}%" if len(v) > 0 else "   N/A")
            else:
                break
        print(f"  {jdg:<8}({len(sub):>2}件): " + "  ".join(f"day{i+1}={v}" for i, v in enumerate(vals)))

    # ── 地合い別 ──
    print("\n=== 地合い別 各日平均リターン ===")
    for cond in ["STRONG", "NORMAL", "WEAK", "PANIC"]:
        sub = df_h[df_h["market_condition"] == cond]
        if sub.empty:
            continue
        vals = []
        for d in range(1, MAX_DAYS + 1):
            col = f"day{d}_pct"
            if col in sub.columns:
                v = sub[col].dropna()
                vals.append(f"{v.mean():>+5.1f}%" if len(v) > 0 else "   N/A")
            else:
                break
        print(f"  {cond:<8}({len(sub):>2}件): " + "  ".join(f"day{i+1}={v}" for i, v in enumerate(vals)))

    # ── BUY × +3%利確/損切り-5% シミュレーション ──
    print("\n=== BUY判定 +3%利確 / 損切り-5% シミュレーション ===")
    buy = df_h[df_h["morning_judgment"] == "BUY"].copy()
    if not buy.empty:
        exit_counts = buy["exit_reason"].value_counts()
        print(f"  件数: {len(buy)}件")
        print(f"  利確+3%到達: {(buy['exit_reason']=='利確+3%').sum()}件")
        print(f"  損切り-5%  : {(buy['exit_reason']=='損切り-5%').sum()}件")
        print(f"  5日後引け  : {buy[buy['exit_reason'].str.startswith('引け', na=False)].shape[0]}件")
        avg_exit = buy["exit_pct"].mean()
        win = (buy["exit_pct"] > 0).sum()
        print(f"  勝率: {win}/{len(buy)}件 ({win/len(buy)*100:.1f}%)")
        print(f"  平均リターン: {avg_exit:+.2f}%")

    # ── 最大リターン到達日の分布 ──
    print("\n=== 最大リターン（5日中）の到達日分布（BUY判定）===")
    if not buy.empty:
        for d in range(1, MAX_DAYS + 1):
            cnt = (buy["max_ret_day"] == d).sum()
            bar = "█" * cnt
            print(f"  {d}日目: {cnt:>2}件 {bar}")

    # ── 個別銘柄詳細（BUY判定のみ）──
    print("\n=== BUY判定 銘柄別詳細 ===")
    print(f"  {'日付':<12} {'コード':<6} {'スコア':>6} "
          f"{'1日':>6} {'2日':>6} {'3日':>6} {'4日':>6} {'5日':>6}  出口")
    print("  " + "─" * 72)
    for _, r in buy.sort_values("scan_date").iterrows():
        days_str = "  ".join(
            f"{r[f'day{d}_pct']:>+5.1f}%" if f"day{d}_pct" in r and pd.notna(r[f"day{d}_pct"])
            else "   N/A"
            for d in range(1, MAX_DAYS + 1)
        )
        print(f"  {r['scan_date']:<12} {str(r['code']):<6} {r['score']:>6.1f}  "
              f"{days_str}  {r.get('exit_reason','')}")


# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("=== 複数日保有バックテスト ===")
    print(f"キャッシュ読込中...")
    df = build_multiday_log()
    analyze(df)
