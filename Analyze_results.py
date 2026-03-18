"""
analyze_results.py - バックテスト結果分析スクリプト

使い方:
    python analyze_results.py

出力:
    1. 地合い別 戦略A成功率
    2. スコア帯別 成功率
    3. 朝予測 vs 実際地合い 一致率
    4. 戦略B 累積成績
    5. 曜日別 成功率
    6. 直近トレンド（週次）
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

BACKTEST_LOG_CSV = "out/backtest_log.csv"
MORNING_LOG_CSV  = "out/morning_log.csv"
MARKET_LOG_CSV   = "out/market_log.csv"
WATCHLIST_CSV    = "out/watchlist.csv"


def load_data():
    """各CSVを読み込む"""
    data = {}

    if os.path.exists(BACKTEST_LOG_CSV):
        df = pd.read_csv(BACKTEST_LOG_CSV, encoding="utf-8-sig")
        df["verify_date"] = pd.to_datetime(df["verify_date"])
        df["scan_date"]   = pd.to_datetime(df["scan_date"])
        data["backtest"]  = df
    else:
        print(f"  ⚠️  {BACKTEST_LOG_CSV} が見つかりません")
        data["backtest"] = pd.DataFrame()

    if os.path.exists(MORNING_LOG_CSV):
        df = pd.read_csv(MORNING_LOG_CSV, encoding="utf-8-sig")
        df["date"]       = pd.to_datetime(df["date"])
        data["morning"]  = df
    else:
        data["morning"] = pd.DataFrame()

    if os.path.exists(MARKET_LOG_CSV):
        df = pd.read_csv(MARKET_LOG_CSV, encoding="utf-8-sig")
        df["date"]      = pd.to_datetime(df["date"])
        data["market"]  = df
    else:
        data["market"] = pd.DataFrame()

    if os.path.exists(WATCHLIST_CSV):
        df = pd.read_csv(WATCHLIST_CSV, encoding="utf-8-sig")
        df["buy_date"]  = pd.to_datetime(df["buy_date"])
        data["watchlist"] = df
    else:
        data["watchlist"] = pd.DataFrame()

    return data


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def analyze_strategy_a(bt):
    """戦略A 分析"""
    if bt.empty:
        print("  データなし")
        return

    total = len(bt)
    print(f"  総サンプル数: {total}件  ({bt['scan_date'].min().date()} 〜 {bt['scan_date'].max().date()})")

    # ── 1. 地合い別成功率
    section("【1】地合い別 成功率（高値+2%以上）")
    for cond in ["STRONG", "NORMAL", "WEAK", "PANIC"]:
        sub = bt[bt["market_condition"] == cond]
        if sub.empty:
            continue
        success = sub[sub["grade"].isin(["S", "A", "B"])].shape[0]
        rate    = round(success / len(sub) * 100, 1)
        avg_max = sub["max_rise_pct"].mean() if "max_rise_pct" in sub.columns else None
        avg_str = f"  平均高値騰落:{avg_max:>+.1f}%" if avg_max is not None else ""
        print(f"  {cond:<8}: {len(sub):>3}件  成功{success:>2}件  成功率{rate:>5.1f}%{avg_str}")

    # ── 2. スコア帯別成功率
    section("【2】スコア帯別 成功率")
    bins   = [0, 7.5, 8.0, 8.5, 9.0, 10.1]
    labels = ["〜7.5", "7.5〜8.0", "8.0〜8.5", "8.5〜9.0", "9.0〜"]
    bt["score_band"] = pd.cut(bt["score"], bins=bins, labels=labels, right=False)
    for band in labels:
        sub = bt[bt["score_band"] == band]
        if sub.empty:
            continue
        success = sub[sub["grade"].isin(["S", "A", "B"])].shape[0]
        rate    = round(success / len(sub) * 100, 1)
        avg_max = sub["max_rise_pct"].mean() if "max_rise_pct" in sub.columns else None
        avg_str = f"  平均高値:{avg_max:>+.1f}%" if avg_max is not None else ""
        print(f"  スコア{band:<10}: {len(sub):>3}件  成功{success:>2}件  成功率{rate:>5.1f}%{avg_str}")

    # ── 3. グレード分布
    section("【3】グレード分布（全期間）")
    grade_map = {"S": "✅ S級(TOP20)", "A": "✨ A級(高値+5%)", "B": "📈 B級(高値+2%)",
                 "FLAT": "➖ 横ばい", "FAIL": "❌ 逆行"}
    for grade, label in grade_map.items():
        count = bt[bt["grade"] == grade].shape[0]
        pct   = round(count / total * 100, 1)
        bar   = "█" * int(pct / 2)
        print(f"  {label:<18}: {count:>3}件 ({pct:>5.1f}%)  {bar}")

    # ── 4. 朝判定別成功率
    if "morning_judgment" in bt.columns:
        section("【4】朝判定別 成功率")
        for j in ["BUY", "CAUTION", "PASS", "UNKNOWN"]:
            sub = bt[bt["morning_judgment"] == j]
            if sub.empty:
                continue
            success = sub[sub["grade"].isin(["S", "A", "B"])].shape[0]
            rate    = round(success / len(sub) * 100, 1)
            print(f"  {j:<10}: {len(sub):>3}件  成功{success:>2}件  成功率{rate:>5.1f}%")

    # ── 5. 曜日別成功率
    section("【5】曜日別 成功率")
    bt["weekday"] = bt["scan_date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    jp    = {"Monday": "月", "Tuesday": "火", "Wednesday": "水",
             "Thursday": "木", "Friday": "金"}
    for day in order:
        sub = bt[bt["weekday"] == day]
        if sub.empty:
            continue
        success = sub[sub["grade"].isin(["S", "A", "B"])].shape[0]
        rate    = round(success / len(sub) * 100, 1)
        print(f"  {jp[day]}曜日: {len(sub):>3}件  成功{success:>2}件  成功率{rate:>5.1f}%")

    # ── 6. 週次トレンド
    section("【6】週次トレンド")
    bt["week"] = bt["scan_date"].dt.to_period("W")
    for week, grp in bt.groupby("week"):
        success = grp[grp["grade"].isin(["S", "A", "B"])].shape[0]
        rate    = round(success / len(grp) * 100, 1)
        conds   = grp["market_condition"].value_counts().to_dict()
        cond_str = "/".join([f"{k}:{v}" for k, v in conds.items()])
        print(f"  {str(week):<20}: {len(grp):>2}件  成功率{rate:>5.1f}%  ({cond_str})")


def analyze_strategy_b(wl):
    """戦略B 分析"""
    if wl.empty:
        print("  データなし")
        return

    verified = wl[wl["next_rise"].notna()].copy()
    if verified.empty:
        print("  検証済みデータなし")
        return

    section("【戦略B】逆張り 累積成績")
    total   = len(verified)
    success = (verified["next_rise"] >= 5.0).sum()
    caution = ((verified["next_rise"] >= 0) & (verified["next_rise"] < 5.0)).sum()
    fail    = (verified["next_rise"] < 0).sum()
    avg     = verified["next_rise"].mean()
    median  = verified["next_rise"].median()

    print(f"  総件数    : {total}件")
    print(f"  ✅ +5%超  : {success}件 ({round(success/total*100,1)}%)")
    print(f"  ⚠️  0〜5%  : {caution}件 ({round(caution/total*100,1)}%)")
    print(f"  ❌ マイナス: {fail}件 ({round(fail/total*100,1)}%)")
    print(f"  平均騰落  : {avg:>+.2f}%")
    print(f"  中央値    : {median:>+.2f}%")

    # RBスコア別
    section("【戦略B】RBスコア別成績")
    if "rebound_score" in verified.columns:
        bins   = [0, 3, 5, 7, 11]
        labels = ["3点以下", "4〜5点", "6〜7点", "8点以上"]
        verified["rb_band"] = pd.cut(verified["rebound_score"], bins=bins, labels=labels)
        for band in labels:
            sub = verified[verified["rb_band"] == band]
            if sub.empty:
                continue
            s   = (sub["next_rise"] >= 5.0).sum()
            avg = sub["next_rise"].mean()
            print(f"  RB{band:<8}: {len(sub):>3}件  +5%達成:{s:>2}件  平均:{avg:>+.2f}%")

    # 地合い別
    section("【戦略B】地合い別成績")
    if "market_condition" in verified.columns:
        for cond in ["NORMAL", "WEAK", "PANIC"]:
            sub = verified[verified["market_condition"] == cond]
            if sub.empty:
                continue
            s   = (sub["next_rise"] >= 5.0).sum()
            avg = sub["next_rise"].mean()
            print(f"  {cond:<8}: {len(sub):>3}件  +5%達成:{s:>2}件  平均:{avg:>+.2f}%")


def analyze_morning_forecast(morning, market):
    """朝予測 vs 実際地合いの一致率"""
    if morning.empty or market.empty:
        return

    section("【朝予測 vs 実際地合い】一致率")
    merged = pd.merge(
        morning[["date", "condition_forecast", "condition_score"]],
        market[["date", "condition"]],
        on="date", how="inner"
    )

    if merged.empty:
        print("  マッチするデータなし")
        return

    total   = len(merged)
    correct = (merged["condition_forecast"] == merged["condition"]).sum()
    rate    = round(correct / total * 100, 1)
    print(f"  一致率: {correct}/{total}件 ({rate}%)\n")

    print(f"  {'予測':<8} {'実際':<8} {'件数':>4}  {'一致':>4}")
    print("  " + "─" * 30)
    for _, row in merged.iterrows():
        match = "✅" if row["condition_forecast"] == row["condition"] else "❌"
        print(f"  {row['condition_forecast']:<8} {row['condition']:<8} "
              f"{row['date'].date()}  {match}")


def main():
    print(f"=== 📊 バックテスト結果分析 ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ===")

    data = load_data()
    bt   = data["backtest"]
    wl   = data["watchlist"]
    mg   = data["morning"]
    mk   = data["market"]

    if bt.empty:
        print("\n⚠️  backtest_log.csvが空です。scan_daily.pyを実行してデータを蓄積してください。")
        return

    section("【戦略A】順張り 分析")
    analyze_strategy_a(bt)

    analyze_strategy_b(wl)

    analyze_morning_forecast(mg, mk)

    print(f"\n{'='*60}")
    print(f"  分析完了")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()