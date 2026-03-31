"""
simulate_compound.py - 複利シミュレーション

STRONG日に低スコア上位5銘柄へ均等投資し、
利益をそのまま再投資した場合の実際の年率リターンを計算する。

使い方:
    python simulate_compound.py
"""

import pandas as pd
import numpy as np

# ══════════════════════════════════════════
# 設定
# ══════════════════════════════════════════
INITIAL_CAPITAL = 1_000_000   # 初期資金（円）
TOP_N           = 5           # 1日あたり最大購入銘柄数
TP              = 3.0         # 利確ライン（%）
SCORE_MIN       = 3.0         # スコア下限
SCORE_MAX       = 7.0         # スコア上限（低スコア優先）
GAP_MAX         = 5.0         # ギャップアップ上限（%）
COMMISSION      = 0.0         # 手数料（%）※0=未考慮、0.2=往復0.2%

SL_CALIBRATION = {
    -1.0: {"tp_prob": 0.101, "sl_prob": 0.810},
    -2.0: {"tp_prob": 0.155, "sl_prob": 0.684},
    -3.0: {"tp_prob": 0.183, "sl_prob": 0.586},
}


def calc_trade_return(row, tp, sl):
    """1トレードのリターンを計算（分足校正込み）"""
    hi = row["ret_max"]
    lo = row["ret_low"]

    if pd.isna(hi) or pd.isna(lo):
        return row["ret_oc"]

    if hi >= tp and lo > sl:
        return tp
    elif lo <= sl and hi < tp:
        return sl
    elif hi >= tp and lo <= sl:
        cal = SL_CALIBRATION.get(sl, {"tp_prob": 0.15, "sl_prob": 0.70})
        total_p = cal["tp_prob"] + cal["sl_prob"]
        tp_w = cal["tp_prob"] / total_p if total_p > 0 else 0.5
        sl_w = cal["sl_prob"] / total_p if total_p > 0 else 0.5
        return tp * tp_w + sl * sl_w
    else:
        return row["ret_oc"]


def run_compound(df, sl, label):
    """複利シミュレーション本体"""

    # フィルタリング
    sub = df[
        (df["trade_cond"] == "STRONG") &
        (df["gap_pct"] < GAP_MAX) &
        (df["score"] >= SCORE_MIN) &
        (df["score"] < SCORE_MAX)
    ].copy()

    sub["sim_ret"] = sub.apply(lambda r: calc_trade_return(r, TP, sl), axis=1)
    sub = sub.sort_values(["trade_date", "score"])

    # STRONG日ごとに上位5銘柄を選択
    capital      = float(INITIAL_CAPITAL)
    capital_log  = []   # 日別資産推移
    trade_log    = []   # トレードごとの記録
    peak_capital = capital
    max_drawdown = 0.0

    trade_dates = sorted(sub["trade_date"].unique())

    for date in trade_dates:
        day_stocks = sub[sub["trade_date"] == date].head(TOP_N)
        n = len(day_stocks)
        if n == 0:
            continue

        # 均等配分（手数料を除いた投資額）
        alloc_per_stock = capital / n

        day_gain = 0.0
        for _, row in day_stocks.iterrows():
            ret_pct  = row["sim_ret"]
            # 手数料控除（往復）
            net_ret  = ret_pct - COMMISSION
            gain     = alloc_per_stock * net_ret / 100
            day_gain += gain
            trade_log.append({
                "date":      date,
                "code":      row["code"],
                "score":     row["score"],
                "ret_pct":   round(ret_pct, 2),
                "net_ret":   round(net_ret, 2),
                "alloc":     round(alloc_per_stock, 0),
                "gain":      round(gain, 0),
            })

        capital += day_gain

        # ドローダウン更新
        if capital > peak_capital:
            peak_capital = capital
        dd = (capital - peak_capital) / peak_capital * 100
        if dd < max_drawdown:
            max_drawdown = dd

        capital_log.append({
            "date":       date,
            "capital":    round(capital, 0),
            "day_gain":   round(day_gain, 0),
            "day_ret":    round(day_gain / (capital - day_gain) * 100, 3),
            "drawdown":   round(dd, 2),
            "n_stocks":   n,
        })

    if not capital_log:
        print(f"  [{label}] データなし")
        return

    cap_df = pd.DataFrame(capital_log)

    # ── 集計 ──────────────────────────────
    total_ret   = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    n_days      = len(trade_dates)
    # 取引日数から年率換算（1年≒252営業日）
    # STRONG日がn_days日あった場合、実際の期間は約1年
    first_date  = pd.to_datetime(cap_df["date"].iloc[0])
    last_date   = pd.to_datetime(cap_df["date"].iloc[-1])
    actual_days = (last_date - first_date).days
    years       = actual_days / 365.0 if actual_days > 0 else 1.0
    annual_ret  = ((capital / INITIAL_CAPITAL) ** (1 / years) - 1) * 100

    # 日次リターンのシャープレシオ
    daily_rets  = cap_df["day_ret"].values
    sharpe      = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0

    print(f"\n{'='*60}")
    print(f"【複利シミュレーション結果】{label}")
    print(f"{'='*60}")
    print(f"  戦略    : STRONG日 × 低スコア(3〜7)上位{TOP_N}銘柄")
    print(f"  TP/SL   : +{TP}% / {sl}%")
    print(f"  手数料  : {COMMISSION}%（往復）")
    print(f"  対象期間: {cap_df['date'].iloc[0]} 〜 {cap_df['date'].iloc[-1]}")
    print(f"  STRONG日: {n_days}日 / トレード数: {len(trade_log)}回")
    print(f"  {'─'*54}")
    print(f"  初期資金      :  {INITIAL_CAPITAL:>12,.0f} 円")
    print(f"  最終資金      :  {capital:>12,.0f} 円")
    print(f"  総利益        :  {capital - INITIAL_CAPITAL:>+12,.0f} 円")
    print(f"  {'─'*54}")
    print(f"  期間リターン  :  {total_ret:>+8.2f}%")
    print(f"  年率リターン  :  {annual_ret:>+8.2f}%  ← 複利換算")
    print(f"  最大ドローダウン: {max_drawdown:>+7.2f}%")
    print(f"  Sharpe比      :  {sharpe:>8.2f}")
    print(f"  {'─'*54}")

    # 月別推移
    print(f"\n  【月別推移】")
    print(f"  {'月':>8}  {'STRONG日':>6}  {'月次損益(円)':>12}  {'月末資産(円)':>12}  {'月次%':>7}")
    cap_df["month"] = cap_df["date"].str[:7]
    monthly = cap_df.groupby("month").agg(
        n_days   =("date",     "count"),
        gain_sum =("day_gain", "sum"),
        last_cap =("capital",  "last"),
    )
    for month, r in monthly.iterrows():
        prev_cap  = cap_df[cap_df["month"] == month]["capital"].iloc[0] - \
                    cap_df[cap_df["month"] == month]["day_gain"].iloc[0]
        month_pct = r["gain_sum"] / prev_cap * 100 if prev_cap > 0 else 0
        print(f"  {month:>8}  {int(r['n_days']):>6}日  "
              f"{r['gain_sum']:>+12,.0f}円  {r['last_cap']:>12,.0f}円  "
              f"{month_pct:>+6.1f}%")

    return capital, annual_ret, max_drawdown, sharpe


def main():
    print("複利シミュレーション読み込み中...")
    df = pd.read_csv("out/sim_precise_trades.csv")
    print(f"  総シグナル数: {len(df):,}")

    # SL違いで比較
    print(f"\n初期資金: {INITIAL_CAPITAL:,}円 / 手数料: {COMMISSION}%（往復）\n")

    results = []
    for sl, label in [(-1.0, "SL-1%"), (-2.0, "SL-2%"), (-3.0, "SL-3%（SLなし相当）")]:
        r = run_compound(df, sl, label)
        if r:
            results.append((label, *r))

    # 比較サマリー
    print(f"\n{'='*60}")
    print("【SL別 比較サマリー】")
    print(f"{'='*60}")
    print(f"  {'条件':<20}  {'最終資産':>12}  {'年率':>8}  {'最大DD':>8}  {'Sharpe':>7}")
    print(f"  {'─'*58}")
    for label, final_cap, ann_ret, mdd, sharpe in results:
        print(f"  {label:<20}  {final_cap:>12,.0f}円  {ann_ret:>+7.2f}%  {mdd:>+7.2f}%  {sharpe:>7.2f}")

    print(f"\n  ※ 手数料・スリッページ未考慮")
    print(f"  ※ 非STRONG日は全て待機（資金は増えない）")
    print(f"  ※ STRONG日は資金を{TOP_N}等分して均等投資")


if __name__ == "__main__":
    main()
