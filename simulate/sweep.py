"""
sweep.py
パラメータスイープによる最適売買条件探索
シャープレシオ最大化を目標に全組み合わせを検証
アウトオブサンプル検証付き（前60%で最適化 → 後40%で検証）
"""
import pandas as pd
import numpy as np
from itertools import product
import warnings
warnings.filterwarnings('ignore')

INITIAL = 1_000_000

# ── SLキャリブレーション（分足実データより） ──────────────────
SL_CAL = {
    -1.0: {"tp_prob": 0.101, "sl_prob": 0.810},
    -2.0: {"tp_prob": 0.155, "sl_prob": 0.684},
    -3.0: {"tp_prob": 0.183, "sl_prob": 0.586},
}

TP_LIST = [1.0, 2.0, 3.0, 4.0, 5.0]
SL_LIST = [-1.0, -2.0, -3.0]


# ── データ読み込み ──────────────────────────────────────────
def load_data():
    df = pd.read_csv("out/sim_precise_trades.csv")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["scan_date"]  = pd.to_datetime(df["scan_date"])

    # 外れ値除外（today_rise が極端な銘柄：ストップ高・ストップ安等）
    df = df[(df["today_rise"] > -30) & (df["today_rise"] < 30)].copy()

    # TP/SL別リターンを事前計算（ベクトル化）
    hi = df["ret_max"].values
    lo = df["ret_low"].values
    oc = df["ret_oc"].values

    for tp in TP_LIST:
        for sl in SL_LIST:
            cal   = SL_CAL[sl]
            tp_p  = cal["tp_prob"]
            sl_p  = cal["sl_prob"]
            t     = tp_p + sl_p
            mixed = tp * tp_p / t + sl * sl_p / t
            ret   = np.where(
                        (hi >= tp) & (lo > sl),   tp,
                np.where(
                        (lo <= sl) & (hi < tp),   sl,
                np.where(
                        (hi >= tp) & (lo <= sl),  mixed,
                        oc)))
            col = _col(tp, sl)
            df[col] = ret

    # スコア昇順でソート（低スコア優先）
    df = df.sort_values(["trade_date", "score"]).reset_index(drop=True)
    return df


def _col(tp, sl):
    return f"r_{int(tp*10)}_{int(abs(sl)*10)}"


# ── 複利シミュレーション（高速版） ────────────────────────────
def compound_sim(sub, top_n, ret_col):
    """
    sub: フィルタ済みDataFrame（trade_dateとscore昇順でソート済み前提）
    top_n: 1日あたりの最大保有銘柄数（スコア低い順）
    """
    if len(sub) < 10:
        return None

    # 1日あたりtop_n件に絞る（score昇順なので先頭がベスト）
    top = sub.groupby("trade_date", sort=True)[ret_col].apply(
        lambda x: x.head(top_n).mean()
    ).reset_index()
    top.columns = ["date", "ret"]

    if len(top) < 5:
        return None

    # 複利計算
    rets   = top["ret"].values
    caps   = INITIAL * np.cumprod(1 + rets / 100)
    peak   = np.maximum.accumulate(caps)
    dds    = (caps - peak) / peak * 100
    max_dd = float(dds.min())
    final  = float(caps[-1])

    dates = top["date"].values
    years = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25
    if years <= 0:
        return None

    ann    = ((final / INITIAL) ** (1 / years) - 1) * 100
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    wr     = (rets > 0).mean() * 100

    return {
        "annual": round(ann, 1),
        "max_dd": round(max_dd, 1),
        "sharpe": round(sharpe, 2),
        "wr":     round(wr, 1),
        "n_days": len(top),
        "final":  round(final, 0),
    }


# ── パラメータグリッド定義 ─────────────────────────────────
COND_SETS = {
    "STRONG":    ["STRONG"],
    "NORMAL":    ["NORMAL"],
    "STR+NOR":   ["STRONG", "NORMAL"],
}

SCORE_RANGES = [
    (3, 6), (3, 7), (3, 8),
    (4, 7), (4, 8),
    (5, 7), (5, 8),
]

RATIO_MAXS  = [3.0, 5.0, 99.0]          # ratio < X
RISE_MAXS   = [0.0, 2.0, 5.0, 99.0]     # today_rise < X
GAP_MAXS    = [2.0, 5.0]                 # gap_pct < X
TOP_NS      = [3, 5, 10]


def main():
    print("データ読み込み中...")
    df = load_data()
    print(f"  有効データ: {len(df):,}件 / "
          f"取引日{df['trade_date'].nunique()}日")
    print(f"  期間: {df['trade_date'].min().date()} 〜 "
          f"{df['trade_date'].max().date()}")

    # ── アウトオブサンプル分割 ────────────────────────────
    all_dates = sorted(df["trade_date"].unique())
    split_idx = int(len(all_dates) * 0.6)
    split_date = all_dates[split_idx]
    df_train = df[df["trade_date"] <  split_date].copy()
    df_test  = df[df["trade_date"] >= split_date].copy()
    print(f"  訓練期間: 〜{pd.Timestamp(split_date).date()} "
          f"({df_train['trade_date'].nunique()}日)")
    print(f"  検証期間: {pd.Timestamp(split_date).date()}〜 "
          f"({df_test['trade_date'].nunique()}日)")

    # ── パラメータスイープ（訓練データで最適化） ───────────
    total = (len(COND_SETS) * len(SCORE_RANGES) * len(RATIO_MAXS) *
             len(RISE_MAXS) * len(GAP_MAXS) * len(TP_LIST) *
             len(SL_LIST) * len(TOP_NS))
    print(f"\n総組み合わせ数: {total:,}件 スイープ開始...")

    results = []
    done = 0

    for (cond_label, conds), (smin, smax), rmax, rise_max, gap_max, tp, sl, top_n in product(
        COND_SETS.items(), SCORE_RANGES, RATIO_MAXS, RISE_MAXS,
        GAP_MAXS, TP_LIST, SL_LIST, TOP_NS
    ):
        done += 1
        if done % 2000 == 0:
            print(f"  {done:,}/{total:,}件処理中...", end="\r", flush=True)

        ret_col = _col(tp, sl)
        mask = (
            (df_train["scan_cond"].isin(conds)) &
            (df_train["trade_cond"].isin(["STRONG", "NORMAL"])) &
            (df_train["score"]      >= smin) &
            (df_train["score"]      <  smax) &
            (df_train["ratio"]      <  rmax) &
            (df_train["today_rise"] <  rise_max) &
            (df_train["gap_pct"]    <  gap_max)
        )
        sub = df_train[mask]
        res = compound_sim(sub, top_n, ret_col)
        if res is None:
            continue

        results.append({
            "cond":      cond_label,
            "score":     f"{smin}〜{smax}",
            "ratio_max": rmax,
            "rise_max":  rise_max,
            "gap_max":   gap_max,
            "tp":        tp,
            "sl":        sl,
            "top_n":     top_n,
            **{f"train_{k}": v for k, v in res.items()},
        })

    print(f"\n完了: {len(results):,}件の有効結果")

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("train_sharpe", ascending=False)

    # ── TOP20の検証データでの成績を計算 ──────────────────
    print("\n上位20件を検証データで確認中...")
    val_results = []
    for _, row in result_df.head(20).iterrows():
        conds   = COND_SETS[row["cond"]]
        ret_col = _col(row["tp"], row["sl"])
        mask = (
            (df_test["scan_cond"].isin(conds)) &
            (df_test["trade_cond"].isin(["STRONG", "NORMAL"])) &
            (df_test["score"]      >= float(row["score"].split("〜")[0])) &
            (df_test["score"]      <  float(row["score"].split("〜")[1])) &
            (df_test["ratio"]      <  row["ratio_max"]) &
            (df_test["today_rise"] <  row["rise_max"]) &
            (df_test["gap_pct"]    <  row["gap_max"])
        )
        sub = df_test[mask]
        res = compound_sim(sub, int(row["top_n"]), ret_col)
        val_results.append(res if res else {
            "annual": None, "max_dd": None, "sharpe": None,
            "wr": None, "n_days": 0, "final": None
        })

    for i, vr in enumerate(val_results):
        for k, v in vr.items():
            result_df.loc[result_df.index[i], f"val_{k}"] = v

    # ── 結果表示 ──────────────────────────────────────
    print("\n" + "=" * 90)
    print("【TOP 20 シャープレシオ上位（訓練データ）】")
    print("=" * 90)
    disp_cols = [
        "cond", "score", "ratio_max", "rise_max", "gap_max",
        "tp", "sl", "top_n",
        "train_sharpe", "train_annual", "train_max_dd", "train_wr", "train_n_days",
        "val_sharpe",   "val_annual",   "val_max_dd",
    ]
    top20 = result_df[disp_cols].head(20)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    print(top20.to_string(index=False))

    # ── TP×SL ヒートマップ ────────────────────────────
    print("\n" + "=" * 60)
    print("【TP × SL ヒートマップ（シャープレシオ）】")
    for cond in ["STRONG", "NORMAL", "STR+NOR"]:
        best_score = (result_df[result_df["cond"] == cond]
                      .sort_values("train_sharpe", ascending=False)
                      .iloc[0][["score","ratio_max","rise_max","top_n"]]
                      if len(result_df[result_df["cond"] == cond]) > 0 else None)
        if best_score is None:
            continue
        filt = (
            (result_df["cond"]      == cond) &
            (result_df["score"]     == best_score["score"]) &
            (result_df["ratio_max"] == best_score["ratio_max"]) &
            (result_df["rise_max"]  == best_score["rise_max"]) &
            (result_df["top_n"]     == best_score["top_n"])
        )
        hm = result_df[filt].pivot_table(
            index="tp", columns="sl", values="train_sharpe"
        )
        print(f"\n  {cond}日 (score={best_score['score']}, "
              f"ratio<{best_score['ratio_max']}, "
              f"rise<{best_score['rise_max']}, "
              f"top{int(best_score['top_n'])})")
        print(hm.to_string())

    # ── ベスト条件サマリー ─────────────────────────────
    print("\n" + "=" * 60)
    print("【条件別ベスト（訓練→検証 比較）】")
    print("=" * 60)
    for cond in ["STRONG", "NORMAL", "STR+NOR"]:
        sub_r = result_df[result_df["cond"] == cond]
        if sub_r.empty:
            continue
        best = sub_r.iloc[0]
        print(f"\n  {cond}:")
        print(f"    条件: score{best['score']} / ratio<{best['ratio_max']} / "
              f"rise<{best['rise_max']} / gap<{best['gap_max']}")
        print(f"    TP:{best['tp']:+.0f}%  SL:{best['sl']:.0f}%  top{int(best['top_n'])}")
        print(f"    訓練: Sharpe{best['train_sharpe']:.2f} / "
              f"年率{best['train_annual']:+.1f}% / DD{best['train_max_dd']:.1f}% / "
              f"WR{best['train_wr']:.1f}%")
        val_s = best.get("val_sharpe")
        val_a = best.get("val_annual")
        if pd.notna(val_s) and pd.notna(val_a):
            print(f"    検証: Sharpe{val_s:.2f} / 年率{val_a:+.1f}% / "
                  f"DD{best['val_max_dd']:.1f}%")
        else:
            print(f"    検証: データ不足")

    # ── 保存 ──────────────────────────────────────────
    result_df.to_csv("out/sweep_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ 全結果保存: out/sweep_results.csv ({len(result_df):,}件)")


if __name__ == "__main__":
    main()
