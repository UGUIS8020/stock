"""
evolve_filtered.py - 銀行系・100円未満除外フィルターを適用した遺伝的最適化

フィルター内容:
  1. 銀行系除外: 銘柄名に「銀行」「フィナンシャルグループ」を含む銘柄
  2. 低位株除外: trade_date当日の終値が 100円未満の銘柄

使用方法:
    cd C:\\...\\stock
    python optimize\\evolve_filtered.py --pop 5000 --gen 50
    python optimize\\evolve_filtered.py --pop 5000 --gen 3    # テスト
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TRADES_CSV          = "out/sim_precise_trades.csv"
FILTERED_TRADES_CSV = "out/sim_precise_trades_filtered.csv"
WF_SPLIT   = 0.60
MIN_SAMPLE = 10

ELITE_RATE   = 0.02
TOURNAMENT_K = 5
CROSS_RATE   = 0.80
MUTATE_RATE  = 0.15
MUTATE_SIGMA = 0.15

#                   [score, r_min, r_max, rise_lo, rise_hi, rb,  gap,  tp,   sl,   cond, topn, ls,  vd,  cd  ]
GENE_MIN = np.array([ 0.0,  0.0,  3.0,  -15.0,   -5.0,   0.0, -5.0,  1.0, -7.0,  0.0,  0.0, 0.0, 0.0, 0.0])
GENE_MAX = np.array([10.0, 10.0, 50.0,    5.0,   10.0,   8.0, 10.0, 10.0, -0.5,  3.0,  3.0, 0.7, 1.0, 5.0])
N_GENES  = len(GENE_MIN)

_CAL_SL  = np.array([-0.5, -1.0, -2.0, -3.0, -5.0, -7.0])
_CAL_TPC = np.array([0.050, 0.101, 0.155, 0.183, 0.220, 0.250])
_CAL_SLC = np.array([0.930, 0.810, 0.684, 0.586, 0.480, 0.400])

COND_LABELS = ["全地合い", "NORMAL", "STRONG", "NORMAL+STRONG"]
TOPN_VALUES = [1, 3, 5, 999]

MIN_PRICE = 100  # 100円未満を除外


# ══════════════════════════════════════════════
# フィルタ済みデータ作成
# ══════════════════════════════════════════════
def build_filtered_trades():
    """銀行系・低位株を除外したトレードデータを作成（キャッシュあり）"""
    import sqlite3

    if os.path.exists(FILTERED_TRADES_CSV):
        mtime_raw  = os.path.getmtime(TRADES_CSV)
        mtime_filt = os.path.getmtime(FILTERED_TRADES_CSV)
        if mtime_filt > mtime_raw:
            df = pd.read_csv(FILTERED_TRADES_CSV, encoding="utf-8-sig", low_memory=False)
            print(f"  フィルタ済みキャッシュ読込: {len(df):,}件")
            return df

    print("  フィルタ済みデータを構築中...")
    df = pd.read_csv(TRADES_CSV, encoding="utf-8-sig", low_memory=False)
    before = len(df)
    print(f"  元データ: {before:,}件")

    db_path = "out/stock.db"
    conn = sqlite3.connect(db_path)

    # ── 銀行系コードを取得 ──
    bank_df = pd.read_sql("""
        SELECT DISTINCT code, name FROM (
            SELECT code, name FROM scan_results WHERE name LIKE '%銀行%' OR name LIKE '%フィナンシャルグループ%'
            UNION
            SELECT CAST(code AS TEXT), name FROM daily_prices
            WHERE (name LIKE '%銀行%' OR name LIKE '%フィナンシャルグループ%')
            LIMIT 5000
        )
    """, conn) if False else None  # daily_prices に name がない場合

    # scan_results から銀行コードを取得
    try:
        bank_codes_df = pd.read_sql("""
            SELECT DISTINCT code FROM scan_results
            WHERE name LIKE '%銀行%' OR name LIKE '%フィナンシャルグループ%'
        """, conn)
        bank_codes = set(bank_codes_df["code"].astype(str).tolist())
        print(f"  銀行系コード: {len(bank_codes)}件")
    except Exception as e:
        print(f"  ⚠️ 銀行コード取得失敗: {e}")
        bank_codes = set()

    # ── trade_date × code の終値を取得 ──
    print("  株価データ取得中（daily_prices）...")
    price_df = pd.read_sql("""
        SELECT code, Date as trade_date, Close as close_price
        FROM daily_prices
        WHERE Close IS NOT NULL
    """, conn)
    price_df["code"] = price_df["code"].astype(str)
    conn.close()

    # マージして価格付与
    df["code"] = df["code"].astype(str)
    df = df.merge(price_df, on=["code", "trade_date"], how="left")

    n_before = len(df)

    # ── フィルター適用 ──
    bank_mask      = df["code"].isin(bank_codes)
    low_price_mask = df["close_price"] < MIN_PRICE
    no_price_mask  = df["close_price"].isna()

    n_bank      = bank_mask.sum()
    n_low       = low_price_mask.sum()
    n_no_price  = no_price_mask.sum()

    df_filtered = df[~bank_mask & ~low_price_mask & ~no_price_mask].copy()

    print(f"  銀行系除外    : {n_bank:,}件")
    print(f"  100円未満除外 : {n_low:,}件")
    print(f"  価格不明除外  : {n_no_price:,}件")
    print(f"  フィルタ後    : {len(df_filtered):,}件 ({len(df_filtered)/n_before*100:.1f}%残存)")

    # 元のカラムのみ保存
    orig_cols = list(pd.read_csv(TRADES_CSV, encoding="utf-8-sig", nrows=0).columns)
    df_filtered[orig_cols].to_csv(FILTERED_TRADES_CSV, index=False, encoding="utf-8-sig")
    print(f"  保存: {FILTERED_TRADES_CSV}")
    return df_filtered[orig_cols]


# ══════════════════════════════════════════════
# データ準備
# ══════════════════════════════════════════════
def load_trades():
    df = build_filtered_trades()
    before = len(df)
    df = df.dropna(subset=["ret_max", "ret_low", "ret_oc", "today_rise", "gap_pct"])
    df = df[(df["today_rise"].abs() < 30) & (df["gap_pct"].abs() < 20)]
    df = df.sort_values(["trade_date", "score"], ascending=[True, False]).reset_index(drop=True)
    df["score_rank"] = df.groupby("trade_date")["score"].rank(ascending=False, method="first")
    print(f"  最終データ: {before:,}件 → {len(df):,}件（欠損・外れ値除外後）")
    print(f"  期間: {df['trade_date'].min()} 〜 {df['trade_date'].max()}")
    return df


def df_to_numpy(df):
    cond_map = {"NORMAL": 1, "STRONG": 2, "WEAK": 3}
    return {
        "score":        df["score"].values.astype(np.float32),
        "ratio":        df["ratio"].values.astype(np.float32),
        "rise":         df["today_rise"].values.astype(np.float32),
        "rb":           df.get("rb_score", pd.Series(np.zeros(len(df)))).values.astype(np.float32),
        "gap":          df["gap_pct"].values.astype(np.float32),
        "cond":         df["trade_cond"].map(cond_map).fillna(0).values.astype(np.int8),
        "score_rank":   df["score_rank"].values.astype(np.float32),
        "ret_oc":       df["ret_oc"].values.astype(np.float32),
        "ret_max":      df["ret_max"].values.astype(np.float32),
        "ret_low":      df["ret_low"].values.astype(np.float32),
        "lower_shadow": (df["lower_shadow_ratio"].values.astype(np.float32)
                         if "lower_shadow_ratio" in df.columns
                         else np.zeros(len(df), dtype=np.float32)),
        "vol_decay":    (df["vol_decay_3d"].values.astype(np.int8)
                         if "vol_decay_3d" in df.columns
                         else np.ones(len(df), dtype=np.int8)),
        "consec_drop":  (df["consec_drop"].values.astype(np.int8)
                         if "consec_drop" in df.columns
                         else np.zeros(len(df), dtype=np.int8)),
    }


def split_numpy(npy, dates, split_date):
    mask = dates < split_date
    def sub(m):
        return {k: v[m] for k, v in npy.items()}
    return sub(mask), sub(~mask)


# ══════════════════════════════════════════════
# 染色体デコード
# ══════════════════════════════════════════════
def decode(ind):
    score_min  = float(np.clip(ind[0],  GENE_MIN[0],  GENE_MAX[0]))
    ratio_min  = float(np.clip(ind[1],  GENE_MIN[1],  GENE_MAX[1]))
    ratio_max  = float(np.clip(ind[2],  GENE_MIN[2],  GENE_MAX[2]))
    rise_lo    = float(np.clip(ind[3],  GENE_MIN[3],  GENE_MAX[3]))
    rise_hi    = float(np.clip(ind[4],  GENE_MIN[4],  GENE_MAX[4]))
    if rise_hi <= rise_lo:
        rise_hi = rise_lo + 1.0
    if ratio_max <= ratio_min:
        ratio_max = ratio_min + 1.0
    rb_min     = float(np.clip(ind[5],  GENE_MIN[5],  GENE_MAX[5]))
    gap_max    = float(np.clip(ind[6],  GENE_MIN[6],  GENE_MAX[6]))
    tp         = round(float(np.clip(ind[7],  GENE_MIN[7],  GENE_MAX[7])), 1)
    sl         = round(float(np.clip(ind[8],  GENE_MIN[8],  GENE_MAX[8])), 1)
    cond       = int(ind[9])  % 4
    top_n      = TOPN_VALUES[int(ind[10]) % 4]
    ls_min     = round(float(np.clip(ind[11], GENE_MIN[11], GENE_MAX[11])), 2)
    vd_req     = int(round(float(np.clip(ind[12], 0, 1))))
    cd_min     = int(round(float(np.clip(ind[13], GENE_MIN[13], GENE_MAX[13]))))
    return {
        "score_min": score_min, "ratio_min": ratio_min, "ratio_max": ratio_max,
        "rise_lo":   rise_lo,   "rise_hi":   rise_hi,
        "rb_min":    rb_min,    "gap_max":   gap_max,
        "tp": tp, "sl": sl,
        "cond": cond, "top_n": top_n,
        "ls_min": ls_min, "vd_req": vd_req, "cd_min": cd_min,
    }


# ══════════════════════════════════════════════
# 適応度計算
# ══════════════════════════════════════════════
def calc_returns(npy, idx, tp, sl):
    tp_p  = float(np.interp(-sl, [-v for v in _CAL_SL[::-1]], _CAL_TPC[::-1]))
    sl_p  = float(np.interp(-sl, [-v for v in _CAL_SL[::-1]], _CAL_SLC[::-1]))
    denom = tp_p + sl_p if (tp_p + sl_p) > 0 else 1.0
    hi, lo, oc = npy["ret_max"][idx], npy["ret_low"][idx], npy["ret_oc"][idx]
    return np.where(
        (hi >= tp) & (lo > sl),   tp,
        np.where(
            (lo <= sl) & (hi < tp), sl,
            np.where(
                (hi >= tp) & (lo <= sl),
                tp * tp_p / denom + sl * sl_p / denom,
                oc
            )
        )
    )


def evaluate_one(ind, npy):
    p = decode(ind)
    mask = (
        (npy["score"]      >= p["score_min"])  &
        (npy["ratio"]      >= p["ratio_min"])  &
        (npy["ratio"]      <  p["ratio_max"])  &
        (npy["rise"]       >= p["rise_lo"])    &
        (npy["rise"]       <= p["rise_hi"])    &
        (npy["rb"]         >= p["rb_min"])     &
        (npy["gap"]        <= p["gap_max"])    &
        (npy["score_rank"] <= p["top_n"])      &
        (npy["lower_shadow"] >= p["ls_min"])   &
        (npy["consec_drop"]  >= p["cd_min"])
    )
    if p["vd_req"] == 1:
        mask &= npy["vol_decay"] == 1
    if p["cond"] == 1:
        mask &= npy["cond"] == 1
    elif p["cond"] == 2:
        mask &= npy["cond"] == 2
    elif p["cond"] == 3:
        mask &= npy["cond"] <= 2

    idx = np.where(mask)[0]
    if len(idx) < MIN_SAMPLE:
        return -99.0

    rets = calc_returns(npy, idx, p["tp"], p["sl"])
    mean, std = rets.mean(), rets.std()
    if std < 1e-9:
        return -99.0
    return float(mean / std * np.sqrt(252))


def evaluate_pop(pop, npy):
    return np.array([evaluate_one(ind, npy) for ind in pop])


# ══════════════════════════════════════════════
# GA 操作
# ══════════════════════════════════════════════
def init_pop(pop_size):
    pop = np.random.rand(pop_size, N_GENES)
    for j in range(N_GENES):
        pop[:, j] = pop[:, j] * (GENE_MAX[j] - GENE_MIN[j]) + GENE_MIN[j]
    return pop


def tournament_select(pop, fitness):
    idx  = np.random.randint(0, len(pop), TOURNAMENT_K)
    best = idx[np.argmax(fitness[idx])]
    return pop[best].copy()


def crossover(p1, p2):
    if np.random.rand() > CROSS_RATE:
        return p1.copy(), p2.copy()
    pt = np.random.randint(1, N_GENES)
    return np.concatenate([p1[:pt], p2[pt:]]), np.concatenate([p2[:pt], p1[pt:]])


def mutate(ind):
    ind = ind.copy()
    for j in range(N_GENES):
        if np.random.rand() < MUTATE_RATE:
            rng = GENE_MAX[j] - GENE_MIN[j]
            ind[j] += np.random.randn() * rng * MUTATE_SIGMA
            ind[j]  = np.clip(ind[j], GENE_MIN[j], GENE_MAX[j])
    return ind


def next_generation(pop, fitness):
    n = len(pop)
    n_elite = max(1, int(n * ELITE_RATE))
    elite_idx = np.argsort(fitness)[::-1][:n_elite]
    new_pop = [pop[i].copy() for i in elite_idx]
    while len(new_pop) < n:
        p1 = tournament_select(pop, fitness)
        p2 = tournament_select(pop, fitness)
        c1, c2 = crossover(p1, p2)
        new_pop.append(mutate(c1))
        if len(new_pop) < n:
            new_pop.append(mutate(c2))
    return np.array(new_pop[:n])


# ══════════════════════════════════════════════
# 結果保存
# ══════════════════════════════════════════════
def format_result(ind, in_sharpe, out_npy, dates_out):
    p = decode(ind)
    idx = np.where(
        (out_npy["score"]      >= p["score_min"])  &
        (out_npy["ratio"]      >= p["ratio_min"])  &
        (out_npy["ratio"]      <  p["ratio_max"])  &
        (out_npy["rise"]       >= p["rise_lo"])    &
        (out_npy["rise"]       <= p["rise_hi"])    &
        (out_npy["rb"]         >= p["rb_min"])     &
        (out_npy["gap"]        <= p["gap_max"])    &
        (out_npy["score_rank"] <= p["top_n"])      &
        (out_npy["lower_shadow"] >= p["ls_min"])   &
        (out_npy["consec_drop"]  >= p["cd_min"])
    )[0]
    if p["vd_req"] == 1:
        idx = idx[out_npy["vol_decay"][idx] == 1]
    if p["cond"] == 1:
        idx = idx[out_npy["cond"][idx] == 1]
    elif p["cond"] == 2:
        idx = idx[out_npy["cond"][idx] == 2]
    elif p["cond"] == 3:
        idx = idx[out_npy["cond"][idx] <= 2]

    out_n, out_wr, out_avg, out_sharpe, out_cum = 0, 0.0, 0.0, -99.0, 0.0
    if len(idx) >= MIN_SAMPLE:
        rets = calc_returns(out_npy, idx, p["tp"], p["sl"])
        out_n   = len(rets)
        out_wr  = round(float((rets > 0).mean() * 100), 1)
        out_avg = round(float(rets.mean()), 3)
        mn, sd  = rets.mean(), rets.std()
        if sd > 1e-9:
            out_sharpe = round(float(mn / sd * np.sqrt(252)), 3)
        out_cum = round(float(rets.sum()), 1)

    # 戦略タイプ判定
    if p["rise_hi"] <= 0 and p["rb_min"] > 0:
        stype = "B(逆張り)"
    elif p["rise_lo"] >= 0:
        stype = "A(順張り)"
    else:
        stype = "混合"

    stable = "★" if (out_sharpe > 1.0 and out_avg > 0) else ""
    return {
        "type": stype,
        "score_min": round(p["score_min"], 1), "ratio_min": round(p["ratio_min"], 1),
        "ratio_max": round(p["ratio_max"], 1),
        "rise_lo": round(p["rise_lo"], 1),     "rise_hi": round(p["rise_hi"], 1),
        "rb_min":  round(p["rb_min"], 1),      "gap_max": round(p["gap_max"], 1),
        "tp": p["tp"], "sl": p["sl"],
        "cond": COND_LABELS[p["cond"]], "top_n": p["top_n"],
        "ls_min": p["ls_min"], "vd_req": p["vd_req"], "cd_min": p["cd_min"],
        "in_sharpe": round(in_sharpe, 3),
        "out_sharpe": out_sharpe, "out_n": out_n,
        "out_wr": out_wr, "out_avg": out_avg, "out_cum": out_cum,
        "stable": stable,
    }


# ══════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop",  type=int, default=5000)
    parser.add_argument("--gen",  type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    print(f"\n{'='*60}")
    print(f"遺伝的最適化 (フィルター適用版: 銀行系・100円未満除外)")
    print(f"  個体数: {args.pop:,}  世代数: {args.gen}  seed: {args.seed}")
    print(f"{'='*60}\n")

    # データ読込
    df = load_trades()
    dates = df["trade_date"].values
    split_date = np.sort(np.unique(dates))[int(len(np.unique(dates)) * WF_SPLIT)]
    print(f"\n  学習期間: {dates.min()} 〜 {split_date}")
    print(f"  検証期間: {split_date} 〜 {dates.max()}")

    npy_all  = df_to_numpy(df)
    npy_in, npy_out = split_numpy(npy_all, dates, split_date)
    dates_out = dates[dates >= split_date]

    # GA実行
    pop     = init_pop(args.pop)
    fitness = evaluate_pop(pop, npy_in)
    history = []

    best_csv_rows = []
    OUT_BEST    = "out/evolve_best_filtered.csv"
    OUT_HISTORY = "out/evolve_history_filtered.csv"
    OUT_REPORT  = "out/evolve_report_filtered.txt"

    print(f"\n{'─'*60}")
    t0 = datetime.now()
    for gen in range(1, args.gen + 1):
        top_idx     = np.argmax(fitness)
        best_sharpe = fitness[top_idx]
        avg_sharpe  = np.mean(fitness[fitness > -90])

        elapsed = (datetime.now() - t0).total_seconds()
        eta     = elapsed / gen * (args.gen - gen) if gen > 1 else 0
        print(f"  世代 {gen:3d}/{args.gen}  best={best_sharpe:.3f}  "
              f"avg={avg_sharpe:.3f}  経過={elapsed:.0f}s  残={eta:.0f}s")

        history.append({"gen": gen, "best_sharpe": best_sharpe, "avg_sharpe": avg_sharpe})

        # エリート戦略を毎世代収集
        elite_n  = max(5, int(args.pop * ELITE_RATE))
        elite_idx = np.argsort(fitness)[::-1][:elite_n]
        for ei in elite_idx:
            row = format_result(pop[ei], fitness[ei], npy_out, dates_out)
            if row["stable"] == "★":
                best_csv_rows.append(row)

        # 途中保存（5世代ごと）
        if gen % 5 == 0 or gen == args.gen:
            if best_csv_rows:
                bdf = pd.DataFrame(best_csv_rows).drop_duplicates(
                    subset=["score_min","ratio_min","ratio_max","rise_lo","rise_hi","tp","sl"]
                ).sort_values("out_sharpe", ascending=False).head(30)
                bdf.to_csv(OUT_BEST, index=False, encoding="utf-8-sig")
            pd.DataFrame(history).to_csv(OUT_HISTORY, index=False, encoding="utf-8-sig")

        pop     = next_generation(pop, fitness)
        fitness = evaluate_pop(pop, npy_in)

    # ── 最終結果 ──
    total_time = (datetime.now() - t0).total_seconds()
    print(f"\n{'='*60}")
    print(f"完了  総時間: {total_time:.0f}秒 ({total_time/60:.1f}分)")

    if best_csv_rows:
        bdf = pd.DataFrame(best_csv_rows).drop_duplicates(
            subset=["score_min","ratio_min","ratio_max","rise_lo","rise_hi","tp","sl"]
        ).sort_values("out_sharpe", ascending=False).head(30)
        bdf.to_csv(OUT_BEST, index=False, encoding="utf-8-sig")
        print(f"\n【Top10 安定戦略（★: out_sharpe>1.0 & out_avg>0）】")
        star = bdf[bdf["stable"] == "★"].head(10)
        for _, r in star.iterrows():
            print(f"  [{r['type']}] score≥{r['score_min']} ratio={r['ratio_min']}~{r['ratio_max']}"
                  f" rise={r['rise_lo']}~{r['rise_hi']} TP={r['tp']}% SL={r['sl']}%"
                  f" cond={r['cond']}  →  OOS: sharpe={r['out_sharpe']} wr={r['out_wr']}%"
                  f" avg={r['out_avg']:+.3f}% ({r['out_n']}件) {r['stable']}")

    # レポート
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(f"遺伝的最適化レポート（銀行系・100円未満除外）\n")
        f.write(f"実行日時: {datetime.now()}\n")
        f.write(f"個体数: {args.pop}  世代数: {args.gen}\n")
        f.write(f"フィルター: 銀行系除外 + 100円未満除外 (MIN_PRICE={MIN_PRICE})\n\n")
        if best_csv_rows:
            f.write(bdf[bdf["stable"]=="★"].to_string(index=False))

    print(f"\n  保存: {OUT_BEST}")
    print(f"  保存: {OUT_HISTORY}")
    print(f"  保存: {OUT_REPORT}")


if __name__ == "__main__":
    main()
