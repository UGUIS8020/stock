# -*- coding: utf-8 -*-
"""
evolve_d.py - 遺伝的アルゴリズムによる戦略D（daytime.py）パラメータ検証

daytime.py はザラ場中の+0.3%モメンタム確認をトリガーに発注するため、
本来は分足データでの検証が望ましいが、intraday_pricesは717銘柄・36日分と
サンプルが薄い。そこで日次データ（sim_precise_trades.csv、evolve.py/evolve_b.pyと
同一ソース）を使い、「スコア・ギャップ・前日騰落・地合い」という
候補選定フィルターとTP/SL幅の妥当性を日次ベースで検証する
（ザラ場内の正確なトリガー時刻までは再現できない点に注意）。

【実行方法】
    python optimize/evolve_d.py --pop 5000 --gen 100

【出力ファイル】
    out/evolve_d_best.csv
    out/evolve_d_history.csv
    out/evolve_d_report.txt

【染色体（1人の戦略）7遺伝子】
    gene[0] score_min : スコア下限（0〜10）
    gene[1] gap_min    : 寄り付きギャップ下限（-10〜0%）
    gene[2] gap_max    : 寄り付きギャップ上限（0〜10%）※daytime.pyのGAP_MIN/MAX_PCTに対応
    gene[3] rise_lo    : 前日騰落率の下限（-10〜0%）
    gene[4] tp         : 利確幅（1〜10%）
    gene[5] sl         : 損切り幅（-10〜-0.5%）
    gene[6] cond       : 対象地合い（0=NORMAL、1=STRONG、2=NORMAL+STRONG）
                          ※daytime.pyは元々STRONG/NORMALのみ許可のためPANIC/WEAKは対象外
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "out")
TRADES_CSV = os.path.join(OUT_DIR, "sim_precise_trades.csv")
WF_SPLIT   = 0.60
MIN_SAMPLE = 30

ELITE_RATE   = 0.02
TOURNAMENT_K = 5
CROSS_RATE   = 0.80
MUTATE_RATE  = 0.15
MUTATE_SIGMA = 0.15

#                   [score, gap_min, gap_max, rise_lo, tp,   sl,   cond]
GENE_MIN = np.array([ 0.0,  -10.0,    0.0,   -10.0,    1.0, -10.0, 0.0])
GENE_MAX = np.array([10.0,    0.0,   10.0,     0.0,   10.0,  -0.5, 2.0])
N_GENES  = len(GENE_MIN)

COND_LABELS = ["NORMAL", "STRONG", "NORMAL+STRONG"]

# 現行daytime.py設定（比較用）
LIVE_SCORE_MIN = None     # daytime.pyはスコアで足切りせずadj_score優先度のみ
LIVE_GAP_MIN   = -5.0
LIVE_GAP_MAX   = 2.0
LIVE_TP        = 3.0
LIVE_SL        = -5.0


# ══════════════════════════════════════════════
# データ準備
# ══════════════════════════════════════════════
def load_trades():
    if not os.path.exists(TRADES_CSV):
        print(f"❌ {TRADES_CSV} が見つかりません。先に simulate/simulate_precise.py を実行してください。")
        sys.exit(1)
    df = pd.read_csv(TRADES_CSV, encoding="utf-8-sig")
    # daytime.pyはPANIC/WEAKを対象外にしているため、あらかじめNORMAL/STRONGのみに絞る
    df = df[df["trade_cond"].isin(["NORMAL", "STRONG"])].copy()
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df


def df_to_numpy(df):
    cond_map = {"NORMAL": 0, "STRONG": 1}
    return {
        "score":   df["score"].values.astype(np.float32),
        "rise":    df["today_rise"].values.astype(np.float32),
        "gap":     df["gap_pct"].values.astype(np.float32),
        "cond":    df["trade_cond"].map(cond_map).values.astype(np.int8),
        "ret_oc":  df["ret_oc"].values.astype(np.float32),
        "ret_max": df["ret_max"].values.astype(np.float32),
        "ret_low": df["ret_low"].values.astype(np.float32),
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
    score_min = round(float(ind[0]), 2)
    gap_min   = round(float(ind[1]), 2)
    gap_max   = round(float(ind[2]), 2)
    if gap_max <= gap_min:
        gap_max = gap_min + 0.5
    rise_lo   = round(float(ind[3]), 2)
    tp        = round(float(ind[4]), 1)
    sl        = round(float(ind[5]), 1)
    cond      = int(round(float(ind[6])))
    cond      = max(0, min(2, cond))
    return {"score_min": score_min, "gap_min": gap_min, "gap_max": gap_max,
            "rise_lo": rise_lo, "tp": tp, "sl": sl, "cond": cond}


# ══════════════════════════════════════════════
# 適応度計算
# ══════════════════════════════════════════════
def calc_tp_sl_returns(npy, idx, tp, sl):
    hi = npy["ret_max"][idx]
    lo = npy["ret_low"][idx]
    oc = npy["ret_oc"][idx]
    return np.where(
        (hi >= tp) & (lo > sl), tp,
        np.where((lo <= sl) & (hi < tp), sl, oc)   # 両方到達 or 未到達 → 引け値で近似
    )


def evaluate_one(ind, npy):
    p = decode(ind)
    mask = (
        (npy["score"] >= p["score_min"]) &
        (npy["gap"]   >= p["gap_min"])   &
        (npy["gap"]   <= p["gap_max"])   &
        (npy["rise"]  >= p["rise_lo"])
    )
    if p["cond"] == 0:
        mask &= npy["cond"] == 0
    elif p["cond"] == 1:
        mask &= npy["cond"] == 1
    # cond==2 は NORMAL+STRONG両方なのでフィルタなし

    idx = np.where(mask)[0]
    if len(idx) < MIN_SAMPLE:
        return -99.0

    rets = calc_tp_sl_returns(npy, idx, p["tp"], p["sl"])
    mean, std = rets.mean(), rets.std()
    if std < 1e-9:
        return -99.0
    sharpe  = min(mean / std * np.sqrt(252), 5.0)
    penalty = min(1.0, len(idx) / 100.0)
    return float(sharpe * penalty)


def evaluate_pop(pop, npy):
    return np.array([evaluate_one(ind, npy) for ind in pop])


# ══════════════════════════════════════════════
# GA オペレータ
# ══════════════════════════════════════════════
def init_population(n):
    raw = np.random.rand(n, N_GENES)
    return GENE_MIN + raw * (GENE_MAX - GENE_MIN)


def tournament_select(pop, fitness, k=TOURNAMENT_K):
    n = len(pop)
    idx = np.random.randint(0, n, size=(n, k))
    winners = np.argmax(fitness[idx], axis=1)
    return pop[idx[np.arange(n), winners]]


def crossover(parents, rate=CROSS_RATE):
    n = len(parents)
    children = parents.copy()
    for i in range(0, n - 1, 2):
        if np.random.rand() < rate:
            mask = np.random.rand(N_GENES) < 0.5
            children[i]   = np.where(mask, parents[i],   parents[i+1])
            children[i+1] = np.where(mask, parents[i+1], parents[i])
    return children


def mutate(pop, rate=MUTATE_RATE, sigma=MUTATE_SIGMA):
    mask  = np.random.rand(*pop.shape) < rate
    noise = np.random.randn(*pop.shape) * sigma * (GENE_MAX - GENE_MIN)
    pop   = pop + mask * noise
    return np.clip(pop, GENE_MIN, GENE_MAX)


def run_ga(npy_in, pop_size, n_gen):
    elite_n = max(1, int(pop_size * ELITE_RATE))
    pop     = init_population(pop_size)
    history = []

    print(f"\n  人口: {pop_size:,}人  世代数: {n_gen}  エリート: {elite_n}人")
    print(f"  {'世代':>5}  {'最高Sharpe':>11}  {'平均Sharpe':>11}  {'上位5平均':>11}  経過時間")
    print("  " + "─" * 60)

    t0 = datetime.now()
    for gen in range(n_gen):
        fitness = evaluate_pop(pop, npy_in)
        valid   = fitness[fitness > -90]
        best    = fitness.max()
        mean_f  = valid.mean() if len(valid) > 0 else -99
        top5    = fitness[np.argsort(fitness)[-5:]].mean()
        elapsed = (datetime.now() - t0).seconds

        history.append({"gen": gen, "best": round(float(best), 3),
                        "mean": round(float(mean_f), 3), "top5": round(float(top5), 3)})

        if gen % 10 == 0 or gen == n_gen - 1:
            print(f"  {gen+1:>5}世代  {best:>+10.3f}  {mean_f:>+10.3f}  {top5:>+10.3f}  {elapsed}秒")

        if gen == n_gen - 1:
            break

        elite_idx = np.argsort(fitness)[-elite_n:]
        elites    = pop[elite_idx]
        parents   = tournament_select(pop, fitness)
        children  = crossover(parents)
        children  = mutate(children)
        children[:elite_n] = elites
        pop = children

    fitness = evaluate_pop(pop, npy_in)
    return pop, fitness, pd.DataFrame(history)


# ══════════════════════════════════════════════
# ウォークフォワード検証
# ══════════════════════════════════════════════
def walkforward_validate(top_inds, npy_in, npy_out):
    results = []
    seen = set()
    for ind in top_inds:
        p = decode(ind)
        key = (p["score_min"], p["gap_min"], p["gap_max"], p["rise_lo"], p["tp"], p["sl"], p["cond"])
        if key in seen:
            continue
        seen.add(key)

        fit_in  = evaluate_one(ind, npy_in)
        fit_out = evaluate_one(ind, npy_out)

        mask = (
            (npy_out["score"] >= p["score_min"]) &
            (npy_out["gap"]   >= p["gap_min"])   &
            (npy_out["gap"]   <= p["gap_max"])   &
            (npy_out["rise"]  >= p["rise_lo"])
        )
        if p["cond"] == 0:
            mask &= npy_out["cond"] == 0
        elif p["cond"] == 1:
            mask &= npy_out["cond"] == 1
        idx = np.where(mask)[0]
        if len(idx) < 10:
            continue

        rets = calc_tp_sl_returns(npy_out, idx, p["tp"], p["sl"])
        wr  = (rets > 0).mean() * 100
        avg = rets.mean()
        stable = "★" if fit_in > 0 and fit_out > 0 else ""

        results.append({
            "score_min": p["score_min"], "gap_min": p["gap_min"], "gap_max": p["gap_max"],
            "rise_lo": p["rise_lo"], "tp": p["tp"], "sl": p["sl"],
            "cond": COND_LABELS[p["cond"]],
            "in_sharpe": round(fit_in, 3), "out_sharpe": round(fit_out, 3),
            "out_n": len(idx), "out_wr": round(wr, 1), "out_avg": round(avg, 3),
            "stable": stable,
        })

    if not results:
        print("  ※ ウォークフォワード検証で有効な個体が見つかりませんでした")
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("out_sharpe", ascending=False)


# ══════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="遺伝的アルゴリズムによる戦略D検証")
    parser.add_argument("--pop", type=int, default=5000)
    parser.add_argument("--gen", type=int, default=100)
    args = parser.parse_args()

    print("=" * 70)
    print(f"  戦略D 遺伝的アルゴリズム検証  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  人口: {args.pop:,}人  世代数: {args.gen}世代")
    print("=" * 70)
    print(f"\n  ※ 日次データによる近似検証（ザラ場内トリガー時刻は再現不可）")
    print(f"  現行ライブ設定: gap{LIVE_GAP_MIN:+.1f}〜{LIVE_GAP_MAX:+.1f}% TP+{LIVE_TP:.1f}% SL{LIVE_SL:+.1f}%")

    df = load_trades()
    dates = df["trade_date"].values
    all_dates  = sorted(df["trade_date"].unique())
    split_date = all_dates[int(len(all_dates) * WF_SPLIT)]
    print(f"\n  全件: {len(df):,}件 / 分割日: {split_date}")
    print(f"  In-sample: {(dates < split_date).sum():,}件 / Out-of-sample: {(dates >= split_date).sum():,}件")

    npy_all = df_to_numpy(df)
    npy_in, npy_out = split_numpy(npy_all, dates, split_date)

    print(f"\n【進化開始】")
    np.random.seed(42)
    t_start = datetime.now()
    pop, fitness, history_df = run_ga(npy_in, args.pop, args.gen)
    elapsed = (datetime.now() - t_start).seconds
    print(f"\n  完了: {elapsed}秒")

    top_n   = min(200, args.pop)
    top_idx = np.argsort(fitness)[-top_n:][::-1]
    top_pop = pop[top_idx]

    print(f"\n【ウォークフォワード検証】上位{top_n}個体をOut-of-sampleで検証中...")
    wf_df = walkforward_validate(top_pop, npy_in, npy_out)

    print(f"\n{'='*70}")
    print("【結果】上位20件")
    print(f"{'='*70}")
    print(f"  {'score':>6} {'gap帯':>12} {'rise_lo':>8} {'TP':>6} {'SL':>7} {'地合い':>14} {'N':>6} {'WR':>6} {'avg%':>8} {'安定':>4}")
    print("  " + "─" * 90)
    shown = 0
    for _, r in wf_df.iterrows():
        if shown >= 20:
            break
        gap_str = f"{r['gap_min']:+.1f}〜{r['gap_max']:+.1f}%"
        print(f"  {r['score_min']:>6.1f} {gap_str:>12} {r['rise_lo']:>+7.1f}% {r['tp']:>+5.1f}% {r['sl']:>+6.1f}% "
              f"{r['cond']:>14} {r['out_n']:>6} {r['out_wr']:>5.1f}% {r['out_avg']:>+7.3f}%  {r['stable']}")
        shown += 1

    stable_df = wf_df[wf_df["stable"] == "★"] if not wf_df.empty else pd.DataFrame()
    print(f"\n  ★ In/Out両方プラス: {len(stable_df)}件 / 上位{top_n}件中（重複除去後 {len(wf_df)}件）")
    if not stable_df.empty:
        print(f"  地合い内訳: " + " / ".join(f"{c}:{(stable_df['cond']==c).sum()}件" for c in COND_LABELS))

    best_path = os.path.join(OUT_DIR, "evolve_d_best.csv")
    hist_path = os.path.join(OUT_DIR, "evolve_d_history.csv")
    wf_df.to_csv(best_path, index=False, encoding="utf-8-sig")
    history_df.to_csv(hist_path, index=False, encoding="utf-8-sig")

    report_lines = [
        "戦略D 遺伝的アルゴリズム検証レポート",
        f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"人口: {args.pop:,}人  世代数: {args.gen}  所要時間: {elapsed}秒",
        f"データ期間: {df['trade_date'].min()} 〜 {df['trade_date'].max()}  (全{len(df):,}件、NORMAL/STRONGのみ)",
        f"ウォークフォワード分割日: {split_date}",
        "",
        f"現行ライブ設定: gap{LIVE_GAP_MIN:+.1f}〜{LIVE_GAP_MAX:+.1f}% TP+{LIVE_TP:.1f}% SL{LIVE_SL:+.1f}%",
        "",
        "【In/Out両方プラス（推奨）上位5件】",
    ]
    for _, r in stable_df.head(5).iterrows():
        report_lines.append(
            f"  score>={r['score_min']:.1f} gap{r['gap_min']:+.1f}〜{r['gap_max']:+.1f}% rise>={r['rise_lo']:+.1f}% "
            f"TP{r['tp']:+.1f}%/SL{r['sl']:+.1f}% {r['cond']} | N={r['out_n']} WR={r['out_wr']}% avg={r['out_avg']:+.3f}%"
        )
    report_lines.append("\n※ 日次データによる近似検証。ザラ場内の+0.3%モメンタムトリガーの正確な時刻は未反映。")
    report_txt = "\n".join(report_lines)

    rep_path = os.path.join(OUT_DIR, "evolve_d_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    print(f"\n保存完了: {best_path}")
    print(f"          {hist_path}")
    print(f"          {rep_path}")
    print(f"\n{report_txt}")


if __name__ == "__main__":
    main()
