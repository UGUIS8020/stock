# -*- coding: utf-8 -*-
"""
evolve_f.py - 遺伝的アルゴリズムによる戦略F（銀行×金利マクロ連動）最適化

optimize_f.py が生成した out/opt_f_trades.csv を使い、
rate閾値・スコア下限・TP・SLの組み合わせを進化的に探索する。

【実行方法】
    python evolve_f.py                    # 3000人・60世代（標準・母数が小さいため軽量設定）
    python evolve_f.py --pop 5000 --gen 100

【出力ファイル】
    out/evolve_f_best.csv     - 上位戦略一覧（ウォークフォワード検証済み）
    out/evolve_f_history.csv  - 世代別Sharpe推移
    out/evolve_f_report.txt   - サマリーレポート

【染色体（1人の戦略）】
    gene[0] rate_min : rate_chg（前日までの米10年債利回り変化率%）の下限（0.0〜3.0）
    gene[1] score_min: calc_score_dスコアの下限（0.0〜10.0）
    gene[2] tp        : 利確幅（0=引け決済、0.5〜8%）
    gene[3] sl        : 損切り幅（0=引け決済、-1〜-6%）

母数が2,000〜数百件程度と小さいため、evolve_b.py よりも人口・世代数を絞り、
サンプル数ペナルティを強めに掛けて過学習を抑制する。
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "out")
TRADES_CSV = os.path.join(OUT_DIR, "opt_f_trades.csv")
WF_SPLIT   = 0.60
MIN_SAMPLE = 20    # 戦略Bより母数が少ないため緩め（30→20）

# ── GA パラメータ ──
ELITE_RATE   = 0.02
TOURNAMENT_K = 5
CROSS_RATE   = 0.80
MUTATE_RATE  = 0.15
MUTATE_SIGMA = 0.15

# [rate_min, score_min, tp, sl]
GENE_MIN = np.array([0.0, 0.0, 0.0, -6.0])
GENE_MAX = np.array([3.0, 10.0, 8.0,  0.0])
N_GENES  = len(GENE_MIN)


# ══════════════════════════════════════════════
# データ準備
# ══════════════════════════════════════════════
def load_trades():
    if not os.path.exists(TRADES_CSV):
        print(f"❌ {TRADES_CSV} が見つかりません。先に optimize_f.py を実行してください。")
        sys.exit(1)
    df = pd.read_csv(TRADES_CSV, encoding="utf-8-sig")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def df_to_numpy(df):
    return {
        "rate_chg": df["rate_chg"].values.astype(np.float32),
        "score":    df["score"].values.astype(np.float32),
        "ret_oc":   df["ret_oc"].values.astype(np.float32),
        "ret_max":  df["ret_max"].values.astype(np.float32),
        "ret_low":  df["ret_low"].values.astype(np.float32),
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
    rate_min  = round(float(ind[0]), 2)
    score_min = round(float(ind[1]), 2)
    tp_raw    = float(ind[2])
    sl_raw    = float(ind[3])
    use_close = (tp_raw < 0.3) or (sl_raw > -0.3)
    tp = None if use_close else round(tp_raw, 1)
    sl = None if use_close else round(sl_raw, 1)
    return {"rate_min": rate_min, "score_min": score_min,
            "tp": tp, "sl": sl, "use_close": use_close}


# ══════════════════════════════════════════════
# 適応度計算
# ══════════════════════════════════════════════
def calc_tp_sl_returns(npy, idx, tp, sl):
    """同日中にTP/SL両方到達した場合は順序不明のため引け値で近似する
    （sim_precise.py 相当の銀行株専用キャリブレーションが無いための簡易近似）。
    """
    hi = npy["ret_max"][idx]
    lo = npy["ret_low"][idx]
    oc = npy["ret_oc"][idx]
    return np.where(
        (hi >= tp) & (lo > sl), tp,
        np.where((lo <= sl) & (hi < tp), sl, oc)
    )


def evaluate_one(ind, npy):
    p = decode(ind)
    mask = (npy["rate_chg"] >= p["rate_min"]) & (npy["score"] >= p["score_min"])
    idx = np.where(mask)[0]
    if len(idx) < MIN_SAMPLE:
        return -999.0

    rets = npy["ret_oc"][idx] if p["use_close"] else calc_tp_sl_returns(npy, idx, p["tp"], p["sl"])
    avg = rets.mean()
    std = rets.std()
    if std < 0.001:
        return -999.0
    sharpe = avg / std * np.sqrt(252)
    sharpe = min(sharpe, 5.0)
    # 母数が小さいため、100件基準のペナルティに加えさらに緩やかに絞る
    penalty = min(1.0, len(idx) / 100.0)
    return float(sharpe * penalty)


def evaluate_population(pop, npy):
    return np.array([evaluate_one(ind, npy) for ind in pop], dtype=np.float32)


# ══════════════════════════════════════════════
# GA オペレータ（evolve_b.pyと同一方式）
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
    print(f"  {'世代':>5}  {'最高Sharpe':>11}  {'平均Sharpe':>11}  {'上位Sharpe5平均':>15}  経過時間")
    print("  " + "─" * 62)

    t0 = datetime.now()
    for gen in range(n_gen):
        fitness = evaluate_population(pop, npy_in)
        valid   = fitness[fitness > -900]
        best    = fitness.max()
        mean_f  = valid.mean() if len(valid) > 0 else -999
        top5_avg = fitness[np.argsort(fitness)[-5:]].mean()
        elapsed  = (datetime.now() - t0).seconds

        history.append({"gen": gen, "best": round(float(best), 3),
                        "mean": round(float(mean_f), 3), "top5": round(float(top5_avg), 3),
                        "valid_n": len(valid)})

        if gen % 10 == 0 or gen == n_gen - 1:
            print(f"  {gen+1:>5}世代  {best:>+10.3f}  {mean_f:>+10.3f}  {top5_avg:>+14.3f}  {elapsed}秒")

        if gen == n_gen - 1:
            break

        elite_idx = np.argsort(fitness)[-elite_n:]
        elites    = pop[elite_idx]
        parents   = tournament_select(pop, fitness)
        children  = crossover(parents)
        children  = mutate(children)
        children[:elite_n] = elites
        pop = children

    fitness = evaluate_population(pop, npy_in)
    return pop, fitness, pd.DataFrame(history)


# ══════════════════════════════════════════════
# ウォークフォワード検証
# ══════════════════════════════════════════════
def walkforward_validate(top_inds, npy_in, npy_out):
    results = []
    seen = set()
    for ind in top_inds:
        p = decode(ind)
        key = (p["rate_min"], p["score_min"], p["tp"], p["sl"])
        if key in seen:
            continue
        seen.add(key)

        fit_in  = evaluate_one(ind, npy_in)
        fit_out = evaluate_one(ind, npy_out)

        mask = (npy_out["rate_chg"] >= p["rate_min"]) & (npy_out["score"] >= p["score_min"])
        idx = np.where(mask)[0]
        if len(idx) < 5:
            continue

        rets = npy_out["ret_oc"][idx] if p["use_close"] else calc_tp_sl_returns(npy_out, idx, p["tp"], p["sl"])
        wr  = (rets > 0).mean() * 100
        avg = rets.mean()
        cum = rets.sum()
        stable = "★" if fit_in > 0 and fit_out > 0 else ""
        exit_str = "引け決済" if p["use_close"] else f"TP{p['tp']:+.1f}%/SL{p['sl']:+.1f}%"

        results.append({
            "rate_min": p["rate_min"], "score_min": p["score_min"], "exit": exit_str,
            "in_sharpe": round(fit_in, 3), "out_sharpe": round(fit_out, 3),
            "out_n": len(idx), "out_wr": round(wr, 1), "out_avg": round(avg, 3),
            "out_cum": round(cum, 1), "stable": stable,
        })

    if not results:
        print("  ※ ウォークフォワード検証で有効な個体が見つかりませんでした")
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("out_sharpe", ascending=False)


# ══════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="遺伝的アルゴリズムによる戦略F最適化")
    parser.add_argument("--pop", type=int, default=3000, help="人口（デフォルト3000・母数が小さいため軽量）")
    parser.add_argument("--gen", type=int, default=60,   help="世代数（デフォルト60）")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  戦略F 遺伝的アルゴリズム最適化  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  人口: {args.pop:,}人  世代数: {args.gen}世代")
    print("=" * 70)

    df = load_trades()
    dates = df["date"].values
    all_dates  = sorted(df["date"].unique())
    split_date = all_dates[int(len(all_dates) * WF_SPLIT)]
    print(f"\n  全件: {len(df):,}件 / 分割日: {split_date}")
    print(f"  In-sample: {(dates < split_date).sum():,}件 / Out-of-sample: {(dates >= split_date).sum():,}件")

    npy_all = df_to_numpy(df)
    npy_in, npy_out = split_numpy(npy_all, dates, split_date)

    print(f"\n【進化開始】In-sampleデータで{args.pop:,}人が{args.gen}世代進化します...")
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
    print(f"  {'rate>=':>7} {'score>=':>8} {'出口':>16} {'件数':>5} {'勝率':>6} {'Out avg':>8} {'Out Sharpe':>10} {'安定':>4}")
    print("  " + "─" * 70)
    shown = 0
    for _, r in wf_df.iterrows():
        if shown >= 20:
            break
        print(f"  {r['rate_min']:>6.2f}% {r['score_min']:>7.2f}  {r['exit']:>16} "
              f"{r['out_n']:>5} {r['out_wr']:>5.1f}% {r['out_avg']:>+7.3f}% "
              f"{r['out_sharpe']:>+9.3f}  {r['stable']}")
        shown += 1

    stable_df = wf_df[wf_df["stable"] == "★"] if not wf_df.empty else pd.DataFrame()
    print(f"\n  ★ In/Out両方プラス: {len(stable_df)}件 / 上位{top_n}件中（重複除去後 {len(wf_df)}件）")

    best_path = os.path.join(OUT_DIR, "evolve_f_best.csv")
    hist_path = os.path.join(OUT_DIR, "evolve_f_history.csv")
    wf_df.to_csv(best_path, index=False, encoding="utf-8-sig")
    history_df.to_csv(hist_path, index=False, encoding="utf-8-sig")

    report_lines = [
        "戦略F 遺伝的アルゴリズム最適化レポート",
        f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"人口: {args.pop:,}人  世代数: {args.gen}  所要時間: {elapsed}秒",
        f"データ期間: {df['date'].min()} 〜 {df['date'].max()}  (全{len(df):,}件)",
        f"ウォークフォワード分割日: {split_date}",
        "",
        f"現行ライブ設定: rate>=1.0% score>=7.5",
        "",
        "【In/Out両方プラス（推奨）上位5件】",
    ]
    for _, r in stable_df.head(5).iterrows():
        report_lines.append(
            f"  rate>={r['rate_min']:.2f}% score>={r['score_min']:.2f} {r['exit']} | "
            f"N={r['out_n']} WR={r['out_wr']}% avg={r['out_avg']:+.3f}% Sharpe={r['out_sharpe']:+.3f}"
        )
    if stable_df.empty and not wf_df.empty:
        best_row = wf_df.iloc[0]
        report_lines.append(
            f"  ※ In/Out両方プラスの個体なし。Out-of-sample最良: "
            f"rate>={best_row['rate_min']:.2f}% score>={best_row['score_min']:.2f} "
            f"{best_row['exit']} avg={best_row['out_avg']:+.3f}%"
        )
    report_lines.append("\n※ 母数が最大でも2,500件程度（5銘柄×約2年）と小さく、過学習リスクが高い点に留意。")
    report_txt = "\n".join(report_lines)

    rep_path = os.path.join(OUT_DIR, "evolve_f_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    print(f"\n保存完了:")
    print(f"  {best_path}")
    print(f"  {hist_path}")
    print(f"  {rep_path}")
    print(f"\n{report_txt}")


if __name__ == "__main__":
    main()
