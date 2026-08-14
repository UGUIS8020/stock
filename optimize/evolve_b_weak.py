"""
evolve_b_weak.py - 遺伝的アルゴリズムによるWEAK日・戦略B候補の探索（2026-08-14新設）

evolve_b.py(既存)は2泊保有バグ修正前のopt_b_trades.csv(NORMAL+STRONG固定)を
使っており、WEAK日は探索対象に入っていなかった。本スクリプトは、正しく2泊保有を
反映したWEAK日限定の候補データ(out/sim_b_weak_ga_input.csv、simulate/内の
アドホック生成スクリプトで作成)に対して、evolve_b.pyと同じGA機構をそのまま流用する。

【染色体】evolve_b.pyと同一(condは全件WEAK固定のためフィルタ対象から除外)
    gene[0] drop_lo / gene[1] drop_hi / gene[2] rb_min / gene[3] gap_max
    gene[4] tp / gene[5] sl / gene[6] top_n / gene[7] lower_shadow_min
    gene[8] vol_decay_req / gene[9] consec_drop_min

実行方法:
    python optimize/evolve_b_weak.py             # 20,000人・100世代（標準）
    python optimize/evolve_b_weak.py --pop 5000 --gen 50
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR    = "out"
TRADES_CSV = "out/sim_b_weak_ga_input.csv"
WF_SPLIT   = 0.60
MIN_SAMPLE = 30

ELITE_RATE   = 0.02
TOURNAMENT_K = 5
CROSS_RATE   = 0.80
MUTATE_RATE  = 0.15
MUTATE_SIGMA = 0.15

#    [drop_lo, drop_hi, rb_min, gap_max, tp,   sl,   top_n, ls_min, vd_req, cd_min]
GENE_MIN = np.array([-10.0, -8.0, 2.0, -3.0,  0.0, -7.0, 0.0,  0.0,  0.0,  1.0])
GENE_MAX = np.array([ -3.0, -3.0, 8.0,  5.0,  8.0,  0.0, 4.0,  0.7,  1.0,  5.0])
N_GENES  = len(GENE_MIN)

_CAL_SL  = np.array([-1.0, -2.0, -3.0, -5.0])
_CAL_TPC = np.array([0.101, 0.155, 0.183, 0.220])
_CAL_SLC = np.array([0.810, 0.684, 0.586, 0.480])

TOPN_VALUES = [1, 3, 5, 999]


def load_trades():
    if not os.path.exists(TRADES_CSV):
        print(f"❌ {TRADES_CSV} が見つかりません。先にWEAK日候補生成スクリプトを実行してください。")
        sys.exit(1)
    df = pd.read_csv(TRADES_CSV, encoding="utf-8-sig")
    df = df.sort_values(["scan_date", "rb_score"], ascending=[True, False]).reset_index(drop=True)
    df["rb_rank"] = df.groupby("scan_date")["rb_score"].rank(ascending=False, method="first")
    return df


def df_to_numpy(df):
    result = {
        "rise":    df["today_rise"].values.astype(np.float32),
        "rb":      df["rb_score"].values.astype(np.float32),
        "gap":     df["gap_pct"].values.astype(np.float32),
        "rb_rank": df["rb_rank"].values.astype(np.float32),
        "ret_oc":  df["ret_oc"].values.astype(np.float32),
        "ret_max": df["ret_max"].values.astype(np.float32),
        "ret_low": df["ret_low"].values.astype(np.float32),
        "lower_shadow": df["lower_shadow_ratio"].values.astype(np.float32),
        "vol_decay":    df["vol_decay_3d"].values.astype(np.int8),
        "consec_drop":  df["consec_drop"].values.astype(np.int8),
    }
    return result


def split_numpy(npy, dates, split_date):
    mask = dates < split_date
    def sub(m):
        return {k: v[m] for k, v in npy.items()}
    return sub(mask), sub(~mask)


def decode(ind):
    drop_lo = float(ind[0])
    drop_hi = float(ind[1])
    if drop_hi <= drop_lo:
        drop_hi = drop_lo + 0.5
    rb_min   = max(2, int(round(ind[2])))
    gap_max  = float(ind[3])
    tp_raw   = float(ind[4])
    sl_raw   = float(ind[5])
    use_close = (tp_raw < 0.5) or (sl_raw > -0.5)
    tp = None if use_close else round(tp_raw, 1)
    sl = None if use_close else round(sl_raw, 1)
    top_n    = TOPN_VALUES[int(ind[6]) % 4]
    lower_shadow_min = round(float(ind[7]), 2)
    vol_decay_req    = int(round(float(ind[8])))
    consec_drop_min  = max(1, int(round(float(ind[9]))))
    return {
        "drop_lo": drop_lo, "drop_hi": drop_hi,
        "rb_min":  rb_min,  "gap_max": gap_max,
        "tp": tp, "sl": sl, "use_close": use_close,
        "top_n": top_n,
        "lower_shadow_min": lower_shadow_min,
        "vol_decay_req":    vol_decay_req,
        "consec_drop_min":  consec_drop_min,
    }


def calc_tp_sl_returns(npy, idx, tp, sl):
    tp_p = np.interp(-sl, -_CAL_SL, _CAL_TPC) if sl is not None else 0
    sl_p = np.interp(-sl, -_CAL_SL, _CAL_SLC) if sl is not None else 0
    hi   = npy["ret_max"][idx]
    lo   = npy["ret_low"][idx]
    oc   = npy["ret_oc"][idx]
    rets = np.where(
        (hi >= tp) & (lo > sl),  tp,
        np.where(
            (lo <= sl) & (hi < tp), sl,
            np.where(
                (hi >= tp) & (lo <= sl),
                tp * tp_p / (tp_p + sl_p) + sl * sl_p / (tp_p + sl_p),
                oc
            )
        )
    )
    return rets


def evaluate_one(ind, npy):
    p = decode(ind)
    mask = (
        (npy["rise"]         <= p["drop_hi"])          &
        (npy["rise"]         >  p["drop_lo"])          &
        (npy["rb"]           >= p["rb_min"])            &
        (npy["gap"]          <= p["gap_max"])           &
        (npy["rb_rank"]      <= p["top_n"])             &
        (npy["lower_shadow"] >= p["lower_shadow_min"]) &
        (npy["consec_drop"]  >= p["consec_drop_min"])
    )
    if p["vol_decay_req"] == 1:
        mask &= npy["vol_decay"] == 1

    idx = np.where(mask)[0]
    if len(idx) < MIN_SAMPLE:
        return -999.0

    if p["use_close"]:
        rets = npy["ret_oc"][idx]
    else:
        rets = calc_tp_sl_returns(npy, idx, p["tp"], p["sl"])

    avg = rets.mean()
    std = rets.std()
    if std < 0.001:
        return -999.0
    sharpe = avg / std * np.sqrt(252)
    sharpe = min(sharpe, 5.0)
    penalty = min(1.0, len(idx) / 100.0)
    return float(sharpe * penalty)


def evaluate_population(pop, npy):
    return np.array([evaluate_one(ind, npy) for ind in pop], dtype=np.float32)


def init_population(n):
    raw = np.random.rand(n, N_GENES)
    return GENE_MIN + raw * (GENE_MAX - GENE_MIN)


def tournament_select(pop, fitness, k=TOURNAMENT_K):
    n = len(pop)
    idx = np.random.randint(0, n, size=(n, k))
    winners = np.argmax(fitness[idx], axis=1)
    selected_idx = idx[np.arange(n), winners]
    return pop[selected_idx]


def crossover(parents, rate=CROSS_RATE):
    n = len(parents)
    children = parents.copy()
    for i in range(0, n - 1, 2):
        if np.random.rand() < rate:
            mask = np.random.rand(N_GENES) < 0.5
            children[i]     = np.where(mask, parents[i],   parents[i+1])
            children[i+1]   = np.where(mask, parents[i+1], parents[i])
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
        valid    = fitness[fitness > -900]
        best     = fitness.max()
        mean_f   = valid.mean() if len(valid) > 0 else -999
        top5_idx = np.argsort(fitness)[-5:]
        top5_avg = fitness[top5_idx].mean()
        elapsed  = (datetime.now() - t0).seconds

        history.append({
            "gen": gen, "best": round(float(best), 3),
            "mean": round(float(mean_f), 3), "top5": round(float(top5_avg), 3),
            "valid_n": len(valid),
        })

        if gen % 10 == 0 or gen == n_gen - 1:
            print(f"  {gen+1:>5}世代  {best:>+10.3f}  {mean_f:>+10.3f}  {top5_avg:>+14.3f}  {elapsed}秒")

        if gen == n_gen - 1:
            break

        elite_idx = np.argsort(fitness)[-elite_n:]
        elites    = pop[elite_idx]
        parents  = tournament_select(pop, fitness)
        children = crossover(parents)
        children = mutate(children)
        children[:elite_n] = elites
        pop = children

    fitness = evaluate_population(pop, npy_in)
    return pop, fitness, pd.DataFrame(history)


def walkforward_validate(top_inds, npy_in, npy_out):
    results = []
    for ind in top_inds:
        p      = decode(ind)
        fit_in = evaluate_one(ind, npy_in)
        fit_out= evaluate_one(ind, npy_out)

        mask = (
            (npy_out["rise"]         <= p["drop_hi"])          &
            (npy_out["rise"]         >  p["drop_lo"])          &
            (npy_out["rb"]           >= p["rb_min"])            &
            (npy_out["gap"]          <= p["gap_max"])           &
            (npy_out["rb_rank"]      <= p["top_n"])             &
            (npy_out["lower_shadow"] >= p["lower_shadow_min"]) &
            (npy_out["consec_drop"]  >= p["consec_drop_min"])
        )
        if p["vol_decay_req"] == 1:
            mask &= npy_out["vol_decay"] == 1
        idx = np.where(mask)[0]

        if len(idx) < 5:
            continue

        rets = npy_out["ret_oc"][idx] if p["use_close"] else calc_tp_sl_returns(npy_out, idx, p["tp"], p["sl"])
        wr   = (rets > 0).mean() * 100
        avg  = rets.mean()
        cum  = rets.sum()
        stable = "★" if fit_in > 0 and fit_out > 0 else ""

        exit_str = "引け決済" if p["use_close"] else f"TP{p['tp']:+.1f}%/SL{p['sl']:+.1f}%"
        topn_str = str(p["top_n"]) if p["top_n"] < 999 else "全件"

        ls_str = f"ls≥{p['lower_shadow_min']:.2f}" if p["lower_shadow_min"] > 0.05 else ""
        vd_str = "売枯必須" if p["vol_decay_req"] == 1 else ""
        cd_str = f"cd≥{p['consec_drop_min']}" if p["consec_drop_min"] > 1 else ""
        pattern_str = " ".join(s for s in [ls_str, vd_str, cd_str] if s) or "-"

        results.append({
            "drop_lo":          round(p["drop_lo"], 1),
            "drop_hi":          round(p["drop_hi"], 1),
            "rb_min":           p["rb_min"],
            "gap_max":          round(p["gap_max"], 1),
            "exit":             exit_str,
            "top_n":            topn_str,
            "lower_shadow_min": p["lower_shadow_min"],
            "vol_decay_req":    p["vol_decay_req"],
            "consec_drop_min":  p["consec_drop_min"],
            "pattern":          pattern_str,
            "in_sharpe":        round(fit_in,  3),
            "out_sharpe":       round(fit_out, 3),
            "out_n":            len(idx),
            "out_wr":           round(wr, 1),
            "out_avg":          round(avg, 3),
            "out_cum":          round(cum, 1),
            "stable":           stable,
        })

    if not results:
        print("  ※ ウォークフォワード検証で有効な個体が見つかりませんでした")
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("out_sharpe", ascending=False)


def main():
    parser = argparse.ArgumentParser(description="遺伝的アルゴリズムによるWEAK日・戦略B最適化")
    parser.add_argument("--pop", type=int, default=20000, help="人口（デフォルト20000）")
    parser.add_argument("--gen", type=int, default=100,   help="世代数（デフォルト100）")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  戦略B（WEAK日限定）遺伝的アルゴリズム最適化  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  人口: {args.pop:,}人  世代数: {args.gen}世代")
    print("=" * 70)

    print(f"\nデータ読み込み中...")
    df    = load_trades()
    dates = df["trade_date"].values
    all_dates  = sorted(df["trade_date"].unique())
    split_date = all_dates[int(len(all_dates) * WF_SPLIT)]
    print(f"  全件: {len(df):,}件 / 分割日: {split_date}")
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
    print("【結果】In/Out両方プラスの戦略（★）上位20件")
    print(f"{'='*70}")
    print(f"  {'drop帯':>10} {'RB':>3} {'gap上限':>6} {'出口':>18} "
          f"{'パターン':>14} {'件数':>5} {'勝率':>6} {'Out avg':>8} {'Out Sharpe':>10} {'安定':>4}")
    print("  " + "─" * 100)

    shown = 0
    for _, r in wf_df.iterrows():
        if shown >= 20:
            break
        band    = f"{r['drop_lo']}〜{r['drop_hi']}%"
        gap_str = f"{r['gap_max']:+.1f}%"
        mark    = r["stable"]
        pattern = r.get("pattern", "-")
        print(f"  {band:>10} {r['rb_min']:>3} {gap_str:>6} "
              f"{r['exit']:>18} {pattern:>14} {r['out_n']:>5} {r['out_wr']:>5.1f}% "
              f"{r['out_avg']:>+7.3f}% {r['out_sharpe']:>+9.3f}  {mark}")
        shown += 1

    stable_df = wf_df[wf_df["stable"] == "★"] if not wf_df.empty else pd.DataFrame()
    print(f"\n  ★ In/Out両方プラス: {len(stable_df)}件 / 上位{top_n}件中")

    best_path = os.path.join(OUT_DIR, "evolve_b_weak_best.csv")
    hist_path = os.path.join(OUT_DIR, "evolve_b_weak_history.csv")
    wf_df.to_csv(best_path,    index=False, encoding="utf-8-sig")
    history_df.to_csv(hist_path, index=False, encoding="utf-8-sig")

    print(f"\n保存完了:")
    print(f"  {best_path}")
    print(f"  {hist_path}")
    # S3アップロードは行わない（過去にoptimize_b.pyで本番ファイルを誤上書きした事故があるため、
    # このアドホック探索スクリプトでは意図的に省略）


if __name__ == "__main__":
    main()
