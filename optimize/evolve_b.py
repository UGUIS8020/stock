"""
evolve_b.py - 遺伝的アルゴリズムによる戦略B最適化

20,000人の「仮想投資家」が各自の戦略で売買し、
利益が出た戦略が生き残り・交配・進化を繰り返す。
何世代も後に「最強の戦略パラメータ」が自然に浮かび上がる。

【実行方法】
    python evolve_b.py                    # 20,000人・100世代（標準）
    python evolve_b.py --pop 5000         # 人数指定
    python evolve_b.py --gen 50           # 世代数指定
    python evolve_b.py --pop 20000 --gen 200  # フル設定

【出力ファイル】
    out/evolve_b_best.csv     - 上位戦略一覧（過学習検証済み）
    out/evolve_b_history.csv  - 世代別Sharpe推移
    out/evolve_b_report.txt   - サマリーレポート

【染色体（1人の戦略）】
    gene[0] drop_lo   : 何%以上の下落は対象外（-10〜-3）
    gene[1] drop_hi   : 何%以下の下落を対象にするか（-8〜-3）
    gene[2] rb_min    : RBスコア最低値（2〜8）
    gene[3] gap_max   : 翌朝ギャップアップ許容上限（-3〜+5%）
    gene[4] tp        : 利確幅（0=引け決済、1〜8%）
    gene[5] sl        : 損切り幅（0=引け決済、-1〜-7%）
    gene[6] cond      : 対象地合い（0=全、1=NORMAL、2=STRONG、3=N+S）
    gene[7] top_n     : 1日に買う最大件数（1/3/5/全件）
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR     = "out"
TRADES_CSV  = "out/opt_b_trades.csv"
WF_SPLIT    = 0.60   # 前60%=学習期間 / 後40%=検証期間
MIN_SAMPLE  = 5      # 適応度計算に必要な最小トレード数（In-sampleが少ないため緩め）

# ── GA パラメータ ──
ELITE_RATE    = 0.02   # 上位2%はそのまま次世代へ
TOURNAMENT_K  = 5      # トーナメント選択のサイズ
CROSS_RATE    = 0.80   # 交叉確率
MUTATE_RATE   = 0.15   # 各遺伝子の突然変異確率
MUTATE_SIGMA  = 0.15   # 突然変異の強さ（正規化座標での標準偏差）

# ── 染色体の定義（各遺伝子の最小・最大） ──
#    [drop_lo, drop_hi, rb_min, gap_max, tp, sl, cond, top_n]
GENE_MIN = np.array([-10.0, -8.0, 2.0, -3.0,  0.0, -7.0, 0.0, 0.0])
GENE_MAX = np.array([ -3.0, -3.0, 8.0,  5.0,  8.0,  0.0, 4.0, 4.0])
N_GENES  = len(GENE_MIN)

# SL キャリブレーション（sim_precise.py 由来）
# 任意の SL に対して線形補間で tp_prob/sl_prob を推定
_CAL_SL  = np.array([-1.0, -2.0, -3.0, -5.0])
_CAL_TPC = np.array([0.101, 0.155, 0.183, 0.220])
_CAL_SLC = np.array([0.810, 0.684, 0.586, 0.480])

COND_LABELS = ["全地合い", "NORMAL", "STRONG", "NORMAL+STRONG"]
TOPN_VALUES = [1, 3, 5, 999]


# ══════════════════════════════════════════════
# データ準備
# ══════════════════════════════════════════════
def load_trades():
    if not os.path.exists(TRADES_CSV):
        print(f"❌ {TRADES_CSV} が見つかりません。先に optimize_b.py を実行してください。")
        sys.exit(1)
    df = pd.read_csv(TRADES_CSV, encoding="utf-8-sig")
    # 日付でソートし、rb_rank（同日内でのRBスコア順位）を付与
    df = df.sort_values(["scan_date", "rb_score"], ascending=[True, False]).reset_index(drop=True)
    df["rb_rank"] = df.groupby("scan_date")["rb_score"].rank(ascending=False, method="first")
    return df


def df_to_numpy(df):
    """高速評価のためにnumpy配列へ変換"""
    cond_map = {"NORMAL": 1, "STRONG": 2, "WEAK": 3}
    return {
        "rise":     df["today_rise"].values.astype(np.float32),
        "rb":       df["rb_score"].values.astype(np.float32),
        "gap":      df["gap_pct"].values.astype(np.float32),
        "cond":     df["trade_cond"].map(cond_map).fillna(0).values.astype(np.int8),
        "rb_rank":  df["rb_rank"].values.astype(np.float32),
        "ret_oc":   df["ret_oc"].values.astype(np.float32),
        "ret_max":  df["ret_max"].values.astype(np.float32),
        "ret_low":  df["ret_low"].values.astype(np.float32),
    }


def split_numpy(npy, dates, split_date):
    """ウォークフォワード用にin/out分割"""
    mask = dates < split_date
    def sub(m):
        return {k: v[m] for k, v in npy.items()}
    return sub(mask), sub(~mask)


# ══════════════════════════════════════════════
# 染色体デコード
# ══════════════════════════════════════════════
def decode(ind):
    """float染色体 → 実際のパラメータ辞書"""
    drop_lo = float(ind[0])
    drop_hi = float(ind[1])
    # drop_hi（上限・less negative）は drop_lo（下限・more negative）より大きくなければならない
    # 例: drop_hi=-4 > drop_lo=-8 が有効。逆なら drop_hi を drop_lo+0.5 に補正。
    if drop_hi <= drop_lo:
        drop_hi = drop_lo + 0.5
    rb_min   = max(2, int(round(ind[2])))
    gap_max  = float(ind[3])
    tp_raw   = float(ind[4])
    sl_raw   = float(ind[5])
    use_close = (tp_raw < 0.5) or (sl_raw > -0.5)  # tp≈0 or sl≈0 → 引け決済
    tp = None if use_close else round(tp_raw, 1)
    sl = None if use_close else round(sl_raw, 1)
    cond   = int(ind[6]) % 4
    top_n  = TOPN_VALUES[int(ind[7]) % 4]
    return {
        "drop_lo": drop_lo, "drop_hi": drop_hi,
        "rb_min":  rb_min,  "gap_max": gap_max,
        "tp": tp, "sl": sl, "use_close": use_close,
        "cond": cond, "top_n": top_n,
    }


# ══════════════════════════════════════════════
# 適応度計算（高速numpy版）
# ══════════════════════════════════════════════
def calc_tp_sl_returns(npy, idx, tp, sl):
    """TP/SL適用後リターンをnumpyで計算"""
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
    """1個体の適応度（Sharpe比）を返す"""
    p = decode(ind)

    # フィルタ
    mask = (
        (npy["rise"] <= p["drop_hi"]) &
        (npy["rise"] >  p["drop_lo"]) &
        (npy["rb"]   >= p["rb_min"])  &
        (npy["gap"]  <= p["gap_max"]) &
        (npy["rb_rank"] <= p["top_n"])
    )
    if p["cond"] == 1:
        mask &= npy["cond"] == 1
    elif p["cond"] == 2:
        mask &= npy["cond"] == 2
    elif p["cond"] == 3:
        mask &= npy["cond"] <= 2   # NORMAL(1) or STRONG(2)

    idx = np.where(mask)[0]
    if len(idx) < MIN_SAMPLE:
        return -999.0

    if p["use_close"]:
        rets = npy["ret_oc"][idx]
    else:
        rets = calc_tp_sl_returns(npy, idx, p["tp"], p["sl"])

    avg = rets.mean()
    std = rets.std()
    if std < 0.001:   # std が極端に小さい（TP均一ヒット等）は除外
        return -999.0
    sharpe = avg / std * np.sqrt(252)
    sharpe = min(sharpe, 15.0)   # 非現実的な高Sharpeをキャップ
    # サンプル数ペナルティ: 少ないサンプルは信頼性低
    penalty = min(1.0, len(idx) / 50.0)
    return float(sharpe * penalty)


def evaluate_population(pop, npy):
    """全個体の適応度をまとめて計算"""
    return np.array([evaluate_one(ind, npy) for ind in pop], dtype=np.float32)


# ══════════════════════════════════════════════
# GA オペレータ
# ══════════════════════════════════════════════
def init_population(n):
    """ランダム初期集団（遺伝子値は0〜1に正規化してから実値へ変換）"""
    raw = np.random.rand(n, N_GENES)
    return GENE_MIN + raw * (GENE_MAX - GENE_MIN)


def tournament_select(pop, fitness, k=TOURNAMENT_K):
    """トーナメント選択: k個からランダム選択し最高適応度を返す"""
    n = len(pop)
    idx = np.random.randint(0, n, size=(n, k))
    winners = np.argmax(fitness[idx], axis=1)
    selected_idx = idx[np.arange(n), winners]
    return pop[selected_idx]


def crossover(parents, rate=CROSS_RATE):
    """一様交叉: 隣り合う2個体をペアに"""
    n = len(parents)
    children = parents.copy()
    for i in range(0, n - 1, 2):
        if np.random.rand() < rate:
            mask = np.random.rand(N_GENES) < 0.5
            children[i]     = np.where(mask, parents[i],   parents[i+1])
            children[i+1]   = np.where(mask, parents[i+1], parents[i])
    return children


def mutate(pop, rate=MUTATE_RATE, sigma=MUTATE_SIGMA):
    """ガウスノイズ突然変異（各遺伝子独立）"""
    mask  = np.random.rand(*pop.shape) < rate
    noise = np.random.randn(*pop.shape) * sigma * (GENE_MAX - GENE_MIN)
    pop   = pop + mask * noise
    return np.clip(pop, GENE_MIN, GENE_MAX)


# ══════════════════════════════════════════════
# メインGA ループ
# ══════════════════════════════════════════════
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

        # 統計
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

        # エリート保存
        elite_idx = np.argsort(fitness)[-elite_n:]
        elites    = pop[elite_idx]

        # 選択 → 交叉 → 突然変異
        parents  = tournament_select(pop, fitness)
        children = crossover(parents)
        children = mutate(children)

        # エリートを差し替え
        children[:elite_n] = elites
        pop = children

    # 最終世代の適応度
    fitness = evaluate_population(pop, npy_in)
    return pop, fitness, pd.DataFrame(history)


# ══════════════════════════════════════════════
# ウォークフォワード検証
# ══════════════════════════════════════════════
def walkforward_validate(top_inds, npy_in, npy_out):
    results = []
    for ind in top_inds:
        p      = decode(ind)
        fit_in = evaluate_one(ind, npy_in)
        fit_out= evaluate_one(ind, npy_out)

        # out-of-sample の詳細指標
        mask = (
            (npy_out["rise"] <= p["drop_hi"]) &
            (npy_out["rise"] >  p["drop_lo"]) &
            (npy_out["rb"]   >= p["rb_min"])  &
            (npy_out["gap"]  <= p["gap_max"]) &
            (npy_out["rb_rank"] <= p["top_n"])
        )
        if p["cond"] == 1:  mask &= npy_out["cond"] == 1
        elif p["cond"] == 2: mask &= npy_out["cond"] == 2
        elif p["cond"] == 3: mask &= npy_out["cond"] <= 2
        idx = np.where(mask)[0]

        if len(idx) < 5:
            continue

        rets = npy_out["ret_oc"][idx] if p["use_close"] else calc_tp_sl_returns(npy_out, idx, p["tp"], p["sl"])
        wr   = (rets > 0).mean() * 100
        avg  = rets.mean()
        cum  = rets.sum()
        stable = "★" if fit_in > 0 and fit_out > 0 else ""

        exit_str = "引け決済" if p["use_close"] else f"TP{p['tp']:+.1f}%/SL{p['sl']:+.1f}%"
        cond_str = COND_LABELS[p["cond"]]
        topn_str = str(p["top_n"]) if p["top_n"] < 999 else "全件"

        results.append({
            "drop_lo":   round(p["drop_lo"], 1),
            "drop_hi":   round(p["drop_hi"], 1),
            "rb_min":    p["rb_min"],
            "gap_max":   round(p["gap_max"], 1),
            "exit":      exit_str,
            "condition": cond_str,
            "top_n":     topn_str,
            "in_sharpe": round(fit_in,  3),
            "out_sharpe":round(fit_out, 3),
            "out_n":     len(idx),
            "out_wr":    round(wr, 1),
            "out_avg":   round(avg, 3),
            "out_cum":   round(cum, 1),
            "stable":    stable,
        })

    if not results:
        print("  ※ ウォークフォワード検証で有効な個体が見つかりませんでした")
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("out_sharpe", ascending=False)


# ══════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="遺伝的アルゴリズムによる戦略B最適化")
    parser.add_argument("--pop", type=int, default=20000, help="人口（デフォルト20000）")
    parser.add_argument("--gen", type=int, default=100,   help="世代数（デフォルト100）")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  戦略B 遺伝的アルゴリズム最適化  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  人口: {args.pop:,}人  世代数: {args.gen}世代")
    print("=" * 70)

    # ── データ読み込み ──
    print("\nデータ読み込み中...")
    df    = load_trades()
    dates = df["trade_date"].values
    all_dates  = sorted(df["trade_date"].unique())
    split_date = all_dates[int(len(all_dates) * WF_SPLIT)]
    print(f"  全件: {len(df):,}件 / 分割日: {split_date}")
    print(f"  In-sample: {(dates < split_date).sum():,}件 / Out-of-sample: {(dates >= split_date).sum():,}件")
    print(f"  地合い: " + " / ".join(f"{c}:{(df['trade_cond']==c).sum()}" for c in ["STRONG","NORMAL","WEAK"]))

    npy_all = df_to_numpy(df)
    npy_in, npy_out = split_numpy(npy_all, dates, split_date)

    # ── GA 実行（In-sampleで進化）──
    print(f"\n【進化開始】In-sampleデータで{args.pop:,}人が{args.gen}世代進化します...")
    np.random.seed(42)
    t_start = datetime.now()
    pop, fitness, history_df = run_ga(npy_in, args.pop, args.gen)
    elapsed = (datetime.now() - t_start).seconds
    print(f"\n  完了: {elapsed}秒")

    # ── 上位個体を取得 ──
    top_n   = min(200, args.pop)
    top_idx = np.argsort(fitness)[-top_n:][::-1]
    top_pop = pop[top_idx]

    # ── ウォークフォワード検証 ──
    print(f"\n【ウォークフォワード検証】上位{top_n}個体をOut-of-sampleで検証中...")
    wf_df = walkforward_validate(top_pop, npy_in, npy_out)

    # ── 結果表示 ──
    print(f"\n{'='*70}")
    print("【結果】In/Out両方プラスの戦略（★）上位20件")
    print(f"{'='*70}")
    print(f"  {'drop帯':>10} {'RB':>3} {'gap上限':>6} {'地合い':>14} {'出口':>18} "
          f"{'件数':>5} {'勝率':>6} {'Out avg':>8} {'Out Sharpe':>10} {'安定':>4}")
    print("  " + "─" * 90)

    shown = 0
    for _, r in wf_df.iterrows():
        if shown >= 20:
            break
        band    = f"{r['drop_lo']}〜{r['drop_hi']}%"
        gap_str = f"{r['gap_max']:+.1f}%"
        mark    = r["stable"]
        print(f"  {band:>10} {r['rb_min']:>3} {gap_str:>6} {r['condition']:>14} "
              f"{r['exit']:>18} {r['out_n']:>5} {r['out_wr']:>5.1f}% "
              f"{r['out_avg']:>+7.3f}% {r['out_sharpe']:>+9.3f}  {mark}")
        shown += 1

    stable_df = wf_df[wf_df["stable"] == "★"] if not wf_df.empty else pd.DataFrame()
    print(f"\n  ★ In/Out両方プラス: {len(stable_df)}件 / 上位{top_n}件中")

    # ── 保存 ──
    best_path = os.path.join(OUT_DIR, "evolve_b_best.csv")
    hist_path = os.path.join(OUT_DIR, "evolve_b_history.csv")
    wf_df.to_csv(best_path,    index=False, encoding="utf-8-sig")
    history_df.to_csv(hist_path, index=False, encoding="utf-8-sig")

    # ── レポート ──
    report_lines = [
        f"戦略B 遺伝的アルゴリズム最適化レポート",
        f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"人口: {args.pop:,}人  世代数: {args.gen}  所要時間: {elapsed}秒",
        f"データ期間: {df['scan_date'].min()} 〜 {df['scan_date'].max()}",
        f"ウォークフォワード分割日: {split_date}",
        "",
        "【In/Out両方プラス（推奨戦略）上位5件】",
    ]
    for _, r in stable_df.head(5).iterrows():
        report_lines.append(
            f"  drop{r['drop_lo']}〜{r['drop_hi']}% / RB>={r['rb_min']} / "
            f"gap<={r['gap_max']:+.1f}% / {r['condition']} / {r['exit']} | "
            f"N={r['out_n']} WR={r['out_wr']}% avg={r['out_avg']:+.3f}% Sharpe={r['out_sharpe']:+.3f}"
        )
    if stable_df.empty:
        best_row = wf_df.iloc[0] if not wf_df.empty else None
        if best_row is not None:
            report_lines.append(
                f"  ※ 過学習なし条件なし。Out-of-sample最良: "
                f"drop{best_row['drop_lo']}〜{best_row['drop_hi']}% / "
                f"{best_row['condition']} / {best_row['exit']} | "
                f"avg={best_row['out_avg']:+.3f}%"
            )

    report_lines += [
        "",
        f"【世代別Sharpe推移（10世代ごと）】",
    ]
    for _, h in history_df[history_df["gen"] % 10 == 0].iterrows():
        report_lines.append(
            f"  {int(h['gen'])+1:>4}世代: best={h['best']:+.3f}  top5avg={h['top5']:+.3f}"
        )

    report_lines.append("\n次回実行推奨: 来月初（データ蓄積後）")
    report_txt = "\n".join(report_lines)

    rep_path = os.path.join(OUT_DIR, "evolve_b_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    print(f"\n保存完了:")
    print(f"  {best_path}  （上位戦略一覧）")
    print(f"  {hist_path} （世代別推移）")
    print(f"  {rep_path}   （レポート）")
    print(f"\n{report_txt}")


if __name__ == "__main__":
    main()
