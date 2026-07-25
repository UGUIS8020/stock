"""
evolve.py - 遺伝的アルゴリズムによる統合戦略最適化

50,000人の「仮想投資家」が各自の戦略パラメータを自由に持って売買する。
戦略A（順張り）・戦略B（逆張り）の区別なく、2年分のdailyデータから
最も利益が出た戦略パラメータが自然に浮かび上がる。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【前提条件（必ず確認）】
  1. 実行ディレクトリ: stock/ フォルダの直下で実行すること
       cd C:\...\stock
       python optimize\evolve.py
  2. 必要ファイル: out/sim_precise_trades.csv（568,000行以上）
       なければ先に: python simulate\simulate_precise.py を実行
  3. 必要ライブラリ: pandas, numpy
       pip install pandas numpy

【実行方法】
    python optimize\evolve.py                      # 50,000人・100世代（重い・数時間）
    python optimize\evolve.py --pop 20000 --gen 100  # 推奨: 1〜2時間
    python optimize\evolve.py --pop 5000  --gen 100  # 軽量: 数十分
    python optimize\evolve.py --pop 5000  --gen 3    # テスト用

【処理時間の目安（1プロセス・CPUシングルコア利用）】
    5,000人 ×   3世代 →  数分
    5,000人 × 100世代 →  数十分〜1時間
   20,000人 × 100世代 →  1〜3時間
   50,000人 × 100世代 →  5〜10時間
   ※複数プロセス同時実行すると共倒れするので1プロセスだけ実行すること

【染色体（1人の戦略）15遺伝子】
    gene[0]  score_min         : A系スコア下限（0〜10）
    gene[1]  ratio_min         : 出来高比率下限（0〜10倍）
    gene[2]  ratio_max         : 出来高比率上限（3〜50倍）
    gene[3]  rise_lo           : 前日騰落の下限（-15〜+5%）← B系は-10前後
    gene[4]  rise_hi           : 前日騰落の上限（-5〜+10%）← A系は+2前後
    gene[5]  rb_min            : B系リバウンドスコア下限（0〜8）
    gene[6]  gap_max           : 翌朝ギャップ上限（-5〜+10%）
    gene[7]  tp                : 利確幅（1〜10%）
    gene[8]  sl                : 損切り幅（-7〜-0.5%）
    gene[9]  cond              : 対象地合い（0=全、1=NORMAL、2=STRONG、3=N+S）
    gene[10] top_n             : 1日の最大買い件数（1/3/5/全件）
    gene[11] lower_shadow_min  : 下ヒゲ比率下限（0〜0.7）
    gene[12] vol_decay_req     : 売り枯れ必須（0=不問、1=必須）
    gene[13] consec_drop_min   : 連続下落日数下限（0〜5）
    gene[14] score_max         : スコア上限（3〜10）★2026-07-24追加
                                  backtest_log.csv（ナイーブ始値終値）ではNORMAL日score>=6.5が
                                  逆効果に見えたが、TP/SLシミュレーション込みの実トレードでは
                                  score>=6.5の39件が平均+0.715%と逆の結果になったため、
                                  正しくTP/SLを考慮したこちらのデータで改めて検証する。

【出力ファイル（stock/out/ 以下に生成）】
    out/evolve_best.csv     - 上位戦略一覧（アウトオブサンプル検証済み）
    out/evolve_history.csv  - 世代別Sharpe推移
    out/evolve_report.txt   - サマリーレポート（人間が読む用）

【ウォークフォワード検証（過学習防止）】
    前60%のデータで学習 → 後40%で検証（out_sharpe）
    out_sharpe > 1.0 かつ out_avg > 0 の戦略に ★ が付く
    ★ の戦略のみ実運用を検討すること
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ★ パスは stock/ フォルダからの相対パス（stock/直下で実行すること）
TRADES_CSV = "out/sim_precise_trades.csv"
WF_SPLIT   = 0.60   # 前60%=学習用、後40%=アウトオブサンプル検証用
MIN_SAMPLE = 10     # 検証期間中の最低トレード件数（これ未満の戦略は除外）

# ── GAパラメータ（通常は変更不要）──
ELITE_RATE   = 0.02   # エリート保存率（上位2%をそのまま次世代へ）
TOURNAMENT_K = 5      # トーナメント選択の対戦人数
CROSS_RATE   = 0.80   # 交叉率（80%の確率で親の遺伝子を組み合わせる）
MUTATE_RATE  = 0.15   # 突然変異率（各遺伝子が15%の確率で変化）
MUTATE_SIGMA = 0.15   # 突然変異の大きさ（遺伝子範囲の15%分ランダムに動く）

#                   [score, r_min, r_max, rise_lo, rise_hi, rb,  gap,  tp,   sl,   cond, topn, ls,  vd,  cd,  score_max]
GENE_MIN = np.array([ 0.0,  0.0,  3.0,  -15.0,   -5.0,   0.0, -5.0,  1.0, -7.0,  0.0,  0.0, 0.0, 0.0, 0.0,  3.0])
GENE_MAX = np.array([10.0, 10.0, 50.0,    5.0,   10.0,   8.0, 10.0, 10.0, -0.5,  3.0,  3.0, 0.7, 1.0, 5.0, 10.0])
N_GENES  = len(GENE_MIN)

_CAL_SL  = np.array([-0.5, -1.0, -2.0, -3.0, -5.0, -7.0])
_CAL_TPC = np.array([0.050, 0.101, 0.155, 0.183, 0.220, 0.250])
_CAL_SLC = np.array([0.930, 0.810, 0.684, 0.586, 0.480, 0.400])

COND_LABELS = ["全地合い", "NORMAL", "STRONG", "NORMAL+STRONG"]
TOPN_VALUES = [1, 3, 5, 999]


# ══════════════════════════════════════════════
# データ準備
# ══════════════════════════════════════════════
def load_trades():
    if not os.path.exists(TRADES_CSV):
        print(f"❌ {TRADES_CSV} が見つかりません。先に simulate/simulate_precise.py を実行してください。")
        sys.exit(1)
    df = pd.read_csv(TRADES_CSV, encoding="utf-8-sig", low_memory=False)
    before = len(df)
    df = df.dropna(subset=["ret_max", "ret_low", "ret_oc", "today_rise", "gap_pct"])
    df = df[(df["today_rise"].abs() < 30) & (df["gap_pct"].abs() < 20)]
    df = df.sort_values(["trade_date", "score"], ascending=[True, False]).reset_index(drop=True)
    df["score_rank"] = df.groupby("trade_date")["score"].rank(ascending=False, method="first")
    print(f"  トレードデータ: {before:,}件 → {len(df):,}件（欠損・外れ値除外後）")
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
    score_max  = float(np.clip(ind[14], GENE_MIN[14], GENE_MAX[14]))
    if score_max <= score_min:
        score_max = score_min + 0.5
    return {
        "score_min": score_min, "score_max": score_max,
        "ratio_min": ratio_min, "ratio_max": ratio_max,
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
        (npy["score"]      <  p["score_max"])  &
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
    sharpe = mean / std * np.sqrt(252)
    sharpe = min(sharpe, 5.0)   # evolve_b.pyと同じキャップ（極小サンプルの暴走防止）
    penalty = min(1.0, len(idx) / 100.0)  # サンプル数ペナルティ（evolve_b.pyと同一方式）
    return float(sharpe * penalty)


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
    for j in range(N_GENES):
        if np.random.rand() < MUTATE_RATE:
            span = GENE_MAX[j] - GENE_MIN[j]
            ind[j] += np.random.randn() * MUTATE_SIGMA * span
            ind[j]  = np.clip(ind[j], GENE_MIN[j], GENE_MAX[j])
    return ind


def next_gen(pop, fitness):
    n       = len(pop)
    n_elite = max(1, int(n * ELITE_RATE))
    elites  = [pop[i].copy() for i in np.argsort(fitness)[-n_elite:]]
    new_pop = elites[:]
    while len(new_pop) < n:
        c1, c2 = crossover(tournament_select(pop, fitness),
                            tournament_select(pop, fitness))
        new_pop.append(mutate(c1))
        if len(new_pop) < n:
            new_pop.append(mutate(c2))
    return np.array(new_pop[:n])


# ══════════════════════════════════════════════
# アウトオブサンプル検証
# ══════════════════════════════════════════════
def out_of_sample_check(pop, fitness, npy_out, top_k=200):
    results = []
    for i in np.argsort(fitness)[-top_k:]:
        p          = decode(pop[i])
        in_sharpe  = fitness[i]
        out_sharpe = evaluate_one(pop[i], npy_out)

        mask = (
            (npy_out["score"]      >= p["score_min"])  &
            (npy_out["score"]      <  p["score_max"])  &
            (npy_out["ratio"]      >= p["ratio_min"])  &
            (npy_out["ratio"]      <  p["ratio_max"])  &
            (npy_out["rise"]       >= p["rise_lo"])    &
            (npy_out["rise"]       <= p["rise_hi"])    &
            (npy_out["rb"]         >= p["rb_min"])     &
            (npy_out["gap"]        <= p["gap_max"])    &
            (npy_out["score_rank"] <= p["top_n"])      &
            (npy_out["lower_shadow"] >= p["ls_min"])   &
            (npy_out["consec_drop"]  >= p["cd_min"])
        )
        if p["vd_req"] == 1:
            mask &= npy_out["vol_decay"] == 1
        if p["cond"] == 1:
            mask &= npy_out["cond"] == 1
        elif p["cond"] == 2:
            mask &= npy_out["cond"] == 2
        elif p["cond"] == 3:
            mask &= npy_out["cond"] <= 2

        idx = np.where(mask)[0]
        if len(idx) < MIN_SAMPLE:
            continue

        rets   = calc_returns(npy_out, idx, p["tp"], p["sl"])
        out_wr  = float((rets > 0).mean() * 100)
        out_avg = float(rets.mean())
        out_cum = float(rets.sum())
        stable  = "★" if out_sharpe > 1.0 and out_avg > 0 else ""

        # 戦略タイプ判定
        if p["rise_hi"] < 0:
            stype = "B(逆張り)"
        elif p["rise_lo"] > -2:
            stype = "A(順張り)"
        else:
            stype = "AB(複合)"

        results.append({
            "type":       stype,
            "score_min":  round(p["score_min"], 1),
            "score_max":  round(p["score_max"], 1),
            "ratio_min":  round(p["ratio_min"], 1),
            "ratio_max":  round(p["ratio_max"], 1),
            "rise_lo":    round(p["rise_lo"],   1),
            "rise_hi":    round(p["rise_hi"],   1),
            "rb_min":     round(p["rb_min"],    1),
            "gap_max":    round(p["gap_max"],   1),
            "tp":  p["tp"],  "sl": p["sl"],
            "cond":   COND_LABELS[p["cond"]],
            "top_n":  p["top_n"] if p["top_n"] < 999 else "全件",
            "ls_min": round(p["ls_min"], 2),
            "vd_req": p["vd_req"],
            "cd_min": p["cd_min"],
            "in_sharpe":  round(in_sharpe,  3),
            "out_sharpe": round(out_sharpe, 3),
            "out_n":   len(idx),
            "out_wr":  round(out_wr,  1),
            "out_avg": round(out_avg, 3),
            "out_cum": round(out_cum, 1),
            "stable":  stable,
        })

    results.sort(key=lambda r: r["out_sharpe"], reverse=True)
    return results


# ══════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    # --pop: 仮想投資家の人数（多いほど精度↑・時間↑）  推奨: 20000〜50000
    parser.add_argument("--pop",  type=int, default=50_000)
    # --gen: 世代数（多いほど進化↑・時間↑）  推奨: 100
    parser.add_argument("--gen",  type=int, default=100)
    # --seed: 乱数シード（同じ値なら再現可能）
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"\n{'='*65}")
    print(f"  evolve.py - 統合戦略 遺伝的アルゴリズム最適化")
    print(f"  人口: {args.pop:,}人  世代: {args.gen}世代")
    print(f"  戦略A(順張り)・B(逆張り)の区別なく最適解を探索")
    print(f"{'='*65}\n")

    print("データ読み込み中...")
    df    = load_trades()
    dates = df["trade_date"].values
    split_date = sorted(dates)[int(len(dates) * WF_SPLIT)]
    print(f"  分割日: {split_date}  (前{WF_SPLIT*100:.0f}%=学習 / 後{(1-WF_SPLIT)*100:.0f}%=検証)\n")

    npy_all = df_to_numpy(df)
    npy_in, npy_out = split_numpy(npy_all, dates, split_date)
    print(f"  学習: {npy_in['score'].shape[0]:,}件  検証: {npy_out['score'].shape[0]:,}件\n")

    # 第1世代: ランダム初期化
    pop     = init_pop(args.pop)
    fitness = evaluate_pop(pop, npy_in)  # ← ここが最も時間がかかる（人数×568,000件の評価）

    history = []
    print(f"  {'世代':>4}  {'最高Sharpe':>10}  {'平均Sharpe':>10}  {'時刻'}")
    print("  " + "─" * 45)

    for gen in range(1, args.gen + 1):
        pop     = next_gen(pop, fitness)   # 選択→交叉→突然変異
        fitness = evaluate_pop(pop, npy_in)
        best_f  = float(fitness.max())
        avg_f   = float(fitness[fitness > -90].mean()) if (fitness > -90).any() else 0.0
        history.append({"gen": gen, "best_sharpe": best_f, "avg_sharpe": avg_f})
        # 10世代ごとに進捗表示（Sharpeが上昇しているか確認できる）
        if gen % 10 == 0 or gen == 1:
            t = datetime.now().strftime("%H:%M:%S")
            print(f"  {gen:>4}  {best_f:>10.3f}  {avg_f:>10.3f}  {t}")

    print(f"\nアウトオブサンプル検証中（上位200戦略）...")
    best_results = out_of_sample_check(pop, fitness, npy_out, top_k=200)

    os.makedirs("out", exist_ok=True)
    pd.DataFrame(best_results).to_csv("out/evolve_best.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv("out/evolve_history.csv",   index=False, encoding="utf-8-sig")

    lines = []
    lines.append("=" * 70)
    lines.append("  統合戦略 遺伝的アルゴリズム最適化レポート")
    lines.append(f"  人口: {args.pop:,}人  世代: {args.gen}  生成日: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  データ: {df['trade_date'].min()} 〜 {df['trade_date'].max()}  ({len(df):,}件)")
    lines.append("=" * 70)
    lines.append(f"\n【アウトオブサンプル上位20戦略】（★=安定）")
    lines.append(f"  {'type':10} {'score':>10} {'ratio':>9} {'rise':>12} {'rb':>4} "
                 f"{'TP':>5} {'SL':>5} {'地合い':>12} | "
                 f"{'in_sh':>6} {'out_sh':>6} {'N':>5} {'WR':>6} {'avg%':>7} {'★'}")
    lines.append("  " + "─" * 110)

    for r in best_results[:20]:
        lines.append(
            f"  {r['type']:10} {r['score_min']:>4.1f}〜{r['score_max']:>4.1f} "
            f"{r['ratio_min']:>4.1f}〜{r['ratio_max']:>4.1f} "
            f"{r['rise_lo']:>+5.1f}〜{r['rise_hi']:>+5.1f}% "
            f"{r['rb_min']:>4.0f} "
            f"{r['tp']:>+5.1f}% {r['sl']:>+5.1f}% "
            f"{r['cond']:>12} | "
            f"{r['in_sharpe']:>6.2f} {r['out_sharpe']:>6.2f} "
            f"{r['out_n']:>5} {r['out_wr']:>5.1f}% {r['out_avg']:>+7.3f}%  {r['stable']}"
        )

    stable = [r for r in best_results if r["stable"] == "★"]
    a_type = [r for r in stable if r["type"] == "A(順張り)"]
    b_type = [r for r in stable if r["type"] == "B(逆張り)"]
    ab_type = [r for r in stable if r["type"] == "AB(複合)"]

    lines.append(f"\n  安定戦略(★): {len(stable)}件 / 200件中")
    lines.append(f"    → A(順張り): {len(a_type)}件  B(逆張り): {len(b_type)}件  AB(複合): {len(ab_type)}件")

    if stable:
        lines.append(f"\n【最優秀戦略】")
        r = stable[0]
        lines.append(f"  タイプ  : {r['type']}")
        lines.append(f"  スコア  : {r['score_min']}〜{r['score_max']}")
        lines.append(f"  出来高比: {r['ratio_min']}〜{r['ratio_max']}倍")
        lines.append(f"  前日騰落: {r['rise_lo']:+}%〜{r['rise_hi']:+}%")
        lines.append(f"  RBスコア: {r['rb_min']}以上")
        lines.append(f"  TP/SL   : {r['tp']:+}% / {r['sl']:+}%")
        lines.append(f"  地合い  : {r['cond']}")
        lines.append(f"  1日上限 : {r['top_n']}件")
        lines.append(f"  out勝率 : {r['out_wr']}%  out平均: {r['out_avg']:+}%  Sharpe: {r['out_sharpe']}")

    lines.append(f"\n{'='*70}")
    report = "\n".join(lines)
    print("\n" + report)

    with open("out/evolve_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n保存完了:")
    print(f"  out/evolve_best.csv")
    print(f"  out/evolve_history.csv")
    print(f"  out/evolve_report.txt")


if __name__ == "__main__":
    main()
