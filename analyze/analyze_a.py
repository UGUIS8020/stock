"""
analyze_a.py - 戦略A（順張り）統計的エッジ分析

遺伝的アルゴリズムではなく統計的手法で
「どの条件で順張りシグナルが有意にプラスになるか」を探す。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【実行方法】
    cd stock/
    python analyze/analyze_a.py

【前提ファイル】
    out/sim_precise_trades.csv  ← simulate_precise.py で生成済みであること

【分析内容】
    1. 特徴量相関分析        どの特徴量がリターンと相関するか
    2. 単変量分析            各特徴量の値域別に勝率・p値を計算
    3. 地合い別分析          NORMAL/STRONG で傾向が異なるか
    4. 多変量条件探索        複数条件の組み合わせで有意な条件を発見
    5. 出来高比率帯分析
    6. TP/SL最適化           有効条件に絞ってTP/SL組み合わせの期待値を計算
                             → daytime.py の設定値に直接反映できる
    7. アウトオブサンプル検証  時系列60/40分割で条件の汎化性能を確認
                             → 過学習チェック（訓練期間で見つけた条件が未知期間でも有効か）
    8. 月別・曜日別分析        月・曜日ごとのWR/Sharpeを計算
                             → 苦手な月や曜日を避けることで安定性が向上する
    9. 銘柄属性分析            価格帯・業種コードで銘柄をグループ化して WR/Sharpe を比較
                             → 得意な価格帯・業種に絞ることでさらに安定性が向上する

【統計手法】
    二項検定（正規近似）:
        帰無仮説 H0: p = 0.50（ランダムと同じ）
        対立仮説 H1: p > 0.50（プラスのエッジあり）
        p値 < 0.05 かつ 勝率 > 55% かつ N >= 30 を「有意」と判定

    TP/SL期待値計算:
        ret_max / ret_low から TP・SL どちらが先に到達したか判定
        両方到達時は分足校正データ（確率）で按分
        期待値 = E[return]、Sharpe = 平均/標準偏差 × √252

【出力ファイル】
    out/analyze_a_report.txt   - 人間が読むサマリーレポート
    out/analyze_a_detail.csv   - 全条件の詳細データ
    out/analyze_a_tpsl.csv     - TP/SL最適化の全組み合わせ結果
    out/analyze_a_oos.csv      - アウトオブサンプル検証の詳細結果
    out/analyze_a_seasonal.csv - 月別・曜日別分析の詳細結果
    out/analyze_a_attr.csv     - 銘柄属性（価格帯・業種）分析の詳細結果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from itertools import product

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TRADES_CSV  = "out/sim_precise_trades.csv"
REPORT_TXT  = "out/analyze_a_report.txt"
DETAIL_CSV  = "out/analyze_a_detail.csv"
TPSL_CSV    = "out/analyze_a_tpsl.csv"
OOS_CSV        = "out/analyze_a_oos.csv"
SEASONAL_CSV   = "out/analyze_a_seasonal.csv"
ATTR_CSV       = "out/analyze_a_attr.csv"

OOS_SPLIT = 0.6   # 訓練データ割合（60%訓練 / 40%検証）

# 有意水準
MIN_N       = 30     # 最低サンプル数
MIN_WR      = 52.0   # 最低勝率（%）
P_THRESHOLD = 0.05   # p値の閾値
WIN_RETURN  = 0.0    # 「勝ち」の定義：終値リターンがこれ以上


# ══════════════════════════════════════════════
# ユーティリティ
# ══════════════════════════════════════════════
def binomial_pvalue(wins, n, p0=0.5):
    """二項検定のp値（正規近似）: H1: p > p0"""
    if n < 5:
        return 1.0
    z = (wins - n * p0) / np.sqrt(n * p0 * (1 - p0))
    # 標準正規分布の上側確率（片側検定）
    return float(1.0 - _norm_cdf(z))


def _norm_cdf(z):
    """標準正規分布のCDF（numpyのみで実装）"""
    return 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))


def _erf(x):
    """誤差関数の近似（Abramowitz & Stegun）"""
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (0.254829592 * t
                - 0.284496736 * t**2
                + 1.421413741 * t**3
                - 1.453152027 * t**4
                + 1.061405429 * t**5) * np.exp(-x * x)
    return sign * y


def stats_row(series_win, label=""):
    """勝率・p値・判定を返す"""
    n    = len(series_win)
    wins = series_win.sum()
    wr   = wins / n * 100 if n > 0 else 0.0
    pval = binomial_pvalue(wins, n)
    sig  = "★" if (pval < P_THRESHOLD and wr >= MIN_WR and n >= MIN_N) else ""
    return {"label": label, "n": n, "wins": int(wins), "wr": round(wr, 1),
            "pval": round(pval, 4), "sig": sig}


# ══════════════════════════════════════════════
# データ読み込み
# ══════════════════════════════════════════════

def _add_upper_shadow(df):
    """daily_prices から前日（scan_date）の上ひげ比率を計算して df に追加する。

    upper_shadow_ratio = (High - Close) / (High - Low)
        0 に近い → 前日が高値引け（買い圧力強・順張りに有利）
        1 に近い → 前日が安値引け（売り圧力強・天井打ちの可能性）

    scan_date = trade_date の前営業日（シグナルスキャン日）の OHLC を使用。
    """
    try:
        import db as _db
        conn = _db.get_conn()

        codes = df["code"].astype(str).unique().tolist()
        dates = df["scan_date"].astype(str).unique().tolist()

        c_ph = ",".join(["?"] * len(codes))
        d_ph = ",".join(["?"] * len(dates))
        price_df = pd.read_sql(
            f"SELECT code, Date, High, Low, Close FROM daily_prices "
            f"WHERE code IN ({c_ph}) AND Date IN ({d_ph})",
            conn, params=codes + dates
        )
        conn.close()

        price_df["code"] = price_df["code"].astype(str)
        price_df["_range"] = price_df["High"] - price_df["Low"]
        price_df["upper_shadow_ratio"] = np.where(
            price_df["_range"] > 0,
            (price_df["High"] - price_df["Close"]) / price_df["_range"],
            np.nan
        )

        # (code, scan_date) でマージ
        df = df.copy()
        df["_code_str"] = df["code"].astype(str)
        df["_scan_str"]  = df["scan_date"].astype(str)
        merged = df.merge(
            price_df[["code", "Date", "upper_shadow_ratio"]],
            left_on=["_code_str", "_scan_str"],
            right_on=["code", "Date"],
            how="left",
            suffixes=("", "_dp")
        )
        df["upper_shadow_ratio"] = merged["upper_shadow_ratio"].values
        df.drop(columns=["_code_str", "_scan_str"], inplace=True)

        n_ok = df["upper_shadow_ratio"].notna().sum()
        print(f"  上ひげ比率（upper_shadow_ratio）: {n_ok:,}/{len(df):,}件 計算完了")
    except Exception as e:
        print(f"  ⚠️  上ひげ比率の取得失敗: {e}")
        df["upper_shadow_ratio"] = np.nan

    return df


def _add_ma_deviation(df):
    """daily_prices から前日（scan_date）終値の移動平均乖離率を計算して df に追加する。

    ma5_dev  = (前日終値 - 5日MA)  / 5日MA  × 100  ← 短期トレンド
    ma25_dev = (前日終値 - 25日MA) / 25日MA × 100  ← 中期トレンド

    プラス → MAより上（上昇トレンド）
    マイナス → MAより下（下降トレンド）
    大きすぎると過熱圏 → 順張りに不利になる可能性
    """
    try:
        import db as _db
        conn = _db.get_conn()

        codes = df["code"].astype(str).unique().tolist()
        c_ph  = ",".join(["?"] * len(codes))

        # 全コードの全期間を一括取得（ローリングMAに過去データが必要）
        price_df = pd.read_sql(
            f"SELECT code, Date, Close FROM daily_prices "
            f"WHERE code IN ({c_ph}) ORDER BY code, Date",
            conn, params=codes
        )
        conn.close()

        price_df["code"]  = price_df["code"].astype(str)
        price_df["Close"] = pd.to_numeric(price_df["Close"], errors="coerce")

        # コードごとにローリングMA計算
        grp = price_df.groupby("code")["Close"]
        price_df["ma5"]  = grp.transform(lambda x: x.rolling(5,  min_periods=5).mean())
        price_df["ma25"] = grp.transform(lambda x: x.rolling(25, min_periods=25).mean())
        price_df["ma5_dev"]  = np.where(
            price_df["ma5"]  > 0,
            (price_df["Close"] - price_df["ma5"])  / price_df["ma5"]  * 100, np.nan)
        price_df["ma25_dev"] = np.where(
            price_df["ma25"] > 0,
            (price_df["Close"] - price_df["ma25"]) / price_df["ma25"] * 100, np.nan)

        # (code, scan_date) でマージ
        df = df.copy()
        df["_code_str"] = df["code"].astype(str)
        df["_scan_str"]  = df["scan_date"].astype(str)
        merged = df.merge(
            price_df[["code", "Date", "ma5_dev", "ma25_dev"]],
            left_on=["_code_str", "_scan_str"],
            right_on=["code", "Date"],
            how="left",
            suffixes=("", "_dp")
        )
        df["ma5_dev"]  = merged["ma5_dev"].values
        df["ma25_dev"] = merged["ma25_dev"].values
        df.drop(columns=["_code_str", "_scan_str"], inplace=True)

        n5  = df["ma5_dev"].notna().sum()
        n25 = df["ma25_dev"].notna().sum()
        print(f"  MA乖離率: 5日={n5:,}件 / 25日={n25:,}件 計算完了")
    except Exception as e:
        print(f"  ⚠️  MA乖離率の取得失敗: {e}")
        df["ma5_dev"]  = np.nan
        df["ma25_dev"] = np.nan

    return df


def load_data():
    if not os.path.exists(TRADES_CSV):
        print(f"❌ {TRADES_CSV} が見つかりません")
        sys.exit(1)

    df = pd.read_csv(TRADES_CSV, encoding="utf-8-sig", low_memory=False)
    df = df.dropna(subset=["ret_oc", "score", "ratio", "today_rise", "gap_pct"])
    df = df[(df["today_rise"].abs() < 30) & (df["gap_pct"].abs() < 20)]

    # 「勝ち」フラグ
    df["win"] = (df["ret_oc"] > WIN_RETURN).astype(int)

    # 地合いコード
    cond_map = {"NORMAL": 1, "STRONG": 2, "WEAK": 0}
    df["cond_code"] = df["trade_cond"].map(cond_map).fillna(-1)

    # 上ひげ比率（daily_prices DB から取得）
    print("  上ひげ比率を計算中（DBアクセス）...")
    df = _add_upper_shadow(df)

    # MA乖離率（5日・25日）
    print("  MA乖離率を計算中（全コード一括）...")
    df = _add_ma_deviation(df)

    print(f"  データ読み込み: {len(df):,}件")
    print(f"  期間: {df['trade_date'].min()} 〜 {df['trade_date'].max()}")
    print(f"  全体勝率: {df['win'].mean()*100:.1f}%  (基準線=50%)")
    return df


# ══════════════════════════════════════════════
# 1. 特徴量相関分析
# ══════════════════════════════════════════════
def analyze_correlation(df):
    features = ["score", "ratio", "today_rise", "rb_score",
                "gap_pct", "lower_shadow_ratio", "upper_shadow_ratio",
                "ma5_dev", "ma25_dev", "vol_decay_3d", "consec_drop"]
    features = [f for f in features if f in df.columns]

    corr = df[features + ["ret_oc", "win"]].corr()
    oc_corr  = corr["ret_oc"].drop(["ret_oc", "win"]).sort_values(key=abs, ascending=False)
    win_corr = corr["win"].drop(["ret_oc", "win"]).sort_values(key=abs, ascending=False)

    lines = []
    lines.append("\n【1. 特徴量とリターンの相関係数】")
    lines.append(f"  {'特徴量':<20} {'ret_oc相関':>10}  {'勝率相関':>10}  解釈")
    lines.append("  " + "─" * 65)
    for feat in oc_corr.index:
        oc  = oc_corr[feat]
        win = win_corr[feat]
        direction = "↑高いほど有利" if oc > 0.02 else ("↓低いほど有利" if oc < -0.02 else "ほぼ無関係")
        lines.append(f"  {feat:<20} {oc:>+10.4f}  {win:>+10.4f}  {direction}")

    return lines, list(oc_corr.index[:4])  # 上位4特徴量を返す


# ══════════════════════════════════════════════
# 2. 単変量分析（特徴量ごとの分位別勝率）
# ══════════════════════════════════════════════
def analyze_univariate(df, features):
    lines = []
    lines.append("\n【2. 特徴量別 勝率分析（★=有意にプラス, p<0.05 & WR≥52% & N≥30）】")

    for feat in features:
        if feat not in df.columns:
            continue
        lines.append(f"\n  ── {feat} ──")
        lines.append(f"  {'範囲':<22} {'件数':>6}  {'勝率':>6}  {'p値':>7}  判定")
        lines.append("  " + "─" * 55)

        try:
            # 5分位に分割
            df["_bin"] = pd.qcut(df[feat], q=5, duplicates="drop")
            for bin_val, grp in df.groupby("_bin", observed=True):
                r = stats_row(grp["win"], str(bin_val))
                lines.append(f"  {r['label']:<22} {r['n']:>6,}  {r['wr']:>5.1f}%  "
                              f"{r['pval']:>7.4f}  {r['sig']}")
        except Exception:
            lines.append(f"  （分位分割できませんでした）")
        finally:
            if "_bin" in df.columns:
                df.drop(columns=["_bin"], inplace=True)

    return lines


# ══════════════════════════════════════════════
# 3. 地合い別分析
# ══════════════════════════════════════════════
def analyze_by_condition(df):
    lines = []
    lines.append("\n【3. 地合い別 勝率分析】")
    lines.append(f"  {'地合い':<14} {'件数':>7}  {'勝率':>6}  {'p値':>7}  判定")
    lines.append("  " + "─" * 50)

    for cond in ["STRONG", "NORMAL", "WEAK"]:
        grp = df[df["trade_cond"] == cond]
        if len(grp) == 0:
            continue
        r = stats_row(grp["win"], cond)
        lines.append(f"  {r['label']:<14} {r['n']:>7,}  {r['wr']:>5.1f}%  "
                     f"{r['pval']:>7.4f}  {r['sig']}")

    # スコア×地合いの組み合わせ
    lines.append(f"\n  ── スコア帯 × 地合い ──")
    lines.append(f"  {'条件':<28} {'件数':>6}  {'勝率':>6}  {'p値':>7}  判定")
    lines.append("  " + "─" * 60)

    score_bins = [(0, 3), (3, 5), (5, 7), (7, 10), (10, 999)]
    for (s_lo, s_hi), cond in product(score_bins, ["STRONG", "NORMAL"]):
        mask = (df["score"] >= s_lo) & (df["score"] < s_hi) & (df["trade_cond"] == cond)
        grp  = df[mask]
        if len(grp) < MIN_N:
            continue
        label = f"score {s_lo}〜{s_hi} × {cond}"
        r = stats_row(grp["win"], label)
        lines.append(f"  {r['label']:<28} {r['n']:>6,}  {r['wr']:>5.1f}%  "
                     f"{r['pval']:>7.4f}  {r['sig']}")

    return lines


# ══════════════════════════════════════════════
# 4. 多変量条件探索
# ══════════════════════════════════════════════
def analyze_multivariate(df):
    lines  = []
    detail = []

    lines.append("\n【4. 多変量条件探索（有意条件のみ表示）】")
    lines.append(f"  基準: p<{P_THRESHOLD} & WR≥{MIN_WR}% & N≥{MIN_N}")
    lines.append(f"  {'条件':<50} {'件数':>6}  {'勝率':>6}  {'p値':>7}")
    lines.append("  " + "─" * 80)

    # 探索する条件の範囲を定義
    score_ranges   = [(0, 3), (3, 5), (5, 7), (7, 9999)]
    rise_ranges    = [(-99, -3), (-3, 0), (0, 3), (3, 99)]
    gap_ranges     = [(-99, -2), (-2, 0), (0, 2), (2, 99)]
    cond_list      = ["STRONG", "NORMAL", "ALL"]

    found = []

    for (s_lo, s_hi), (r_lo, r_hi), (g_lo, g_hi), cond in product(
            score_ranges, rise_ranges, gap_ranges, cond_list):

        mask = (
            (df["score"]      >= s_lo) & (df["score"]      <  s_hi) &
            (df["today_rise"] >= r_lo) & (df["today_rise"] <  r_hi) &
            (df["gap_pct"]    >= g_lo) & (df["gap_pct"]    <  g_hi)
        )
        if cond != "ALL":
            mask &= (df["trade_cond"] == cond)

        grp = df[mask]
        if len(grp) < MIN_N:
            continue

        r = stats_row(grp["win"])
        r["score"]      = f"{s_lo}〜{s_hi}"
        r["rise"]       = f"{r_lo:+}〜{r_hi:+}%"
        r["gap"]        = f"{g_lo:+}〜{g_hi:+}%"
        r["cond"]       = cond
        detail.append(r)

        if r["sig"] == "★":
            label = (f"score{s_lo}〜{s_hi} / 前日{r_lo:+}〜{r_hi:+}% "
                     f"/ gap{g_lo:+}〜{g_hi:+}% / {cond}")
            lines.append(f"  ★ {label:<48} {r['n']:>6,}  {r['wr']:>5.1f}%  {r['pval']:>7.4f}")
            found.append(r)

    if not found:
        lines.append("  （有意な多変量条件は見つかりませんでした）")

    return lines, detail, found


# ══════════════════════════════════════════════
# 5. 上位条件ランキング
# ══════════════════════════════════════════════
def analyze_ratio(df):
    """出来高比率帯別分析"""
    lines = []
    lines.append("\n【5. 出来高比率帯 × 地合い 分析】")
    lines.append(f"  {'条件':<32} {'件数':>6}  {'勝率':>6}  {'p値':>7}  判定")
    lines.append("  " + "─" * 65)

    ratio_bins = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 999)]
    for (r_lo, r_hi), cond in product(ratio_bins, ["STRONG", "NORMAL", "ALL"]):
        mask = (df["ratio"] >= r_lo) & (df["ratio"] < r_hi)
        if cond != "ALL":
            mask &= (df["trade_cond"] == cond)
        grp = df[mask]
        if len(grp) < MIN_N:
            continue
        label = f"ratio {r_lo}〜{r_hi}x / {cond}"
        r = stats_row(grp["win"], label)
        lines.append(f"  {r['label']:<32} {r['n']:>6,}  {r['wr']:>5.1f}%  "
                     f"{r['pval']:>7.4f}  {r['sig']}")

    return lines


# ══════════════════════════════════════════════
# 6. TP/SL最適化
# ══════════════════════════════════════════════

# 分足校正データ（evolve.py と共通）: SL水準ごとのTP/SL先着確率
_CAL_SL  = np.array([-0.5, -1.0, -2.0, -3.0, -5.0, -7.0])
_CAL_TPC = np.array([0.050, 0.101, 0.155, 0.183, 0.220, 0.250])  # TP先着確率
_CAL_SLC = np.array([0.930, 0.810, 0.684, 0.586, 0.480, 0.400])  # SL先着確率


def calc_trade_returns(npy_max, npy_low, npy_oc, tp, sl):
    """ret_max / ret_low / ret_oc から TP/SL を考慮した1トレードあたりリターンを計算。

    判定ルール:
        TP先着: ret_max >= tp かつ ret_low > sl  → +tp
        SL先着: ret_low <= sl かつ ret_max < tp  → sl
        両方到達: 分足校正確率で按分
        どちらも未到達: ret_oc（終値リターン）
    """
    # 両到達時の按分確率（SL水準によって変わる）
    tp_p = float(np.interp(-sl, _CAL_SL[::-1] * -1, _CAL_TPC[::-1]))
    sl_p = float(np.interp(-sl, _CAL_SL[::-1] * -1, _CAL_SLC[::-1]))
    denom = tp_p + sl_p if (tp_p + sl_p) > 0 else 1.0

    tp_hit  = (npy_max >= tp) & (npy_low > sl)
    sl_hit  = (npy_low <= sl) & (npy_max < tp)
    both    = (npy_max >= tp) & (npy_low <= sl)
    neither = ~tp_hit & ~sl_hit & ~both

    rets = np.where(tp_hit,  tp,
           np.where(sl_hit,  sl,
           np.where(both,    tp * tp_p / denom + sl * sl_p / denom,
                              npy_oc)))  # neither → 終値で決済
    return rets


def analyze_tp_sl(df):
    """有効条件（STRONG × gap小 × 前日プラス）に絞ってTP/SLを最適化する。

    daytime.py の TP_PCT / SL_PCT に直接反映できる推奨値を出力する。
    """
    lines = []
    lines.append("\n【6. TP/SL最適化】有効条件内でのTP/SL組み合わせ期待値")
    lines.append("  対象条件: STRONG地合い × gap -5〜+2% × 前日プラス（daytime.py 新条件）")

    # 有効条件でフィルタ（daytime.py の新パラメータに合わせる）
    mask = (
        (df["trade_cond"] == "STRONG") &
        (df["gap_pct"]    >= -5.0)     &
        (df["gap_pct"]    <=  2.0)     &
        (df["today_rise"] >=  0.0)
    )
    df_f = df[mask].copy()
    n_total = len(df_f)

    if n_total < MIN_N:
        lines.append(f"  ⚠️  サンプル不足（{n_total}件）")
        return lines, []

    lines.append(f"  対象件数: {n_total:,}件\n")

    npy_max = df_f["ret_max"].values.astype(np.float32)
    npy_low = df_f["ret_low"].values.astype(np.float32)
    npy_oc  = df_f["ret_oc"].values.astype(np.float32)

    # TP/SL の探索範囲
    tp_list = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]
    sl_list = [-0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -5.0]

    results = []
    for tp in tp_list:
        for sl in sl_list:
            rets = calc_trade_returns(npy_max, npy_low, npy_oc, tp, sl)
            n    = len(rets)
            wr   = float((rets > 0).mean() * 100)
            avg  = float(rets.mean())
            std  = float(rets.std())
            sharpe = avg / std * np.sqrt(252) if std > 1e-9 else 0.0
            tp_rate = float((npy_max >= tp).mean() * 100)   # TP到達率
            sl_rate = float((npy_low <= sl).mean() * 100)   # SL到達率
            results.append({
                "tp": tp, "sl": sl, "n": n,
                "wr": round(wr, 1), "avg": round(avg, 3),
                "sharpe": round(sharpe, 3),
                "tp_reach": round(tp_rate, 1),
                "sl_reach": round(sl_rate, 1),
            })

    results.sort(key=lambda x: -x["sharpe"])

    # テーブル表示
    lines.append(f"  {'TP':>5} {'SL':>5} {'勝率':>6} {'平均%':>7} {'Sharpe':>8}"
                 f" {'TP到達%':>7} {'SL到達%':>7}  判定")
    lines.append("  " + "─" * 70)
    for r in results:
        mark = "★★" if r["sharpe"] >= 3.0 else ("★" if r["sharpe"] >= 1.5 else "")
        lines.append(
            f"  {r['tp']:>+4.1f}% {r['sl']:>+4.1f}%"
            f"  {r['wr']:>5.1f}%  {r['avg']:>+6.3f}%  {r['sharpe']:>8.3f}"
            f"  {r['tp_reach']:>6.1f}%  {r['sl_reach']:>6.1f}%  {mark}"
        )

    # 推奨値
    best = results[0]
    lines.append(f"\n  【推奨 TP/SL】")
    lines.append(f"  TP: +{best['tp']}%  SL: {best['sl']}%")
    lines.append(f"  → 勝率{best['wr']}% / 平均{best['avg']:+.3f}% / Sharpe={best['sharpe']}")
    lines.append(f"\n  daytime.py への反映:")
    lines.append(f"    TP_PCT = {best['tp']/100:.3f}   # +{best['tp']}%")
    lines.append(f"    SL_PCT = {abs(best['sl'])/100:.3f}   # {best['sl']}%")

    return lines, results


# ══════════════════════════════════════════════
# 7. アウトオブサンプル検証
# ══════════════════════════════════════════════

def _apply_cond(d, cond_id):
    """条件IDに対応するブールマスクを返す"""
    if cond_id == "main":
        return (
            (d["trade_cond"] == "STRONG") &
            (d["gap_pct"]    >= -5.0) &
            (d["gap_pct"]    <=  2.0) &
            (d["today_rise"] >=  0.0)
        )
    elif cond_id == "tight":
        return (
            (d["trade_cond"] == "STRONG") &
            (d["gap_pct"]    >= -2.0) &
            (d["gap_pct"]    <=  0.0) &
            (d["today_rise"] >=  0.0)
        )
    else:  # "all"
        return pd.Series(True, index=d.index)


def _oos_stats(grp, tp=3.0, sl=-5.0):
    """グループのTP/SL考慮リターン統計を返す"""
    if len(grp) < MIN_N:
        return None
    npy_max = grp["ret_max"].values.astype(np.float32)
    npy_low = grp["ret_low"].values.astype(np.float32)
    npy_oc  = grp["ret_oc"].values.astype(np.float32)
    rets = calc_trade_returns(npy_max, npy_low, npy_oc, tp, sl)
    n      = len(rets)
    wr     = float((rets > 0).mean() * 100)
    avg    = float(rets.mean())
    std    = float(rets.std())
    sharpe = avg / std * np.sqrt(252) if std > 1e-9 else 0.0
    pval   = binomial_pvalue(int((rets > 0).sum()), n)
    sig    = "★" if (pval < P_THRESHOLD and wr >= MIN_WR and n >= MIN_N) else ""
    return {"n": n, "wr": round(wr, 1), "avg": round(avg, 3),
            "sharpe": round(sharpe, 3), "pval": round(pval, 4), "sig": sig}


def analyze_oos(df, tp=3.0, sl=-5.0):
    """時系列 60/40 分割でアウトオブサンプル検証を行う。

    発見した条件が訓練データへの過学習ではなく、未知の期間でも
    実際にエッジがあるかを確認する。
    """
    lines = []
    lines.append(f"\n【7. アウトオブサンプル検証（時系列{int(OOS_SPLIT*100)}/{int((1-OOS_SPLIT)*100)}分割）】")
    lines.append(f"  評価設定: TP={tp:+.1f}% / SL={sl:+.1f}%（daytime.py 現在値）")

    # ── 時系列分割 ──────────────────────────────
    dates     = sorted(df["trade_date"].unique())
    n_dates   = len(dates)
    split_idx = int(n_dates * OOS_SPLIT)
    cutoff    = dates[split_idx]

    df_train = df[df["trade_date"] <  cutoff].copy()
    df_oos   = df[df["trade_date"] >= cutoff].copy()

    lines.append(f"\n  訓練期間: {dates[0]} 〜 {dates[split_idx-1]}"
                 f"  ({len(df_train):,}件 / {split_idx}日)")
    lines.append(f"  検証期間: {cutoff} 〜 {dates[-1]}"
                 f"  ({len(df_oos):,}件 / {n_dates - split_idx}日)")

    # ── 条件ごとに訓練 vs 検証を比較 ─────────────
    cond_defs = [
        ("main",  "STRONG × gap-5〜+2% × 前日プラス（基本条件）"),
        ("tight", "STRONG × gap-2〜 0% × 前日プラス（高WR帯）"),
        ("all",   "全体（ベースライン）"),
    ]

    rows = []
    lines.append(f"\n  {'条件':<40} {'期間':<5} {'件数':>6}  {'勝率':>6}  {'平均%':>7}  {'Sharpe':>8}  判定")
    lines.append("  " + "─" * 85)

    for cond_id, cond_name in cond_defs:
        for period_label, dset in [("訓練", df_train), ("検証", df_oos)]:
            mask = _apply_cond(dset, cond_id)
            grp  = dset[mask]
            st   = _oos_stats(grp, tp=tp, sl=sl)
            if st is None:
                lines.append(f"  {cond_name:<40} {period_label:<5} {'(N不足)'!s}")
                continue
            rows.append({"condition": cond_name, "period": period_label, **st})
            lines.append(
                f"  {cond_name:<40} {period_label:<5} {st['n']:>6,}  {st['wr']:>5.1f}%"
                f"  {st['avg']:>+6.3f}%  {st['sharpe']:>8.3f}  {st['sig']}"
            )

    # ── 判定 ─────────────────────────────────────
    lines.append(f"\n  【判定】")

    def _find(period, cond_id):
        name = next(n for c, n in cond_defs if c == cond_id)
        return next((r for r in rows if r["period"] == period and r["condition"] == name), None)

    tr = _find("訓練", "main")
    oo = _find("検証", "main")

    if tr and oo:
        d_sharpe = tr["sharpe"] - oo["sharpe"]
        d_wr     = tr["wr"]     - oo["wr"]
        lines.append(f"  基本条件（daytime.py）:")
        lines.append(f"    訓練  Sharpe={tr['sharpe']:>6.3f}  WR={tr['wr']:.1f}%  avg={tr['avg']:>+.3f}%")
        lines.append(f"    検証  Sharpe={oo['sharpe']:>6.3f}  WR={oo['wr']:.1f}%  avg={oo['avg']:>+.3f}%")
        lines.append(f"    劣化  ΔSharpe={-d_sharpe:>+.3f}  ΔWR={-d_wr:>+.1f}%")

        if oo["sharpe"] >= 1.5 and oo["wr"] >= MIN_WR:
            verdict = "✅ 汎化OK"
            detail  = (f"検証期間でも Sharpe={oo['sharpe']:.2f} / WR={oo['wr']:.1f}%\n"
                       "     → 過学習ではなく実際のエッジが存在する可能性が高い")
        elif oo["avg"] > 0:
            verdict = "⚠️  弱い汎化"
            detail  = (f"検証期間の期待値はプラス（avg={oo['avg']:>+.3f}%）だが WR/Sharpe が低下\n"
                       "     → 追加フィルタや特徴量の強化で改善できる可能性あり")
        else:
            verdict = "❌ 過学習の疑い"
            detail  = (f"検証期間で期待値マイナス（avg={oo['avg']:>+.3f}%）\n"
                       "     → 条件を見直し、特徴量③〜⑥の追加を優先してください")

        lines.append(f"\n  {verdict}")
        lines.append(f"     {detail}")

    return lines, rows


# ══════════════════════════════════════════════
# 8. 月別・曜日別分析
# ══════════════════════════════════════════════

_DOW_NAMES = {0: "月", 1: "火", 2: "水", 3: "木", 4: "金"}
_MON_NAMES = {1:"1月",2:"2月",3:"3月",4:"4月",5:"5月",6:"6月",
              7:"7月",8:"8月",9:"9月",10:"10月",11:"11月",12:"12月"}


def _seasonal_stats(grp, tp, sl):
    """グループのTP/SL考慮リターン統計を返す（サンプル不足時はNone）"""
    if len(grp) < MIN_N:
        return None
    npy_max = grp["ret_max"].values.astype(np.float32)
    npy_low = grp["ret_low"].values.astype(np.float32)
    npy_oc  = grp["ret_oc"].values.astype(np.float32)
    rets    = calc_trade_returns(npy_max, npy_low, npy_oc, tp, sl)
    n      = len(rets)
    wr     = float((rets > 0).mean() * 100)
    avg    = float(rets.mean())
    std    = float(rets.std())
    sharpe = avg / std * np.sqrt(252) if std > 1e-9 else 0.0
    return {"n": n, "wr": round(wr, 1), "avg": round(avg, 3),
            "sharpe": round(sharpe, 3)}


def analyze_seasonal(df, tp=3.0, sl=-5.0):
    """月別・曜日別に勝率とSharpeを集計し、苦手な時期を特定する。

    有効条件（STRONG × gap≤+2% × 前日プラス）に絞って集計するため、
    daytime.py が実際に発注する場面での傾向を反映している。
    """
    lines = []
    lines.append(f"\n【8. 月別・曜日別分析】")
    lines.append(f"  対象: STRONG × gap-5〜+2% × 前日プラス")
    lines.append(f"  評価: TP={tp:+.1f}% / SL={sl:+.1f}%")

    # 有効条件でフィルタ
    mask = _apply_cond(df, "main")
    df_f = df[mask].copy()

    if len(df_f) < MIN_N:
        lines.append("  ⚠️  サンプル不足")
        return lines, []

    # 日付型に変換
    df_f["_date"] = pd.to_datetime(df_f["trade_date"], errors="coerce")
    df_f["_month"] = df_f["_date"].dt.month
    df_f["_dow"]   = df_f["_date"].dt.dayofweek   # 0=月 〜 4=金

    rows = []

    # ── 月別 ─────────────────────────────────────
    lines.append(f"\n  ─── 月別（1月〜12月） ───")
    lines.append(f"  {'月':<6} {'件数':>6}  {'勝率':>6}  {'平均%':>7}  {'Sharpe':>8}  評価")
    lines.append("  " + "─" * 55)

    month_stats = []
    for m in range(1, 13):
        grp = df_f[df_f["_month"] == m]
        st  = _seasonal_stats(grp, tp, sl)
        if st is None:
            continue
        verdict = "◎" if st["sharpe"] >= 5.0 else ("○" if st["sharpe"] >= 2.0 else
                  ("△" if st["avg"] > 0 else "✗"))
        lines.append(
            f"  {_MON_NAMES[m]:<6} {st['n']:>6,}  {st['wr']:>5.1f}%"
            f"  {st['avg']:>+6.3f}%  {st['sharpe']:>8.3f}  {verdict}"
        )
        month_stats.append({"unit": "month", "key": m, "label": _MON_NAMES[m], **st})
        rows.append({"type": "月別", "label": _MON_NAMES[m], **st})

    # 苦手月・得意月のサマリ
    good_months = [r["label"] for r in rows if r["type"] == "月別" and r["sharpe"] >= 5.0]
    bad_months  = [r["label"] for r in rows if r["type"] == "月別" and r["avg"] <= 0]
    if good_months:
        lines.append(f"\n  ◎ 得意月（Sharpe≥5）: {' / '.join(good_months)}")
    if bad_months:
        lines.append(f"  ✗ 苦手月（avg≤0）  : {' / '.join(bad_months)}")

    # ── 曜日別 ───────────────────────────────────
    lines.append(f"\n  ─── 曜日別 ───")
    lines.append(f"  {'曜日':<4} {'件数':>6}  {'勝率':>6}  {'平均%':>7}  {'Sharpe':>8}  評価")
    lines.append("  " + "─" * 50)

    for dow in range(5):   # 0=月〜4=金
        grp = df_f[df_f["_dow"] == dow]
        st  = _seasonal_stats(grp, tp, sl)
        if st is None:
            continue
        verdict = "◎" if st["sharpe"] >= 5.0 else ("○" if st["sharpe"] >= 2.0 else
                  ("△" if st["avg"] > 0 else "✗"))
        lines.append(
            f"  {_DOW_NAMES[dow]}曜 {st['n']:>6,}  {st['wr']:>5.1f}%"
            f"  {st['avg']:>+6.3f}%  {st['sharpe']:>8.3f}  {verdict}"
        )
        rows.append({"type": "曜日別", "label": f"{_DOW_NAMES[dow]}曜", **st})

    # ── 月×曜日 クロス集計（上位・下位のみ） ──────
    lines.append(f"\n  ─── 月×曜日 クロス（WR上位5・下位3） ───")
    lines.append(f"  {'月×曜日':<10} {'件数':>6}  {'勝率':>6}  {'平均%':>7}  {'Sharpe':>8}")
    lines.append("  " + "─" * 55)

    cross = []
    for m in range(1, 13):
        for dow in range(5):
            grp = df_f[(df_f["_month"] == m) & (df_f["_dow"] == dow)]
            st  = _seasonal_stats(grp, tp, sl)
            if st is None:
                continue
            cross.append({"label": f"{_MON_NAMES[m]}×{_DOW_NAMES[dow]}曜", **st})

    cross.sort(key=lambda x: -x["wr"])
    for r in cross[:5]:
        lines.append(f"  {r['label']:<10} {r['n']:>6,}  {r['wr']:>5.1f}%"
                     f"  {r['avg']:>+6.3f}%  {r['sharpe']:>8.3f}  ◎")
    lines.append("  ...")
    for r in cross[-3:]:
        lines.append(f"  {r['label']:<10} {r['n']:>6,}  {r['wr']:>5.1f}%"
                     f"  {r['avg']:>+6.3f}%  {r['sharpe']:>8.3f}  ✗")

    # ── 弱い曜日の厳しいフィルタ探索 ─────────────
    weak_dows = [dow for dow in range(5)
                 if _seasonal_stats(df_f[df_f["_dow"] == dow], tp, sl) is not None
                 and _seasonal_stats(df_f[df_f["_dow"] == dow], tp, sl)["sharpe"] < 3.0]

    if weak_dows:
        lines.append(f"\n  ─── 弱い曜日の厳しいフィルタ探索 ───")
        lines.append(f"  対象曜日: {' / '.join(_DOW_NAMES[d]+'曜' for d in weak_dows)}")
        lines.append(f"  基準: Sharpe≥3.0 かつ WR≥{MIN_WR}% かつ N≥{MIN_N}")
        lines.append(f"  {'条件':<48} {'件数':>6}  {'勝率':>6}  {'平均%':>7}  {'Sharpe':>8}")
        lines.append("  " + "─" * 82)

        gap_ranges  = [(-5.0, -2.0), (-2.0, 0.0), (0.0, 2.0), (-5.0, 2.0)]
        score_ranges = [(0, 3), (3, 7), (0, 7)]
        rise_ranges  = [(0.0, 99.0), (-99.0, 99.0)]

        dow_filter_candidates = []
        for dow in weak_dows:
            df_dow = df_f[df_f["_dow"] == dow]
            for (g_lo, g_hi), (s_lo, s_hi), (r_lo, r_hi) in [
                    (g, s, r)
                    for g in gap_ranges
                    for s in score_ranges
                    for r in rise_ranges]:
                sub = df_dow[
                    (df_dow["gap_pct"]    >= g_lo) & (df_dow["gap_pct"]    < g_hi) &
                    (df_dow["score"]      >= s_lo) & (df_dow["score"]      < s_hi) &
                    (df_dow["today_rise"] >= r_lo)
                ]
                st = _seasonal_stats(sub, tp, sl)
                if st is None or st["sharpe"] < 3.0 or st["wr"] < MIN_WR:
                    continue
                label = (f"{_DOW_NAMES[dow]}曜 / gap{g_lo:+.0f}〜{g_hi:+.0f}% "
                         f"/ score{s_lo}〜{s_hi} / 前日{'+' if r_lo>=0 else ''}")
                lines.append(f"  {label:<48} {st['n']:>6,}  {st['wr']:>5.1f}%"
                              f"  {st['avg']:>+6.3f}%  {st['sharpe']:>8.3f}")
                dow_filter_candidates.append({
                    "dow": dow, "dow_name": _DOW_NAMES[dow],
                    "gap_lo": g_lo, "gap_hi": g_hi,
                    "score_lo": s_lo, "score_hi": s_hi,
                    "rise_lo": r_lo, **st
                })

        if not dow_filter_candidates:
            # 閾値を下げてベスト候補を表示（参考情報として）
            lines.append(f"  ※ Sharpe≥3.0 の条件なし → ベスト候補（参考）を表示")
            lines.append(f"  {'条件':<48} {'件数':>6}  {'勝率':>6}  {'平均%':>7}  {'Sharpe':>8}")
            lines.append("  " + "─" * 82)
            fallback = []
            for dow in weak_dows:
                df_dow = df_f[df_f["_dow"] == dow]
                for (g_lo, g_hi), (s_lo, s_hi), (r_lo, r_hi) in [
                        (g, s, r)
                        for g in gap_ranges
                        for s in score_ranges
                        for r in rise_ranges]:
                    sub = df_dow[
                        (df_dow["gap_pct"]    >= g_lo) & (df_dow["gap_pct"]    < g_hi) &
                        (df_dow["score"]      >= s_lo) & (df_dow["score"]      < s_hi) &
                        (df_dow["today_rise"] >= r_lo)
                    ]
                    st = _seasonal_stats(sub, tp, sl)
                    if st is None:
                        continue
                    fallback.append({
                        "dow": dow, "dow_name": _DOW_NAMES[dow],
                        "gap_lo": g_lo, "gap_hi": g_hi,
                        "score_lo": s_lo, "score_hi": s_hi,
                        "rise_lo": r_lo, **st
                    })
            fallback.sort(key=lambda x: -x["sharpe"])
            for r in fallback[:5]:
                label = (f"{r['dow_name']}曜 / gap{r['gap_lo']:+.0f}〜{r['gap_hi']:+.0f}% "
                         f"/ score{r['score_lo']}〜{r['score_hi']}"
                         f" / 前日{'+' if r['rise_lo']>=0 else '全'}")
                lines.append(f"  {label:<48} {r['n']:>6,}  {r['wr']:>5.1f}%"
                              f"  {r['avg']:>+6.3f}%  {r['sharpe']:>8.3f}")

            if fallback:
                best = fallback[0]
                lines.append(f"\n  【月曜の最善条件（Sharpe<3 のため要注意）】")
                lines.append(f"    gap {best['gap_lo']:+.0f}〜{best['gap_hi']:+.0f}%"
                             f" / score {best['score_lo']}〜{best['score_hi']}"
                             f" / 前日{'プラス' if best['rise_lo'] >= 0 else '全て'}")
                lines.append(f"    → WR={best['wr']:.1f}% / avg={best['avg']:+.3f}%"
                             f" / Sharpe={best['sharpe']:.3f}  N={best['n']:,}")
                lines.append(f"\n  ⚠️  月曜は構造的に弱く、フィルタで改善困難。")
                lines.append(f"     選択肢: (A) 月曜は見送り  (B) gap -2〜0% に絞って慎重発注")
        else:
            # 最良条件のサマリ
            best = max(dow_filter_candidates, key=lambda x: x["sharpe"])
            lines.append(f"\n  【推奨フィルタ（{best['dow_name']}曜）】")
            lines.append(f"    gap {best['gap_lo']:+.0f}〜{best['gap_hi']:+.0f}%"
                         f" / score {best['score_lo']}〜{best['score_hi']}"
                         f" / 前日{'プラス' if best['rise_lo'] >= 0 else '全て'}")
            lines.append(f"    → WR={best['wr']:.1f}% / avg={best['avg']:+.3f}%"
                         f" / Sharpe={best['sharpe']:.3f}  N={best['n']:,}")
            lines.append(f"\n  daytime.py への反映案:")
            lines.append(f"    MON_GAP_MAX = {best['gap_hi']:.1f}   # 月曜の gap 上限を厳しく")
            lines.append(f"    MON_GAP_MIN = {best['gap_lo']:.1f}   # 月曜の gap 下限")

        rows.extend(dow_filter_candidates)

    return lines, rows


# ══════════════════════════════════════════════
# 9. 銘柄属性分析（価格帯・業種）
# ══════════════════════════════════════════════

# 業種コード: 日本株コードの先頭2桁で粗い業種グループを推定
_SECTOR_MAP = [
    (20,  "農林・鉱業"),
    (30,  "食品・繊維"),
    (40,  "化学・ゴム"),
    (50,  "薬品・IT"),
    (60,  "非鉄・窯業"),
    (70,  "機械・電機"),
    (80,  "輸送・精密"),
    (90,  "金融・不動産"),
    (100, "サービス・IT"),
]

def _code_to_sector(code):
    """4桁の銘柄コード先頭2桁で業種を粗く推定する（東証分類の近似）。"""
    try:
        c = int(code) // 100
        for threshold, name in _SECTOR_MAP:
            if c < threshold:
                return name
        return "サービス・IT"
    except Exception:
        return "不明"


def _load_avg_price_map():
    """daily_prices の平均終値を銘柄ごとに取得して価格帯マップを返す。"""
    try:
        import db as _db
        conn = _db.get_conn()
        df_p = pd.read_sql(
            "SELECT code, AVG(Close) as avg_close FROM daily_prices GROUP BY code",
            conn
        )
        conn.close()
        return df_p.set_index("code")["avg_close"].to_dict()
    except Exception:
        return {}


def _price_label(avg_close):
    """平均株価から価格帯ラベルを返す。"""
    if avg_close is None or pd.isna(avg_close):
        return None
    if avg_close < 500:
        return "①~500円"
    elif avg_close < 1000:
        return "②500~1000円"
    elif avg_close < 2000:
        return "③1000~2000円"
    elif avg_close < 5000:
        return "④2000~5000円"
    else:
        return "⑤5000円~"


def analyze_attributes(df, tp=3.0, sl=-5.0):
    """価格帯・業種コード別に WR/Sharpe を集計し、得意な銘柄属性を特定する。

    価格帯: daily_prices の平均終値から5段階に分類
    業種  : 銘柄コード先頭2桁から粗い業種グループを推定（東証33業種の近似）
    """
    lines = []
    lines.append(f"\n【9. 銘柄属性分析（価格帯・業種）】")
    lines.append(f"  対象: STRONG × gap-5〜+2% × 前日プラス")
    lines.append(f"  評価: TP={tp:+.1f}% / SL={sl:+.1f}%")
    lines.append(f"  ※業種は銘柄コード先頭桁から推定（粗い近似）")

    mask = _apply_cond(df, "main")
    df_f = df[mask].copy()

    rows = []

    # ── 価格帯付与 ────────────────────────────────
    print("    DBから平均株価を取得中...")
    price_map = _load_avg_price_map()
    df_f["avg_close"]   = df_f["code"].map(price_map)
    df_f["price_range"] = df_f["avg_close"].map(_price_label)

    # 業種付与
    df_f["sector"] = df_f["code"].astype(str).map(_code_to_sector)

    # ── 価格帯別 ─────────────────────────────────
    lines.append(f"\n  ─── 価格帯別 ───")
    lines.append(f"  {'価格帯':<16} {'件数':>6}  {'勝率':>6}  {'平均%':>7}  {'Sharpe':>8}  評価")
    lines.append("  " + "─" * 60)

    for label in ["①~500円", "②500~1000円", "③1000~2000円", "④2000~5000円", "⑤5000円~"]:
        grp = df_f[df_f["price_range"] == label]
        st  = _seasonal_stats(grp, tp, sl)
        if st is None:
            continue
        verdict = "◎" if st["sharpe"] >= 7.0 else ("○" if st["sharpe"] >= 4.0 else
                  ("△" if st["avg"] > 0 else "✗"))
        lines.append(
            f"  {label:<16} {st['n']:>6,}  {st['wr']:>5.1f}%"
            f"  {st['avg']:>+6.3f}%  {st['sharpe']:>8.3f}  {verdict}"
        )
        rows.append({"type": "価格帯", "label": label, **st})

    n_no_price = df_f["price_range"].isna().sum()
    if n_no_price > 0:
        lines.append(f"  （価格データなし: {n_no_price:,}件）")

    # ── 業種別 ───────────────────────────────────
    lines.append(f"\n  ─── 業種別（コード先頭桁推定） ───")
    lines.append(f"  {'業種':<16} {'件数':>6}  {'勝率':>6}  {'平均%':>7}  {'Sharpe':>8}  評価")
    lines.append("  " + "─" * 60)

    sector_stats = []
    for sector in df_f["sector"].dropna().unique():
        grp = df_f[df_f["sector"] == sector]
        st  = _seasonal_stats(grp, tp, sl)
        if st is None:
            continue
        sector_stats.append({"sector": sector, **st})

    sector_stats.sort(key=lambda x: -x["sharpe"])
    for s in sector_stats:
        verdict = "◎" if s["sharpe"] >= 7.0 else ("○" if s["sharpe"] >= 4.0 else
                  ("△" if s["avg"] > 0 else "✗"))
        lines.append(
            f"  {s['sector']:<16} {s['n']:>6,}  {s['wr']:>5.1f}%"
            f"  {s['avg']:>+6.3f}%  {s['sharpe']:>8.3f}  {verdict}"
        )
        rows.append({"type": "業種", "label": s["sector"], **s})

    # ── 価格帯 × 業種 クロス（上位5・下位3） ────────
    lines.append(f"\n  ─── 価格帯 × 業種 クロス（WR上位5・下位3） ───")
    lines.append(f"  {'組み合わせ':<26} {'件数':>6}  {'勝率':>6}  {'平均%':>7}  {'Sharpe':>8}")
    lines.append("  " + "─" * 68)

    cross = []
    for pr_label in ["①~500円", "②500~1000円", "③1000~2000円", "④2000~5000円", "⑤5000円~"]:
        for sec in df_f["sector"].dropna().unique():
            grp = df_f[(df_f["price_range"] == pr_label) & (df_f["sector"] == sec)]
            st  = _seasonal_stats(grp, tp, sl)
            if st is None:
                continue
            cross.append({"label": f"{pr_label}×{sec}", **st})

    cross.sort(key=lambda x: -x["sharpe"])
    for r in cross[:5]:
        lines.append(f"  {r['label']:<26} {r['n']:>6,}  {r['wr']:>5.1f}%"
                     f"  {r['avg']:>+6.3f}%  {r['sharpe']:>8.3f}  ◎")
    lines.append("  ...")
    for r in cross[-3:]:
        lines.append(f"  {r['label']:<26} {r['n']:>6,}  {r['wr']:>5.1f}%"
                     f"  {r['avg']:>+6.3f}%  {r['sharpe']:>8.3f}  ✗")

    # ── サマリ ────────────────────────────────────
    best_price  = max((r for r in rows if r["type"] == "価格帯"), key=lambda x: x["sharpe"], default=None)
    best_sector = max((r for r in rows if r["type"] == "業種"),   key=lambda x: x["sharpe"], default=None)
    worst_price = min((r for r in rows if r["type"] == "価格帯"), key=lambda x: x["sharpe"], default=None)

    lines.append(f"\n  【サマリ】")
    if best_price:
        lines.append(f"  ◎ 最良価格帯: {best_price['label']}"
                     f"  WR={best_price['wr']:.1f}% / Sharpe={best_price['sharpe']:.3f}")
    if worst_price and worst_price["avg"] <= 0:
        lines.append(f"  ✗ 苦手価格帯: {worst_price['label']}"
                     f"  WR={worst_price['wr']:.1f}% / Sharpe={worst_price['sharpe']:.3f}")
    if best_sector:
        lines.append(f"  ◎ 最良業種  : {best_sector['label']}"
                     f"  WR={best_sector['wr']:.1f}% / Sharpe={best_sector['sharpe']:.3f}")

    return lines, rows


# ══════════════════════════════════════════════
# 10. 上ひげ比率詳細分析
# ══════════════════════════════════════════════

def analyze_upper_shadow(df, tp=3.0, sl=-5.0):
    """上ひげ比率（upper_shadow_ratio）の詳細分析。

    全体では相関が低いが、有効条件（STRONG × gap≤+2% × 前日プラス）内では
    上ひげ比率が高いほど WR が高くなる傾向がある。

    upper_shadow_ratio = (High - Close) / (High - Low)
        0 → 高値引け（前日の売り圧力なし）
        1 → 安値引け（前日の売り圧力大→翌日 STRONG 地合いで再上昇）
    """
    lines = []
    lines.append(f"\n【10. 上ひげ比率詳細分析（有効条件内）】")
    lines.append(f"  upper_shadow_ratio = (High - Close) / (High - Low)  ← 前日（scan_date）")
    lines.append(f"  0=高値引け / 1=安値引け（大きな上ひげ）")

    if "upper_shadow_ratio" not in df.columns or df["upper_shadow_ratio"].isna().all():
        lines.append("  ⚠️  upper_shadow_ratio が利用不可（DBアクセスを確認）")
        return lines, []

    rows        = []
    all_bin_stats = {}   # label → bin_stats リスト（後で閾値計算に使用）

    # ── 有効条件全体 ─────────────────────────────
    mask_main = _apply_cond(df, "main") & df["upper_shadow_ratio"].notna()
    df_main   = df[mask_main].copy()

    lines.append(f"\n  対象: STRONG × gap-5〜+2% × 前日プラス  ({len(df_main):,}件)")
    lines.append(f"  評価: TP={tp:+.1f}% / SL={sl:+.1f}%")

    subsets = [
        ("有効条件全体", df_main),
        ("gap -2〜0%",   df_main[(df_main["gap_pct"] >= -2.0) & (df_main["gap_pct"] <= 0.0)]),
    ]
    for label, dset in subsets:
        if len(dset) < MIN_N:
            continue
        try:
            dset = dset.copy()
            dset["_usr_bin"] = pd.qcut(dset["upper_shadow_ratio"], q=5, duplicates="drop")
        except Exception:
            continue

        lines.append(f"\n  ─── {label} ───")
        lines.append(f"  {'上ひげ比率範囲':<24} {'件数':>6}  {'勝率':>6}  {'平均%':>7}  {'Sharpe':>8}  評価")
        lines.append("  " + "─" * 70)

        bin_stats = []
        for bin_val, grp in dset.groupby("_usr_bin", observed=True):
            st = _seasonal_stats(grp, tp, sl)
            if st is None:
                continue
            verdict = "◎" if st["sharpe"] >= 7.0 else ("○" if st["sharpe"] >= 4.0 else
                      ("△" if st["avg"] > 0 else "✗"))
            lines.append(f"  {str(bin_val):<24} {st['n']:>6,}  {st['wr']:>5.1f}%"
                         f"  {st['avg']:>+6.3f}%  {st['sharpe']:>8.3f}  {verdict}")
            lo = float(bin_val.left)
            hi = float(bin_val.right)
            bin_stats.append({"label": label, "usr_lo": round(lo, 3),
                               "usr_hi": round(hi, 3), **st})
            rows.append({"group": label, "usr_range": str(bin_val), **st})

        all_bin_stats[label] = bin_stats

        if bin_stats:
            best  = max(bin_stats, key=lambda x: x["sharpe"])
            worst = min(bin_stats, key=lambda x: x["sharpe"])
            lines.append(f"\n  ◎ 最良帯（Sharpe最大）: usr {best['usr_lo']:.2f}〜  "
                         f"WR={best['wr']:.1f}% / Sharpe={best['sharpe']:.3f}")
            lines.append(f"  ✗ 最弱帯（Sharpe最小）: usr 0〜{worst['usr_hi']:.2f}  "
                         f"WR={worst['wr']:.1f}% / Sharpe={worst['sharpe']:.3f}")

    # ── daytime.py への反映案 ─────────────────────
    main_bins = all_bin_stats.get("有効条件全体", [])
    if len(main_bins) >= 2:
        sorted_bins = sorted(main_bins, key=lambda x: x["usr_lo"])
        worst_wr    = min(b["wr"] for b in sorted_bins)
        best_wr     = max(b["wr"] for b in sorted_bins)
        diff        = best_wr - worst_wr
        threshold   = sorted_bins[len(sorted_bins) // 2]["usr_lo"]   # 中央ビン下限

        lines.append(f"\n  【daytime.py への反映案】")
        lines.append(f"  上ひげ比率が高いほど有利: 最大{diff:.1f}% の WR 差（単調増加）")
        lines.append(f"  → calc_attr_weight() に線形重みを追加:")
        lines.append(f"    usr_weight = 1.0 + 0.15 * upper_shadow_ratio  # 0→×1.0、1→×1.15")
        lines.append(f"  → daytime.py でのコード例:")
        lines.append(f"    usr = prev_ohlcv.get(code, {{}}).get('upper_shadow_ratio', 0.5)")
        lines.append(f"    attr_w *= (1.0 + 0.15 * usr)")

    return lines, rows


# ══════════════════════════════════════════════
# 11. MA乖離率詳細分析
# ══════════════════════════════════════════════

def analyze_ma_deviation(df, tp=3.0, sl=-5.0):
    """MA乖離率（5日・25日）の詳細分析。

    有効条件（STRONG × gap≤+2% × 前日プラス）内で
    乖離率の大きさと WR/Sharpe の関係を確認する。
    過熱圏（乖離率が大きすぎる）で WR が下がるなら、上限フィルタを設定できる。
    """
    lines = []
    lines.append(f"\n【11. MA乖離率詳細分析（有効条件内）】")
    lines.append(f"  ma5_dev  = (前日終値 - 5日MA)  / 5日MA  × 100  （短期トレンド）")
    lines.append(f"  ma25_dev = (前日終値 - 25日MA) / 25日MA × 100  （中期トレンド）")
    lines.append(f"  評価: TP={tp:+.1f}% / SL={sl:+.1f}%")

    mask_main = _apply_cond(df, "main")
    df_main   = df[mask_main].copy()

    ma_cols = [c for c in ["ma5_dev", "ma25_dev"] if c in df.columns
               and df[c].notna().sum() > MIN_N]

    if not ma_cols:
        lines.append("  ⚠️  MA乖離率データなし")
        return lines, []

    rows = []
    all_bin_stats = {}

    for col in ma_cols:
        label = "5日MA乖離率" if "5" in col else "25日MA乖離率"
        dset  = df_main[df_main[col].notna()].copy()

        if len(dset) < MIN_N:
            continue

        try:
            dset["_bin"] = pd.qcut(dset[col], q=5, duplicates="drop")
        except Exception:
            continue

        lines.append(f"\n  ─── {label}（{col}） ───")
        lines.append(f"  {'乖離率範囲':<26} {'件数':>6}  {'勝率':>6}  {'平均%':>7}  {'Sharpe':>8}  評価")
        lines.append("  " + "─" * 72)

        bin_stats = []
        for bin_val, grp in dset.groupby("_bin", observed=True):
            st = _seasonal_stats(grp, tp, sl)
            if st is None:
                continue
            verdict = "◎" if st["sharpe"] >= 7.0 else ("○" if st["sharpe"] >= 4.0 else
                      ("△" if st["avg"] > 0 else "✗"))
            lines.append(f"  {str(bin_val):<26} {st['n']:>6,}  {st['wr']:>5.1f}%"
                         f"  {st['avg']:>+6.3f}%  {st['sharpe']:>8.3f}  {verdict}")
            lo = float(bin_val.left)
            hi = float(bin_val.right)
            bin_stats.append({"col": col, "lo": round(lo, 2), "hi": round(hi, 2), **st})
            rows.append({"feature": col, "range": str(bin_val), **st})

        all_bin_stats[col] = bin_stats

        if bin_stats:
            best  = max(bin_stats, key=lambda x: x["sharpe"])
            worst = min(bin_stats, key=lambda x: x["sharpe"])
            lines.append(f"\n  ◎ 最良: {col} {best['lo']:+.1f}〜{best['hi']:+.1f}%  "
                         f"WR={best['wr']:.1f}% / Sharpe={best['sharpe']:.3f}")
            lines.append(f"  ✗ 最弱: {col} {worst['lo']:+.1f}〜{worst['hi']:+.1f}%  "
                         f"WR={worst['wr']:.1f}% / Sharpe={worst['sharpe']:.3f}")

    # ── 最適フィルタ範囲の提案 ────────────────────
    lines.append(f"\n  ─── daytime.py への反映案 ───")
    for col, bins in all_bin_stats.items():
        if len(bins) < 3:
            continue
        sorted_b  = sorted(bins, key=lambda x: x["lo"])
        worst_wr  = min(b["wr"] for b in sorted_b)
        best_wr   = max(b["wr"] for b in sorted_b)
        diff      = best_wr - worst_wr

        # 過熱圏の判定: 最上位ビン（最大乖離）で WR が下がるか確認
        top_bin   = sorted_b[-1]
        mid_bins  = sorted_b[1:-1]
        mid_best  = max(mid_bins, key=lambda x: x["wr"]) if mid_bins else None

        label = "5日MA" if "5" in col else "25日MA"
        if mid_best and top_bin["wr"] < mid_best["wr"] - 1.0:
            lines.append(f"  {label}: 過熱圏（{top_bin['lo']:+.0f}%超）で WR が低下")
            lines.append(f"    → check_signal() に上限フィルタ追加を推奨:")
            lines.append(f"      {'MA5_DEV_MAX' if '5' in col else 'MA25_DEV_MAX'}"
                         f" = {top_bin['lo']:.0f}   # {top_bin['lo']:+.0f}%超は除外")
        else:
            lines.append(f"  {label}: 乖離率が大きいほど WR が高い（最大差{diff:.1f}%）")
            lines.append(f"    → フィルタより adj_score への加味を推奨")
            if sorted_b:
                bottom = sorted_b[0]
                lines.append(f"      乖離率 {bottom['lo']:+.0f}〜{bottom['hi']:+.0f}% の銘柄は優先度を下げる")

    return lines, rows


# ══════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════
def main():
    print(f"\n{'='*65}")
    print(f"  analyze_a.py - 戦略A（順張り）統計的エッジ分析")
    print(f"  実行日: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*65}\n")

    print("データ読み込み中...")
    df = load_data()

    all_lines = []
    all_lines.append("=" * 70)
    all_lines.append("  戦略A（順張り）統計的エッジ分析レポート")
    all_lines.append(f"  実行日: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    all_lines.append(f"  データ: {df['trade_date'].min()} 〜 {df['trade_date'].max()}"
                     f"  ({len(df):,}件)")
    all_lines.append(f"  全体勝率: {df['win'].mean()*100:.1f}%  (p=0.05の閾値=50%)")
    all_lines.append("=" * 70)

    # 1. 相関分析
    print("相関分析中...")
    corr_lines, top_features = analyze_correlation(df)
    all_lines += corr_lines

    # 2. 単変量分析
    print("単変量分析中...")
    uni_lines = analyze_univariate(df, ["score", "ratio", "today_rise", "gap_pct",
                                        "rb_score", "lower_shadow_ratio",
                                        "upper_shadow_ratio",
                                        "ma5_dev", "ma25_dev"])
    all_lines += uni_lines

    # 3. 地合い別分析
    print("地合い別分析中...")
    cond_lines = analyze_by_condition(df)
    all_lines += cond_lines

    # 4. 多変量探索
    print("多変量条件探索中（しばらくかかります）...")
    multi_lines, detail_rows, found = analyze_multivariate(df)
    all_lines += multi_lines

    # 5. 出来高比率分析
    print("出来高比率分析中...")
    ratio_lines = analyze_ratio(df)
    all_lines += ratio_lines

    # 6. TP/SL最適化
    print("TP/SL最適化中...")
    tpsl_lines, tpsl_results = analyze_tp_sl(df)
    all_lines += tpsl_lines

    # 7. アウトオブサンプル検証
    print("アウトオブサンプル検証中...")
    best_tp = tpsl_results[0]["tp"] if tpsl_results else 3.0
    best_sl = tpsl_results[0]["sl"] if tpsl_results else -5.0
    oos_lines, oos_rows = analyze_oos(df, tp=best_tp, sl=best_sl)
    all_lines += oos_lines

    # 8. 月別・曜日別分析
    print("月別・曜日別分析中...")
    seasonal_lines, seasonal_rows = analyze_seasonal(df, tp=best_tp, sl=best_sl)
    all_lines += seasonal_lines

    # 9. 銘柄属性分析
    print("銘柄属性分析中（DBから価格取得）...")
    attr_lines, attr_rows = analyze_attributes(df, tp=best_tp, sl=best_sl)
    all_lines += attr_lines

    # 10. 上ひげ比率詳細分析
    print("上ひげ比率詳細分析中...")
    usr_lines, usr_rows = analyze_upper_shadow(df, tp=best_tp, sl=best_sl)
    all_lines += usr_lines

    # 11. MA乖離率詳細分析
    print("MA乖離率詳細分析中...")
    ma_lines, ma_rows = analyze_ma_deviation(df, tp=best_tp, sl=best_sl)
    all_lines += ma_lines

    # ── まとめ ──
    all_lines.append(f"\n{'='*70}")
    all_lines.append("  【まとめ】戦略Aとして使える条件")
    all_lines.append(f"{'='*70}")
    if found:
        all_lines.append(f"  有意な条件が {len(found)} 件見つかりました:")
        found_sorted = sorted(found, key=lambda x: -x["wr"])
        for r in found_sorted[:10]:
            all_lines.append(
                f"  ★ score{r['score']} / 前日{r['rise']} / gap{r['gap']} / {r['cond']}"
                f"  → WR={r['wr']}% / N={r['n']} / p={r['pval']}"
            )
    else:
        all_lines.append("  ★ 有意な条件は見つかりませんでした。")
        all_lines.append("  → 現在の特徴量では順張りシグナルに統計的エッジがありません。")
        all_lines.append("  → 別の特徴量（IR・セクター・時間帯など）の追加を検討してください。")

    all_lines.append(f"\n  詳細データ → {DETAIL_CSV}")
    all_lines.append("=" * 70)

    report = "\n".join(all_lines)
    print("\n" + report)

    # 保存
    os.makedirs("out", exist_ok=True)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)

    if detail_rows:
        pd.DataFrame(detail_rows).sort_values("wr", ascending=False).to_csv(
            DETAIL_CSV, index=False, encoding="utf-8-sig"
        )

    if tpsl_results:
        pd.DataFrame(tpsl_results).sort_values("sharpe", ascending=False).to_csv(
            TPSL_CSV, index=False, encoding="utf-8-sig"
        )

    if oos_rows:
        pd.DataFrame(oos_rows).to_csv(OOS_CSV, index=False, encoding="utf-8-sig")

    if seasonal_rows:
        pd.DataFrame(seasonal_rows).to_csv(SEASONAL_CSV, index=False, encoding="utf-8-sig")

    if attr_rows:
        pd.DataFrame(attr_rows).to_csv(ATTR_CSV, index=False, encoding="utf-8-sig")

    print(f"\n保存完了:")
    print(f"  {REPORT_TXT}")
    if detail_rows:
        print(f"  {DETAIL_CSV}")
    if tpsl_results:
        print(f"  {TPSL_CSV}")
    if oos_rows:
        print(f"  {OOS_CSV}")
    if attr_rows:
        print(f"  {ATTR_CSV}")
    if seasonal_rows:
        print(f"  {SEASONAL_CSV}")


if __name__ == "__main__":
    main()
