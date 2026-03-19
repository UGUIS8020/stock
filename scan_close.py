"""
scan_close.py
=============
実行タイミング : 毎営業日 15:00（cron）
目的          : 当日の日中値動きを分析し、15:20の引け注文 or PTS用の候補銘柄を抽出
戦略          : 「夜間保有優位」を活かすため、引け or PTS で仕込み → 翌朝売り
データソース  : J-Quants（scan_daily.py と同じ認証情報を使用）
出力          : コンソール + close_candidates_log.csv
"""

import os
import json
import math
import datetime
import pandas as pd
import numpy as np

# J-Quants SDK（scan_daily.py と同じ）
try:
    import jquantsapi
except ImportError:
    print("[ERROR] jquantsapi がインストールされていません: pip install jquants-api-client")
    raise

# ─────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────
CONFIG = {
    # スコア閾値
    "STRATEGY_A_THRESHOLD": 5,   # モメンタム（日中強かった銘柄）
    "STRATEGY_B_THRESHOLD": 5,   # リバウンド（売られすぎた銘柄）

    # 最低出来高（流動性フィルター）
    "MIN_VOLUME": 50_000,

    # 株価レンジ（低位株・超高額株を除外）
    "MIN_PRICE": 300,
    "MAX_PRICE": 50_000,

    # 出力ファイル
    "LOG_FILE": "close_candidates_log.csv",
    "MARKET_LOG": "market_log.csv",       # scan_morning.py が書き込む市場環境ログ

    # 最大候補表示数
    "TOP_N": 10,
}

# ─────────────────────────────────────────────
# J-Quants 認証
# ─────────────────────────────────────────────
def get_jquants_client():
    """scan_daily.py と同じ認証方法を使用"""
    refresh_token = os.environ.get("JQUANTS_REFRESH_TOKEN", "")
    if not refresh_token:
        # ローカル設定ファイルからも試みる
        config_path = os.path.expanduser("~/.jquants/config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
                refresh_token = cfg.get("refresh_token", "")
    if not refresh_token:
        raise ValueError("JQUANTS_REFRESH_TOKEN が設定されていません")
    client = jquantsapi.Client(refresh_token=refresh_token)
    return client


# ─────────────────────────────────────────────
# 市場環境の読み込み（scan_morning.py の結果を流用）
# ─────────────────────────────────────────────
def load_market_condition():
    """
    scan_morning.py が market_log.csv に書き込んだ当日の市場環境を読む。
    ファイルがなければ NORMAL として扱う。
    """
    try:
        df = pd.read_csv(CONFIG["MARKET_LOG"])
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        today_rows = df[df["date"] == today_str]
        if not today_rows.empty:
            condition = today_rows.iloc[-1]["market_condition"]
            print(f"[市場環境] {condition}（scan_morning.py より）")
            return condition
    except Exception:
        pass
    print("[市場環境] NORMAL（market_log.csv が読めないためデフォルト）")
    return "NORMAL"


# ─────────────────────────────────────────────
# データ取得
# ─────────────────────────────────────────────
def fetch_daily_quotes(client, today_str):
    """当日の日足データを取得（J-Quants /prices/daily_quotes）"""
    try:
        df = client.get_prices_daily_quotes(date_yyyymmdd=today_str.replace("-", ""))
        return df
    except Exception as e:
        print(f"[ERROR] 日足データ取得失敗: {e}")
        return pd.DataFrame()


def fetch_historical_for_universe(client, codes, lookback_days=40):
    """
    移動平均・平均出来高計算用の過去データを取得。
    コードリストをまとめて取得して効率化。
    """
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=lookback_days * 2)  # 営業日考慮で余裕を持つ
    results = {}
    for code in codes:
        try:
            df = client.get_prices_daily_quotes(
                code=code,
                from_yyyymmdd=start_date.strftime("%Y%m%d"),
                to_yyyymmdd=end_date.strftime("%Y%m%d"),
            )
            if not df.empty:
                df = df.sort_values("Date").tail(lookback_days)
                results[code] = df
        except Exception:
            pass
    return results


# ─────────────────────────────────────────────
# Strategy A：モメンタム（日中強い銘柄を引けで仕込む）
# ─────────────────────────────────────────────
def calc_strategy_a_score(row, hist_df):
    """
    日中の強さを評価するスコア（0〜10点）
    「今日強かった銘柄は翌朝もギャップアップしやすい」
    """
    score = 0
    reasons = []

    # ① 日中上昇率（始値→現値）
    if row["Open"] and row["Open"] > 0:
        day_return = (row["Close"] - row["Open"]) / row["Open"] * 100
        if day_return >= 3.0:
            score += 3; reasons.append(f"日中+{day_return:.1f}%")
        elif day_return >= 1.5:
            score += 2; reasons.append(f"日中+{day_return:.1f}%")
        elif day_return >= 0.5:
            score += 1; reasons.append(f"日中+{day_return:.1f}%")

    # ② 高値引け度（現値が日中レンジのどこにいるか）
    price_range = row["High"] - row["Low"]
    if price_range > 0:
        high_close_ratio = (row["Close"] - row["Low"]) / price_range
        if high_close_ratio >= 0.85:
            score += 3; reasons.append(f"高値引け度{high_close_ratio:.0%}")
        elif high_close_ratio >= 0.65:
            score += 1; reasons.append(f"高値引け度{high_close_ratio:.0%}")

    # ③ 出来高ペース（過去20日平均比）
    if len(hist_df) >= 20:
        avg_vol = hist_df["Volume"].iloc[-20:].mean()
        # 15:00時点の出来高は1日の約93%（6時間/6.5時間）
        estimated_full_vol = row["Volume"] / 0.93
        vol_ratio = estimated_full_vol / avg_vol if avg_vol > 0 else 0
        if vol_ratio >= 2.0:
            score += 2; reasons.append(f"出来高{vol_ratio:.1f}倍")
        elif vol_ratio >= 1.5:
            score += 1; reasons.append(f"出来高{vol_ratio:.1f}倍")

    # ④ MA25上での推移
    if len(hist_df) >= 25:
        ma25 = hist_df["Close"].iloc[-25:].mean()
        if row["Close"] > ma25:
            score += 1; reasons.append("MA25上")
        # MA25を今日上抜けたかチェック
        prev_close = hist_df["Close"].iloc[-2] if len(hist_df) >= 2 else None
        if prev_close and prev_close < ma25 <= row["Close"]:
            score += 1; reasons.append("MA25上抜け★")

    # ⑤ 前日比騰落率（前日引けから今日の現値）
    if len(hist_df) >= 2:
        prev_close = hist_df["Close"].iloc[-2]
        if prev_close > 0:
            change_pct = (row["Close"] - prev_close) / prev_close * 100
            if change_pct >= 5.0:
                score -= 1  # 急騰しすぎは追いかけない（一部減点）
                reasons.append(f"前日比{change_pct:.1f}%（過熱注意）")

    return score, reasons


# ─────────────────────────────────────────────
# Strategy B：リバウンド（売られすぎた銘柄を引けで仕込む）
# ─────────────────────────────────────────────
def calc_strategy_b_score(row, hist_df):
    """
    売られすぎ → 翌朝リバウンド期待のスコア（0〜10点）
    scan_daily.py のStrategy Bと同じ思想
    """
    score = 0
    reasons = []

    if len(hist_df) < 14:
        return 0, []

    close_series = pd.concat([hist_df["Close"], pd.Series([row["Close"]])])

    # ① RSI（14日）
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]

    if not math.isnan(current_rsi):
        if current_rsi <= 25:
            score += 3; reasons.append(f"RSI{current_rsi:.0f}（売られすぎ）")
        elif current_rsi <= 35:
            score += 2; reasons.append(f"RSI{current_rsi:.0f}")
        elif current_rsi <= 45:
            score += 1; reasons.append(f"RSI{current_rsi:.0f}")

    # ② MA25乖離率（マイナス乖離 = 下に離れすぎ）
    if len(hist_df) >= 25:
        ma25 = hist_df["Close"].iloc[-25:].mean()
        deviation = (row["Close"] - ma25) / ma25 * 100
        if deviation <= -10:
            score += 3; reasons.append(f"MA25乖離{deviation:.1f}%")
        elif deviation <= -5:
            score += 2; reasons.append(f"MA25乖離{deviation:.1f}%")
        elif deviation <= -2:
            score += 1; reasons.append(f"MA25乖離{deviation:.1f}%")

    # ③ 日中の下落率（本日の売られ度）
    if row["Open"] and row["Open"] > 0:
        day_return = (row["Close"] - row["Open"]) / row["Open"] * 100
        if day_return <= -3.0:
            score += 2; reasons.append(f"日中{day_return:.1f}%（本日急落）")
        elif day_return <= -1.5:
            score += 1; reasons.append(f"日中{day_return:.1f}%")

    # ④ 出来高増加（売りが出尽くしのサイン）
    if len(hist_df) >= 20:
        avg_vol = hist_df["Volume"].iloc[-20:].mean()
        estimated_full_vol = row["Volume"] / 0.93
        vol_ratio = estimated_full_vol / avg_vol if avg_vol > 0 else 0
        if vol_ratio >= 2.0:
            score += 2; reasons.append(f"出来高{vol_ratio:.1f}倍（売り出尽くし？）")
        elif vol_ratio >= 1.5:
            score += 1; reasons.append(f"出来高{vol_ratio:.1f}倍")

    return score, reasons


# ─────────────────────────────────────────────
# スコア閾値の市場環境調整
# ─────────────────────────────────────────────
def adjust_threshold(base_threshold, market_condition, strategy):
    """market_conditionに応じてスコア閾値を調整"""
    if market_condition == "STRONG":
        # 強い日はA有利、Bは少し厳しく
        return base_threshold - 1 if strategy == "A" else base_threshold + 1
    elif market_condition == "WEAK":
        # 弱い日はAを厳しく、Bは緩める
        return base_threshold + 1 if strategy == "A" else base_threshold - 1
    elif market_condition == "PANIC":
        # PANICはA不可、Bのみ（しかし高閾値）
        return 999 if strategy == "A" else base_threshold + 2
    return base_threshold  # NORMAL


# ─────────────────────────────────────────────
# メイン処理
# ─────────────────────────────────────────────
def main():
    now = datetime.datetime.now()
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    print(f"\n{'='*60}")
    print(f"  scan_close.py  [{now.strftime('%Y-%m-%d %H:%M')}]")
    print(f"{'='*60}")

    # 1. 市場環境チェック
    market_cond = load_market_condition()
    if market_cond == "PANIC":
        print("[PANIC] 市場PANIC：Strategy A スキップ。Strategy B のみ高閾値で実施。")

    # 2. J-Quants クライアント初期化
    client = get_jquants_client()

    # 3. 当日の全銘柄日足データ取得
    print("\n[1/3] 当日の日足データ取得中...")
    quotes_df = fetch_daily_quotes(client, today_str)
    if quotes_df.empty:
        print("[ERROR] データ取得失敗。終了します。")
        return

    # 4. フィルタリング（流動性・株価レンジ）
    quotes_df = quotes_df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    quotes_df = quotes_df[
        (quotes_df["Volume"] >= CONFIG["MIN_VOLUME"]) &
        (quotes_df["Close"] >= CONFIG["MIN_PRICE"]) &
        (quotes_df["Close"] <= CONFIG["MAX_PRICE"])
    ]
    codes = quotes_df["Code"].tolist()
    print(f"  フィルター通過銘柄数: {len(codes)}")

    # 5. 過去データ取得（MA・出来高平均計算用）
    print("\n[2/3] 過去データ取得中（移動平均・出来高平均の計算用）...")
    hist_data = fetch_historical_for_universe(client, codes, lookback_days=40)
    print(f"  過去データ取得完了: {len(hist_data)} 銘柄")

    # 6. スコアリング
    print("\n[3/3] スコアリング中...")
    threshold_a = adjust_threshold(CONFIG["STRATEGY_A_THRESHOLD"], market_cond, "A")
    threshold_b = adjust_threshold(CONFIG["STRATEGY_B_THRESHOLD"], market_cond, "B")

    candidates = []
    for _, row in quotes_df.iterrows():
        code = row["Code"]
        hist_df = hist_data.get(code, pd.DataFrame())

        # Strategy A
        score_a, reasons_a = calc_strategy_a_score(row, hist_df)
        if score_a >= threshold_a:
            candidates.append({
                "date": today_str,
                "code": code,
                "name": row.get("Name", ""),
                "close": row["Close"],
                "strategy": "A",
                "score": score_a,
                "reasons": " / ".join(reasons_a),
                "entry_timing": "15:20引け or PTS",
                "exit_timing": "翌朝寄り付き売り",
                "market_condition": market_cond,
            })

        # Strategy B
        score_b, reasons_b = calc_strategy_b_score(row, hist_df)
        if score_b >= threshold_b:
            candidates.append({
                "date": today_str,
                "code": code,
                "name": row.get("Name", ""),
                "close": row["Close"],
                "strategy": "B",
                "score": score_b,
                "reasons": " / ".join(reasons_b),
                "entry_timing": "15:20引け or PTS",
                "exit_timing": "翌朝〜翌々日スイング",
                "market_condition": market_cond,
            })

    # 同一銘柄でA・B両方ヒットした場合はスコアの高い方を残す
    seen = {}
    for c in candidates:
        key = c["code"]
        if key not in seen or c["score"] > seen[key]["score"]:
            seen[key] = c
    candidates = sorted(seen.values(), key=lambda x: -x["score"])

    # 7. 結果表示
    print(f"\n{'='*60}")
    print(f"  【引け前候補】{today_str}  市場環境: {market_cond}")
    print(f"  候補数: {len(candidates)}  （閾値 A≥{threshold_a}, B≥{threshold_b}）")
    print(f"{'='*60}")

    for i, c in enumerate(candidates[:CONFIG["TOP_N"]], 1):
        print(f"  {i:2d}. [{c['strategy']}] {c['code']} {c['name'][:10]:<10s}"
              f"  ¥{c['close']:,}  スコア:{c['score']}")
        print(f"       {c['reasons']}")
        print(f"       → {c['entry_timing']} / 出口: {c['exit_timing']}")
        print()

    if not candidates:
        print("  該当銘柄なし")

    # 8. ログ保存
    if candidates:
        result_df = pd.DataFrame(candidates)
        log_path = CONFIG["LOG_FILE"]
        if os.path.exists(log_path):
            existing = pd.read_csv(log_path)
            result_df = pd.concat([existing, result_df], ignore_index=True)
        result_df.to_csv(log_path, index=False)
        print(f"[LOG] {log_path} に保存しました")

    print(f"\n  ⏰ 15:20〜15:25 に上位銘柄の引け注文 or PTS発注を検討してください")
    print(f"     ⚠️  米国株急落リスクがある場合は持ち越しを再考")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()