"""
verify_results.py - 判定結果の事後検証

candidates_log.csv の BUY/CAUTION/PASS 判定に対して
実際の値動き（寄り付き→引け）を yfinance で照合し、
的中率・平均リターンを集計する。

使い方:
    python verify_results.py          # 全期間
    python verify_results.py --days 30  # 直近30日
    python verify_results.py --strategy A  # 戦略Aのみ
"""

import os
import sys
import argparse
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

CANDIDATES_LOG_CSV = "out/candidates_log.csv"
MORNING_LOG_CSV    = "out/morning_log.csv"
VERIFY_CACHE_CSV   = "out/verify_cache.csv"  # 取得済み株価をキャッシュ


# ══════════════════════════════════════════════════════
# 株価取得（キャッシュ付き）
# ══════════════════════════════════════════════════════
def load_price_cache():
    if os.path.exists(VERIFY_CACHE_CSV):
        return pd.read_csv(VERIFY_CACHE_CSV, dtype={"code": str})
    return pd.DataFrame(columns=["code", "date", "open", "close", "high", "low"])


def save_price_cache(cache_df):
    cache_df.to_csv(VERIFY_CACHE_CSV, index=False)


def fetch_ohlc(code, date_str, cache_df):
    """
    指定銘柄・日付の OHLC を返す。
    キャッシュにあればそれを使い、なければ yfinance から取得。
    戻り値: {"open": float, "close": float, "high": float, "low": float} or None
    """
    hit = cache_df[(cache_df["code"] == str(code)) & (cache_df["date"] == date_str)]
    if not hit.empty:
        row = hit.iloc[0]
        return {"open": row["open"], "close": row["close"],
                "high": row["high"], "low": row["low"]}

    try:
        target = datetime.strptime(date_str, "%Y-%m-%d")
        start  = (target - timedelta(days=3)).strftime("%Y-%m-%d")
        end    = (target + timedelta(days=2)).strftime("%Y-%m-%d")
        ticker = yf.Ticker(f"{code}.T")
        hist   = ticker.history(start=start, end=end)

        if hist.empty:
            return None

        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        target_dt  = pd.Timestamp(date_str)
        day_data   = hist[hist.index.date == target_dt.date()]

        if day_data.empty:
            return None

        row = day_data.iloc[0]
        result = {
            "open":  round(float(row["Open"]),  2),
            "close": round(float(row["Close"]), 2),
            "high":  round(float(row["High"]),  2),
            "low":   round(float(row["Low"]),   2),
        }
        return result

    except Exception:
        return None


# ══════════════════════════════════════════════════════
# メイン集計
# ══════════════════════════════════════════════════════
def run_verify(days=None, strategy_filter=None):
    if not os.path.exists(CANDIDATES_LOG_CSV):
        print("❌ out/candidates_log.csv が見つかりません")
        print("   先に scan_morning.py を数日実行してください")
        return

    df = pd.read_csv(CANDIDATES_LOG_CSV, dtype={"code": str})
    df["date"] = pd.to_datetime(df["date"])

    # フィルタ
    if days:
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df["date"] >= cutoff]
    if strategy_filter:
        df = df[df["strategy"] == strategy_filter.upper()]

    if df.empty:
        print("⚠️  対象データがありません")
        return

    # 地合いスコアを結合
    if os.path.exists(MORNING_LOG_CSV):
        mlog = pd.read_csv(MORNING_LOG_CSV)
        mlog["date"] = pd.to_datetime(mlog["date"])
        df = df.merge(mlog[["date", "condition_forecast", "condition_score"]],
                      on="date", how="left")
    else:
        df["condition_forecast"] = "UNKNOWN"
        df["condition_score"]    = None

    # ── 株価取得 ──
    print(f"\n  株価データ取得中（{len(df)}件）...")
    cache_df   = load_price_cache()
    new_rows   = []
    ohlc_map   = {}
    total      = len(df)
    fetched    = 0
    cache_hits = 0

    unique_pairs = df[["code", "date"]].drop_duplicates()
    for _, r in unique_pairs.iterrows():
        code     = str(r["code"])
        date_str = r["date"].strftime("%Y-%m-%d")
        key      = (code, date_str)

        # キャッシュ確認
        hit = cache_df[(cache_df["code"] == code) & (cache_df["date"] == date_str)]
        if not hit.empty:
            row = hit.iloc[0]
            ohlc_map[key] = {"open":  row["open"],  "close": row["close"],
                             "high":  row["high"],  "low":   row["low"]}
            cache_hits += 1
            continue

        # yfinance 取得
        ohlc = fetch_ohlc(code, date_str, cache_df)
        if ohlc:
            ohlc_map[key] = ohlc
            new_rows.append({"code": code, "date": date_str, **ohlc})
        fetched += 1
        if fetched % 10 == 0:
            print(f"    {fetched + cache_hits}/{len(unique_pairs)} 件処理済み...")
        time.sleep(0.2)  # API 負荷軽減

    # キャッシュ保存
    if new_rows:
        new_cache = pd.DataFrame(new_rows)
        cache_df  = pd.concat([cache_df, new_cache], ignore_index=True).drop_duplicates(
            subset=["code", "date"])
        save_price_cache(cache_df)
        print(f"  キャッシュ更新: {len(new_rows)}件追加（既存{cache_hits}件ヒット）\n")

    # ── リターン計算 ──
    returns = []
    for _, row in df.iterrows():
        key  = (str(row["code"]), row["date"].strftime("%Y-%m-%d"))
        ohlc = ohlc_map.get(key)
        if ohlc is None or ohlc["open"] == 0:
            pct    = None
            result = "データなし"
        else:
            pct = round((ohlc["close"] - ohlc["open"]) / ohlc["open"] * 100, 2)
            if pct >= 3.0:
                result = "大幅上昇"
            elif pct >= 0.5:
                result = "上昇"
            elif pct >= -0.5:
                result = "横ばい"
            elif pct >= -3.0:
                result = "下落"
            else:
                result = "大幅下落"

        returns.append({
            "date":       row["date"].strftime("%Y-%m-%d"),
            "strategy":   row["strategy"],
            "code":       row["code"],
            "name":       row["name"],
            "judgment":   row["judgment"],
            "score":      row.get("score"),
            "condition":  row.get("condition_forecast", "UNKNOWN"),
            "open":       ohlc["open"]  if ohlc else None,
            "close":      ohlc["close"] if ohlc else None,
            "return_pct": pct,
            "result":     result,
        })

    result_df = pd.DataFrame(returns)
    valid_df  = result_df.dropna(subset=["return_pct"])

    if valid_df.empty:
        print("⚠️  株価データが取得できませんでした（日本株のみ対応）")
        return

    # ══════════════════════════════════════════════════════
    # 表示
    # ══════════════════════════════════════════════════════
    title = "全期間"
    if days:
        title = f"直近{days}日"
    if strategy_filter:
        title += f" / 戦略{strategy_filter.upper()}"

    print(f"\n{'='*62}")
    print(f"【判定結果の事後検証】{title}")
    print(f"  対象: {valid_df['date'].min()} 〜 {valid_df['date'].max()}")
    print(f"  総件数: {len(valid_df)}件（データなし除外済み）")
    print(f"{'='*62}")

    # ── 1. 判定別サマリー ──
    print(f"\n■ 判定別 的中率・平均リターン")
    print(f"  {'判定':<10} {'件数':>4}  {'的中率':>7}  {'平均リターン':>10}  {'平均(勝ち)':>9}  {'平均(負け)':>9}")
    print(f"  {'─'*58}")

    judgment_order = ["BUY", "CAUTION", "PASS"]
    for jdg in judgment_order:
        sub = valid_df[valid_df["judgment"] == jdg]
        if sub.empty:
            continue
        n        = len(sub)
        win_rate = len(sub[sub["return_pct"] > 0]) / n * 100
        avg_ret  = sub["return_pct"].mean()
        wins     = sub[sub["return_pct"] > 0]["return_pct"]
        losses   = sub[sub["return_pct"] <= 0]["return_pct"]
        avg_win  = wins.mean()  if not wins.empty  else 0
        avg_loss = losses.mean() if not losses.empty else 0

        icon = {"BUY": "✅", "CAUTION": "⚠️ ", "PASS": "❌"}.get(jdg, "  ")
        print(f"  {icon} {jdg:<8} {n:>4}件  {win_rate:>6.1f}%  {avg_ret:>+9.2f}%  "
              f"{avg_win:>+8.2f}%  {avg_loss:>+8.2f}%")

    # ── 2. 戦略別サマリー ──
    print(f"\n■ 戦略別 BUY判定の的中率")
    print(f"  {'戦略':<6} {'件数':>4}  {'的中率':>7}  {'平均リターン':>10}  {'最大':>6}  {'最小':>6}")
    print(f"  {'─'*50}")

    buy_df = valid_df[valid_df["judgment"] == "BUY"]
    for strat in ["A", "B"]:
        sub = buy_df[buy_df["strategy"] == strat]
        if sub.empty:
            continue
        n        = len(sub)
        win_rate = len(sub[sub["return_pct"] > 0]) / n * 100
        avg_ret  = sub["return_pct"].mean()
        max_ret  = sub["return_pct"].max()
        min_ret  = sub["return_pct"].min()
        print(f"  戦略{strat}   {n:>4}件  {win_rate:>6.1f}%  {avg_ret:>+9.2f}%  "
              f"{max_ret:>+5.1f}%  {min_ret:>+5.1f}%")

    # ── 3. 地合い別サマリー ──
    if "condition" in valid_df.columns:
        print(f"\n■ 地合い別 BUY判定の的中率")
        print(f"  {'地合い':<8} {'件数':>4}  {'的中率':>7}  {'平均リターン':>10}")
        print(f"  {'─'*38}")
        for cond in ["STRONG", "NORMAL", "WEAK", "PANIC"]:
            sub = buy_df[buy_df["condition"] == cond]
            if sub.empty:
                continue
            n        = len(sub)
            win_rate = len(sub[sub["return_pct"] > 0]) / n * 100
            avg_ret  = sub["return_pct"].mean()
            icon = {"STRONG": "🚀", "NORMAL": "✅", "WEAK": "⚠️ ", "PANIC": "🚨"}.get(cond, "  ")
            print(f"  {icon} {cond:<6} {n:>4}件  {win_rate:>6.1f}%  {avg_ret:>+9.2f}%")

    # ── 4. スコア帯別 BUY判定の成績（戦略Aのみ）──
    buy_a_df = buy_df[buy_df["strategy"] == "A"].copy()
    buy_a_df["score"] = pd.to_numeric(buy_a_df["score"], errors="coerce")
    buy_a_valid = buy_a_df.dropna(subset=["score"])
    if not buy_a_valid.empty:
        print(f"\n■ スコア帯別 BUY判定の成績（戦略A）")
        print(f"  {'スコア帯':<10} {'件数':>4}  {'的中率':>7}  {'平均リターン':>10}  最適閾値の参考に")
        print(f"  {'─'*52}")
        bins   = [0, 7.5, 8.0, 8.5, 9.0, 99]
        labels = ["〜7.5", "7.5-8.0", "8.0-8.5", "8.5-9.0", "9.0〜"]
        buy_a_valid["band"] = pd.cut(buy_a_valid["score"], bins=bins, labels=labels, right=False)
        for band in labels:
            sub = buy_a_valid[buy_a_valid["band"] == band]
            if sub.empty:
                continue
            n        = len(sub)
            win_rate = len(sub[sub["return_pct"] > 0]) / n * 100
            avg_ret  = sub["return_pct"].mean()
            bar      = "█" * int(win_rate / 10)
            print(f"  スコア{band:<8} {n:>4}件  {win_rate:>6.1f}%  {avg_ret:>+9.2f}%  {bar}")

    # ── 5. PASS銘柄の実際の動き（機会損失確認）──
    pass_df = valid_df[valid_df["judgment"] == "PASS"]
    if not pass_df.empty:
        missed = pass_df[pass_df["return_pct"] >= 5.0].sort_values("return_pct", ascending=False)
        print(f"\n■ 見送り(PASS)のうち +5%以上だった銘柄（機会損失）")
        if missed.empty:
            print(f"  なし（大きな機会損失はありませんでした）")
        else:
            print(f"  {'日付':<12} {'戦略':<4} {'コード':<6} {'銘柄名':<18} {'リターン':>8}  スコア")
            print(f"  {'─'*58}")
            for _, r in missed.head(10).iterrows():
                print(f"  {r['date']:<12} {r['strategy']:<4} {str(r['code']):<6} "
                      f"{str(r['name'])[:16]:<18} {r['return_pct']:>+7.2f}%  {r['score']}")

    # ── 5. BUY判定の詳細リスト ──
    print(f"\n■ BUY判定の詳細（直近20件）")
    print(f"  {'日付':<12} {'戦略':<4} {'コード':<6} {'銘柄名':<18} {'始値':>6} {'終値':>6} {'リターン':>8}  判定")
    print(f"  {'─'*70}")

    recent_buy = buy_df.sort_values("date", ascending=False).head(20)
    for _, r in recent_buy.iterrows():
        ret  = r["return_pct"]
        icon = "✅" if ret > 0 else "❌"
        print(f"  {r['date']:<12} {r['strategy']:<4} {str(r['code']):<6} "
              f"{str(r['name'])[:16]:<18} {r['open']:>6.0f} {r['close']:>6.0f} "
              f"{ret:>+7.2f}%  {icon}")

    print(f"\n{'='*62}")
    print(f"✅ 検証完了")
    print(f"{'='*62}\n")


# ══════════════════════════════════════════════════════
# エントリーポイント
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="判定結果の事後検証")
    parser.add_argument("--days",     type=int,  default=None, help="直近N日間のみ検証")
    parser.add_argument("--strategy", type=str,  default=None, help="戦略フィルタ（A or B）")
    args = parser.parse_args()

    run_verify(days=args.days, strategy=args.strategy)
