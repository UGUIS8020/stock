"""
market_watch.py - 寄り付き後リアルタイム監視 + AI買い価格判断

scan_morning.py → ai_filter.py → market_watch.py の順で自動起動される。

9:00〜9:10の間、候補銘柄の価格を15秒ごとに取得・表示し、
9:05頃にClaudeへ価格推移を送信して「いくらで買うべきか」を判断させる。

実行方法:
    python market_watch.py          # 9:00まで待機して自動開始
    python market_watch.py --now    # 即時開始（テスト用）

前提:
    - scan_morning.py を先に実行済みであること
    - .env に ANTHROPIC_API_KEY が設定されていること
"""

import os
import sys
import json
import time
import argparse
import requests
import pandas as pd
import anthropic
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

CANDIDATES_LOG_CSV = "out/candidates_log.csv"
MORNING_LOG_CSV    = "out/morning_log.csv"
JST                = timezone(timedelta(hours=9))
TODAY              = datetime.now(JST).strftime("%Y-%m-%d")
MODEL              = "claude-haiku-4-5-20251001"

# 監視設定
POLL_INTERVAL_SEC  = 15    # 価格取得間隔（秒）
WATCH_START_HOUR   = 9     # 監視開始時刻
WATCH_START_MIN    = 0
AI_JUDGE_MIN       = 5     # AI判断開始（9:05）
WATCH_END_MIN      = 12    # 監視終了（9:12）


# ══════════════════════════════════════════════
# エントリータイミング判断
# ══════════════════════════════════════════════
# 【バックテスト根拠】2026-03-10〜03-25（47件・BUY判定）
# ─────────────────────────────────────────────
# WEAK日（12件）:
#   寄り付き→終値 平均 +1.49%
#   上昇継続: 6件(50%) / 反落: 0件 / 終日下落: 0件
#   → 寄り付き成行が最適。待っても乗れるが逃す心配なし。
#
# NORMAL日（24件）:
#   寄り付き→終値 平均 -0.78%
#   上昇継続: 2件(8%) / 反落: 2件 / 終日下落: 10件(42%)
#   → 9:05以降に上昇確認してからエントリー。
#     上昇していなければ見送りが正解。
#
# STRONG日:
#   サンプル少数のため NORMAL日と同等に扱う（暫定）
#   → 1ヶ月データ蓄積後に再評価予定
#
# ※ 要再評価タイミング: 約2026-04-25（データ100件超）
# ─────────────────────────────────────────────

def decide_timing(condition, judgment):
    """地合い・判定からエントリータイミング戦略を返す。

    戻り値:
        {
          "style":        "OPEN_MARKET" | "WAIT_CONFIRM" | "SKIP",
          "ai_trigger_min": AI判断を起動する経過分数,
          "confirm_threshold_pct": 上昇確認に必要な前日比（%）,
          "description":  表示用テキスト,
        }
    """
    if condition == "PANIC":
        # PANIC日: 全見送り
        # 【根拠】PANIC日 BUY avg -0.27%、終日下落多数
        return {
            "style":                 "SKIP",
            "ai_trigger_min":        None,
            "confirm_threshold_pct": None,
            "description":           "🚨 PANIC日 → 全見送り",
        }

    if condition == "WEAK":
        # WEAK日: 寄り付き成行（8:55までに注文）
        # 【根拠】上昇継続50%・反落0件・終日下落0件（12件）
        # ※ 要再評価: 2026-04-25頃（WEAK日サンプル30件超になったら）
        return {
            "style":                 "OPEN_MARKET",
            "ai_trigger_min":        3,   # 9:03に最終確認
            "confirm_threshold_pct": 0.0, # プラスならOK
            "description":           "⚡ WEAK日 → 寄り付き成行推奨（上昇継続率50%・反落ゼロ）",
        }

    if condition in ("NORMAL", "STRONG"):
        # NORMAL/STRONG日: 9:05以降に上昇確認後エントリー
        # 【根拠】NORMAL日は終日下落42%。上昇確認なしの寄り付き買いは損失リスク大。
        # STRONG日はサンプル不足のため暫定的にNORMALと同扱い。
        # ※ 要再評価: 2026-04-25頃（NORMAL日サンプル50件超になったら）
        threshold = 0.3 if condition == "STRONG" else 0.5
        return {
            "style":                 "WAIT_CONFIRM",
            "ai_trigger_min":        5,        # 9:05にAI判断
            "confirm_threshold_pct": threshold, # この%以上の上昇で買い確認
            "description":           f"⏳ {condition}日 → 9:05以降 前日比+{threshold}%以上を確認後エントリー（終日下落42%のため慎重に）",
        }

    # UNKNOWN など
    return {
        "style":                 "WAIT_CONFIRM",
        "ai_trigger_min":        5,
        "confirm_threshold_pct": 0.5,
        "description":           "❓ 地合い不明 → 9:05以降 上昇確認後エントリー",
    }


# ══════════════════════════════════════════════
# 価格取得（Yahoo Finance v8 API）
# ══════════════════════════════════════════════
def fetch_realtime_price(code):
    """Yahoo Finance v8 APIから直近の株価を取得する。
    遅延は約15〜30秒。立花証券API接続後はそちらに差し替え予定。
    戻り値: {"price": 現在値, "prev_close": 前日終値, "change_pct": 前日比%, "volume": 出来高}
    """
    url     = f"https://query1.finance.yahoo.com/v8/finance/chart/{code}.T"
    params  = {"interval": "1m", "range": "1d"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=8)
        if resp.status_code != 200:
            return None

        data   = resp.json()
        result = data["chart"]["result"][0]
        meta   = result["meta"]

        price      = meta.get("regularMarketPrice")
        prev_close = meta.get("previousClose") or meta.get("chartPreviousClose")

        if not price or not prev_close:
            return None

        change_pct = round((price - prev_close) / prev_close * 100, 2)

        # 直近の出来高（分足の合計）
        volumes   = result["indicators"]["quote"][0].get("volume", [])
        vol_total = sum(v for v in volumes if v) if volumes else 0

        return {
            "price":      price,
            "prev_close": prev_close,
            "change_pct": change_pct,
            "volume":     vol_total,
        }
    except Exception:
        return None


# ══════════════════════════════════════════════
# 候補銘柄の読み込み
# ══════════════════════════════════════════════
def load_candidates():
    """本日のBUY/CAUTION候補を読み込む"""
    if not os.path.exists(CANDIDATES_LOG_CSV):
        return [], "UNKNOWN"

    df        = pd.read_csv(CANDIDATES_LOG_CSV, encoding="utf-8-sig")
    today_df  = df[df["date"] == TODAY]
    condition = today_df["condition"].iloc[0] if not today_df.empty else "UNKNOWN"
    targets   = today_df[today_df["judgment"].isin(["BUY", "CAUTION"])].copy()

    candidates = []
    for _, row in targets.sort_values("score", ascending=False).iterrows():
        candidates.append({
            "code":     str(row["code"]),
            "name":     str(row["name"]),
            "strategy": str(row["strategy"]),
            "score":    float(row["score"]),
            "judgment": str(row["judgment"]),
            "reason":   str(row["reason"]),
        })
    return candidates, condition


def load_market_info():
    """morning_log.csvから地合い情報を読み込む"""
    if not os.path.exists(MORNING_LOG_CSV):
        return {}
    try:
        df    = pd.read_csv(MORNING_LOG_CSV, encoding="utf-8-sig")
        today = df[df["date"] == TODAY]
        if not today.empty:
            return today.iloc[0].to_dict()
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════
# AI買い価格判断
# ══════════════════════════════════════════════
def ask_claude_entry(candidate, price_history, condition, market_info, timing=None):
    """5分間の価格推移をClaudeに渡し、買い価格・戦略を判断させる。"""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # 価格推移をテキスト化
    history_text = "\n".join([
        f"  {r['time']}  {r['price']:,.0f}円  ({r['change_pct']:+.2f}%)  "
        f"出来高:{r['volume']:,}株  勢い:{r['momentum']}"
        for r in price_history
    ])

    first = price_history[0]  if price_history else {}
    last  = price_history[-1] if price_history else {}

    prompt = f"""あなたは日本株のデイトレード補助AIです。
寄り付き後の価格推移を分析し、エントリー判断をしてください。

## 銘柄情報
- コード: {candidate['code']}  銘柄名: {candidate['name']}
- 戦略: {candidate['strategy']}  スコア: {candidate['score']}
- スキャン判定: {candidate['judgment']}  理由: {candidate['reason']}
- 前日終値: {first.get('prev_close', '不明'):,.0f}円

## 今朝の地合い
- 地合い: {condition}
- 日経先物: {market_info.get('nikkei_change', '不明')}%  ドル円: {market_info.get('usdjpy', '不明')}円

## 寄り付き後の価格推移（{len(price_history)}ポイント）
{history_text}

## 現時点のサマリー
- 寄り付き: {first.get('price', '不明'):,.0f}円 ({first.get('change_pct', 0):+.2f}%)
- 現在値:   {last.get('price', '不明'):,.0f}円 ({last.get('change_pct', 0):+.2f}%)
- 値幅:     {round(last.get('price',0) - first.get('price',0), 0):+,.0f}円
- 出来高:   {last.get('volume', 0):,}株

## 判断基準（戦略{candidate['strategy']}）
- 戦略A（順張り）: 上昇トレンド確認後にエントリー。+3%指値売り、-5%損切り
- 戦略B（逆張り）: 下落一服・反転確認後にエントリー。+3%指値売り、-3%損切り

## 本日の推奨タイミング戦略（バックテスト根拠）
{timing.get('description', '不明') if timing else '不明'}
{"- WEAK日は寄り付き成行推奨: 上昇継続50%・反落ゼロ（12件）" if condition == "WEAK" else ""}
{"- NORMAL日は終日下落42%のため、上昇確認なしの場合は見送りを優先すること" if condition in ("NORMAL","STRONG") else ""}

## お願い
以下をJSON形式で回答してください：
{{
  "判断": "買い実行" or "見送り" or "様子見（継続監視）",
  "推奨買い価格": 数値（円）or null,
  "指値売り価格": 数値（円）or null,
  "損切り価格":   数値（円）or null,
  "根拠": "判断の理由（2〜3文）",
  "リスク": "注意すべき点（1文）"
}}

注意: 現在の価格トレンドと出来高の増減を重視して判断してください。"""

    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.content[0].text
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < 2:
                print(f"  ⚠️ APIが混雑中。10秒後にリトライ（{attempt+1}/3）...")
                time.sleep(10)
            else:
                raise


def parse_ai_entry(text):
    """AIの回答をパースしてdictで返す"""
    t = text.strip()
    if "```" in t:
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
    try:
        return json.loads(t)
    except Exception:
        return {"判断": "解析失敗", "根拠": text[:200]}


# ══════════════════════════════════════════════
# リアルタイム監視メインループ
# ══════════════════════════════════════════════
def watch_loop(candidates, condition, market_info, start_now=False):
    """9:00〜9:12の間、価格を監視してAI判断を行う"""

    # 銘柄ごとにタイミング戦略を決定
    timings    = {c["code"]: decide_timing(condition, c["judgment"]) for c in candidates}
    histories  = {c["code"]: [] for c in candidates}
    ai_results = {c["code"]: None for c in candidates}
    ai_done    = {c["code"]: False for c in candidates}

    print(f"\n{'='*60}")
    print(f"【リアルタイム監視】{len(candidates)}銘柄  地合い:{condition}")
    for c in candidates:
        t = timings[c["code"]]
        print(f"  {c['code']} {c['name']}: {t['description']}")
    print(f"{'='*60}")
    print(f"\n  {'時刻':<8}", end="")
    for c in candidates:
        print(f"  {c['code']}({c['name'][:6]})", end="")
    print()
    print("  " + "─" * (8 + len(candidates) * 18))

    while True:
        now     = datetime.now(JST)
        now_min = now.hour * 60 + now.minute

        market_start = WATCH_START_HOUR * 60 + WATCH_START_MIN
        ai_start     = WATCH_START_HOUR * 60 + AI_JUDGE_MIN
        watch_end    = WATCH_START_HOUR * 60 + WATCH_END_MIN

        # 終了判定
        if now_min >= watch_end:
            print(f"\n  ⏰ 9:{WATCH_END_MIN:02d}になりました。監視を終了します。")
            break

        # 市場開始前は待機
        if not start_now and now_min < market_start:
            wait_sec = (market_start - now_min) * 60 - now.second
            print(f"  市場開始まで {wait_sec}秒待機中... ({now.strftime('%H:%M:%S')})", end="\r")
            time.sleep(5)
            continue

        # 価格取得・表示
        time_str = now.strftime("%H:%M:%S")
        print(f"  {time_str}", end="")

        for c in candidates:
            code   = c["code"]
            result = fetch_realtime_price(code)

            if result:
                price      = result["price"]
                change_pct = result["change_pct"]
                volume     = result["volume"]

                # 勢い判定（前回からの変化）
                prev_history = histories[code]
                if prev_history:
                    prev_price = prev_history[-1]["price"]
                    momentum   = "↑↑" if price > prev_price * 1.002 else \
                                 "↑"  if price > prev_price else \
                                 "↓↓" if price < prev_price * 0.998 else \
                                 "↓"  if price < prev_price else "→"
                else:
                    momentum = "→"

                histories[code].append({
                    "time":       time_str,
                    "price":      price,
                    "prev_close": result["prev_close"],
                    "change_pct": change_pct,
                    "volume":     volume,
                    "momentum":   momentum,
                })

                chg_icon = "📈" if change_pct >= 0 else "📉"
                print(f"  {chg_icon}{price:>7,.0f}円({change_pct:>+.1f}%)", end="")
            else:
                print(f"  {'取得失敗':>14}", end="")

        print()  # 改行

        # タイミング戦略に応じてAI判断を実施
        for c in candidates:
            code   = c["code"]
            timing = timings[code]

            if ai_done[code]:
                continue

            # SKIPは即判断
            if timing["style"] == "SKIP":
                ai_results[code] = {"判断": "見送り", "根拠": timing["description"]}
                ai_done[code]    = True
                continue

            # OPEN_MARKET: ai_trigger_min 経過後に上昇確認 → AI判断
            # WAIT_CONFIRM: ai_trigger_min 経過後に閾値確認 → AI判断
            trigger_min = timing.get("ai_trigger_min", AI_JUDGE_MIN)
            threshold   = timing.get("confirm_threshold_pct", 0.0)

            if now_min < WATCH_START_HOUR * 60 + trigger_min:
                continue
            if len(histories[code]) < 3:
                continue

            # 上昇確認チェック（WAIT_CONFIRMの場合）
            latest_chg = histories[code][-1]["change_pct"] if histories[code] else 0
            if timing["style"] == "WAIT_CONFIRM" and latest_chg < threshold:
                # 上昇確認できなかった場合、まだ監視継続（終了時間まで待つ）
                if now_min < WATCH_START_HOUR * 60 + WATCH_END_MIN - 1:
                    continue
                # 終了間際も確認できなければ自動見送り
                ai_results[code] = {
                    "判断":   "見送り",
                    "根拠":   f"9:{WATCH_END_MIN:02d}まで前日比+{threshold}%の上昇確認できず。"
                              f"現在{latest_chg:+.1f}%。NORMAL日終日下落リスクのため見送り。",
                }
                ai_done[code] = True
                _print_ai_result(c, ai_results[code])
                continue

            # AI判断実行
            print(f"\n  🤖 AI判断中: {code} {c['name']}  "
                  f"（{timing['style']} / 現在{latest_chg:+.1f}%）...")
            try:
                raw    = ask_claude_entry(c, histories[code], condition, market_info, timing)
                result = parse_ai_entry(raw)
                ai_results[code] = result
                ai_done[code]    = True
                _print_ai_result(c, result)
            except Exception as e:
                print(f"  ❌ AI判断エラー: {e}")
                ai_done[code] = True

        time.sleep(POLL_INTERVAL_SEC)

    # 最終サマリー表示
    _print_final_summary(candidates, ai_results, histories)


def _print_ai_result(candidate, result):
    """AI判断結果を表示"""
    judgment = result.get("判断", "不明")
    price    = result.get("推奨買い価格")
    sell     = result.get("指値売り価格")
    stop     = result.get("損切り価格")
    reason   = result.get("根拠", "")
    risk     = result.get("リスク", "")

    icon = {"買い実行": "🟢", "見送り": "🔴", "様子見（継続監視）": "🟡"}.get(judgment, "⚪")

    print(f"\n  {'─'*56}")
    print(f"  {icon} AI判断: {judgment}  [{candidate['code']} {candidate['name']}]")
    if price:
        print(f"     推奨買い価格: {price:,.0f}円")
    if sell:
        print(f"     指値売り   : {sell:,.0f}円（+{round((sell/price-1)*100,1)}%）" if price else f"     指値売り: {sell:,.0f}円")
    if stop:
        print(f"     損切り     : {stop:,.0f}円（{round((stop/price-1)*100,1)}%）" if price else f"     損切り: {stop:,.0f}円")
    print(f"     根拠: {reason}")
    if risk:
        print(f"     リスク: {risk}")
    print(f"  {'─'*56}\n")


def _print_final_summary(candidates, ai_results, histories):
    """全銘柄の最終サマリーを表示"""
    print(f"\n{'='*60}")
    print(f"【最終サマリー】")
    print(f"{'='*60}")

    for c in candidates:
        code   = c["code"]
        result = ai_results.get(code)
        hist   = histories.get(code, [])

        if not result:
            print(f"\n  {code} {c['name']}: AI判断なし（データ不足）")
            continue

        judgment = result.get("判断", "不明")
        price    = result.get("推奨買い価格")
        sell     = result.get("指値売り価格")
        stop     = result.get("損切り価格")
        icon     = {"買い実行": "🟢", "見送り": "🔴", "様子見（継続監視）": "🟡"}.get(judgment, "⚪")

        first_price = hist[0]["price"] if hist else None
        last_price  = hist[-1]["price"] if hist else None
        last_chg    = hist[-1]["change_pct"] if hist else None

        print(f"\n  {icon} {code} {c['name']}")
        print(f"     判断    : {judgment}")
        if first_price and last_price:
            print(f"     値動き  : {first_price:,.0f}円 → {last_price:,.0f}円  ({last_chg:+.2f}%)")
        if price:
            print(f"     買い価格: {price:,.0f}円  売り:{sell:,.0f}円  損切:{stop:,.0f}円")

    print(f"\n  ⚠️  AIの判断は参考情報です。最終判断はご自身で行ってください。")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════
def main(start_now=False):
    print(f"=== 📡 リアルタイム監視（{TODAY}）===\n")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY が設定されていません")
        return

    candidates, condition = load_candidates()
    if not candidates:
        print("  本日のBUY/CAUTION候補がありません → 監視を終了します")
        return

    if condition == "PANIC":
        print("  🚨 地合いPANIC → 監視を終了します")
        return

    market_info = load_market_info()

    print(f"  監視対象: {len(candidates)}銘柄  地合い: {condition}")
    for c in candidates:
        print(f"    {c['judgment']:<8} {c['code']} {c['name']}（スコア{c['score']:.1f}）")

    now = datetime.now(JST)
    if not start_now and now.hour < WATCH_START_HOUR:
        target = now.replace(hour=WATCH_START_HOUR, minute=WATCH_START_MIN,
                             second=0, microsecond=0)
        wait   = (target - now).seconds
        print(f"\n  ⏰ 9:00まで {wait//60}分{wait%60}秒 待機します...")

    watch_loop(candidates, condition, market_info, start_now=start_now)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", action="store_true", help="即時開始（テスト用）")
    args = parser.parse_args()
    main(start_now=args.now)
