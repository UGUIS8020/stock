"""
market_watch.py - 寄り付き後リアルタイム監視 + ルールベース発注

scan_morning.py → market_watch.py の順で自動起動される。

9:00〜9:30の間、候補銘柄の価格を15秒ごとに取得・表示し、
タイミング戦略（地合い・判定）に応じてルールベースで発注する。

実行方法:
    python market_watch.py          # 9:00まで待機して自動開始
    python market_watch.py --now    # 即時開始（テスト用）

前提:
    - scan_morning.py を先に実行済みであること
"""

import os
import re
import sys
import json
import time
import argparse
import requests
import urllib3
import pandas as pd
import tachibana_order
import db
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

_BASE_DIR            = Path(__file__).parent
CANDIDATES_LOG_CSV   = str(_BASE_DIR / "out" / "candidates_log.csv")
MORNING_LOG_CSV      = str(_BASE_DIR / "out" / "morning_log.csv")
TACHIBANA_LOGIN_FILE = str(_BASE_DIR / "tachibana_login_response.json")
OBSERVE_LOG_CSV      = str(_BASE_DIR / "out" / "observe_log.csv")
JST                = timezone(timedelta(hours=9))
TODAY              = datetime.now(JST).strftime("%Y-%m-%d")
# 監視設定
POLL_INTERVAL_SEC  = 15    # 価格取得間隔（秒）
WATCH_START_HOUR   = 9     # 監視開始時刻
WATCH_START_MIN    = 0
AI_JUDGE_MIN       = 5     # 発注判断開始（9:05）
WATCH_END_MIN      = 30    # 監視終了（9:30）

# TP/SL設定
TP_PCT = 0.03   # 利確 +3%
SL_PCT = 0.01   # 損切 -1%（最適化結果: SL=-1%でSharpe14倍改善）

# 監視銘柄数の上限（絞り込み）
MAX_WATCH_A = 10   # 戦略A: scoreで降順

# 発注設定
DEFAULT_SHARES   = 100      # 高値株のデフォルト株数
CHEAP_THRESHOLD  = 1_000   # この価格未満は最大株数モード
MAX_ORDER_AMOUNT = 100_000  # 安い株の1発注上限額


def calc_shares(price):
    """価格に応じた発注株数を返す（100株単位）。
    1,000円未満の安い株は100,000円以内で買えるだけ。"""
    if price and price < CHEAP_THRESHOLD:
        lots = int(MAX_ORDER_AMOUNT / price / 100)
        return max(lots, 1) * 100
    return DEFAULT_SHARES


# ══════════════════════════════════════════════
# エントリータイミング判断
# ══════════════════════════════════════════════
# 【戦略A】バックテスト根拠 2026-03-10〜03-25（47件・BUY判定）
# ─────────────────────────────────────────────
# WEAK日（12件）:
#   寄り付き→終値 平均 +1.49%
#   → 寄り付き成行が最適。
#
# NORMAL日（24件）:
#   寄り付き→終値 平均 -0.78%
#   → 9:05以降に上昇確認してからエントリー。
#     上昇していなければ見送りが正解。
# ─────────────────────────────────────────────

def decide_timing(condition, judgment, ai_recommendation="", strategy="A"):
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
            "description":           "PANIC日 → 全見送り",
        }

    # WEAK日のCAUTION → 9:05以降に成行買い
    # 【根拠】WEAK×CAUTION N=44 avg+0.97% Sharpe=0.43（SL=-1%）が全セグメント中最高
    #         旧ロジック（OBSERVE）はこの利益を全て取り逃していた
    if condition == "WEAK" and judgment == "CAUTION":
        return {
            "style":                 "WAIT_CONFIRM",
            "ai_trigger_min":        5,
            "confirm_threshold_pct": 0.0,
            "description":           "WEAK日CAUTION → 9:05以降 成行買い（avg+0.97%・N=44）",
        }

    if condition == "WEAK":
        # WEAK日BUY: 寄り付き成行（8:55までに注文）
        # 【根拠】上昇継続50%・反落0件・終日下落0件（12件）
        # ※ 要再評価: 2026-04-25頃（WEAK日サンプル30件超になったら）
        return {
            "style":                 "OPEN_MARKET",
            "ai_trigger_min":        3,
            "confirm_threshold_pct": 0.0,
            "description":           "WEAK日BUY → 寄り付き成行推奨（上昇継続率50%・反落ゼロ）",
        }

    if condition in ("NORMAL", "STRONG"):
        if judgment == "CAUTION":
            # STRONG日は全銘柄CAUTION（BUYなし）。0.3%で取りこぼしを防ぐ
            # NORMAL日CAUTIONは1.0%上昇確認（有効ゾーン外の慎重銘柄）
            threshold = 0.3 if condition == "STRONG" else 1.0
            return {
                "style":                 "WAIT_CONFIRM",
                "ai_trigger_min":        5,
                "confirm_threshold_pct": threshold,
                "description":           f"{condition}日CAUTION → 9:05以降 前日比+{threshold}%以上を確認後エントリー",
            }
        else:
            # NORMAL日BUY（最良ゾーン銘柄）
            return {
                "style":                 "WAIT_CONFIRM",
                "ai_trigger_min":        5,
                "confirm_threshold_pct": 0.5,
                "description":           f"{condition}日BUY → 9:05以降 前日比+0.5%以上を確認後エントリー",
            }

    # UNKNOWN など
    return {
        "style":                 "WAIT_CONFIRM",
        "ai_trigger_min":        5,
        "confirm_threshold_pct": 0.5,
        "description":           "地合い不明 → 9:05以降 上昇確認後エントリー",
    }


def is_v_recovery(history, dip_threshold=-0.3, recovery_min=0.5):
    """寄り付きダウン→V字回復パターンを検出する。
    STRONG日BUYで一時的にマイナスになっても回復中の銘柄を拾うために使用。

    条件:
      - 過去データのうち最安値が dip_threshold% 以下（一度下落した）
      - 直近の前日比 - 最安値前日比 >= recovery_min%（V字で回復中）
    """
    if len(history) < 3:
        return False
    changes  = [h["change_pct"] for h in history]
    min_chg  = min(changes[:-1])   # 最終点を除いた最安値
    latest   = changes[-1]
    return min_chg <= dip_threshold and (latest - min_chg) >= recovery_min


# ══════════════════════════════════════════════
# Tachibana API（立花証券）リアルタイム価格取得
# ══════════════════════════════════════════════
def load_tachibana_url():
    """tachibana_login_response.json から株価URLを読み込む。失敗時はNone。"""
    try:
        with open(TACHIBANA_LOGIN_FILE, encoding="utf-8") as f:
            data = json.load(f)
        url = data.get("sUrlPrice", "")
        return url if url else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def fetch_tachibana_prices(url_price, codes):
    """Tachibana APIから複数銘柄の現在値・気配値・出来高を一括取得。
    戻り値: {code: {"price", "ask", "bid", "ask_vol", "bid_vol", "prev_close", "volume"}}
    APIの上限120件/回をバッチ分割して全件取得する。
    """
    if not url_price or not codes:
        return {}
    batch_size = 120
    http = urllib3.PoolManager()
    quotes = {}
    p_no_base = int(time.time()) % 100000
    for i, start in enumerate(range(0, len(codes), batch_size)):
        batch = codes[start:start + batch_size]
        code_list = ",".join(str(c) for c in batch)
        t = datetime.now(JST)
        p_sd_date = (f"{t.year}.{t.month:02}.{t.day:02}"
                     f"-{t.hour:02}:{t.minute:02}:{t.second:02}"
                     f".{t.microsecond // 1000:03}")
        params = (
            "{"
            f'"p_no":"{p_no_base + i}",'
            f'"p_sd_date":"{p_sd_date}",'
            '"sCLMID":"CLMMfdsGetMarketPrice",'
            f'"sTargetIssueCode":"{code_list}",'
            '"sTargetColumn":"pDPP,pPRP,pQAP,pQBP,pAV,pBV,pDV",'
            '"sJsonOfmt":"5"'
            "}"
        )
        try:
            resp = http.request("GET", url_price + "?" + params,
                                timeout=urllib3.Timeout(connect=3, read=5))
            text = resp.data.decode("shift-jis", errors="ignore")
            result = json.loads(text)
            for item in result.get("aCLMMfdsMarketPrice", []):
                code = item.get("sIssueCode", "").strip('"')
                if not code or code == "stock_code":
                    continue
                def _f(key):
                    v = item.get(key, "")
                    if isinstance(v, str):
                        v = v.strip('"')
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        return None
                price = _f("pDPP") or _f("pPRP")
                quotes[code] = {
                    "price":      price,
                    "ask":        _f("pQAP"),
                    "bid":        _f("pQBP"),
                    "ask_vol":    _f("pAV"),
                    "bid_vol":    _f("pBV"),
                    "prev_close": _f("pPRP"),
                    "volume":     int(_f("pDV") or 0),
                }
        except Exception:
            pass
    return quotes


# ══════════════════════════════════════════════
# 価格取得（Yahoo Finance v8 API・フォールバック）
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
def _extract_rb_score(reason):
    """reason文字列から RBスコア（例: '(9点)'）を抽出する。"""
    m = re.search(r'\((\d+)点\)', str(reason))
    return int(m.group(1)) if m else 0


def load_candidates():
    """本日のBUY/CAUTION候補を読み込み、MAX_WATCH_A/B件に絞り込む。"""
    today_df = pd.DataFrame()
    try:
        import db as _db, sqlite3
        conn = sqlite3.connect(_db.DB_PATH)
        today_df = pd.read_sql(
            f'SELECT date,strategy,code,condition,name,score,ratio,judgment,reason '
            f'FROM candidates_log WHERE date="{TODAY}"', conn)
        conn.close()
    except Exception:
        pass

    if today_df.empty:
        print("  ⚠️  candidates_log にデータがありません。scan_morning.py を先に実行してください。")
        return [], "UNKNOWN"

    condition = today_df["condition"].iloc[0] if not today_df.empty else "UNKNOWN"
    targets   = today_df[today_df["judgment"].isin(["BUY", "CAUTION"])].copy()
    original  = len(targets)

    # ── 戦略A: scoreで降順、上限MAX_WATCH_A件（戦略Bはclosing_watchに移管）
    df_a = (targets[targets["strategy"] == "A"]
            .sort_values("score", ascending=False)
            .head(MAX_WATCH_A))

    targets = df_a.drop_duplicates(subset=["code"])

    if len(targets) < original:
        print(f"  📌 監視対象を絞り込み: {original}件 → {len(targets)}件（A:{len(df_a)}件）")

    candidates = []
    for _, row in targets.iterrows():
        candidates.append({
            "code":              str(row["code"]),
            "name":              str(row["name"]),
            "strategy":          str(row["strategy"]),
            "score":             float(row["score"]),
            "ratio":             float(row["ratio"]) if "ratio" in row and pd.notna(row["ratio"]) else 0.0,
            "today_rise":        float(row["today_rise"]) if "today_rise" in row and pd.notna(row["today_rise"]) else 0.0,
            "judgment":          str(row["judgment"]),
            "reason":            str(row["reason"]),
            "ai_recommendation": str(row["ai_recommendation"]) if "ai_recommendation" in row and pd.notna(row["ai_recommendation"]) else "",
        })
    return candidates, condition


def load_market_info():
    """morning_log テーブルから本日の地合い情報を読み込む"""
    try:
        import db as _db, sqlite3
        conn = sqlite3.connect(_db.DB_PATH)
        df = pd.read_sql(
            'SELECT * FROM morning_log WHERE date=?', conn, params=[TODAY])
        conn.close()
        if not df.empty:
            return df.iloc[0].to_dict()
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════
# 異常検知（ルールベース）
# ══════════════════════════════════════════════
def check_anomaly(candidate):
    """異常な銘柄を検出してメッセージを返す。正常ならNone。"""
    price      = candidate.get("price") or 0
    ratio      = candidate.get("ratio") or 0
    today_rise = candidate.get("today_rise") or 0

    if 0 < price < 200:
        return f"株価{price:.0f}円（極端に安い・上場廃止リスク）"
    if ratio >= 30:
        return f"出来高比率{ratio:.0f}倍（薄商い銘柄の異常値の可能性）"
    if today_rise >= 15:
        return f"前日+{today_rise:.1f}%急騰済み（高値追いリスク）"
    if today_rise <= -15:
        return f"前日{today_rise:.1f}%暴落（業績悪化・売り継続リスク）"
    return None


def ask_claude_entry(candidate, price_history, condition, market_info, timing=None):
    """互換性のためのスタブ（AI判断は無効化済み）"""
    raise NotImplementedError("AI判断は無効化されています")

def confirm_and_order(candidate, price, url_request):
    """
    発注確認プロンプトを表示し、ユーザーの yes/no に応じて Tachibana API で発注する。
    """
    code      = candidate["code"]
    name      = candidate["name"]
    rec_price = price
    sell_p    = round(price * (1 + TP_PCT)) if price else None
    stop_p    = round(price * (1 - SL_PCT)) if price else None

    shares    = calc_shares(rec_price or price)
    estimated = int((rec_price or price or 0) * shares)

    # 買い余力確認
    buying_power = None
    if url_request:
        buying_power = tachibana_order.get_buying_power(url_request)

    print(f"\n\a{'━'*56}")
    print(f"  🔔🔔🔔  発注確認  [{code}] {name}  🔔🔔🔔")
    print(f"{'━'*56}")
    print(f"     現在値   : {price:>8,.0f}円" if price else "     現在値   : 取得中")
    print(f"     推奨買い : {rec_price:>8,.0f}円" if rec_price else "")
    print(f"     指値売り : {sell_p:>8,.0f}円" if sell_p else "")
    print(f"     損切り   : {stop_p:>8,.0f}円" if stop_p else "")
    print(f"     発注株数 : {shares}株  （変更: 数字を入力）")
    print(f"     概算金額 : {estimated:>8,.0f}円")
    if buying_power is not None:
        print(f"     買余力   : {buying_power:>8,.0f}円")
        if estimated > buying_power:
            print(f"  ⚠️  買余力不足の可能性があります")
    mode = "本番" if tachibana_order.LIVE_TRADING else "モック（実発注なし）"
    print(f"     モード   : {mode}")
    print(f"{'━'*56}")

    while True:
        ans = input(">>> [y=発注 / n=見送り / 数字=株数変更] : ").strip().lower()

        if ans == "n" or ans == "":
            print(f"  ↩️  {code} 見送りました。")
            return

        if ans.isdigit():
            shares    = int(ans)
            estimated = int(rec_price * shares) if rec_price else 0
            print(f"     株数を {shares}株（概算 {estimated:,.0f}円）に変更しました。")
            print(f"  [y=発注 / n=見送り] > ", end="", flush=True)
            continue

        if ans == "y":
            if not url_request:
                print(f"  ❌ 業務URLが取得できません。再ログインしてください。")
                return
            print(f"  📤 発注中... {code} {shares}株 成行買い")
            mkt_code = db.get_market_code_db(code)
            result = tachibana_order.place_buy_order(url_request, code, shares, market_code=mkt_code)
            if result["success"]:
                print(f"  ✅ {result['message']}")
                buy_px = int(rec_price or price or 0)
                if buy_px > 0:
                    tachibana_order.save_position(code, name, shares, buy_px,
                                                  strategy="A", tp_pct=TP_PCT, sl_pct=SL_PCT)
            else:
                print(f"  ❌ 発注失敗: {result['message']}")
                print(f"     APIレスポンス: {result['raw']}")
            return



# ══════════════════════════════════════════════
# リアルタイム監視メインループ
# ══════════════════════════════════════════════
def watch_loop(candidates, condition, market_info, start_now=False, url_price=None, url_request=None):
    """9:00〜9:30の間、価格を監視してルールベースで発注する"""

    # 銘柄ごとにタイミング戦略を決定
    timings   = {c["code"]: decide_timing(condition, c["judgment"], "", c.get("strategy", "A")) for c in candidates}
    histories = {c["code"]: [] for c in candidates}
    ordered   = {c["code"]: False for c in candidates}   # 発注済みフラグ
    results   = {c["code"]: "監視中" for c in candidates}

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

        market_start  = WATCH_START_HOUR * 60 + WATCH_START_MIN
        order_start   = WATCH_START_HOUR * 60 + AI_JUDGE_MIN
        watch_end     = WATCH_START_HOUR * 60 + WATCH_END_MIN

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

        # Tachibana API で全候補を一括取得（失敗時は空dict → Yahoo Financeへ）
        tachibana_batch = {}
        if url_price:
            try:
                tachibana_batch = fetch_tachibana_prices(
                    url_price, [c["code"] for c in candidates]
                )
            except Exception:
                tachibana_batch = {}

        for c in candidates:
            code = c["code"]

            # Tachibana 優先 → Yahoo Finance フォールバック
            tq = tachibana_batch.get(str(code))
            if tq and tq.get("price"):
                prev_close = tq["prev_close"] or tq["price"]
                change_pct = (
                    round((tq["price"] - prev_close) / prev_close * 100, 2)
                    if prev_close else 0
                )
                result = {
                    "price":      tq["price"],
                    "prev_close": prev_close,
                    "change_pct": change_pct,
                    "volume":     tq.get("volume") or 0,
                    "ask":        tq.get("ask"),
                    "bid":        tq.get("bid"),
                }
            else:
                yf_result = fetch_realtime_price(code)
                result = yf_result  # None の場合もある

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
                    "ask":        result.get("ask"),
                    "bid":        result.get("bid"),
                })

                chg_icon = "📈" if change_pct >= 0 else "📉"
                src_mark = "📡" if (tq and tq.get("price")) else "🌐"
                print(f"  {src_mark}{chg_icon}{price:>7,.0f}円({change_pct:>+.1f}%)", end="")
            else:
                print(f"  {'取得失敗':>14}", end="")

        print()  # 改行

        # タイミング戦略に応じてルールベースで発注判断
        for c in candidates:
            code   = c["code"]
            timing = timings[code]

            if ordered[code]:
                continue

            # SKIP: PANIC日など全見送り
            if timing["style"] == "SKIP":
                ordered[code]  = True
                results[code]  = "見送り(PANIC)"
                print(f"\n  ⏭️  {code} {c['name']}: PANIC日 → 見送り")
                continue

            # OBSERVE: データ蓄積のみ（発注しない）
            if timing["style"] == "OBSERVE":
                continue

            trigger_min = timing.get("ai_trigger_min", AI_JUDGE_MIN)
            threshold   = timing.get("confirm_threshold_pct", 0.0)

            if now_min < WATCH_START_HOUR * 60 + trigger_min:
                continue
            if len(histories[code]) < 2:
                continue

            latest_chg = histories[code][-1]["change_pct"] if histories[code] else 0
            latest_px  = histories[code][-1]["price"]       if histories[code] else None

            # 異常検知 → 強制見送り
            anomaly = check_anomaly(c)
            if anomaly:
                ordered[code] = True
                results[code] = f"見送り(異常検知)"
                print(f"\n  🚫 {code} {c['name']}: 異常検知 → 強制見送り（{anomaly}）")
                continue

            # OPEN_MARKET: トリガー時刻到達で即発注
            if timing["style"] == "OPEN_MARKET":
                ordered[code] = True
                results[code] = "発注"
                print(f"\n  🟢 {code} {c['name']}: {timing['description']}")
                confirm_and_order(c, latest_px, url_request)
                continue

            # WAIT_CONFIRM: 上昇閾値を超えたら発注
            if timing["style"] == "WAIT_CONFIRM":
                if latest_chg >= threshold:
                    ordered[code] = True
                    results[code] = "発注"
                    print(f"\n  🟢 {code} {c['name']}: 前日比{latest_chg:+.1f}% ≥ 閾値{threshold:+.1f}% → 発注")
                    confirm_and_order(c, latest_px, url_request)
                elif now_min >= WATCH_START_HOUR * 60 + WATCH_END_MIN - 1:
                    ordered[code] = True
                    results[code] = "見送り(上昇未確認)"
                    print(f"\n  🔴 {code} {c['name']}: 9:{WATCH_END_MIN:02d}まで前日比+{threshold}%未達"
                          f"（現在{latest_chg:+.1f}%）→ 見送り")
                continue

        time.sleep(POLL_INTERVAL_SEC)

    # 観察データ保存 → 最終サマリー表示
    _save_observe_log(candidates, timings, histories)
    _print_final_summary(candidates, results, histories)



def _save_observe_log(candidates, timings, histories):
    """OBSERVEスタイルの銘柄の価格推移をCSVに保存する（後日検証用）"""
    observe_candidates = [
        c for c in candidates if timings[c["code"]]["style"] == "OBSERVE"
    ]
    if not observe_candidates:
        return

    rows = []
    for c in observe_candidates:
        code = c["code"]
        for h in histories.get(code, []):
            rows.append({
                "date":       TODAY,
                "code":       code,
                "name":       c["name"],
                "condition":  "WEAK",
                "judgment":   c["judgment"],
                "score":      c["score"],
                "time":       h["time"],
                "price":      h["price"],
                "change_pct": h["change_pct"],
                "volume":     h["volume"],
                "momentum":   h["momentum"],
            })

    if not rows:
        return

    df     = pd.DataFrame(rows)
    exists = os.path.exists(OBSERVE_LOG_CSV)
    df.to_csv(OBSERVE_LOG_CSV, mode="a", header=not exists,
              index=False, encoding="utf-8-sig")
    print(f"\n  💾 観察データ保存: {OBSERVE_LOG_CSV}（{len(observe_candidates)}銘柄 × {len(rows)//max(len(observe_candidates),1)}ポイント）")


def _print_final_summary(candidates, results, histories):
    """全銘柄の最終サマリーを表示"""
    print(f"\n{'='*60}")
    print(f"【最終サマリー】")
    print(f"{'='*60}")

    for c in candidates:
        code        = c["code"]
        judgment    = results.get(code, "監視中")
        hist        = histories.get(code, [])
        first_price = hist[0]["price"]  if hist else None
        last_price  = hist[-1]["price"] if hist else None
        last_chg    = hist[-1]["change_pct"] if hist else None

        icon = "🟢" if judgment == "発注" else ("🔵" if "観察" in judgment else "🔴")
        print(f"\n  {icon} {code} {c['name']}")
        print(f"     結果    : {judgment}")
        if first_price and last_price:
            print(f"     値動き  : {first_price:,.0f}円 → {last_price:,.0f}円  ({last_chg:+.2f}%)")
        if judgment == "発注" and last_price:
            print(f"     TP目標  : {round(last_price*(1+TP_PCT)):,.0f}円（+{TP_PCT*100:.0f}%）"
                  f"  SL: {round(last_price*(1-SL_PCT)):,.0f}円（-{SL_PCT*100:.0f}%）")

    print(f"{'='*60}\n")


# ══════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════
def main(start_now=False):
    print(f"=== 📡 リアルタイム監視（{TODAY}）===\n")

    candidates, condition = load_candidates()
    if not candidates:
        print("  本日のBUY/CAUTION候補がありません → 監視を終了します")
        return

    # 念のため重複コードを除去
    seen_codes: set = set()
    candidates = [c for c in candidates if not (c["code"] in seen_codes or seen_codes.add(c["code"]))]

    if condition == "PANIC":
        print("  🚨 地合いPANIC → 監視を終了します")
        return

    market_info = load_market_info()

    url_price   = load_tachibana_url()
    url_request = tachibana_order.load_url_request()
    if url_price:
        print(f"  📡 Tachibana API: リアルタイム価格取得モード（遅延なし）")
    else:
        print(f"  🌐 Yahoo Finance: 遅延価格取得モード（立花APIにログインすると精度向上）")
    mode_str = "本番発注モード" if tachibana_order.LIVE_TRADING else "モックモード（実発注なし）"
    print(f"  💼 発注: {mode_str}  1,000円未満→最大{MAX_ORDER_AMOUNT//10000}万円分 / 1,000円以上→{DEFAULT_SHARES}株")

    print(f"  監視対象: {len(candidates)}銘柄  地合い: {condition}")
    for c in candidates:
        print(f"    {c['judgment']:<8} {c['code']} {c['name']}（スコア{c['score']:.1f}）")

    now = datetime.now(JST)
    if not start_now and now.hour < WATCH_START_HOUR:
        target = now.replace(hour=WATCH_START_HOUR, minute=WATCH_START_MIN,
                             second=0, microsecond=0)
        wait   = (target - now).seconds
        print(f"\n  ⏰ 9:00まで {wait//60}分{wait%60}秒 待機します...")

    watch_loop(candidates, condition, market_info, start_now=start_now,
               url_price=url_price, url_request=url_request)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", action="store_true", help="即時開始（テスト用）")
    args = parser.parse_args()
    main(start_now=args.now)
