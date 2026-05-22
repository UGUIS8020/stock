"""
closing_watch.py - 引け前（14:43〜14:58）戦略B下落銘柄スキャン + 買い判断

【試算根拠（2026-05-16）】
STRONG地合い / -4〜-7% / RB>=3 / 引け前買い:
  WR 69.5%  avg +1.482%  Sharpe +6.01（翌朝寄り買いの2.5倍）
  ※翌朝ギャップアップ平均+0.88%を仕込み段階で確保

実行方法:
    python closing_watch.py       # 14:43まで待機して自動開始
    python closing_watch.py --now # 即時開始（テスト用）

前提:
    - tachibana_login_response.json が存在すること（scan_morning.py 実行後）
    - STRONG地合いの日のみ有効（それ以外は自動終了）
"""

import os
import sys
import json
import time
import argparse
import urllib3
import pandas as pd
import numpy as np
import anthropic
import tachibana_order
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()
import db

JST   = timezone(timedelta(hours=9))
TODAY = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
MODEL = "claude-haiku-4-5-20251001"

CANDIDATES_LOG_CSV   = "out/candidates_log.csv"
MORNING_LOG_CSV      = "out/morning_log.csv"
MARKET_LOG_CSV       = "out/market_log.csv"
CLOSING_LOG_CSV      = "out/closing_log.csv"
TACHIBANA_LOGIN_FILE = "tachibana_login_response.json"

SCAN_START_HOUR    = 14
SCAN_START_MIN     = 30
ORDER_DEADLINE_MIN = 15 * 60 + 15  # 15:15 以降は発注しない

DROP_LO      = -6.0   # 下落下限（GA再検証2026-05-22）
DROP_HI      = -4.0   # 下落上限（GA再検証2026-05-22）
RB_MIN       = 4      # RBスコア最小値（GA再検証2026-05-22）
MIN_VOLUME   = 50_000
MAX_AI_JUDGE = 10     # AI判断する最大件数

DEFAULT_SHARES   = 100
CHEAP_THRESHOLD  = 1_000
MAX_ORDER_AMOUNT = 100_000


def calc_shares(price):
    """価格に応じた発注株数を返す（100株単位）。
    1,000円未満の安い株は100,000円以内で買えるだけ。"""
    if price and price < CHEAP_THRESHOLD:
        lots = int(MAX_ORDER_AMOUNT / price / 100)
        return max(lots, 1) * 100
    return DEFAULT_SHARES

PANIC_NIKKEI  = -2.0;  PANIC_AD  = 0.20
WEAK_NIKKEI   = -1.0;  WEAK_AD   = 0.35
STRONG_NIKKEI =  0.5;  STRONG_AD = 0.60


# ══════════════════════════════════════════════
# テクニカル指標
# ══════════════════════════════════════════════
def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    d  = np.diff(closes)
    ag = np.where(d > 0, d, 0.0)[-period:].mean()
    al = np.where(d < 0, -d, 0.0)[-period:].mean()
    return round(100 - 100 / (1 + ag / al), 1) if al > 0 else 100.0


def calc_rebound_score(hist):
    if len(hist) < 26:
        return 0
    score   = 0
    closes  = pd.to_numeric(hist["Close"], errors="coerce").fillna(0).values
    vol_ser = pd.to_numeric(hist["Volume"], errors="coerce").fillna(0)

    rsi = calc_rsi(closes, 14)
    if rsi is not None:
        if   rsi < 20: score += 4
        elif rsi < 30: score += 3
        elif rsi < 40: score += 2
        elif rsi < 50: score += 1

    if len(hist) >= 21:
        avg20 = vol_ser.iloc[-21:-1].mean()
        vol   = float(vol_ser.iloc[-1])
        if avg20 > 0:
            r = vol / avg20
            if   r >= 3.0: score += 3
            elif r >= 2.0: score += 2
            elif r >= 1.5: score += 1

    ma25 = pd.to_numeric(hist["Close"], errors="coerce").iloc[-25:].mean()
    dev  = (float(hist["Close"].iloc[-1]) / ma25 - 1) * 100 if ma25 > 0 else 0
    if   dev < -15: score += 3
    elif dev < -10: score += 2
    elif dev <  -5: score += 1

    return score


# ══════════════════════════════════════════════
# Tachibana API
# ══════════════════════════════════════════════
def load_tachibana_url():
    try:
        with open(TACHIBANA_LOGIN_FILE, encoding="utf-8") as f:
            data = json.load(f)
        url = data.get("sUrlPrice", "")
        return url if url else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _fetch_price_batch(url_price, codes, http):
    """120件以内のコードリストを一括取得して {code: quote_dict} を返す"""
    code_list = ",".join(str(c) for c in codes)
    t = datetime.now(JST)
    p_sd_date = (f"{t.year}.{t.month:02}.{t.day:02}"
                 f"-{t.hour:02}:{t.minute:02}:{t.second:02}"
                 f".{t.microsecond // 1000:03}")
    params = (
        "{"
        f'"p_no":"{int(time.time()) % 100000}",'
        f'"p_sd_date":"{p_sd_date}",'
        '"sCLMID":"CLMMfdsGetMarketPrice",'
        f'"sTargetIssueCode":"{code_list}",'
        '"sTargetColumn":"pDPP,pPRP,pDV",'
        '"sJsonOfmt":"5"'
        "}"
    )
    try:
        resp = http.request("GET", url_price + "?" + params,
                            timeout=urllib3.Timeout(connect=3, read=8))
        text   = resp.data.decode("shift-jis", errors="ignore")
        result = json.loads(text)
        out    = {}
        for item in result.get("aCLMMfdsMarketPrice", []):
            code = item.get("sIssueCode", "").strip('"')
            if not code:
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
            prev  = _f("pPRP")
            vol   = int(_f("pDV") or 0)
            if price:
                out[code] = {"price": price, "prev_close": prev, "volume": vol}
        return out
    except Exception:
        return {}


# ══════════════════════════════════════════════
# 地合い判定
# ══════════════════════════════════════════════
def get_today_condition():
    """今日の地合いを取得。market_log.csv → candidates_log.csv → 計算の順で試みる。"""
    # market_log.csv（scan_daily.py が更新）
    for csv_path in [MARKET_LOG_CSV, MORNING_LOG_CSV]:
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
                row = df[df["date"] == TODAY]
                if not row.empty:
                    return str(row.iloc[0]["condition"])
            except Exception:
                pass

    # candidates_log.csv（scan_morning.py が更新）
    if os.path.exists(CANDIDATES_LOG_CSV):
        try:
            df  = pd.read_csv(CANDIDATES_LOG_CSV, encoding="utf-8-sig")
            row = df[df["date"] == TODAY]
            if not row.empty:
                return str(row.iloc[0]["condition"])
        except Exception:
            pass

    return "UNKNOWN"


def _classify_condition(ad_ratio):
    """AD比率のみで地合いを分類（nikkei_changeなし簡易版）"""
    if   ad_ratio <= PANIC_AD:  return "PANIC"
    elif ad_ratio <= WEAK_AD:   return "WEAK"
    elif ad_ratio >= STRONG_AD: return "STRONG"
    else:                       return "NORMAL"


# ══════════════════════════════════════════════
# 全銘柄スキャン
# ══════════════════════════════════════════════
def scan_dropping_stocks(url_price):
    """
    全銘柄を Tachibana API でスキャンし、
    ・市場全体の AD比率（地合い計算用）
    ・-4〜-7% 下落中の銘柄リスト
    を返す。
    """
    http    = urllib3.PoolManager()
    codes   = db.get_all_codes()
    total   = len(codes)
    batch   = 120
    n_batch = (total + batch - 1) // batch

    print(f"  全{total:,}銘柄をスキャン中（{n_batch}バッチ）...", flush=True)

    up_count = down_count = 0
    dropping = []

    for i in range(0, total, batch):
        chunk  = codes[i:i + batch]
        quotes = _fetch_price_batch(url_price, chunk, http)

        for code, q in quotes.items():
            price = q.get("price")
            prev  = q.get("prev_close")
            vol   = q.get("volume", 0)
            if not price or not prev or prev <= 0:
                continue

            chg = (price - prev) / prev * 100
            if chg > 0:
                up_count += 1
            else:
                down_count += 1

            if DROP_LO <= chg <= DROP_HI and vol >= MIN_VOLUME:
                dropping.append({
                    "code":       code,
                    "price":      price,
                    "prev_close": prev,
                    "change_pct": round(chg, 2),
                    "volume":     vol,
                    "rb_score":   0,  # 後で計算
                })

        done = min(i + batch, total)
        if done % (batch * 8) == 0 or done == total:
            print(f"  {done:,}/{total:,}件 完了... 候補{len(dropping)}件", flush=True)

    tot      = up_count + down_count
    ad_ratio = up_count / tot if tot > 0 else 0.5
    scanned_cond = _classify_condition(ad_ratio)

    print(f"  スキャン完了: 上昇{up_count:,}件 / 下落{down_count:,}件  AD比率{ad_ratio:.2f} → {scanned_cond}")
    print(f"  -4〜-7%下落候補: {len(dropping)}件")
    return ad_ratio, scanned_cond, dropping


def enrich_with_rb(candidates):
    """候補銘柄のRBスコアをSQLiteの履歴から計算して追加"""
    print(f"  RBスコア計算中（{len(candidates)}件）...", flush=True)
    for c in candidates:
        try:
            hist = db.get_stock_history(c["code"])
            if len(hist) >= 26:
                c["rb_score"] = calc_rebound_score(hist)
        except Exception:
            c["rb_score"] = 0
    return [c for c in candidates if c["rb_score"] >= RB_MIN]


# ══════════════════════════════════════════════
# AI判断
# ══════════════════════════════════════════════
def ask_claude_closing(candidate, condition, market_info):
    """引け前買い用AI判断プロンプト"""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    price     = candidate["price"]
    prev      = candidate["prev_close"]
    change    = candidate["change_pct"]
    rb        = candidate["rb_score"]
    code      = candidate["code"]
    name      = candidate.get("name", code)
    vol       = candidate["volume"]

    # 異常フラグ事前判定
    anomaly_hints = []
    if price < 200:
        anomaly_hints.append(f"⚠️ 株価{price:.0f}円と極端に安い（倒産・上場廃止リスク）")
    if change <= -15:
        anomaly_hints.append(f"⚠️ 本日{change:.1f}%暴落（決算・不祥事の可能性）")
    anomaly_section = ""
    if anomaly_hints:
        anomaly_section = "\n## ⚠️ 異常検知\n" + "\n".join(anomaly_hints) + "\n"

    prompt = f"""あなたは日本株のデイトレード補助AIです。
引け前（14:45〜15:00）の逆張り買いについて判断してください。

## 銘柄情報
- コード: {code}  銘柄名: {name}
- 前日終値: {prev:,.0f}円
- 現在値:   {price:,.0f}円（本日{change:+.2f}%）
- 本日出来高: {vol:,}株
- RBスコア: {rb}点（反発シグナル強度）
{anomaly_section}
## 地合い: {condition}（STRONG確認済み）

## 戦略（引け前買い・逆張り）
- 今日の引け（15:00）で成行き買い
- 翌日に引け決済（または TP+3% / SL-6%）
- 【試算根拠】STRONG×-4〜-7%×RB>=3: WR 69.5%、avg +1.482%

## 判断基準
- 「買い実行」: 反発期待が高い（RB高い・出来高増・売り枯れの兆し）
- 「見送り」: 続落リスクが高い（異常フラグ・業績悪化示唆）
- ⚠️異常検知がある場合は必ず「見送り」

## 回答形式（JSON）
{{
  "判断": "買い実行" or "見送り",
  "推奨買い価格": 数値（現在値付近）or null,
  "指値売り価格": 数値（翌日TP目安）or null,
  "損切り価格":   数値（翌日SL目安）or null,
  "根拠": "2〜3文",
  "異常フラグ": true or false
}}"""

    for attempt in range(5):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=384,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.content[0].text
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < 4:
                print(f"  ⚠️ API混雑。10秒後にリトライ（{attempt+1}/5）...")
                time.sleep(10)
            else:
                raise


def parse_ai(text):
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
# 発注確認
# ══════════════════════════════════════════════
def confirm_and_order(candidate, ai_result, url_request):
    code      = candidate["code"]
    name      = candidate.get("name", code)
    price     = candidate["price"]
    rec_price = ai_result.get("推奨買い価格") or price
    sell_p    = ai_result.get("指値売り価格")
    stop_p    = ai_result.get("損切り価格")
    shares    = calc_shares(rec_price or price)
    estimated = int((rec_price or price or 0) * shares)

    buying_power = None
    if url_request:
        buying_power = tachibana_order.get_buying_power(url_request)

    print(f"\n\a{'━'*56}")
    print(f"  🔔🔔🔔  引け前発注確認  [{code}] {name}  🔔🔔🔔")
    print(f"{'━'*56}")
    print(f"     現在値   : {price:>8,.0f}円  （本日{candidate['change_pct']:+.2f}%）")
    print(f"     推奨買い : {rec_price:>8,.0f}円" if rec_price else "")
    print(f"     翌日売り : {sell_p:>8,.0f}円（TP+3%目安）" if sell_p else "")
    print(f"     損切り   : {stop_p:>8,.0f}円（SL-5.8%目安）" if stop_p else "")
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
        if ans in ("n", ""):
            print(f"  ↩️  {code} 見送りました。")
            return False
        if ans.isdigit():
            shares    = int(ans)
            estimated = int(rec_price * shares) if rec_price else 0
            print(f"  株数を{shares}株（概算{estimated:,.0f}円）に変更。[y=発注/n=見送り] > ", end="", flush=True)
            continue
        if ans == "y":
            if not url_request:
                print(f"  ❌ 業務URLが取得できません。再ログインしてください。")
                return False
            print(f"  📤 発注中... {code} {shares}株 成行買い（引け）")
            result = tachibana_order.place_buy_order(url_request, code, shares)
            if result["success"]:
                print(f"  ✅ {result['message']}")
                buy_px = int(rec_price or price or 0)
                if buy_px > 0:
                    tachibana_order.save_position(code, name, shares, buy_px,
                                                  strategy="B", tp_pct=0.03, sl_pct=0.058)
            else:
                print(f"  ❌ 発注失敗: {result['message']}")
            return result.get("success", False)


# ══════════════════════════════════════════════
# 結果保存
# ══════════════════════════════════════════════
def save_closing_log(candidates, ai_results):
    """AI判断結果を closing_log.csv に保存"""
    rows = []
    for c in candidates:
        code   = c["code"]
        result = ai_results.get(code, {})
        rows.append({
            "date":           TODAY,
            "code":           code,
            "name":           c.get("name", ""),
            "change_pct":     c["change_pct"],
            "rb_score":       c["rb_score"],
            "price":          c["price"],
            "ai_judgment":    result.get("判断", ""),
            "ai_sell_price":  result.get("指値売り価格", ""),
            "ai_stop_price":  result.get("損切り価格", ""),
            "ai_reason":      str(result.get("根拠", ""))[:120],
            "ai_anomaly":     "1" if result.get("異常フラグ") else "",
        })
    df     = pd.DataFrame(rows)
    exists = os.path.exists(CLOSING_LOG_CSV)
    df.to_csv(CLOSING_LOG_CSV, mode="a", header=not exists,
              index=False, encoding="utf-8-sig")
    print(f"\n  💾 ログ保存: {CLOSING_LOG_CSV}（{len(rows)}件）")


# ══════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════
def main(start_now=False):
    db.init_db()
    print(f"=== 📉 引け前スキャン（{TODAY}）===\n")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY が設定されていません")
        return

    # Tachibana API 確認
    url_price   = load_tachibana_url()
    url_request = tachibana_order.load_url_request()
    if not url_price:
        print("❌ Tachibana APIにログインしていません。scan_morning.py を先に実行してください。")
        return
    print(f"  📡 Tachibana API: リアルタイム価格取得モード")
    mode_str = "本番発注モード" if tachibana_order.LIVE_TRADING else "モックモード（実発注なし）"
    print(f"  💼 発注: {mode_str}  1,000円未満→最大{MAX_ORDER_AMOUNT//1000}万円分 / 1,000円以上→{DEFAULT_SHARES}株\n")

    # 開始待機
    now = datetime.now(JST)
    if not start_now and (now.hour < SCAN_START_HOUR or
                          (now.hour == SCAN_START_HOUR and now.minute < SCAN_START_MIN)):
        target = now.replace(hour=SCAN_START_HOUR, minute=SCAN_START_MIN, second=0, microsecond=0)
        wait   = int((target - now).total_seconds())
        print(f"  ⏰ {SCAN_START_HOUR}:{SCAN_START_MIN:02d}まで {wait//60}分{wait%60}秒 待機します...")
        while True:
            now = datetime.now(JST)
            if now.hour > SCAN_START_HOUR or (now.hour == SCAN_START_HOUR and now.minute >= SCAN_START_MIN):
                break
            time.sleep(10)

    # 地合い確認（CSVから読む → スキャン後に上書き）
    condition = get_today_condition()
    print(f"  地合い（事前）: {condition}")

    # ── 全銘柄スキャン ──
    print(f"\n【ステップ1】全銘柄スキャン開始...")
    ad_ratio, scanned_cond, raw_candidates = scan_dropping_stocks(url_price)

    # スキャン結果で地合いを補正（より正確）
    if condition == "UNKNOWN":
        condition = scanned_cond
    print(f"  地合い（スキャン補正後）: {condition}  AD比率: {ad_ratio:.2f}")

    # STRONG以外は終了
    if condition != "STRONG":
        print(f"\n  ⚠️  本日は{condition}地合いです。")
        print(f"     戦略B引け前買いはSTRONG地合いのみ有効のため終了します。")
        print(f"     （STRONG: AD比率>={STRONG_AD} 必要 / 現在{ad_ratio:.2f}）")
        return

    if not raw_candidates:
        print(f"\n  該当銘柄なし（-4〜-7%下落 × 出来高{MIN_VOLUME:,}株以上）")
        return

    # ── RBスコア計算 ──
    print(f"\n【ステップ2】RBスコア計算...")
    candidates = enrich_with_rb(raw_candidates)

    if not candidates:
        print(f"  RBスコア{RB_MIN}以上の銘柄なし")
        return

    # RBスコア降順 → 下落幅降順でソート
    candidates.sort(key=lambda c: (-c["rb_score"], c["change_pct"]))

    # 銘柄名を取得（可能な場合）
    try:
        conn = db.get_conn()
        code_names = dict(pd.read_sql(
            "SELECT DISTINCT code, name FROM scan_results WHERE name != ''", conn
        ).values.tolist())
        conn.close()
        for c in candidates:
            c["name"] = code_names.get(c["code"], c["code"])
    except Exception:
        for c in candidates:
            c["name"] = c.get("name", c["code"])

    print(f"\n{'='*62}")
    print(f"【引け前候補】STRONG地合い × -4〜-7% × RB>={RB_MIN}  {len(candidates)}件")
    print(f"{'='*62}")
    print(f"  {'コード':<7} {'銘柄名':<14} {'下落率':>7} {'現在値':>8} {'RB':>4} {'出来高':>10}")
    print(f"  {'─'*60}")
    for c in candidates[:MAX_AI_JUDGE]:
        print(f"  {c['code']:<7} {c.get('name',''):<14} {c['change_pct']:>+6.2f}% "
              f"{c['price']:>8,.0f}円  {c['rb_score']:>3}点  {c['volume']:>10,}株")
    if len(candidates) > MAX_AI_JUDGE:
        print(f"  ... 他{len(candidates)-MAX_AI_JUDGE}件（上位{MAX_AI_JUDGE}件をAI判断）")

    # ── AI判断 ──
    print(f"\n【ステップ3】AI判断（上位{min(len(candidates), MAX_AI_JUDGE)}件）...")
    market_info = {}
    ai_results  = {}
    buy_list    = []

    now_min = datetime.now(JST)
    now_min = now_min.hour * 60 + now_min.minute

    for c in candidates[:MAX_AI_JUDGE]:
        code = c["code"]

        if now_min >= ORDER_DEADLINE_MIN:
            print(f"  ⏰ 14:58を過ぎました。残り銘柄の判断を省略します。")
            break

        print(f"\n  🤖 AI判断中: [{code}] {c.get('name', '')}  {c['change_pct']:+.2f}%  RB{c['rb_score']}点...")
        try:
            raw    = ask_claude_closing(c, condition, market_info)
            result = parse_ai(raw)
            ai_results[code] = result

            judgment = result.get("判断", "不明")
            anomaly  = result.get("異常フラグ", False)

            # 異常フラグ強制見送り
            if anomaly and judgment == "買い実行":
                result["判断"] = "見送り"
                result["根拠"] = f"[異常フラグ強制見送り] {result.get('根拠', '')}"
                judgment = "見送り"
                print(f"  🚫 異常フラグにより強制見送り")

            icon = {"買い実行": "🟢", "見送り": "🔴"}.get(judgment, "⚪")
            price_info = ""
            rec_p = result.get("推奨買い価格")
            sel_p = result.get("指値売り価格")
            stp_p = result.get("損切り価格")
            if rec_p:
                price_info = f"  買:{rec_p:,.0f}円"
            if sel_p:
                price_info += f" → 売:{sel_p:,.0f}円"
            if stp_p:
                price_info += f" / 損:{stp_p:,.0f}円"
            print(f"  {icon} {judgment}{price_info}")
            print(f"     {result.get('根拠', '')}")

            if judgment == "買い実行":
                buy_list.append(c)

        except Exception as e:
            print(f"  ❌ AI判断エラー: {e}")
            ai_results[code] = {"判断": "エラー", "根拠": str(e)}

        now_min = datetime.now(JST)
        now_min = now_min.hour * 60 + now_min.minute

    # ── 発注確認 ──
    if buy_list:
        print(f"\n{'='*62}")
        print(f"【発注候補】AI「買い実行」: {len(buy_list)}件")
        print(f"{'='*62}")
        for c in buy_list:
            code   = c["code"]
            result = ai_results.get(code, {})
            now_min = datetime.now(JST)
            now_min = now_min.hour * 60 + now_min.minute
            if now_min >= ORDER_DEADLINE_MIN:
                print(f"  ⏰ 14:58を過ぎました。発注を中止します。")
                break
            confirm_and_order(c, result, url_request)
    else:
        print(f"\n  AI「買い実行」なし。本日の引け前スキャン終了。")

    # ── ログ保存 ──
    save_closing_log(candidates[:MAX_AI_JUDGE], ai_results)

    print(f"\n{'='*62}")
    print(f"  引け前スキャン完了  {datetime.now(JST).strftime('%H:%M:%S')}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="引け前下落銘柄スキャン + 買い判断")
    parser.add_argument("--now", action="store_true", help="即時開始（テスト用）")
    args = parser.parse_args()
    main(start_now=args.now)
