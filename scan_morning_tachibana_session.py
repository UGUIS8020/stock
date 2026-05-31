# -*- coding: utf-8 -*-
"""
scan_morning_tachibana_session.py - 立花証券APIセッション管理モジュール

scan_morning.py から分割。今後ほぼ変更不要な安定ロジック。
担当: ログイン・セッション確認・リアルタイム価格取得・株価履歴取得
"""

import json
import os
import time
import urllib3
import pandas as pd
import yfinance as yf

import db as _db

from datetime import datetime
from pathlib import Path

_BASE_DIR            = Path(__file__).parent
TACHIBANA_LOGIN_FILE = str(_BASE_DIR / "tachibana_login_response.json")
TACHIBANA_API_BASE   = "https://kabuka.e-shiten.jp/e_api_v4r8/"
TACHIBANA_USER_ID    = os.getenv("TACHIBANA_LOGIN_ID")
TACHIBANA_PASSWORD   = os.getenv("TACHIBANA_LOGIN_PASS")


def load_tachibana_url():
    """tachibana_login_response.json から株価URLを読み込む。失敗時はNone。"""
    try:
        with open(TACHIBANA_LOGIN_FILE, encoding="utf-8") as f:
            data = json.load(f)
        url = data.get("sUrlPrice", "")
        return url if url else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _url_encode(s):
    """パスワード等に含まれる記号をURLエンコードする"""
    table = {
        ' ': '%20', '!': '%21', '"': '%22', '#': '%23', '$': '%24',
        '%': '%25', '&': '%26', "'": '%27', '(': '%28', ')': '%29',
        '*': '%2A', '+': '%2B', ',': '%2C', '/': '%2F', ':': '%3A',
        ';': '%3B', '<': '%3C', '=': '%3D', '>': '%3E', '?': '%3F',
        '@': '%40', '[': '%5B', ']': '%5D', '^': '%5E', '`': '%60',
        '{': '%7B', '|': '%7C', '}': '%7D', '~': '%7E',
    }
    return ''.join(table[c] if c in table else c for c in s)


def _check_tachibana_session(url_price):
    """株価URLに試験リクエストを送り、セッションが有効かどうか確認する。"""
    if not url_price:
        return False
    try:
        t = datetime.now()
        p_sd = (f"{t.year}.{t.month:02}.{t.day:02}"
                f"-{t.hour:02}:{t.minute:02}:{t.second:02}"
                f".{t.microsecond // 1000:03}")
        params = (
            '{'
            '"p_no":"0",'
            f'"p_sd_date":"{p_sd}",'
            '"sCLMID":"CLMMfdsGetMarketPrice",'
            '"sTargetIssueCode":"7203",'
            '"sTargetColumn":"pDPP,pPRP",'
            '"sJsonOfmt":"5"'
            '}'
        )
        http = urllib3.PoolManager()
        resp = http.request("GET", url_price + "?" + params,
                            timeout=urllib3.Timeout(connect=3, read=5))
        result = json.loads(resp.data.decode("shift-jis", errors="ignore"))
        # p_errno="0" であればセッション有効
        # aCLMMfdsMarketPrice が空でも市場時間外では正常（セッション切れではない）
        return result.get("p_errno", "") == "0"
    except Exception:
        return False


def _tachibana_login():
    """CLMAuthLoginRequest を送信し、セッションURLをJSONに保存して url_price を返す。"""
    t = datetime.now()
    p_sd = (f"{t.year}.{t.month:02}.{t.day:02}"
            f"-{t.hour:02}:{t.minute:02}:{t.second:02}"
            f".{t.microsecond // 1000:03}")
    params = (
        '{'
        '"p_no":"1",'
        f'"p_sd_date":"{p_sd}",'
        '"sCLMID":"CLMAuthLoginRequest",'
        f'"sUserId":"{TACHIBANA_USER_ID}",'
        f'"sPassword":"{_url_encode(TACHIBANA_PASSWORD)}",'
        '"sJsonOfmt":"5"'
        '}'
    )
    url = TACHIBANA_API_BASE + "auth/?" + params
    try:
        http = urllib3.PoolManager()
        resp = http.request("GET", url, timeout=urllib3.Timeout(connect=10, read=15))
        result = json.loads(resp.data.decode("shift-jis", errors="ignore"))
    except Exception as e:
        print(f"  ❌ 通信エラー: {e}")
        return None

    result_code = result.get("sResultCode", "")
    if result_code != "0":
        err = result.get("sResultText", "不明")
        print(f"  ❌ ログイン失敗（コード: {result_code}、内容: {err}）")
        if result_code == "10088":
            print("  → 電話認証が完了していない可能性があります。再度お電話ください。")
        return None

    login_data = {
        "sResultCode":        result_code,
        "sUrlRequest":        result.get("sUrlRequest", ""),
        "sUrlPrice":          result.get("sUrlPrice", ""),
        "sUrlMaster":         result.get("sUrlMaster", ""),
        "sUrlEvent":          result.get("sUrlEvent", ""),
        "sUrlEventWebSocket": result.get("sUrlEventWebSocket", ""),
        "sZyoutoekiKazeiC":   result.get("sZyoutoekiKazeiC", ""),
        "sLastLoginDate":     result.get("sLastLoginDate", ""),
    }
    with open(TACHIBANA_LOGIN_FILE, "w", encoding="utf-8") as f:
        json.dump(login_data, f, ensure_ascii=False, indent=2)

    url_price = login_data["sUrlPrice"]
    print(f"  ✅ 接続成功（{TACHIBANA_LOGIN_FILE} 保存済）")
    return url_price if url_price else None


def ensure_tachibana_session():
    """Tachibanaセッションを確保する。期限切れなら電話認証フローを案内する。"""
    if not TACHIBANA_USER_ID or not TACHIBANA_PASSWORD:
        print("  ⚠️  .env に TACHIBANA_LOGIN_ID / TACHIBANA_LOGIN_PASS が未設定です")
        print("     → Tachibana API なしで続行します")
        return None

    today_str = datetime.now().strftime("%Y%m%d")

    # まず実際にセッションが有効かどうかを確認（日付に関係なく）
    existing_url = load_tachibana_url()
    if existing_url and _check_tachibana_session(existing_url):
        # ファイルの更新日時が今日なら「本日ログイン済み」として表示
        try:
            mtime = os.path.getmtime(TACHIBANA_LOGIN_FILE)
            file_date = datetime.fromtimestamp(mtime).strftime("%Y%m%d")
            if file_date == today_str:
                login_time = datetime.fromtimestamp(mtime).strftime("%H:%M")
                print(f"  ✅ Tachibana API セッション有効（本日 {login_time} ログイン済み）")
            else:
                print("  ✅ Tachibana API セッション有効（セッション確認済み）")
        except Exception:
            print("  ✅ Tachibana API セッション有効")
        return existing_url

    # セッションが無効 → 電話認証フロー
    if existing_url:
        print("  ⚠️  Tachibana API セッションが期限切れです（電話認証が必要）")
    else:
        print("  ℹ️  Tachibana API 未接続（初回認証が必要）")
        print("  ℹ️  ※ 毎朝 python tachibana_login.py を先に実行することを推奨します")

    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  【立花証券e支店API 電話認証】                    │")
    print("  │                                                   │")
    print("  │  登録済み電話番号から今すぐ発信してください:       │")
    print("  │    📞 0120-28-6592（無料）                        │")
    print("  │    📞 050-3102-6575（通話料有料）                 │")
    print("  │                                                   │")
    print("  │  ※ 非通知・184発信は認証されません               │")
    print("  │  ※ 音声が流れ、自動的に切れたらOKです            │")
    print("  └─────────────────────────────────────────────────┘")
    print()
    input("  電話が切れたら Enter キーを押してください...")
    print()
    print("  ログインリクエスト送信中...")

    url_price = _tachibana_login()
    if not url_price:
        print("  ⚠️  認証に失敗しました。Tachibana API なしで続行します。")
    return url_price


def fetch_tachibana_prices(url_price, codes):
    """Tachibana APIから複数銘柄の気配値・現在値を一括取得。
    APIの上限120件/回をバッチ分割して全件取得する。
    戻り値: {code: {"price", "ask", "bid", "ask_vol", "bid_vol", "prev_close", "volume"}}
    """
    if not url_price or not codes:
        return {}
    batch_size = 120
    http = urllib3.PoolManager()
    quotes = {}
    p_no_base = int(time.time()) % 100000
    for i, start in enumerate(range(0, len(codes), batch_size)):
        batch     = codes[start:start + batch_size]
        code_list = ",".join(str(c) for c in batch)
        t = datetime.now()
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
            result = json.loads(resp.data.decode("shift-jis", errors="ignore"))
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


def _tprice(quotes, code):
    """tachibana_prices辞書から現在値floatを取得するヘルパー。"""
    q = (quotes or {}).get(str(code))
    return q["price"] if q else None


def _tquote(quotes, code):
    """tachibana_prices辞書から気配dict全体を取得するヘルパー。"""
    return (quotes or {}).get(str(code))


def _quote_summary(q):
    """気配dictから表示用の (売気配str, 買気配str, 需給比str) を返す。"""
    if not q:
        return "    ---", "    ---", " ---"
    ask = q.get("ask")
    bid = q.get("bid")
    av  = q.get("ask_vol") or 0
    bv  = q.get("bid_vol") or 0
    ask_str   = f"{ask:>7,.0f}" if ask else "    ---"
    bid_str   = f"{bid:>7,.0f}" if bid else "    ---"
    total     = av + bv
    ratio_str = f"{bv/total*100:>4.0f}%" if total > 0 else " ---"
    return ask_str, bid_str, ratio_str


def fetch_stock_hist(code, realtime_price=None):
    """銘柄の株価履歴を取得する。
    優先順: ① Tachibana API（realtime_price引数） → ② SQLite DB → ③ yfinance
    """
    hist = _db.get_stock_history(code)
    if len(hist) >= 22:
        if realtime_price is not None:
            return hist, realtime_price, "tachibana"
        return hist, float(hist["Close"].iloc[-1]), "cache"

    try:
        ticker = yf.Ticker(f"{code}.T")
        hist   = ticker.history(period="3mo")
        if len(hist) >= 22:
            hist = hist.reset_index()
            hist["Date"] = pd.to_datetime(hist["Date"])
            if realtime_price is not None:
                return hist, realtime_price, "tachibana"
            return hist, float(hist["Close"].iloc[-1]), "yfinance"
    except Exception:
        pass

    if realtime_price is not None:
        return None, realtime_price, "tachibana"
    return None, None, None
