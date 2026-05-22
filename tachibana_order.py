# -*- coding: utf-8 -*-
"""
tachibana_order.py - 立花証券API 発注モジュール

対応: 現物 成行買い / 成行売り
使用前に tachibana_connect_test.py でログイン済みであること。
"""

import json
import os
import time
import urllib3
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

LOGIN_FILE      = "tachibana_login_response.json"
SECOND_PASSWORD = os.getenv("TACHIBANA_SECOND_PASSWORD", "")

# ── 安全装置 ────────────────────────────────────────────
LIVE_TRADING = True

# 1発注あたりの上限金額（安全装置）
MAX_ORDER_AMOUNT = 500_000   # 50万円

_p_no = int(time.time()) % 100000


def _next_p_no():
    global _p_no
    _p_no += 1
    return str(_p_no)


def _p_sd_date():
    t = datetime.now()
    return (f"{t.year}.{t.month:02}.{t.day:02}"
            f"-{t.hour:02}:{t.minute:02}:{t.second:02}"
            f".{t.microsecond // 1000:03}")


def _api_get(url):
    http = urllib3.PoolManager()
    resp = http.request("GET", url, timeout=urllib3.Timeout(connect=5, read=10))
    text = resp.data.decode("shift-jis", errors="ignore")
    return json.loads(text)


def load_url_request():
    """tachibana_login_response.json から業務URLを読み込む。"""
    try:
        with open(LOGIN_FILE, encoding="utf-8") as f:
            data = json.load(f)
        url = data.get("sUrlRequest", "")
        return url if url else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def get_buying_power(url_request):
    """現物買余力を照会して残高(円)を返す。取得失敗時はNone。"""
    params = (
        "{"
        f'"p_no":"{_next_p_no()}",'
        f'"p_sd_date":"{_p_sd_date()}",'
        '"sCLMID":"CLMZanKaiKanougaku",'
        '"sJsonOfmt":"5"'
        "}"
    )
    try:
        result = _api_get(url_request + "?" + params)
        if result.get("p_errno", "0") != "0":
            return None
        d = result.get("CLMZanKaiKanougaku", result)
        val = d.get("sSummaryGenkabuKaituke", "")
        return int(float(val)) if val else None
    except Exception:
        return None


def place_buy_order(url_request, code, shares, price=None):
    """
    現物 成行買い注文を発注する。

    Args:
        url_request: 業務URL（tachibana_login_response.json の sUrlRequest）
        code: 銘柄コード（例: "8289"）
        shares: 発注株数（例: 100）
        price: None=成行、数値=指値

    Returns:
        dict: {
            "success": bool,
            "order_no": str or None,  # 注文番号
            "message": str,           # 結果メッセージ
            "raw": dict,              # APIレスポンス生データ
        }
    """
    if not LIVE_TRADING:
        # モックモード: 実際の注文は出さない
        print(f"  🔧 [モックモード] 実際の注文は出ません（LIVE_TRADING=False）")
        return {
            "success":  True,
            "order_no": "MOCK-0000",
            "message":  f"[モック] {code} {shares}株 成行買い（発注なし）",
            "raw":      {},
        }

    # 成行: sSasiNeKubun="1", sSasiNe="0"
    # 指値: sSasiNeKubun="2", sSasiNe=str(price)
    if price is None:
        sasi_kubun = "1"   # 成行
        sasi_ne    = "0"
    else:
        sasi_kubun = "2"   # 指値
        sasi_ne    = str(int(price))

    today = datetime.now().strftime("%Y%m%d")

    params = (
        "{"
        f'"p_no":"{_next_p_no()}",'
        f'"p_sd_date":"{_p_sd_date()}",'
        '"sCLMID":"CLMKabuNewOrder",'
        f'"sIssueCode":"{code}",'
        '"sBaibaikubun":"1",'        # 1=買
        '"sOrderSinkubun":"0",'      # 0=現物
        f'"sSasiNeKubun":"{sasi_kubun}",'
        f'"sSasiNe":"{sasi_ne}",'
        f'"sOrdersu":"{shares}",'
        f'"sExpireDate":"{today}",'  # 当日有効
        '"sGyakusasiOrderType":"0",' # 逆指値なし
        '"sGyakusasiZyouken":"0",'
        '"sGyakusasiSasiNe":"0",'
        '"sEigyouDay":"0",'
        '"sZyoutoekiKazeiC":"0",'
        '"sUkewatasiDay":"0",'
        '"sTatebiType":"0",'
        '"sTategyokuZansu":"0",'
        f'"sSecondPassword":"{SECOND_PASSWORD}",'
        '"sJsonOfmt":"5"'
        "}"
    )
    try:
        result = _api_get(url_request + "?" + params)
        p_errno = result.get("p_errno", "0")
        if p_errno == "0":
            order_no = result.get("sOrderNo", result.get("CLMKabuNewOrder", {}).get("sOrderNo", ""))
            return {
                "success":  True,
                "order_no": order_no,
                "message":  f"注文受付完了（注文番号: {order_no}）",
                "raw":      result,
            }
        else:
            return {
                "success":  False,
                "order_no": None,
                "message":  f"注文エラー (p_errno={p_errno}): {result.get('p_err', '不明')}",
                "raw":      result,
            }
    except Exception as e:
        return {
            "success":  False,
            "order_no": None,
            "message":  f"通信エラー: {e}",
            "raw":      {},
        }


POSITIONS_CSV = "out/positions.csv"

_POSITIONS_FIELDS = [
    "date", "code", "name", "shares", "buy_price", "tp_price", "sl_price",
    "strategy", "buy_time", "status", "sell_price", "sell_time", "pnl_pct",
]


def save_position(code, name, shares, buy_price, strategy, tp_pct=0.03, sl_pct=0.05):
    """買い成功後にポジションを out/positions.csv へ追記する。"""
    import csv as _csv
    tp_price = round(buy_price * (1 + tp_pct))
    sl_price = round(buy_price * (1 - sl_pct))
    today    = datetime.now().strftime("%Y-%m-%d")
    buy_time = datetime.now().strftime("%H:%M:%S")
    row = {
        "date": today, "code": code, "name": name, "shares": shares,
        "buy_price": buy_price, "tp_price": tp_price, "sl_price": sl_price,
        "strategy": strategy, "buy_time": buy_time, "status": "open",
        "sell_price": "", "sell_time": "", "pnl_pct": "",
    }
    file_exists = os.path.exists(POSITIONS_CSV)
    with open(POSITIONS_CSV, "a", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=_POSITIONS_FIELDS)
        if not file_exists:
            w.writeheader()
        w.writerow(row)
    print(f"  💾 ポジション記録: {code} {name} {shares}株 "
          f"買値{buy_price}円 TP:{tp_price}円(+{tp_pct*100:.0f}%) "
          f"SL:{sl_price}円(-{sl_pct*100:.1f}%)")


def place_sell_order(url_request, code, shares, price=None):
    """現物 成行売り注文を発注する。戻り値は place_buy_order と同じ構造。"""
    if not LIVE_TRADING:
        print(f"  🔧 [モックモード] 実際の注文は出ません（LIVE_TRADING=False）")
        return {
            "success":  True,
            "order_no": "MOCK-SELL-0000",
            "message":  f"[モック] {code} {shares}株 成行売り（発注なし）",
            "raw":      {},
        }

    if price is None:
        sasi_kubun = "1"
        sasi_ne    = "0"
    else:
        sasi_kubun = "2"
        sasi_ne    = str(int(price))

    today = datetime.now().strftime("%Y%m%d")
    params = (
        "{"
        f'"p_no":"{_next_p_no()}",'
        f'"p_sd_date":"{_p_sd_date()}",'
        '"sCLMID":"CLMKabuNewOrder",'
        f'"sIssueCode":"{code}",'
        '"sBaibaikubun":"2",'
        '"sOrderSinkubun":"0",'
        f'"sSasiNeKubun":"{sasi_kubun}",'
        f'"sSasiNe":"{sasi_ne}",'
        f'"sOrdersu":"{shares}",'
        f'"sExpireDate":"{today}",'
        '"sGyakusasiOrderType":"0",'
        '"sGyakusasiZyouken":"0",'
        '"sGyakusasiSasiNe":"0",'
        '"sEigyouDay":"0",'
        '"sZyoutoekiKazeiC":"0",'
        '"sUkewatasiDay":"0",'
        '"sTatebiType":"0",'
        '"sTategyokuZansu":"0",'
        f'"sSecondPassword":"{SECOND_PASSWORD}",'
        '"sJsonOfmt":"5"'
        "}"
    )
    try:
        result = _api_get(url_request + "?" + params)
        p_errno = result.get("p_errno", "0")
        if p_errno == "0":
            order_no = result.get("sOrderNo", result.get("CLMKabuNewOrder", {}).get("sOrderNo", ""))
            return {
                "success":  True,
                "order_no": order_no,
                "message":  f"売り注文受付完了（注文番号: {order_no}）",
                "raw":      result,
            }
        else:
            return {
                "success":  False,
                "order_no": None,
                "message":  f"売り注文エラー (p_errno={p_errno}): {result.get('p_err', '不明')}",
                "raw":      result,
            }
    except Exception as e:
        return {
            "success":  False,
            "order_no": None,
            "message":  f"通信エラー: {e}",
            "raw":      {},
        }


def check_order(url_request, order_no):
    """注文照会（約定確認）。約定済みなら True を返す。"""
    params = (
        "{"
        f'"p_no":"{_next_p_no()}",'
        f'"p_sd_date":"{_p_sd_date()}",'
        '"sCLMID":"CLMKabuOrderList",'
        '"sTargetOrderNo":' + f'"{order_no}",'
        '"sJsonOfmt":"5"'
        "}"
    )
    try:
        result = _api_get(url_request + "?" + params)
        orders = result.get("aCLMKabuOrderList", [])
        for o in orders:
            if o.get("sOrderNo", "") == order_no:
                status = o.get("sOrderStatus", "")
                yakujо = o.get("sYakujosu", "0")
                return {
                    "status":   status,
                    "filled":   int(yakujо) if yakujо else 0,
                    "raw":      o,
                }
    except Exception:
        pass
    return {"status": "不明", "filled": 0, "raw": {}}
