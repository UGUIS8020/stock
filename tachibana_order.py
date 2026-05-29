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
from pathlib import Path
import db

load_dotenv()

LOGIN_FILE      = str(Path(__file__).parent / "tachibana_login_response.json")
SECOND_PASSWORD = os.getenv("TACHIBANA_SECOND_PASSWORD", "")

def _load_zyoutoeki_c():
    try:
        with open(LOGIN_FILE, encoding="utf-8") as f:
            return json.load(f).get("sZyoutoekiKazeiC", "1")
    except Exception:
        return "1"

ZYOUTOEKI_C = _load_zyoutoeki_c()

# ── 安全装置 ────────────────────────────────────────────
LIVE_TRADING = True

# 1発注あたりの上限金額（安全装置）
MAX_ORDER_AMOUNT = 500_000   # 50万円

_P_NO_FILE = Path(__file__).parent / "out" / "last_p_no.txt"


def _init_p_no():
    time_based = int(time.time())
    try:
        saved = int(_P_NO_FILE.read_text().strip())
        return max(saved + 1, time_based)
    except Exception:
        return time_based


_p_no = _init_p_no()


def _next_p_no():
    global _p_no
    try:
        saved = int(_P_NO_FILE.read_text().strip())
        _p_no = max(_p_no, saved) + 1
    except Exception:
        _p_no += 1
    try:
        _P_NO_FILE.write_text(str(_p_no))
    except Exception:
        pass
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


def check_kisoku_before_order(url_request, code):
    """
    発注前に立花APIで取引規制を確認する。

    Returns:
        {"blocked": bool, "reason": str}
        APIエラー時は blocked=False（確認できなかった旨のみ警告）
    """
    params = (
        "{"
        f'"p_no":"{_next_p_no()}",'
        f'"p_sd_date":"{_p_sd_date()}",'
        '"sCLMID":"CLMIssueSizyouKiseiKabu",'
        f'"sIssueCode":"{code}",'
        '"sJsonOfmt":"5"'
        "}"
    )
    try:
        result = _api_get(url_request + "?" + params)
        p_errno = result.get("p_errno", "")
        if p_errno != "0":
            print(f"  [規制チェック] {code}: API応答 p_errno={p_errno} → 確認スキップ")
            return {"blocked": False, "reason": f"規制チェックAPI失敗(p_errno={p_errno})"}

        # トップレベルの sTeisiKubun
        teisi = result.get("sTeisiKubun", "0") or "0"
        if teisi != "0":
            return {"blocked": True, "reason": f"取引停止・制限中 (sTeisiKubun={teisi})"}

        # リスト形式で返ってくる場合
        items = result.get("aCLMIssueSizyouKiseiKabu", [])
        for item in items:
            teisi_i = item.get("sTeisiKubun", "0") or "0"
            if teisi_i != "0":
                return {"blocked": True, "reason": f"取引停止・制限中 (sTeisiKubun={teisi_i})"}

        print(f"  [規制チェック] {code}: 規制なし ✅")
        return {"blocked": False, "reason": "規制なし"}

    except Exception as e:
        print(f"  [規制チェック] {code}: 通信エラー({e}) → 確認スキップ")
        return {"blocked": False, "reason": f"規制チェック通信エラー"}


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


def place_buy_order(url_request, code, shares, price=None, market_code=None):
    """現物買い注文を発注する（price=None で成行、数値で指値）。"""
    if not LIVE_TRADING:
        print(f"  🔧 [モックモード] 実際の注文は出ません（LIVE_TRADING=False）")
        return {
            "success":  True,
            "order_no": "MOCK-0000",
            "message":  f"[モック] {code} {shares}株 成行買い（発注なし）",
            "raw":      {},
        }

    # ── 発注前 取引規制チェック ──────────────────────────
    kisoku = check_kisoku_before_order(url_request, code)
    if kisoku["blocked"]:
        print(f"  ⛔ [{code}] 規制銘柄のため発注中止: {kisoku['reason']}")
        return {
            "success":  False,
            "order_no": None,
            "message":  f"規制銘柄のため発注中止: {kisoku['reason']}",
            "raw":      {},
        }

    # 成行: sCondition="0", sOrderPrice="0"
    # 指値: sCondition="2"（未確認）, sOrderPrice=str(price)
    if price is None:
        condition   = "0"
        order_price = "0"
    else:
        condition   = "2"
        order_price = str(int(price))

    params = (
        "{"
        f'"p_no":"{_next_p_no()}",'
        f'"p_sd_date":"{_p_sd_date()}",'
        '"sCLMID":"CLMKabuNewOrder",'
        f'"sZyoutoekiKazeiC":"{ZYOUTOEKI_C}",'
        f'"sIssueCode":"{code}",'
        '"sSizyouC":"00",'
        '"sBaibaiKubun":"3",'
        f'"sCondition":"{condition}",'
        f'"sOrderPrice":"{order_price}",'
        f'"sOrderSuryou":"{shares}",'
        '"sGenkinShinyouKubun":"0",'
        '"sOrderExpireDay":"0",'
        '"sGyakusasiOrderType":"0",'
        '"sGyakusasiZyouken":"0",'
        '"sGyakusasiPrice":"*",'
        '"sTatebiType":"*",'
        f'"sSecondPassword":"{SECOND_PASSWORD}",'
        '"sJsonOfmt":"5"'
        "}"
    )
    try:
        result = _api_get(url_request + "?" + params)
        p_errno  = result.get("p_errno")
        s_result = result.get("sResultCode", "")
        if p_errno is None:
            return {
                "success":  False,
                "order_no": None,
                "message":  f"APIレスポンス異常: {str(result)[:200]}",
                "raw":      result,
            }
        if p_errno == "0" and s_result == "0":
            order_no = result.get("sOrderNumber", "")
            return {
                "success":  True,
                "order_no": order_no,
                "message":  f"注文受付完了（注文番号: {order_no}）",
                "raw":      result,
            }
        else:
            err_msg = result.get("sResultText") or result.get("p_err") or "不明"
            return {
                "success":  False,
                "order_no": None,
                "message":  f"注文エラー (p_errno={p_errno} sResultCode={s_result}): {err_msg}",
                "raw":      result,
            }
    except Exception as e:
        return {
            "success":  False,
            "order_no": None,
            "message":  f"通信エラー: {e}",
            "raw":      {},
        }


def save_position(code, name, shares, buy_price, strategy, tp_pct=0.03, sl_pct=0.05,
                  entry_change_pct=None, rb_score=None, condition=None):
    """買い成功後にポジションを DB へ記録する。"""
    db.init_db()
    db.save_position_db(
        code, name, shares, buy_price, strategy,
        tp_pct=tp_pct, sl_pct=sl_pct,
        entry_change_pct=entry_change_pct,
        rb_score=rb_score,
        condition=condition,
    )


def place_sell_order(url_request, code, shares, price=None, market_code=None):
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
        condition   = "0"
        order_price = "0"
    else:
        condition   = "2"
        order_price = str(int(price))

    params = (
        "{"
        f'"p_no":"{_next_p_no()}",'
        f'"p_sd_date":"{_p_sd_date()}",'
        '"sCLMID":"CLMKabuNewOrder",'
        f'"sZyoutoekiKazeiC":"{ZYOUTOEKI_C}",'
        f'"sIssueCode":"{code}",'
        '"sSizyouC":"00",'
        '"sBaibaiKubun":"1",'
        f'"sCondition":"{condition}",'
        f'"sOrderPrice":"{order_price}",'
        f'"sOrderSuryou":"{shares}",'
        '"sGenkinShinyouKubun":"0",'
        '"sOrderExpireDay":"0",'
        '"sGyakusasiOrderType":"0",'
        '"sGyakusasiZyouken":"0",'
        '"sGyakusasiPrice":"*",'
        '"sTatebiType":"*",'
        f'"sSecondPassword":"{SECOND_PASSWORD}",'
        '"sJsonOfmt":"5"'
        "}"
    )
    try:
        result = _api_get(url_request + "?" + params)
        p_errno = result.get("p_errno", "")
        s_result = result.get("sResultCode", "")
        if p_errno == "0" and s_result == "0":
            order_no = result.get("sOrderNumber", "")
            return {
                "success":  True,
                "order_no": order_no,
                "message":  f"売り注文受付完了（注文番号: {order_no}）",
                "raw":      result,
            }
        else:
            err_msg = result.get("sResultText") or result.get("p_err") or "不明"
            return {
                "success":  False,
                "order_no": None,
                "message":  f"売り注文エラー (p_errno={p_errno} sResultCode={s_result}): {err_msg}",
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
