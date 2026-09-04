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
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from pathlib import Path
import db

load_dotenv()

JST = timezone(timedelta(hours=9))

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
# .envのLIVE_TRADINGで環境ごとに切り替え可能（未設定時は従来通りTrue）
LIVE_TRADING = os.getenv("LIVE_TRADING", "true").strip().lower() in ("1", "true", "yes")

# 1発注あたりの上限金額（安全装置）
MAX_ORDER_AMOUNT = 500_000   # 50万円

_P_NO_FILE = Path(__file__).parent / "out" / "last_p_no.txt"
_P_NO_LOCK = _P_NO_FILE.with_suffix(".lock")


def _init_p_no():
    time_based = int(time.time())
    try:
        saved = int(_P_NO_FILE.read_text().strip())
        return max(saved + 1, time_based)
    except Exception:
        return time_based


_p_no = _init_p_no()


def _acquire_p_no_lock(timeout=2.0, stale_after=5.0):
    """out/last_p_no.txt.lock の排他生成でプロセス間ロックを取る。
    market_watch/closing_watch/daytime/position_monitorが同時にp_noを
    払い出すとTachibana APIがp_no競合(p_errno=6)で弾くため、
    読み込み→+1→書き込みをアトミックにする。
    """
    start = time.time()
    while True:
        try:
            fd = os.open(str(_P_NO_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except (FileExistsError, PermissionError):
            # PermissionError: Windowsでは削除直後の再作成が一時的に拒否されることがある
            try:
                age = time.time() - _P_NO_LOCK.stat().st_mtime
                if age > stale_after:
                    # 異常終了したプロセスの残留ロックとみなして解除
                    _P_NO_LOCK.unlink()
                    continue
            except FileNotFoundError:
                continue
            except PermissionError:
                pass
            if time.time() - start > timeout:
                return False
            time.sleep(0.02)


def _release_p_no_lock():
    try:
        _P_NO_LOCK.unlink()
    except FileNotFoundError:
        pass


def _next_p_no():
    """複数プロセスから同時に呼ばれてもp_noが重複しないよう、
    ファイルロックで排他制御してから読み込み→+1→書き込みする。
    ロックが取れない異常時は、共有カウンタに頼らずミリ秒時刻ベースの
    値にフォールバックし、重複よりも安全側に倒す。
    """
    global _p_no
    got_lock = _acquire_p_no_lock(timeout=5.0)
    try:
        if got_lock:
            try:
                saved = int(_P_NO_FILE.read_text().strip())
                _p_no = max(_p_no, saved) + 1
            except Exception:
                _p_no += 1
            try:
                _P_NO_FILE.write_text(str(_p_no))
            except Exception:
                pass
        else:
            _p_no = max(_p_no + 1, int(time.time() * 1000))
    finally:
        if got_lock:
            _release_p_no_lock()
    return str(_p_no)


def _p_sd_date():
    t = datetime.now()
    return (f"{t.year}.{t.month:02}.{t.day:02}"
            f"-{t.hour:02}:{t.minute:02}:{t.second:02}"
            f".{t.microsecond // 1000:03}")


_API_CALL_LOG_DIR = Path(__file__).parent / "logs"


def log_api_call(source):
    """立花証券APIへの呼び出し1回ごとに日次ログへ追記する（2026-08-12追加）。
    daytime.py等の複数プロセスから同時に呼ばれるため、DB書き込みではなく
    単純な追記のみのテキストファイルにして競合を避ける。件数確認は
    check_api_calls.py で当日ログの行数を数えるだけでよい。"""
    try:
        now = datetime.now(JST)
        _API_CALL_LOG_DIR.mkdir(exist_ok=True)
        path = _API_CALL_LOG_DIR / f"api_calls_{now.strftime('%Y%m%d')}.log"
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{now.strftime('%Y-%m-%d %H:%M:%S')},{source}\n")
    except Exception:
        pass  # カウント失敗が本番取引を止めてはならない


def _api_get(url, source="tachibana_order"):
    http = urllib3.PoolManager()
    resp = http.request("GET", url, timeout=urllib3.Timeout(connect=5, read=10))
    log_api_call(source)
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


def load_url_master():
    """tachibana_login_response.json からマスタ問合せURLを読み込む（v4r10）。"""
    try:
        with open(LOGIN_FILE, encoding="utf-8") as f:
            data = json.load(f)
        url = data.get("sUrlMaster", "")
        return url if url else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# 規制情報は v4r10 で銘柄指定不可の全銘柄一括取得になったため、
# closing_watch が複数銘柄を連続発注する間だけ結果をキャッシュする。
_KISEI_CACHE = {"ts": 0.0, "by_code": None}
_KISEI_TTL_SEC = 120


def _fetch_kisei_map():
    """CLMStkGetIssueSizyouKiseiKabu（全銘柄・市場別規制情報）を取得し
    {code: {...}} を返す。取得失敗時は None。"""
    now = time.time()
    if _KISEI_CACHE["by_code"] is not None and now - _KISEI_CACHE["ts"] < _KISEI_TTL_SEC:
        return _KISEI_CACHE["by_code"]
    url_master = load_url_master()
    if not url_master:
        return None
    params = (
        "{"
        f'"p_no":"{_next_p_no()}",'
        f'"p_sd_date":"{_p_sd_date()}",'
        '"sCLMID":"CLMStkGetIssueSizyouKiseiKabu",'
        '"sJsonOfmt":"5"'
        "}"
    )
    try:
        result = _api_get(url_master + "?" + params, source="tachibana_order.check_kisoku")
        if result.get("p_errno", "0") != "0":
            return None
        by_code = {}
        for item in result.get("aCLMStkIssueSizyouKiseiKabu", []):
            c = str(item.get("sIssueCode", "")).strip('"')
            if c:
                by_code[c] = item
        _KISEI_CACHE.update(ts=now, by_code=by_code)
        return by_code
    except Exception:
        return None


def check_kisoku_before_order(url_request, code):
    """
    発注前に立花APIで現物買付の取引規制を確認する。

    v4r10: CLMIssueSizyouKiseiKabu（銘柄指定）→ CLMStkGetIssueSizyouKiseiKabu
    （全銘柄一括・マスタURL）。応答の該当銘柄行の現物買付区分を見る。

    Returns:
        {"blocked": bool, "reason": str}
        API取得失敗時は blocked=False（確認できなかった旨のみ警告）
    """
    code = str(code)
    kisei = _fetch_kisei_map()
    if kisei is None:
        print(f"  [規制チェック] {code}: 規制情報取得不可 → 発注許可")
        return {"blocked": False, "reason": "規制情報取得不可（発注許可）"}

    row = kisei.get(code)
    if row is None:
        print(f"  [規制チェック] {code}: 規制情報に該当なし → 発注許可")
        return {"blocked": False, "reason": "規制情報に該当なし（発注許可）"}

    # sTeisiKubun: 0=通常 1=取引禁止 2=成行禁止 3=端株禁止
    # 発注はすべて成行買いのため、現物買付が 1（禁止）または 2（成行禁止）なら中止。
    genbutu_kai = str(row.get("sGenbutuKaituke", "0") or "0")
    teisi       = str(row.get("sTeisiKubun", "0") or "0")
    if genbutu_kai in ("1", "2") or teisi in ("1", "2"):
        return {"blocked": True,
                "reason": f"取引停止・制限中 (sTeisiKubun={teisi} sGenbutuKaituke={genbutu_kai})"}

    print(f"  [規制チェック] {code}: 規制なし ✅")
    return {"blocked": False, "reason": "規制なし"}


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


def get_positions(url_request):
    """現物保有株一覧を照会して返す。
    戻り値: [{"code", "name", "shares", "buy_price", "current_price", "pnl_pct"}, ...]

    v4r10: CLMKabuZanList → CLMGenbutuKabuList（配列 aGenbutuKabuList）。
    項目も sUriOrder* prefix に変更。銘柄名は応答に含まれないため "" を返す。
    """
    params = (
        "{"
        f'"p_no":"{_next_p_no()}",'
        f'"p_sd_date":"{_p_sd_date()}",'
        '"sCLMID":"CLMGenbutuKabuList",'
        '"sIssueCode":"",'
        '"sJsonOfmt":"5"'
        "}"
    )
    try:
        result = _api_get(url_request + "?" + params)
        if result.get("p_errno", "0") != "0":
            return []
        positions = []
        for item in result.get("aGenbutuKabuList", []):
            def _f(k):
                v = item.get(k, "")
                if isinstance(v, str):
                    v = v.strip('"')
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return None
            code        = str(item.get("sUriOrderIssueCode", "")).strip('"')
            name        = ""
            shares      = int(_f("sUriOrderZanKabuSuryou") or 0)
            buy_price   = _f("sUriOrderGaisanBokaTanka")
            current     = _f("sUriOrderHyoukaTanka") or buy_price
            pnl_pct     = round((current - buy_price) / buy_price * 100, 2) if (buy_price and current) else None
            if code and shares > 0:
                positions.append({
                    "code":          code,
                    "name":          name,
                    "shares":        shares,
                    "buy_price":     buy_price,
                    "current_price": current,
                    "pnl_pct":       pnl_pct,
                })
        return positions
    except Exception:
        return []


def get_margin_positions(url_request):
    """信用建玉一覧を照会して返す。
    戻り値: [{"code", "name", "shares", "buy_price", "current_price", "pnl_pct", "account_type", "tategyoku_no"}, ...]
    account_type は常に "shinyou"。

    v4r10: CLMShinyouZanList → CLMShinyouTategyokuList（配列 aShinyouTategyokuList）。
    項目も sOrder* prefix に変更。銘柄名は応答に含まれないため "" を返す。
    """
    params = (
        "{"
        f'"p_no":"{_next_p_no()}",'
        f'"p_sd_date":"{_p_sd_date()}",'
        '"sCLMID":"CLMShinyouTategyokuList",'
        '"sIssueCode":"",'
        '"sJsonOfmt":"5"'
        "}"
    )
    try:
        result = _api_get(url_request + "?" + params)
        if result.get("p_errno", "0") != "0":
            return []
        positions = []
        for item in result.get("aShinyouTategyokuList", []):
            def _f(k):
                v = item.get(k, "")
                if isinstance(v, str):
                    v = v.strip('"')
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return None
            code      = str(item.get("sOrderIssueCode", "")).strip('"')
            name      = ""
            shares    = int(_f("sOrderTategyokuSuryou") or 0)
            buy_price = _f("sOrderTategyokuTanka")
            current   = _f("sOrderHyoukaTanka") or buy_price
            tategyoku_no = str(item.get("sOrderTategyokuNumber", "")).strip('"')
            pnl_pct   = round((current - buy_price) / buy_price * 100, 2) if (buy_price and current) else None
            if code and shares > 0:
                positions.append({
                    "code":          code,
                    "name":          name,
                    "shares":        shares,
                    "buy_price":     buy_price,
                    "current_price": current,
                    "pnl_pct":       pnl_pct,
                    "account_type":  "shinyou",
                    "tategyoku_no":  tategyoku_no,
                })
        return positions
    except Exception:
        return []


def get_all_positions(url_request):
    """現物・信用の全保有株を統合して返す。"""
    genbutsu = get_positions(url_request)
    for p in genbutsu:
        p["account_type"] = "genbutsu"
    shinyou = get_margin_positions(url_request)
    return genbutsu + shinyou


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
    # 指値: sCondition="1", sOrderPrice=str(price)
    if price is None:
        condition   = "0"
        order_price = "0"
    else:
        condition   = "1"
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


def get_true_buy_price(url_request, code, account_type="genbutsu", max_retry=3, wait_secs=3):
    """買い注文の実際の平均取得単価をAPIから取得する（2026-08-14追加、Fix②）。

    position_monitor.fetch_api_holdings()が使っているのと同じCLMKabuZanList/
    CLMShinyouZanListのsHiritsuTanka/sTategyokuTanka（分割約定も加味した真の
    平均取得単価）を、発注直後に1銘柄だけ照会する。反映までに数秒かかることが
    あるためリトライする。取得できなければNoneを返し、呼び出し側は気配値の
    ままフォールバックする（[[bug_buy_price_inaccuracy]]のFix①と同じ仕組みを
    日中の発注直後にも適用したもの）。
    """
    # v4r10: CLMKabuZanList→CLMGenbutuKabuList / CLMShinyouZanList→CLMShinyouTategyokuList
    clm_id, list_key, code_key, price_key = (
        ("CLMShinyouTategyokuList", "aShinyouTategyokuList", "sOrderIssueCode", "sOrderTategyokuTanka")
        if account_type == "shinyou" else
        ("CLMGenbutuKabuList", "aGenbutuKabuList", "sUriOrderIssueCode", "sUriOrderGaisanBokaTanka")
    )
    for attempt in range(max_retry):
        if attempt > 0:
            time.sleep(wait_secs)
        time.sleep(0.6)   # p_sd_date競合（p_errno=6/8）を避けるため他プロセスと時間をずらす
                          # （position_monitor.fetch_api_holdings()と同じ対策）
        params = (
            "{"
            f'"p_no":"{_next_p_no()}",'
            f'"p_sd_date":"{_p_sd_date()}",'
            f'"sCLMID":"{clm_id}",'
            '"sIssueCode":"",'
            '"sJsonOfmt":"5"'
            "}"
        )
        try:
            result = _api_get(url_request + "?" + params, source="tachibana_order.get_true_buy_price")
            if result.get("p_errno", "0") != "0":
                continue
            for item in result.get(list_key, []):
                item_code = str(item.get(code_key, "")).strip('"')
                if item_code != str(code):
                    continue
                v = item.get(price_key, "")
                if isinstance(v, str):
                    v = v.strip('"')
                try:
                    price = float(v)
                    if price > 0:
                        return price
                except (TypeError, ValueError):
                    continue
        except Exception:
            continue
    return None


def get_true_execution_price(url_request, order_no, eigyou_day, max_retry=3, wait_secs=3):
    """売り注文の実際の約定単価をCLMOrderListDetailから取得する（2026-08-15追加、Fix③）。

    Fix②(get_true_buy_price)はCLMKabuZanList（保有残高）を使うため、全数売却して
    残高がゼロになる売り注文には使えない。代わりに注文番号で約定明細を直接照会する
    CLMOrderListDetailを使う。aYakuzyouSikkouList（約定失効リスト）を数量加重平均する
    ことで、分割約定（例: 200株@298円+100株@297円）も加味した真の平均約定単価になる。
    取得できなければNoneを返し、呼び出し側は気配値のままフォールバックする
    （[[bug_buy_price_inaccuracy]]参照）。
    """
    for attempt in range(max_retry):
        if attempt > 0:
            time.sleep(wait_secs)
        time.sleep(0.6)   # p_sd_date競合（p_errno=6/8）を避けるため他プロセスと時間をずらす
        params = (
            "{"
            f'"p_no":"{_next_p_no()}",'
            f'"p_sd_date":"{_p_sd_date()}",'
            '"sCLMID":"CLMOrderListDetail",'
            f'"sOrderNumber":"{order_no}",'
            f'"sEigyouDay":"{eigyou_day}",'
            '"sJsonOfmt":"5"'
            "}"
        )
        try:
            result = _api_get(url_request + "?" + params, source="tachibana_order.get_true_execution_price")
            if result.get("p_errno", "0") != "0":
                continue
            total_qty = 0.0
            total_amt = 0.0
            for fill in result.get("aYakuzyouSikkouList", []):
                try:
                    qty = float(str(fill.get("sYakuzyouSuryou", "0")).strip('"'))
                    px  = float(str(fill.get("sYakuzyouPrice", "0")).strip('"'))
                except (TypeError, ValueError):
                    continue
                if qty > 0 and px > 0:
                    total_qty += qty
                    total_amt += qty * px
            if total_qty > 0:
                return total_amt / total_qty
            # 約定失効リストが空の場合のフォールバック（トップレベルのsYakuzyouPrice）
            v = result.get("sYakuzyouPrice", "")
            if isinstance(v, str):
                v = v.strip('"')
            try:
                price = float(v)
                if price > 0:
                    return price
            except (TypeError, ValueError):
                pass
        except Exception:
            continue
    return None


def save_position(code, name, shares, buy_price, strategy, tp_pct=0.03, sl_pct=0.05,
                  entry_change_pct=None, rb_score=None, condition=None,
                  url_request=None, account_type="genbutsu",
                  order_no=None, eigyou_day=None):
    """買い成功後にポジションを DB へ記録する。

    url_request指定時は、発注直前の気配値ではなく実際の約定平均単価で
    buy_price/tp_price/sl_priceを補正する（2026-08-14追加、Fix②）。
    当日完結取引（戦略A/AN/AS/D）はこれまで気配値のまま記録され、実額と
    ズレていた（[[bug_buy_price_inaccuracy]]参照）。補正に失敗しても
    本体の発注処理は継続する（気配値のまま記録・従来動作にフォールバック）。

    order_no指定時は、CLMKabuZanList（保有残高照会、get_true_buy_price）ではなく
    CLMOrderListDetail（注文明細照会、get_true_execution_price）で補正する
    （2026-08-25修正）。CLMKabuZanListは取引時間中は使用不可（p_errno=8）と
    実機で確認されたため、Fix②は8/17のデプロイ以来一度も補正に成功していなかった。
    CLMOrderListDetailは売り側のFix③で取引時間中の動作が実証済みのため、
    買い側もこちらに統一する。order_no未指定時（呼び出し元が未対応）は
    従来のget_true_buy_price（日をまたいだポジション向け）にフォールバックする。
    """
    db.init_db()
    db.save_position_db(
        code, name, shares, buy_price, strategy,
        tp_pct=tp_pct, sl_pct=sl_pct,
        entry_change_pct=entry_change_pct,
        rb_score=rb_score,
        condition=condition,
        account_type=account_type,
    )

    if url_request:
        try:
            if order_no:
                eday = eigyou_day or datetime.now().strftime("%Y%m%d")
                true_price = get_true_execution_price(url_request, order_no, eday)
            else:
                true_price = get_true_buy_price(url_request, code, account_type=account_type)
            if true_price and abs(true_price - float(buy_price)) >= 0.5:
                today  = datetime.now().strftime("%Y-%m-%d")
                new_tp = round(true_price * (1 + tp_pct))
                new_sl = round(true_price * (1 - sl_pct))
                db.update_position(today, code, buy_price=true_price,
                                   tp_price=new_tp, sl_price=new_sl)
                print(f"  🔧 [{code}] 約定単価を補正: {float(buy_price):.1f}円 → {true_price:.1f}円 "
                      f"（TP:{new_tp}円 SL:{new_sl}円 に再計算）")
        except Exception as e:
            print(f"  ⚠️ [{code}] 約定単価の補正確認に失敗（気配値のまま記録・処理は継続）: {e}")


def place_sell_order(url_request, code, shares, price=None, market_code=None,
                     account_type="genbutsu", zyoutoeki_c=None):
    """成行売り注文を発注する。戻り値は place_buy_order と同じ構造。
    account_type: "genbutsu"=現物売り, "shinyou"=信用返済売り
    zyoutoeki_c : "1"=特定(源泉あり), "2"=特定(源泉なし), "3"=一般, "0"=NISA
                  None の場合はグローバル設定（ZYOUTOEKI_C）を使用
    """
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
        condition   = "1"
        order_price = str(int(price))

    # 現物="0", 信用返済="2"
    genkin_shinyou = "2" if account_type == "shinyou" else "0"
    tatebi_type    = "*"
    # 税区分: 引数指定があればそちらを優先、なければグローバル設定
    zyoutoeki = zyoutoeki_c if zyoutoeki_c is not None else ZYOUTOEKI_C

    params = (
        "{"
        f'"p_no":"{_next_p_no()}",'
        f'"p_sd_date":"{_p_sd_date()}",'
        '"sCLMID":"CLMKabuNewOrder",'
        f'"sZyoutoekiKazeiC":"{zyoutoeki}",'
        f'"sIssueCode":"{code}",'
        '"sSizyouC":"00",'
        '"sBaibaiKubun":"1",'
        f'"sCondition":"{condition}",'
        f'"sOrderPrice":"{order_price}",'
        f'"sOrderSuryou":"{shares}",'
        f'"sGenkinShinyouKubun":"{genkin_shinyou}",'
        '"sOrderExpireDay":"0",'
        '"sGyakusasiOrderType":"0",'
        '"sGyakusasiZyouken":"0",'
        '"sGyakusasiPrice":"*",'
        f'"sTatebiType":"{tatebi_type}",'
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
    """注文照会（約定確認）。約定済みなら True を返す。

    v4r10: CLMKabuOrderList → CLMOrderList（配列 aOrderList）。
    注文番号での絞り込み引数が無くなったため全件取得し
    sOrderOrderNumber で突き合わせる。
    """
    params = (
        "{"
        f'"p_no":"{_next_p_no()}",'
        f'"p_sd_date":"{_p_sd_date()}",'
        '"sCLMID":"CLMOrderList",'
        '"sIssueCode":"",'
        '"sSikkouDay":"",'
        '"sOrderSyoukaiStatus":"",'
        '"sJsonOfmt":"5"'
        "}"
    )
    try:
        result = _api_get(url_request + "?" + params)
        orders = result.get("aOrderList", [])
        for o in orders:
            if str(o.get("sOrderOrderNumber", "")).strip('"') == str(order_no):
                status = o.get("sOrderStatus", "")
                yakujo = str(o.get("sOrderYakuzyouSuryo", "0")).strip('"')
                return {
                    "status":   status,
                    "filled":   int(float(yakujo)) if yakujo else 0,
                    "raw":      o,
                }
    except Exception:
        pass
    return {"status": "不明", "filled": 0, "raw": {}}


def cancel_order(url_request, order_no, max_retry=5, eigyou_day=None):
    """注文取消。p_errno=8（タイムスタンプ競合）は最大 max_retry 回リトライ。
    戻り値: {"success": bool, "message": str, "raw": dict}

    v4r10: CLMKabuCancel → CLMKabuCancelOrder。sEigyouDay が必須になったため
    指定が無ければ当日（JST）を使う（指値フォールバックは当日中の取消のみ）。
    """
    if not eigyou_day:
        eigyou_day = datetime.now(JST).strftime("%Y%m%d")
    for attempt in range(max_retry):
        if attempt > 0:
            time.sleep(2.0)   # position_monitor(60秒ループ)との競合を回避するため2秒待機
        params = (
            "{"
            f'"p_no":"{_next_p_no()}",'
            f'"p_sd_date":"{_p_sd_date()}",'
            '"sCLMID":"CLMKabuCancelOrder",'
            f'"sOrderNumber":"{order_no}",'
            f'"sEigyouDay":"{eigyou_day}",'
            f'"sSecondPassword":"{SECOND_PASSWORD}",'
            '"sJsonOfmt":"5"'
            "}"
        )
        try:
            result   = _api_get(url_request + "?" + params)
            p_errno  = result.get("p_errno", "")
            s_result = result.get("sResultCode", "")
            if p_errno == "0" and s_result == "0":
                return {"success": True, "message": f"注文 {order_no} を取消しました", "raw": result}
            if p_errno == "8" and attempt < max_retry - 1:
                continue   # タイムスタンプ競合 → リトライ
            err_msg = result.get("sResultText", "") or result.get("sErrMessage", "")
            return {"success": False,
                    "message": f"取消エラー (p_errno={p_errno} sResultCode={s_result}): {err_msg}",
                    "raw": result}
        except Exception as e:
            return {"success": False, "message": f"通信エラー: {e}", "raw": {}}


def place_buy_limit_with_fallback(url_request, code, shares, price, market_code,
                                  wait_secs=30, strategy="?"):
    """指値買い→未約定時に成行フォールバック。
    戻り値:
      {"success": bool, "fill_price": int, "fill_type": "limit"|"market"|"partial",
       "filled": int, "order_no": str, "message": str}
    """
    # ── 指値発注 ──────────────────────────────────────────
    r = place_buy_order(url_request, code, shares, price=price, market_code=market_code)
    if not r["success"]:
        return {"success": False, "fill_price": 0, "fill_type": "none",
                "filled": 0, "order_no": "", "message": r["message"]}

    order_no = r.get("order_no", "")
    eigyou_day = (r.get("raw") or {}).get("sEigyouDay") or None
    print(f"     指値発注: 注文番号 {order_no}  {wait_secs}秒後に約定確認...")
    time.sleep(wait_secs)

    check  = check_order(url_request, order_no)
    filled = check.get("filled", 0)

    if filled >= shares:
        return {"success": True, "fill_price": int(price), "fill_type": "limit",
                "filled": filled, "order_no": order_no,
                "message": f"指値約定: {shares}株 @ {price:.0f}円"}

    if filled > 0:
        return {"success": True, "fill_price": int(price), "fill_type": "partial",
                "filled": filled, "order_no": order_no,
                "message": f"一部約定: {filled}株 / {shares}株 @ {price:.0f}円"}

    # ── 未約定 → 取消して成行フォールバック ──────────────
    order_status = check.get("status", "")
    print(f"     {wait_secs}秒経過しても未約定（sOrderStatus={order_status!r}）→ 取消して成行に切り替え")
    cancel = cancel_order(url_request, order_no, eigyou_day=eigyou_day)
    if not cancel["success"]:
        # キャンセル失敗時は注文状態を再確認する
        # 受付エラー・既取消など「非アクティブ」ならそのまま成行発注して問題ない
        recheck = check_order(url_request, order_no)
        re_status  = recheck.get("status", "")
        re_filled  = recheck.get("filled", 0)
        if re_filled > 0:
            # 再確認で約定が判明したケース（稀）
            return {"success": True, "fill_price": int(price), "fill_type": "limit",
                    "filled": re_filled, "order_no": order_no,
                    "message": f"遅延約定検出: {re_filled}株 @ {price:.0f}円"}
        # 受付エラー(4)・取消済み(3)・失効(5)等の終端ステータス、または注文が見つからない場合は
        # 注文はすでに非アクティブなので手動取消不要・成行フォールバック続行
        print(f"     取消失敗（再確認ステータス={re_status!r}）→ 注文は非アクティブと判断し成行フォールバック続行")

    r2 = place_buy_order(url_request, code, shares, market_code=market_code)
    if r2["success"]:
        return {"success": True, "fill_price": int(price), "fill_type": "market",
                "filled": shares, "order_no": r2.get("order_no", ""),
                "message": f"成行発注（フォールバック）: {shares}株"}
    return {"success": False, "fill_price": 0, "fill_type": "none",
            "filled": 0, "order_no": "", "message": f"成行発注失敗: {r2['message']}"}
