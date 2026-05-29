# -*- coding: utf-8 -*-
"""
check_market_code.py - Tachibana APIから正しい市場コードを確認する診断スクリプト

実行方法:
    python check_market_code.py

事前条件: tachibana_login.py でログイン済みであること
"""
import sys, json, time
import urllib3
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

LOGIN_FILE = Path(__file__).parent / "tachibana_login_response.json"
P_NO_FILE  = Path(__file__).parent / "out" / "last_p_no.txt"

TEST_CODES = [
    ("9204", "スカイマーク",    "Growth/0113"),
    ("4885", "室町ケミカル",    "Standard/0112"),
    ("9278", "ブックオフ",      "Prime/0111"),
    ("7370", "Enjin",          "Growth/0113"),
    ("8908", "毎日コムネット",   "Standard/0112"),
]

def load_urls():
    data = json.loads(LOGIN_FILE.read_text(encoding="utf-8"))
    return data.get("sUrlRequest", ""), data.get("sUrlMaster", "")

def _p_no():
    try:
        n = int(P_NO_FILE.read_text().strip()) + 1
    except Exception:
        n = int(time.time())
    P_NO_FILE.write_text(str(n))
    return str(n)

def _p_sd():
    from datetime import datetime
    t = datetime.now()
    return f"{t.year}.{t.month:02}.{t.day:02}-{t.hour:02}:{t.minute:02}:{t.second:02}.{t.microsecond//1000:03}"

def api_get(url):
    http = urllib3.PoolManager()
    resp = http.request("GET", url, timeout=urllib3.Timeout(connect=5, read=10))
    return json.loads(resp.data.decode("shift-jis", errors="ignore"))

def query_price_with_market(url_request, code):
    """pMKC フィールドで市場コードを価格APIから取得する"""
    params = (
        "{"
        f'"p_no":"{_p_no()}",'
        f'"p_sd_date":"{_p_sd()}",'
        '"sCLMID":"CLMMfdsGetMarketPrice",'
        f'"sTargetIssueCode":"{code}",'
        '"sTargetColumn":"pMKC,pDPP,pPRP,pITN,pQAP",'
        '"sJsonOfmt":"5"'
        "}"
    )
    try:
        return api_get(url_request + "?" + params)
    except Exception as e:
        return {"error": str(e)}

def query_issue_master(url_master, code):
    """CLMMfdsGetIssue で銘柄マスターから市場コードを取得"""
    params = (
        "{"
        f'"p_no":"{_p_no()}",'
        f'"p_sd_date":"{_p_sd()}",'
        '"sCLMID":"CLMMfdsGetIssue",'
        f'"sIssueCode":"{code}",'
        '"sJsonOfmt":"5"'
        "}"
    )
    try:
        return api_get(url_master + "?" + params)
    except Exception as e:
        return {"error": str(e)}

def main():
    print("=" * 60)
    print("  Tachibana API 市場コード確認スクリプト")
    print("=" * 60)

    try:
        url_request, url_master = load_urls()
    except Exception as e:
        print(f"ログインファイル読み込み失敗: {e}")
        print("先に tachibana_login.py を実行してください")
        return

    if not url_request:
        print("sUrlRequest が空です。再ログインしてください")
        return

    print(f"\n  sUrlRequest: {url_request[:60]}...")
    print(f"  sUrlMaster:  {url_master[:60]}..." if url_master else "  sUrlMaster: なし")

    # 1. pMKC フィールドで市場コードを価格APIから確認
    print(f"\n【1】価格API（pMKC フィールド）")
    print("-" * 60)
    for code, name, mkt_info in TEST_CODES:
        result = query_price_with_market(url_request, code)
        p_errno = result.get("p_errno", "?")
        if p_errno == "0":
            items = result.get("aCLMMfdsMarketPrice", [])
            if items:
                item = items[0]
                pmkc = item.get("pMKC", "(なし)")
                pdpp = item.get("pDPP", "")
                print(f"  {code} {name} ({mkt_info}): pMKC={pmkc}  現在値={pdpp}")
            else:
                print(f"  {code} {name}: aCLMMfdsMarketPrice が空  raw={str(result)[:120]}")
        else:
            err = result.get("p_err", "")
            print(f"  {code} {name}: 失敗 p_errno={p_errno} {err}")
        time.sleep(0.5)

    # 2. マスターAPIから確認（利用可能な場合）
    if url_master:
        print(f"\n【2】マスターAPI（CLMMfdsGetIssue）")
        print("-" * 60)
        for code, name, mkt_info in TEST_CODES[:3]:
            result = query_issue_master(url_master, code)
            p_errno = result.get("p_errno", "?")
            if p_errno == "0":
                issue = result.get("CLMMfdsIssue", result.get("CLMMfdsGetIssue", {}))
                # マーケット関連フィールドを全て出力
                mkt_fields = {k: v for k, v in (issue if isinstance(issue, dict) else result).items()
                              if any(x in k.lower() for x in ["mkt", "mrc", "mrc", "shij"])}
                print(f"  {code} {name}: {mkt_fields}")
                if not mkt_fields:
                    print(f"    全フィールド: {list((issue if isinstance(issue, dict) else result).keys())[:20]}")
            else:
                err = result.get("p_err", "")
                print(f"  {code} {name}: 失敗 p_errno={p_errno} {err}  raw={str(result)[:150]}")
            time.sleep(0.5)

    print(f"\n{'=' * 60}")
    print("完了")
    print()
    print("pMKC の値が確認できた場合、それが Tachibana の正しい sMarketCode です。")
    print("db.py の _JQUANTS_TO_TACHIBANA マッピングをその値で更新してください。")

if __name__ == "__main__":
    main()
