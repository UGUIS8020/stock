# -*- coding: utf-8 -*-
"""
立花APIで銘柄の取引規制情報を取得・確認するスクリプト
CLMIssueSizyouKiseiKabu / CLMIssueSizyouMstKabu / CLMHosyoukinMst を使用
"""
import sys, json, urllib3, time
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

LOGIN_FILE = Path(__file__).parent / "tachibana_login_response.json"
P_NO_FILE  = Path(__file__).parent / "out" / "last_p_no.txt"

data = json.loads(LOGIN_FILE.read_text(encoding="utf-8"))
url_request = data["sUrlRequest"]
url_master  = data.get("sUrlMaster", "")

http = urllib3.PoolManager()

def next_pno():
    try:
        n = int(P_NO_FILE.read_text().strip()) + 1
    except Exception:
        n = int(time.time())
    P_NO_FILE.write_text(str(n))
    t = datetime.now()
    psd = f"{t.year}.{t.month:02}.{t.day:02}-{t.hour:02}:{t.minute:02}:{t.second:02}.{t.microsecond//1000:03}"
    return str(n), psd

def api_get(url):
    resp = http.request("GET", url, timeout=urllib3.Timeout(connect=5, read=10))
    return json.loads(resp.data.decode("shift-jis", errors="ignore"))

TEST_CODES = [
    ("6081", "アライドアーキテクツ", "規制中？"),
    ("9278", "ブックオフ",           "Prime 正常"),
    ("4885", "室町ケミカル",         "Standard 正常"),
]

print("=" * 70)
print("  立花API 取引規制情報 確認 (専用API使用)")
print("=" * 70)

# 【1】CLMIssueSizyouKiseiKabu（市場別銘柄規制）
print("\n【1】CLMIssueSizyouKiseiKabu（規制情報）")
print("-" * 70)
for code, name, note in TEST_CODES:
    pno, psd = next_pno()
    params = (
        "{"
        f'"p_no":"{pno}",'
        f'"p_sd_date":"{psd}",'
        '"sCLMID":"CLMIssueSizyouKiseiKabu",'
        f'"sIssueCode":"{code}",'
        '"sJsonOfmt":"5"'
        "}"
    )
    r = api_get(url_request + "?" + params)
    p_errno = r.get("p_errno", "?")
    print(f"  {code} {name} ({note}): p_errno={p_errno}")
    if p_errno == "0":
        # レスポンスのキーを全部表示
        keys = [k for k in r.keys() if k not in ("p_no", "p_sd_date", "p_rv_date", "p_errno", "p_err", "sCLMID")]
        print(f"    レスポンスキー: {keys}")
        for k in keys:
            print(f"    {k}: {r[k]}")
    else:
        print(f"    エラー: {r.get('p_err', str(r)[:150])}")
    print()
    time.sleep(0.8)

# 【2】CLMIssueSizyouMstKabu（市場別銘柄マスター）
print("\n【2】CLMIssueSizyouMstKabu（信用区分）")
print("-" * 70)
for code, name, note in TEST_CODES:
    pno, psd = next_pno()
    params = (
        "{"
        f'"p_no":"{pno}",'
        f'"p_sd_date":"{psd}",'
        '"sCLMID":"CLMIssueSizyouMstKabu",'
        f'"sIssueCode":"{code}",'
        '"sJsonOfmt":"5"'
        "}"
    )
    r = api_get(url_request + "?" + params)
    p_errno = r.get("p_errno", "?")
    print(f"  {code} {name} ({note}): p_errno={p_errno}")
    if p_errno == "0":
        keys = [k for k in r.keys() if k not in ("p_no", "p_sd_date", "p_rv_date", "p_errno", "p_err", "sCLMID")]
        for k in keys:
            print(f"    {k}: {r[k]}")
    else:
        print(f"    エラー: {r.get('p_err', str(r)[:150])}")
    print()
    time.sleep(0.8)

# 【3】CLMHosyoukinMst（保証金マスター）
print("\n【3】CLMHosyoukinMst（担保率・増担保規制）")
print("-" * 70)
for code, name, note in TEST_CODES[:2]:
    pno, psd = next_pno()
    params = (
        "{"
        f'"p_no":"{pno}",'
        f'"p_sd_date":"{psd}",'
        '"sCLMID":"CLMHosyoukinMst",'
        f'"sIssueCode":"{code}",'
        '"sJsonOfmt":"5"'
        "}"
    )
    r = api_get(url_request + "?" + params)
    p_errno = r.get("p_errno", "?")
    print(f"  {code} {name} ({note}): p_errno={p_errno}")
    if p_errno == "0":
        keys = [k for k in r.keys() if k not in ("p_no", "p_sd_date", "p_rv_date", "p_errno", "p_err", "sCLMID")]
        for k in keys:
            print(f"    {k}: {r[k]}")
    else:
        print(f"    エラー: {r.get('p_err', str(r)[:150])}")
    print()
    time.sleep(0.8)

print("=" * 70)
