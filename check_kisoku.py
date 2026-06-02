# -*- coding: utf-8 -*-
"""
CLMIssueSizyouKiseiKabu 動作確認スクリプト
"""
import sys, json, urllib3, time
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")

JST = timezone(timedelta(hours=9))
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
    t = datetime.now(JST)
    psd = f"{t.year}.{t.month:02}.{t.day:02}-{t.hour:02}:{t.minute:02}:{t.second:02}.{t.microsecond//1000:03}"
    return str(n), psd

def api_get(url):
    resp = http.request("GET", url, timeout=urllib3.Timeout(connect=5, read=10))
    return json.loads(resp.data.decode("shift-jis", errors="ignore"))

# テスト銘柄
TEST_CODES = [
    ("6081", "アライドアーキテクツ", "信用規制中のはず"),
    ("7203", "トヨタ自動車",         "正常銘柄"),
    ("9278", "ブックオフ",           "正常銘柄"),
]

print("=" * 60)
print("  CLMIssueSizyouKiseiKabu 規制情報API 動作確認")
print("=" * 60)

for code, name, note in TEST_CODES:
    for label, base_url in [("sUrlRequest", url_request), ("sUrlMaster", url_master)]:
        if not base_url:
            continue
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
        r = api_get(base_url + "?" + params)
        p_errno = r.get("p_errno", "?")
        print(f"\n{code} {name}（{note}）[{label}]")
        print(f"  p_errno={p_errno}")
        if p_errno == "0":
            keys = [k for k in r.keys() if k not in ("p_no","p_sd_date","p_rv_date","p_errno","p_err","sCLMID")]
            for k in keys:
                print(f"  {k}: {r[k]}")
        else:
            print(f"  エラー: {r.get('p_err','')[:120]}")
        time.sleep(0.8)
    break  # 最初の銘柄だけ両URL試す

print("\n" + "=" * 60)
