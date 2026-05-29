# -*- coding: utf-8 -*-
"""
正しいパラメータ名でCLMKabuNewOrderをテスト
sSizyouC / sBaibaiKubun / sCondition / sOrderPrice / sOrderSuryou / sGenkinShinyouKubun
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import json, urllib3, time, os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

LOGIN_FILE = Path(__file__).parent / "tachibana_login_response.json"
P_NO_FILE  = Path(__file__).parent / "out" / "last_p_no.txt"

data = json.loads(LOGIN_FILE.read_text(encoding="utf-8"))
url_request = data["sUrlRequest"]
zyoutoeki_c = data.get("sZyoutoekiKazeiC", "1")
second_pass = os.getenv("TACHIBANA_SECOND_PASSWORD", "")

print(f"sZyoutoekiKazeiC: {zyoutoeki_c}")

http = urllib3.PoolManager()

def next_pno():
    try:
        n = int(P_NO_FILE.read_text().strip()) + 1
    except Exception:
        n = int(time.time())
    # daytime.pyと競合しないよう少し大きい値にする
    n = max(n, int(time.time() % 1_000_000) + 1_780_015_000)
    P_NO_FILE.write_text(str(n))
    t = datetime.now()
    psd = f"{t.year}.{t.month:02}.{t.day:02}-{t.hour:02}:{t.minute:02}:{t.second:02}.{t.microsecond//1000:03}"
    return str(n), psd

def api_get(url):
    resp = http.request("GET", url, timeout=urllib3.Timeout(connect=5, read=10))
    return json.loads(resp.data.decode("shift-jis", errors="ignore"))

today = datetime.now().strftime("%Y%m%d")

# 正しいパラメータで注文テスト
# sSizyouC="00" (東証), sBaibaiKubun="3" (買), sCondition="0" (成行), sGenkinShinyouKubun="0" (現物)
pno, psd = next_pno()
params = (
    "{"
    f'"p_no":"{pno}",'
    f'"p_sd_date":"{psd}",'
    '"sCLMID":"CLMKabuNewOrder",'
    f'"sZyoutoekiKazeiC":"{zyoutoeki_c}",'
    '"sIssueCode":"6081",'
    '"sSizyouC":"00",'
    '"sBaibaiKubun":"3",'
    '"sCondition":"0",'
    '"sOrderPrice":"0",'
    '"sOrderSuryou":"100",'
    '"sGenkinShinyouKubun":"0",'
    '"sOrderExpireDay":"0",'
    '"sGyakusasiOrderType":"0",'
    '"sGyakusasiZyouken":"0",'
    '"sGyakusasiPrice":"*",'
    '"sTatebiType":"*",'
    f'"sSecondPassword":"{second_pass}",'
    '"sJsonOfmt":"5"'
    "}"
)
print(f"\n[テスト1] 正しいパラメータ (sSizyouC=00, sBaibaiKubun=3, sGyakusasiSasiNe=*)")
r = api_get(url_request + "?" + params)
print(f"  p_errno={r.get('p_errno')} sResultCode={r.get('sResultCode')} sResultText={r.get('sResultText', '')}")
print(f"  sOrderNumber={r.get('sOrderNumber', '（なし）')}")
print(f"  raw: {r}")
