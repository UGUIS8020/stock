# -*- coding: utf-8 -*-
import sys, json, urllib3, time
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

LOGIN_FILE = Path(__file__).parent / "tachibana_login_response.json"
P_NO_FILE  = Path(__file__).parent / "out" / "last_p_no.txt"

data = json.loads(LOGIN_FILE.read_text(encoding="utf-8"))
url_request = data["sUrlRequest"]

http = urllib3.PoolManager()

try:
    n = int(P_NO_FILE.read_text().strip()) + 1
except Exception:
    n = int(time.time())
P_NO_FILE.write_text(str(n))
t = datetime.now()
psd = f"{t.year}.{t.month:02}.{t.day:02}-{t.hour:02}:{t.minute:02}:{t.second:02}.{t.microsecond//1000:03}"

# 注文一覧取得
params = (
    "{"
    f'"p_no":"{n}",'
    f'"p_sd_date":"{psd}",'
    '"sCLMID":"CLMKabuOrderList",'
    '"sJsonOfmt":"5"'
    "}"
)
resp = http.request("GET", url_request + "?" + params, timeout=urllib3.Timeout(connect=5, read=10))
result = json.loads(resp.data.decode("shift-jis", errors="ignore"))
print(f"p_errno={result.get('p_errno')}")
orders = result.get("aCLMKabuOrderList", [])
print(f"注文件数: {len(orders)}")
for o in orders:
    print(f"  注文番号={o.get('sOrderNo')} 銘柄={o.get('sIssueCode')} 状態={o.get('sOrderStatus')} "
          f"注文数={o.get('sOrderSuryou')} 約定数={o.get('sYakujosu')} "
          f"売買={o.get('sBaibaiKubun')} 価格={o.get('sOrderPrice')}")
