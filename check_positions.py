# -*- coding: utf-8 -*-
import sys, json, urllib3, time
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
JST = timezone(timedelta(hours=9))

d = json.load(open('tachibana_login_response.json', encoding='utf-8'))
url_price = d['sUrlPrice']

PM_FILE = Path('out/last_p_no.txt')
try:
    n = int(PM_FILE.read_text().strip()) + 1
except Exception:
    n = int(time.time())
PM_FILE.write_text(str(n))

t = datetime.now(JST)
psd = f'{t.year}.{t.month:02}.{t.day:02}-{t.hour:02}:{t.minute:02}:{t.second:02}.{t.microsecond//1000:03}'
params = (
    '{'
    f'"p_no":"{n}",'
    f'"p_sd_date":"{psd}",'
    '"sCLMID":"CLMMfdsGetMarketPrice",'
    '"sTargetIssueCode":"2462,6203",'
    '"sTargetColumn":"pDPP,pPRP",'
    '"sJsonOfmt":"5"'
    '}'
)

http = urllib3.PoolManager()
resp = http.request('GET', url_price + '?' + params, timeout=urllib3.Timeout(connect=5, read=10))
r = json.loads(resp.data.decode('shift-jis', errors='ignore'))

print(f'p_errno={r.get("p_errno")}')
for item in r.get('aCLMMfdsMarketPrice', []):
    code  = item.get('sIssueCode', '')
    price = float(item.get('pDPP') or item.get('pPRP') or 0)
    names = {'2462': 'ライク', '6203': '豊和工業'}
    buy   = {'2462': 1493, '6203': 1783}
    tp    = {'2462': 1538, '6203': 1836}
    sl    = {'2462': 1478, '6203': 1765}
    if price:
        pct = (price - buy[code]) / buy[code] * 100
        status = '✅TP圏' if price >= tp[code] else ('🔴SL圏' if price <= sl[code] else '🟡保有中')
        print(f'  {code} {names.get(code,"")} : {price:.0f}円 ({pct:+.2f}%) {status}')
        print(f'    TP={tp[code]}円  SL={sl[code]}円')
