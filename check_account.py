# -*- coding: utf-8 -*-
import sys, json, urllib3, time
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
JST = timezone(timedelta(hours=9))

d = json.load(open('tachibana_login_response.json', encoding='utf-8'))
url_request = d['sUrlRequest']
P_NO = Path('out/last_p_no.txt')

def next_pno():
    try:
        n = int(P_NO.read_text().strip()) + 1
    except Exception:
        n = int(time.time())
    P_NO.write_text(str(n))
    t = datetime.now(JST)
    return str(n), f'{t.year}.{t.month:02}.{t.day:02}-{t.hour:02}:{t.minute:02}:{t.second:02}.{t.microsecond//1000:03}'

http = urllib3.PoolManager()

def api(pno, psd, clmid, extra=''):
    params = '{' + f'"p_no":"{pno}","p_sd_date":"{psd}","sCLMID":"{clmid}"{extra},"sJsonOfmt":"5"' + '}'
    resp = http.request('GET', url_request + '?' + params, timeout=urllib3.Timeout(connect=5, read=15))
    return json.loads(resp.data.decode('shift-jis', errors='ignore'))

# 1. 今日の注文一覧
print('=== 本日の注文一覧 ===')
pno, psd = next_pno()
r = api(pno, psd, 'CLMKabuOrderList')
orders = r.get('aCLMKabuOrderList', [])
print(f'注文件数: {len(orders)}')
for o in orders:
    code   = o.get('sIssueCode', '')
    baibai = '買' if o.get('sBaibaiKubun') == '3' else '売'
    status = o.get('sOrderStatus', '')
    qty    = o.get('sOrderSuryou', '')
    yakujo = o.get('sYakujosu', '0')
    price  = o.get('sYakujoPrice', '') or o.get('sOrderPrice', '')
    order_no = o.get('sOrderNo', '')
    print(f'  [{order_no}] {code} {baibai} {qty}株 約定{yakujo}株 価格{price}円 状態:{status}')

time.sleep(0.5)

# 2. 現物保有残高
print('\n=== 現物保有残高 ===')
pno, psd = next_pno()
r = api(pno, psd, 'CLMGenkabuZan')
items = r.get('aCLMGenkabuZan', [])
print(f'p_errno={r.get("p_errno")} 保有銘柄: {len(items)}')
for item in items:
    code  = item.get('sIssueCode', '')
    name  = item.get('sIssueName', '')
    qty   = item.get('sZansuKabu', '')
    avg   = item.get('sTyokkaHeikinTanka', '') or item.get('sHeikinTanka', '')
    print(f'  {code} {name} {qty}株 平均{avg}円')
