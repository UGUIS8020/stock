# -*- coding: utf-8 -*-
"""
verify_order_roundtrip.py - v4r10移行の「書き込み系」実地検証（発注→照会→取消）

【背景・目的】
tools/probe_v4r10.py は新旧I/Fの読み取り比較のみで、発注・取消（実際に注文を
出す書き込み系API）はカバーしていない。CLMKabuNewOrder自体はv4r10で無変更だが、
check_order（CLMOrderList）・cancel_order（CLMKabuCancelOrder、sEigyouDay必須化）
はv4r10でリネームされており、実データでの検証が済んでいなかった。
このスクリプトは、その空白を埋めるために作成した。

【なぜ約定しない価格で発注するのか】
デモ環境は本番と別のAuthIDが必要で未取得のため、実際に本番アカウントで
発注するしかない。誤って本当に約定してしまわないよう、直近値から-20%の
指値（買いなので下、絶対に約定しない水準）を使う。

【既知の注意点】
- 週末（取引セッションが存在しない日）に発注すると sResultCode=11010
  「執行条件に誤りがあります」で拒否される（2026-09-06に実測確認）。
  平日の市場が開いている時間帯に実行すること。
- 本番の稼働アカウントで実行すると、position_monitor.py 等が使っている
  セッションと競合する可能性がある。本番稼働中のアカウントとは別の口座
  （このリポジトリでは奥様名義の別口座を想定）で実行するのが安全。
- LIVE_TRADING=False の場合は place_buy_order がモック応答を返すだけで
  実際には発注されない（このスクリプト冒頭で状態を表示する）。

【使い方】
    python verify_order_roundtrip.py
"""
import sys, json, time
sys.stdout.reconfigure(encoding="utf-8")
from tachibana_login import do_login, LOGIN_FILE
import tachibana_order as to

print("LIVE_TRADING:", to.LIVE_TRADING)

do_login()
with open(LOGIN_FILE, encoding="utf-8") as f:
    login_data = json.load(f)
url_request = login_data.get("sUrlRequest")
url_price = login_data.get("sUrlPrice")

CODE = "7203"
SHARES = 100

print(f"\n=== 1. {CODE} 現在値取得 ===")
params = (
    "{"
    f'"p_no":"{to._next_p_no()}",'
    f'"p_sd_date":"{to._p_sd_date()}",'
    '"sCLMID":"CLMMfdsGetMarketPrice",'
    f'"sTargetIssueCode":"{CODE}",'
    '"sTargetColumn":"pDPP,pPRP",'
    '"sJsonOfmt":"5"'
    "}"
)
r = to._api_get(url_price + "?" + params)
print("p_errno:", r.get("p_errno"))
items = r.get("aCLMMfdsGetMarketPrice", [])
print(json.dumps(items, ensure_ascii=False))
last = None
if items:
    for k in ("pDPP", "pPRP"):
        v = items[0].get(k)
        if v:
            try:
                last = float(str(v).strip('"'))
                break
            except Exception:
                pass
if not last:
    last = 3000.0
    print("価格取得失敗、フォールバック値を使用:", last)
else:
    print("参考値:", last)

safe_price = int(last * 0.8)
print(f"安全指値（約定しない想定の-20%）: {safe_price}円")

print(f"\n=== 2. 買い指値発注 {CODE} {SHARES}株 @ {safe_price}円 ===")
result = to.place_buy_order(url_request, CODE, SHARES, price=safe_price)
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

order_no = result.get("order_no")
# cancel_orderはsEigyouDay必須(v4r10)。未指定だと当日日付にフォールバックするが、
# 深夜に発注すると実際のsEigyouDay(翌営業日)とズレて取消が失敗する
# (2026-09-07 23時台に実地で確認済み: sResultCode=13100「株式サマリにデータがありません」)。
# 本番のplace_buy_limit_with_fallbackと同様、買い注文のraw応答から取得して明示的に渡す。
eigyou_day = (result.get("raw") or {}).get("sEigyouDay") or None
if not result.get("success") or not order_no:
    print("\n発注失敗のためテスト終了")
    sys.exit(0)

print(f"\n=== 3. 発注直後の注文照会 (order_no={order_no}) ===")
time.sleep(2)
check = to.check_order(url_request, order_no)
print(json.dumps(check, ensure_ascii=False, indent=2, default=str))

print(f"\n=== 4. 取消 ===")
cancel = to.cancel_order(url_request, order_no, eigyou_day=eigyou_day)
print(json.dumps(cancel, ensure_ascii=False, indent=2, default=str))

print(f"\n=== 5. 取消後の注文照会（再確認） ===")
time.sleep(2)
check2 = to.check_order(url_request, order_no)
print(json.dumps(check2, ensure_ascii=False, indent=2, default=str))

print(f"\n=== 6. 保有株一覧（新規ポジションが出来ていないことの確認） ===")
positions = to.get_positions(url_request)
print(f"件数: {len(positions)}")
for p in positions:
    print(" ", p)

print("\n=== 完了 ===")
