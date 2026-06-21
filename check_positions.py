# -*- coding: utf-8 -*-
"""
check_positions.py - 立花証券APIで実際の保有株を照会する

使い方:
    python check_positions.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import tachibana_order
import db

db.init_db()

url_request = tachibana_order.load_url_request()
if not url_request:
    print("❌ ログインが必要です。先に tachibana_login.py を実行してください。")
    sys.exit(1)

print("\n  立花証券 保有株照会中...")
api_positions = tachibana_order.get_all_positions(url_request)

if not api_positions:
    print("  ⚠️  APIから保有株を取得できませんでした（セッション切れ or 保有なし）")
else:
    print(f"\n  {'─'*68}")
    print(f"  {'コード':<6} {'銘柄名':<20} {'株数':>5} {'買値':>7} {'現在値':>7} {'損益':>7} {'口座'}")
    print(f"  {'─'*68}")
    total_invest = 0
    total_pnl    = 0
    for p in api_positions:
        invest   = (p['buy_price'] or 0) * p['shares']
        pnl_yen  = invest * (p['pnl_pct'] or 0) / 100
        total_invest += invest
        total_pnl    += pnl_yen
        icon = "✅" if (p['pnl_pct'] or 0) >= 3.0 else ("🔴" if (p['pnl_pct'] or 0) <= -1.0 else "🟡")
        acct_label = "信用" if p.get("account_type") == "shinyou" else "現物"
        print(f"  {p['code']:<6} {p['name'][:18]:<20} {p['shares']:>5}株"
              f"  {p['buy_price']:>7,.0f}円"
              f"  {p['current_price']:>7,.0f}円"
              f"  {p['pnl_pct']:>+6.2f}%  {icon}  {acct_label}")
    print(f"  {'─'*68}")
    print(f"  合計: {len(api_positions)}銘柄  投資額 {total_invest:,.0f}円  評価損益 {total_pnl:+,.0f}円")

# システムDBの保有株と比較・account_type 自動同期
db_positions = db.load_open_positions()
print(f"\n  システムDB保有: {len(db_positions)}件")

if api_positions and db_positions:
    api_map  = {p['code']: p for p in api_positions}
    db_codes = {p['code'] for p in db_positions}
    api_codes = set(api_map.keys())

    only_api = api_codes - db_codes
    only_db  = db_codes  - api_codes

    if only_api:
        print(f"  ⚠️  APIのみ（DBに未登録）: {sorted(only_api)}")
        print(f"     → register_positions.py で登録してください")
    if only_db:
        print(f"  ⚠️  DBのみ（既に売却済み？）: {sorted(only_db)}")

    # account_type 自動同期: APIの口座区分とDBが食い違う場合に修正
    synced = []
    for p in db_positions:
        api_p = api_map.get(p['code'])
        if api_p and api_p.get('account_type') != p.get('account_type'):
            new_type = api_p['account_type']
            db.update_position(p['date'], p['code'], account_type=new_type)
            synced.append(f"{p['code']}({p.get('account_type')}→{new_type})")
    if synced:
        print(f"\n  🔄 account_type 自動同期: {', '.join(synced)}")
    else:
        print(f"  ✅ DBの口座区分は最新です")
