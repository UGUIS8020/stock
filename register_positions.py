# -*- coding: utf-8 -*-
import db, sys
sys.stdout.reconfigure(encoding="utf-8")
db.init_db()

# ライク 2462 100株 買値1493円
db.save_position_db('2462', 'ライク', 100, 1493, 'A',
    tp_pct=0.03, sl_pct=0.01, entry_change_pct=0.5)

# 豊和工業 6203 100株 買値1783円
db.save_position_db('6203', '豊和工業', 100, 1783, 'A',
    tp_pct=0.03, sl_pct=0.01, entry_change_pct=6.8)

pos = db.load_open_positions()
print(f'登録後 open positions: {len(pos)}')
for p in pos:
    print(f'  {p["code"]} {p["name"]} 買値{p["buy_price"]}円 TP{p["tp_price"]}円 SL{p["sl_price"]}円')
