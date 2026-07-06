# -*- coding: utf-8 -*-
import db, sys, pandas as pd
from datetime import datetime, timedelta, timezone

sys.stdout.reconfigure(encoding="utf-8")
db.init_db()

conn = db.get_conn()

JST = timezone(timedelta(hours=9))
today = datetime.now(JST)
days_since_monday = today.weekday()

# 先週月曜〜金曜
last_monday = (today - timedelta(days=days_since_monday + 7)).strftime("%Y-%m-%d")
last_friday  = (today - timedelta(days=days_since_monday + 3)).strftime("%Y-%m-%d")
print(f"先週の期間: {last_monday} 〜 {last_friday}")

df = pd.read_sql(
    "SELECT date, code, name, shares, buy_price, sell_price, pnl_pct, status, exit_reason, strategy "
    "FROM positions "
    "WHERE date >= ? AND date <= ? "
    "ORDER BY date, code",
    conn, params=[last_monday, last_friday]
)
conn.close()

print(f"取引件数: {len(df)}件\n")

if df.empty:
    print("先週の取引データがありません")
else:
    closed = df[df["status"] == "closed"].copy()
    open_p = df[df["status"] == "open"].copy()
    print(f"決済済み: {len(closed)}件  未決済: {len(open_p)}件\n")

    wins = 0
    total_pnl = 0.0
    total_amount = 0.0

    for _, r in closed.iterrows():
        pnl = float(r["pnl_pct"]) if r["pnl_pct"] else 0.0
        buy  = float(r["buy_price"]) if r["buy_price"] else 0.0
        sell = float(r["sell_price"]) if r["sell_price"] else 0.0
        shares = int(r["shares"]) if r["shares"] else 0
        profit_yen = (sell - buy) * shares
        total_pnl += pnl
        total_amount += profit_yen
        if pnl > 0:
            wins += 1
        mark   = "✅" if pnl > 0 else "❌"
        reason = str(r["exit_reason"] or "")[:10]
        name   = str(r["name"] or "")[:14]
        print(f"{mark} {r['date']} {str(r['code']):<6} {name:<14} "
              f"{shares:>5}株 {buy:>7,.0f}→{sell:>7,.0f}円  {pnl:>+6.2f}%  "
              f"({profit_yen:>+7,.0f}円)  {reason}")

    print()
    if len(closed) > 0:
        print(f"勝率   : {wins}/{len(closed)} ({wins/len(closed)*100:.1f}%)")
        print(f"平均%  : {total_pnl/len(closed):+.2f}%")
        print(f"合計損益: {total_amount:+,.0f}円")

    if not open_p.empty:
        print("\n--- 未決済ポジション ---")
        for _, r in open_p.iterrows():
            name = str(r["name"] or "")[:14]
            print(f"  {r['date']} {str(r['code']):<6} {name:<14} {int(r['shares'])}株 買値{float(r['buy_price']):,.0f}円")
