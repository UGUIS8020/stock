# -*- coding: utf-8 -*-
import db, sys, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
db.init_db()
conn = db.get_conn()

df = pd.read_sql(
    "SELECT date, code, name, exit_reason, pnl_pct, strategy, buy_price, sell_price, shares "
    "FROM positions WHERE status='closed' ORDER BY date DESC LIMIT 50",
    conn
)
conn.close()

print("=== 決済理由の内訳（直近50件）===")
for reason, grp in df.groupby("exit_reason"):
    pnl = grp["pnl_pct"].astype(float)
    wins = (pnl > 0).sum()
    n = len(grp)
    print(f"  {str(reason):<14}: {n}件  勝率{wins/n*100:.1f}%  平均{pnl.mean():+.2f}%")

print()
print("=== 直近10件の取引 ===")
for _, r in df.head(10).iterrows():
    pnl = float(r["pnl_pct"]) if r["pnl_pct"] else 0
    mark = "✅" if pnl > 0 else "❌"
    name = str(r["name"] or "")[:12]
    reason = str(r["exit_reason"] or "")
    strat = str(r["strategy"] or "")
    print(f"{mark} {r['date']} {str(r['code']):<6} {name:<12} {pnl:>+6.2f}%  [{reason:<12}] 戦略{strat}")

# 未決済ポジション確認
open_pos = pd.read_sql(
    "SELECT date, code, name, shares, buy_price, tp_price, sl_price FROM positions WHERE status='open'",
    conn if not conn.closed else db.get_conn()
)
conn2 = db.get_conn()
open_pos = pd.read_sql(
    "SELECT date, code, name, shares, buy_price, tp_price, sl_price FROM positions WHERE status='open'",
    conn2
)
conn2.close()

print(f"\n=== 現在の未決済ポジション: {len(open_pos)}件 ===")
for _, r in open_pos.iterrows():
    name = str(r["name"] or "")[:14]
    print(f"  {r['date']} {str(r['code']):<6} {name:<14} {int(r['shares'])}株  "
          f"買値{float(r['buy_price']):,.0f}円  TP{float(r['tp_price']):,.0f}円  SL{float(r['sl_price']):,.0f}円")
if open_pos.empty:
    print("  （なし）")
