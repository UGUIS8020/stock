# -*- coding: utf-8 -*-
"""
analyze_entry_timing.py - エントリー時刻別 利益率比較

9:00〜9:10の各分でエントリーした場合の勝率・平均利益率を比較する。
candidates_log(BUY判定) × intraday_prices(分足) を使用。

使い方:
    python analyze/analyze_entry_timing.py
    python analyze/analyze_entry_timing.py --judgment BUY        # BUYのみ
    python analyze/analyze_entry_timing.py --judgment CAUTION    # CAUTIONのみ
    python analyze/analyze_entry_timing.py --condition NORMAL    # NORMAL日のみ
    python analyze/analyze_entry_timing.py --tp 4.0 --sl 6.0    # TP/SL変更
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import db

# デフォルト設定
DEFAULT_TP_PCT = 4.0   # 利確 +4%
DEFAULT_SL_PCT = 6.0   # 損切 -6%
ENTRY_TIMES    = [f"09:{m:02d}" for m in range(0, 11)]  # 09:00〜09:10


def simulate_tpsl(prices_after_entry, entry_price, tp_pct, sl_pct):
    """
    エントリー後の分足データでTP/SL到達を判定する。
    prices_after_entry: [(time, open, high, low, close), ...] 時刻順
    戻り値: (pnl_pct, exit_reason, exit_time)
    """
    tp_price = entry_price * (1 + tp_pct / 100)
    sl_price = entry_price * (1 - sl_pct / 100)

    for row in prices_after_entry:
        t, o, h, l, c = row
        if h >= tp_price and l <= sl_price:
            # 同一分足内でTP/SL両方 → 先に開値側で判断
            if abs(o - tp_price) <= abs(o - sl_price):
                return tp_pct, "TP", t
            else:
                return -sl_pct, "SL", t
        if h >= tp_price:
            return tp_pct, "TP", t
        if l <= sl_price:
            return -sl_pct, "SL", t

    # 当日中に到達しなかった → 終値で決済
    last_close = prices_after_entry[-1][4] if prices_after_entry else entry_price
    pnl = (last_close - entry_price) / entry_price * 100
    return pnl, "close", prices_after_entry[-1][0] if prices_after_entry else "15:30"


def run(judgment_filter=None, condition_filter=None, tp_pct=DEFAULT_TP_PCT, sl_pct=DEFAULT_SL_PCT):
    db.init_db()
    conn = db.get_conn()
    cur  = conn.cursor()

    # candidates_log から対象銘柄を取得
    where_clauses = ["strategy = 'A'", "judgment != 'PASS'"]
    params = []
    if judgment_filter:
        where_clauses.append("judgment = ?")
        params.append(judgment_filter)
    if condition_filter:
        where_clauses.append("condition = ?")
        params.append(condition_filter)

    where = " AND ".join(where_clauses)
    cur.execute(f"""
        SELECT date, code, name, condition, judgment, score, ratio, open_price, close_price
        FROM candidates_log
        WHERE {where}
        ORDER BY date, code
    """, params)
    candidates = cur.fetchall()

    print(f"=== エントリー時刻別 利益率分析 ===")
    print(f"  対象: 戦略A  judgment={judgment_filter or '全て(BUY+CAUTION)'}  condition={condition_filter or '全て'}")
    print(f"  TP={tp_pct}%  SL={sl_pct}%")
    print(f"  候補数: {len(candidates)}件\n")

    # 時刻ごとの結果集計
    results = {t: [] for t in ENTRY_TIMES}

    for cand in candidates:
        date, code, name, condition, judgment, score, ratio, open_px, close_px = cand

        # 当日の全分足データを取得（9:00以降）
        cur.execute("""
            SELECT time, open, high, low, close
            FROM intraday_prices
            WHERE code = ? AND date = ? AND time >= '09:00' AND time <= '15:30'
            ORDER BY time
        """, (code, date))
        minute_bars = cur.fetchall()

        if not minute_bars:
            continue

        # 分足をdictに変換（時刻→行データ）
        bar_dict = {row[0]: row for row in minute_bars}
        bar_list = minute_bars  # 時刻順リスト

        for entry_time in ENTRY_TIMES:
            # エントリー価格の決定
            if entry_time == "09:00":
                # 9:00は寄り付き始値（candidates_logのopen_price優先）
                if open_px:
                    entry_price = float(open_px)
                elif entry_time in bar_dict:
                    entry_price = float(bar_dict[entry_time][1])  # open
                else:
                    continue
            else:
                # 9:01以降 → その分足のclose価格でエントリー
                if entry_time not in bar_dict:
                    continue
                entry_price = float(bar_dict[entry_time][4])  # close

            if entry_price <= 0:
                continue

            # エントリー後の分足データ（エントリー時刻の次の分から）
            entry_idx = next((i for i, b in enumerate(bar_list) if b[0] > entry_time), None)
            if entry_idx is None:
                continue
            bars_after = bar_list[entry_idx:]
            if not bars_after:
                continue

            pnl, exit_reason, exit_time = simulate_tpsl(bars_after, entry_price, tp_pct, sl_pct)
            results[entry_time].append({
                "date":       date,
                "code":       code,
                "name":       name,
                "condition":  condition,
                "judgment":   judgment,
                "pnl":        pnl,
                "exit_reason": exit_reason,
                "exit_time":  exit_time,
                "entry_price": entry_price,
            })

    conn.close()

    # 結果表示
    print(f"{'時刻':<8} {'N':>5} {'勝率':>7} {'平均損益':>9} {'中央値':>8} {'TP率':>7} {'SL率':>7} {'close率':>8}")
    print("─" * 65)

    best_time = None
    best_avg  = -999

    for t in ENTRY_TIMES:
        data = results[t]
        if not data:
            print(f"{t}   データなし")
            continue
        n    = len(data)
        pnls = [d["pnl"] for d in data]
        wins = sum(1 for p in pnls if p > 0)
        avg  = sum(pnls) / n
        med  = sorted(pnls)[n // 2]
        tp_n = sum(1 for d in data if d["exit_reason"] == "TP")
        sl_n = sum(1 for d in data if d["exit_reason"] == "SL")
        cl_n = sum(1 for d in data if d["exit_reason"] == "close")

        marker = " ← 最良" if avg == max(
            sum(r["pnl"] for r in v) / len(v) for v in results.values() if v
        ) else ""

        print(f"{t}  {n:>5}件  {wins/n*100:>6.1f}%  {avg:>+8.2f}%  {med:>+7.2f}%"
              f"  {tp_n/n*100:>6.1f}%  {sl_n/n*100:>6.1f}%  {cl_n/n*100:>7.1f}%{marker}")

        if avg > best_avg:
            best_avg  = avg
            best_time = t

    print()
    print(f"  最良エントリー時刻: {best_time}  (平均{best_avg:+.2f}%)")

    # 地合い別クロス集計
    print(f"\n=== 地合い×時刻 クロス集計（平均損益%） ===")
    all_conditions = sorted(set(d["condition"] for t in ENTRY_TIMES for d in results[t]))
    header = f"{'時刻':<8}" + "".join(f"{c:>10}" for c in all_conditions) + f"{'合計':>10}"
    print(header)
    print("─" * (8 + 10 * (len(all_conditions) + 1)))
    for t in ENTRY_TIMES:
        data = results[t]
        if not data:
            continue
        row = f"{t:<8}"
        for cond in all_conditions:
            sub = [d["pnl"] for d in data if d["condition"] == cond]
            row += f"  {sum(sub)/len(sub):>+6.2f}({len(sub)})" if sub else f"  {'---':>9}"
        all_pnl = [d["pnl"] for d in data]
        row += f"  {sum(all_pnl)/len(all_pnl):>+6.2f}({len(all_pnl)})"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgment",  default=None, help="BUY / CAUTION")
    parser.add_argument("--condition", default=None, help="NORMAL / STRONG / WEAK")
    parser.add_argument("--tp",        type=float, default=DEFAULT_TP_PCT)
    parser.add_argument("--sl",        type=float, default=DEFAULT_SL_PCT)
    args = parser.parse_args()
    run(args.judgment, args.condition, args.tp, args.sl)


if __name__ == "__main__":
    main()
