"""
ナンピン戦略(下落トレンド中に買い増し、平均取得単価+X%で全売却)の候補銘柄を
毎月チェックするためのスクリーニングツール。オンデマンド実行専用（自動送信はしない）。

方式:
    1. 直近の全期間データを「選定期間」(古い方、約75%)と「検証期間」(新しい方、約25%)に分割
    2. 選定期間の統計(売買代金・年率ボラティリティ・最大下落率)だけで
       「大型・購入しやすい価格・低ボラ」上位20銘柄を選ぶ(検証期間のデータは選定に使わない)
    3. 検証期間でナンピン戦略を実際にシミュレーションし、ROI(投入資金に対する利回り)で
       ランキングする(選定に使っていない期間での事後検証)

実行方法:
    python analyze/screen_nanpin_candidates.py
    python analyze/screen_nanpin_candidates.py --max-price 200000  # 100株あたりの上限金額(円)
"""
import argparse
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "out" / "stock.db"

ETF_KEYWORDS = [
    'ＥＴＦ', 'ETF', 'ｉＦｒｅｅ', 'ｉシェアーズ', 'ＮＥＸＴ　ＦＵＮＤＳ', 'NEXT FUNDS',
    '上場投信', '上場インデックスファンド', 'ＭＡＸＩＳ', 'listed index', 'リート', 'ＲＥＩＴ',
    '投資法人', 'ファンド', '証券投資信託',
]


def is_etf_like(name):
    return any(kw in name for kw in ETF_KEYWORDS)


def has_nulls(bars, keys=('Open', 'High', 'Low', 'Close')):
    return any(any(b[k] is None for k in keys) for b in bars)


def sma(closes, n):
    if len(closes) < n:
        return None
    return sum(closes[-n:]) / n


def efficiency_ratio_stats(closes):
    """年率ボラティリティ(%)と最大下落率(%)を返す。"""
    returns = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes)) if closes[i - 1]]
    if len(returns) < 100:
        return None, None
    vol_annualized = statistics.pstdev(returns) * (250 ** 0.5) * 100
    peak = closes[0]
    max_dd = 0.0
    for c in closes:
        peak = max(peak, c)
        max_dd = min(max_dd, (c - peak) / peak * 100)
    return vol_annualized, max_dd


def backtest_nanpin(bars, trend_n=20, buy_interval=5, buy_shares=100, tp_pct=2.0, max_buys=30):
    """下落トレンド(終値<trend_n日MA)中は buy_interval営業日ごとに買い増し、
    平均取得単価×(1+tp_pct%)まで戻ったら全株売却する。"""
    closes = []
    shares_held = 0
    total_cost = 0.0
    last_buy_idx = None
    cycles = []
    max_capital_used = 0.0
    buys_in_cycle = 0

    for i, bar in enumerate(bars):
        closes.append(bar['Close'])
        ma = sma(closes, trend_n)
        if ma is None:
            continue
        downtrend = bar['Close'] < ma

        if shares_held > 0:
            avg_cost = total_cost / shares_held
            sell_target = avg_cost * (1 + tp_pct / 100)
            if bar['High'] >= sell_target:
                pnl = sell_target * shares_held - total_cost
                cycles.append({'pnl_yen': pnl, 'capital_used': total_cost})
                shares_held = 0
                total_cost = 0.0
                last_buy_idx = None
                buys_in_cycle = 0
                continue

        if downtrend and buys_in_cycle < max_buys:
            if last_buy_idx is None or i - last_buy_idx >= buy_interval:
                total_cost += bar['Close'] * buy_shares
                shares_held += buy_shares
                last_buy_idx = i
                buys_in_cycle += 1
                max_capital_used = max(max_capital_used, total_cost)

    open_position = None
    if shares_held > 0:
        avg_cost = total_cost / shares_held
        last_close = bars[-1]['Close']
        open_position = {
            'capital_used': total_cost,
            'unrealized_pnl_pct': (last_close - avg_cost) / avg_cost * 100,
        }
    return cycles, open_position, max_capital_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-price", type=float, default=200_000,
                     help="100株あたりの購入上限金額(円)。デフォルト20万円")
    ap.add_argument("--top-n", type=int, default=20, help="候補として表示する銘柄数")
    args = ap.parse_args()
    max_share_price = args.max_price / 100

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    names = {r['code']: r['name'] for r in conn.execute("SELECT code, name FROM stock_master")}
    price_rows = conn.execute(
        "SELECT code, Date, Open, High, Low, Close, Volume FROM daily_prices ORDER BY code, Date"
    ).fetchall()
    by_code = defaultdict(list)
    for r in price_rows:
        by_code[r['code']].append(dict(r))

    all_dates = [r['Date'] for r in max(by_code.values(), key=len)]
    n_total = len(all_dates)
    split_idx = int(n_total * 0.75)
    screen_dates = set(all_dates[:split_idx])
    valid_dates = set(all_dates[split_idx:])
    print(f"選定期間: {all_dates[0]} 〜 {all_dates[split_idx - 1]} ({split_idx}日)")
    print(f"検証期間: {all_dates[split_idx]} 〜 {all_dates[-1]} ({n_total - split_idx}日)  ※選定には未使用")
    print(f"購入上限: 100株あたり{args.max_price:,.0f}円(株価{max_share_price:,.0f}円以下)\n")

    candidates = []
    for code, bars in by_code.items():
        name = names.get(code, '')
        if not name or is_etf_like(name):
            continue
        if len(bars) != n_total or has_nulls(bars) or any(b['Volume'] is None for b in bars):
            continue
        screen_bars = [b for b in bars if b['Date'] in screen_dates]
        closes = [b['Close'] for b in screen_bars]
        vols = [b['Volume'] for b in screen_bars]
        avg_turnover = sum(c * v for c, v in zip(closes, vols)) / len(closes)
        vol_annualized, max_dd = efficiency_ratio_stats(closes)
        if vol_annualized is None:
            continue
        candidates.append({
            'code': code, 'name': name, 'screen_end_price': closes[-1],
            'avg_turnover': avg_turnover, 'vol_annualized': vol_annualized, 'max_dd': max_dd,
        })

    candidates.sort(key=lambda x: -x['avg_turnover'])
    large_cap_pool = candidates[:len(candidates) // 5]  # 売買代金上位20% = 大型株の代理指標
    affordable_pool = [c for c in large_cap_pool if c['screen_end_price'] <= max_share_price]
    affordable_pool.sort(key=lambda x: x['vol_annualized'])
    selected = affordable_pool[:args.top_n]

    print(f"大型株プール: {len(large_cap_pool)}銘柄 → 購入しやすい価格帯: {len(affordable_pool)}銘柄 "
          f"→ 低ボラ上位{len(selected)}銘柄を検証\n")

    results = []
    for c in selected:
        code = c['code']
        test_bars = [b for b in by_code[code] if b['Date'] in valid_dates]
        roi_by_tp = {}
        max_cap_by_tp = {}
        unresolved_pct = None
        for tp in (1.0, 2.0, 3.0):
            cycles, open_pos, max_cap = backtest_nanpin(test_bars, tp_pct=tp)
            pnl = sum(x['pnl_yen'] for x in cycles)
            roi_by_tp[tp] = pnl / max_cap * 100 if max_cap else 0.0
            max_cap_by_tp[tp] = max_cap
            if open_pos:
                unresolved_pct = open_pos['unrealized_pnl_pct']
        avg_roi = sum(roi_by_tp.values()) / len(roi_by_tp)
        current_price = by_code[code][-1]['Close']
        results.append({
            **c, 'current_price': current_price, 'avg_roi': avg_roi,
            'roi_by_tp': roi_by_tp, 'max_cap': max_cap_by_tp[3.0], 'unresolved_pct': unresolved_pct,
        })

    results.sort(key=lambda x: -x['avg_roi'])

    print(f"{'コード':<6}{'銘柄名':<16}{'現在値':>8}{'Vol':>7}{'最大DD':>8}"
          f"{'ROI+1%':>8}{'ROI+2%':>8}{'ROI+3%':>8}{'平均ROI':>9}{'必要資金':>10}")
    for r in results:
        note = ""
        if r['unresolved_pct'] is not None and r['unresolved_pct'] < -5:
            note = f"  ⚠️未決済{r['unresolved_pct']:.1f}%"
        print(f"{r['code']:<6}{r['name'][:14]:<16}{r['current_price']:>7,.0f}円{r['vol_annualized']:>6.1f}%"
              f"{r['max_dd']:>7.1f}%{r['roi_by_tp'][1.0]:>7.1f}%{r['roi_by_tp'][2.0]:>7.1f}%"
              f"{r['roi_by_tp'][3.0]:>7.1f}%{r['avg_roi']:>8.1f}%{r['max_cap']:>9,.0f}円{note}")


if __name__ == "__main__":
    main()
