"""
ナンピン戦略(下落トレンド中に買い増し、平均取得単価+X%で全売却)の候補銘柄を
毎月チェックするためのスクリーニングツール。オンデマンド実行専用（自動送信はしない）。

方式:
    1. 直近の全期間データを「選定期間」(古い方、約40%)と「検証fold1」「検証fold2」
       (残り60%を2等分)に分割
    2. 選定期間の統計(売買代金・年率ボラティリティ・最大下落率)だけで
       「大型・購入しやすい価格・低ボラ」上位20銘柄を選ぶ(検証foldのデータは選定に使わない)
    3. fold1・fold2それぞれでナンピン戦略を実際にシミュレーションしてROI(投入資金に
       対する利回り)を計算し、「両fold平均 − fold間のブレ」でランキングする
       (選定に使っていない2つの独立した期間で一貫して機能するかを見る)

実行方法:
    python analyze/screen_nanpin_candidates.py
    python analyze/screen_nanpin_candidates.py --max-price 200000  # 100株あたりの上限金額(円)
"""
import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "out" / "stock.db"

ETF_KEYWORDS = [
    'ＥＴＦ', 'ETF', 'ｉＦｒｅｅ', 'ｉシェアーズ', 'ＮＥＸＴ　ＦＵＮＤＳ', 'NEXT FUNDS',
    '上場投信', '上場インデックスファンド', 'ＭＡＸＩＳ', 'listed index', 'リート', 'ＲＥＩＴ',
    '投資法人', 'ファンド', '証券投資信託',
]


def is_etf_like(name):
    return any(kw in name for kw in ETF_KEYWORDS)


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


def drawdown_recovery_check(closes):
    """最大下落が「下落前の高値」まで戻ったこと(=循環的な下落)があるかを判定する。
    トラフ以降ずっと下落前高値を上回れていない銘柄は、構造的な右肩下がりの疑いとして
    フラグを立てる（2026-09-18追加、日本精工の目視チェックを自動化）。
    全期間(選定期間+検証期間)の値動きを使う: これは銘柄選定のROIランキングに使う
    統計とは別の「安全確認」目的のため、選定期間限定にする必要はない。
    """
    if len(closes) < 2:
        return {'recovered_from_max_dd': True, 'days_to_recover': None, 'off_all_time_high_pct': 0.0}

    peak = closes[0]
    max_dd = 0.0
    dd_peak_price = closes[0]
    dd_trough_idx = 0
    for i, c in enumerate(closes):
        if c > peak:
            peak = c
        dd = (c - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd
            dd_peak_price = peak
            dd_trough_idx = i

    recovery_idx = next(
        (i for i in range(dd_trough_idx, len(closes)) if closes[i] >= dd_peak_price), None
    )
    all_time_high = max(closes)
    off_high_pct = (closes[-1] - all_time_high) / all_time_high * 100

    return {
        'recovered_from_max_dd': recovery_idx is not None,
        'days_to_recover': (recovery_idx - dd_trough_idx) if recovery_idx is not None else None,
        'off_all_time_high_pct': off_high_pct,
    }


def backtest_nanpin(bars, trend_n=20, buy_shares=100, tp_pct=2.0, max_buys=30, warmup=0):
    """下落トレンド(終値<trend_n日MA)中は暦週1回(祝日で営業日が減っても週1回)買い増し、
    平均取得単価×(1+tp_pct%)まで戻ったら全株売却する。
    2026-09-18改善: 営業日を5回数える方式(buy_interval)から暦週ベースに変更。
    祝日を挟むたびに次の買いタイミングが暦上ずれていく問題があり、暦週で
    リセットする方式の方が実測で良好だったため(4銘柄中3銘柄でROI改善)。
    warmup: 先頭warmup件は移動平均の計算(closesへの蓄積)にのみ使い、
    売買判断の対象外にする(fold境界をまたいだMAの準備用、2026-09-18追加)。"""
    closes = []
    shares_held = 0
    total_cost = 0.0
    last_buy_week = None
    cycles = []
    max_capital_used = 0.0
    buys_in_cycle = 0

    for i, bar in enumerate(bars):
        closes.append(bar['Close'])
        ma = sma(closes, trend_n)
        if ma is None or i < warmup:
            continue
        downtrend = bar['Close'] < ma
        week_key = date.fromisoformat(bar['Date']).isocalendar()[:2]  # (年, 週番号)

        if shares_held > 0:
            avg_cost = total_cost / shares_held
            sell_target = avg_cost * (1 + tp_pct / 100)
            if bar['High'] >= sell_target:
                pnl = sell_target * shares_held - total_cost
                cycles.append({'pnl_yen': pnl, 'capital_used': total_cost})
                shares_held = 0
                total_cost = 0.0
                last_buy_week = None
                buys_in_cycle = 0
                continue

        if downtrend and buys_in_cycle < max_buys:
            if last_buy_week is None or week_key != last_buy_week:
                total_cost += bar['Close'] * buy_shares
                shares_held += buy_shares
                last_buy_week = week_key
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

    # 2026-09-20改善: 以前は全銘柄・全期間のデータ(約493万行)を丸ごとメモリに読み込んで
    # から絞り込んでいたため、EC2(RAM3.7GB)でピーク約1.46GBまで膨張していた
    # (stock_usa側で実際にフリーズ事故を起こした構造と同じ、[[stock_usa_ec2_freeze_incident]])。
    # 大型株・低ボラの絞り込みは「選定期間(全体の約40%)だけのデータ」で完結できるため、
    # まず日付範囲・銘柄の完全性をSQL側の集計だけで確認し(個々の行はPython側に読み込まない)、
    # 選定期間分のデータだけで絞り込みを行い、最終候補(上位N銘柄)だけ改めてfold検証用の
    # 全期間データを取得する2段階方式に変更した。
    max_code_row = conn.execute(
        "SELECT code FROM daily_prices GROUP BY code ORDER BY COUNT(*) DESC LIMIT 1"
    ).fetchone()
    all_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT Date FROM daily_prices WHERE code=? ORDER BY Date", [max_code_row['code']]
    )]
    n_total = len(all_dates)
    split1 = int(n_total * 0.40)   # 選定期間(過去40%)
    split2 = int(n_total * 0.70)   # 検証fold1(次の30%) / fold2(直近30%)
    screen_start, screen_end = all_dates[0], all_dates[split1 - 1]
    screen_dates = set(all_dates[:split1])
    fold1_dates = set(all_dates[split1:split2])
    fold2_dates = set(all_dates[split2:])
    print(f"選定期間: {screen_start} 〜 {screen_end} ({split1}日)")
    print(f"検証fold1: {all_dates[split1]} 〜 {all_dates[split2 - 1]} ({split2 - split1}日)  ※選定には未使用")
    print(f"検証fold2: {all_dates[split2]} 〜 {all_dates[-1]} ({n_total - split2}日)  ※選定には未使用")
    print(f"購入上限: 100株あたり{args.max_price:,.0f}円(株価{max_share_price:,.0f}円以下)\n")

    # 銘柄ごとの完全性(全期間分のデータが揃っていて欠損が無いか)をSQL集計だけで判定する
    # (以前のhas_nulls()相当の判定を、個々の行を読まずに済むよう書き換えたもの)。
    completeness = conn.execute("""
        SELECT code, COUNT(*) as cnt,
               SUM(CASE WHEN Open IS NULL OR High IS NULL OR Low IS NULL
                          OR Close IS NULL OR Volume IS NULL THEN 1 ELSE 0 END) as null_cnt
        FROM daily_prices GROUP BY code
    """).fetchall()
    complete_codes = {
        row['code'] for row in completeness
        if row['cnt'] == n_total and row['null_cnt'] == 0
    }
    # ETF等・マスタ未登録銘柄はここで既に除外できる(価格データを見る必要が無いため)
    complete_codes = {c for c in complete_codes if names.get(c) and not is_etf_like(names[c])}

    # 選定期間分のデータだけを取得する(SQLiteのIN句パラメータ上限を避けるため、
    # 銘柄コードでの絞り込みはSQLではなくPython側の集合判定で行う)。
    screen_rows = conn.execute(
        "SELECT code, Date, Close, Volume FROM daily_prices WHERE Date BETWEEN ? AND ? ORDER BY code, Date",
        [screen_start, screen_end]
    ).fetchall()
    by_code_screen = defaultdict(list)
    for r in screen_rows:
        if r['code'] in complete_codes:
            by_code_screen[r['code']].append(r)
    del screen_rows

    candidates = []
    for code, screen_bars in by_code_screen.items():
        closes = [b['Close'] for b in screen_bars]
        vols = [b['Volume'] for b in screen_bars]
        avg_turnover = sum(c * v for c, v in zip(closes, vols)) / len(closes)
        vol_annualized, max_dd = efficiency_ratio_stats(closes)
        if vol_annualized is None:
            continue
        candidates.append({
            'code': code, 'name': names[code], 'screen_end_price': closes[-1],
            'avg_turnover': avg_turnover, 'vol_annualized': vol_annualized, 'max_dd': max_dd,
        })
    del by_code_screen

    candidates.sort(key=lambda x: -x['avg_turnover'])
    large_cap_pool = candidates[:len(candidates) // 5]  # 売買代金上位20% = 大型株の代理指標
    affordable_pool = [c for c in large_cap_pool if c['screen_end_price'] <= max_share_price]
    affordable_pool.sort(key=lambda x: x['vol_annualized'])
    selected = affordable_pool[:args.top_n]

    print(f"大型株プール: {len(large_cap_pool)}銘柄 → 購入しやすい価格帯: {len(affordable_pool)}銘柄 "
          f"→ 低ボラ上位{len(selected)}銘柄を検証\n")

    # ここでようやく、最終候補(上位N銘柄、通常20件程度)だけ全期間のデータを取得する。
    # 件数が少ないためIN句のパラメータ数は問題にならない。
    selected_codes = [c['code'] for c in selected]
    placeholders = ",".join("?" * len(selected_codes))
    full_rows = conn.execute(
        f"SELECT code, Date, Open, High, Low, Close, Volume FROM daily_prices "
        f"WHERE code IN ({placeholders}) ORDER BY code, Date",
        selected_codes,
    ).fetchall()
    by_code = defaultdict(list)
    for r in full_rows:
        by_code[r['code']].append(dict(r))
    del full_rows

    def fold_avg_roi(code, dates):
        """1つのfoldでTP+1/2/3%を回し、平均ROIと最大投入資金・末端含み損益を返す。
        fold開始直前の実データ(最大20営業日)を移動平均のウォームアップとして
        渡す。渡さないとfold冒頭の約1ヶ月が移動平均未計算で売買判断できなく
        なってしまうため(2026-09-18修正)。"""
        all_bars = by_code[code]
        fold_indices = [i for i, b in enumerate(all_bars) if b['Date'] in dates]
        if not fold_indices:
            return 0.0, 0.0, None
        start_idx, end_idx = fold_indices[0], fold_indices[-1]
        warmup = min(20, start_idx)
        test_bars = all_bars[start_idx - warmup: end_idx + 1]

        roi_by_tp = {}
        max_cap_by_tp = {}
        unresolved_pct = None
        for tp in (1.0, 2.0, 3.0):
            cycles, open_pos, max_cap = backtest_nanpin(test_bars, tp_pct=tp, warmup=warmup)
            pnl = sum(x['pnl_yen'] for x in cycles)
            roi_by_tp[tp] = pnl / max_cap * 100 if max_cap else 0.0
            max_cap_by_tp[tp] = max_cap
            if open_pos:
                unresolved_pct = open_pos['unrealized_pnl_pct']
        avg = sum(roi_by_tp.values()) / len(roi_by_tp)
        return avg, max_cap_by_tp[3.0], unresolved_pct

    results = []
    for c in selected:
        code = c['code']
        roi_fold1, cap_fold1, unresolved1 = fold_avg_roi(code, fold1_dates)
        roi_fold2, cap_fold2, unresolved2 = fold_avg_roi(code, fold2_dates)
        # 2026-09-18改善: 検証を2fold化し、単純平均ではなく
        # 「両fold平均 − fold間のブレ」でランキングする(ブレが大きい=再現性が
        # 低い銘柄を、見かけの平均ROIだけで高評価しないようにするため)。
        combined_avg = (roi_fold1 + roi_fold2) / 2
        consistency_penalty = abs(roi_fold1 - roi_fold2) / 2
        score = combined_avg - consistency_penalty
        unresolved_pct = min((v for v in (unresolved1, unresolved2) if v is not None), default=None)
        current_price = by_code[code][-1]['Close']
        full_closes = [b['Close'] for b in by_code[code]]
        recovery = drawdown_recovery_check(full_closes)
        # 2026-09-20追加: 記録用のトレンド参考情報(stock_usa側と同じ、[[stock_usa_nanpin_rotation]]
        # 参照)。実際の売買判断はstrategy_n.py側でローテーション実行時点の最新データを見て
        # 別途行う。ここでのtrend値はランキング作成時点のスナップショットで、時間経過とともに
        # 古くなる。
        ma20 = sma(full_closes, 20)
        trend = ("down" if current_price < ma20 else "up") if ma20 else None
        results.append({
            **c, 'current_price': current_price,
            'roi_fold1': roi_fold1, 'roi_fold2': roi_fold2, 'score': score,
            'max_cap': max(cap_fold1, cap_fold2), 'unresolved_pct': unresolved_pct,
            'trend': trend,
            **recovery,
        })

    results.sort(key=lambda x: -x['score'])

    print(f"{'コード':<6}{'銘柄名':<16}{'現在値':>8}{'Vol':>7}{'最大DD':>8}"
          f"{'ROI(fold1)':>11}{'ROI(fold2)':>11}{'スコア':>8}{'必要資金':>10}")
    unrecovered = []
    for r in results:
        note = ""
        if r['unresolved_pct'] is not None and r['unresolved_pct'] < -5:
            note += f"  ⚠️未決済{r['unresolved_pct']:.1f}%"
        if not r['recovered_from_max_dd']:
            note += f"  ⚠️最大下落から未回復(現在値は過去最高値比{r['off_all_time_high_pct']:.1f}%)"
            unrecovered.append(r)
        print(f"{r['code']:<6}{r['name'][:14]:<16}{r['current_price']:>7,.0f}円{r['vol_annualized']:>6.1f}%"
              f"{r['max_dd']:>7.1f}%{r['roi_fold1']:>10.1f}%{r['roi_fold2']:>10.1f}%"
              f"{r['score']:>7.1f}%{r['max_cap']:>9,.0f}円{note}")

    if unrecovered:
        print(f"\n⚠️ {len(unrecovered)}銘柄が、過去最大の下落からまだ一度も高値を更新できていません"
              f"（構造的な右肩下がりの疑い、要個別確認）:")
        for r in unrecovered:
            print(f"  {r['code']} {r['name']}: 過去最高値比{r['off_all_time_high_pct']:.1f}%")

    # 2026-09-20追加: strategy_n.pyの新規キャンペーン銘柄ローテーションが読み込む
    # 機械可読な最新ランキング(stock_usa側と同じ設計、[[stock_usa_nanpin_rotation]]参照)。
    # healthy(過去最大下落から回復済みの候補)のみを対象にし、毎月上書きする。
    healthy = [r for r in results if r['recovered_from_max_dd']]
    out_dir = Path(__file__).parent.parent / "out"
    ranking_path = out_dir / "nanpin_candidates_ranking_latest.json"
    ranking_path.write_text(
        json.dumps(
            [{"code": r["code"], "name": r["name"], "score": r["score"],
              "current_price": r["current_price"], "trend": r["trend"]}
             for r in healthy],
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n(機械可読ランキングを{ranking_path}に出力しました)")

    # 事後評価用に、上位10銘柄のスコア・株価・トレンドを月ごとに上書きされない形で記録する。
    archive_dir = out_dir / "nanpin_candidates_ranking_archive"
    archive_dir.mkdir(exist_ok=True)
    archive_path = archive_dir / f"{date.today().strftime('%Y%m')}.json"
    archive_path.write_text(
        json.dumps(
            {
                "generated_date": date.today().isoformat(),
                "top10": [{"code": r["code"], "name": r["name"], "score": r["score"],
                           "current_price": r["current_price"], "trend": r["trend"]}
                          for r in healthy[:10]],
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"(上位10銘柄の記録を{archive_path}に保存しました)")


if __name__ == "__main__":
    main()
