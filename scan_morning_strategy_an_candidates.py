# -*- coding: utf-8 -*-
"""
scan_morning_strategy_an_candidates.py - 戦略AN（戦略A・NORMAL日専用）候補スキャン

戦略A(market_watch.py)は、SCORE_MIN_A(3.0)→TOP_N(30)→MAX_WATCH_A(10)という
3段階の絞り込みを持ち、いずれもscore降順で上位を優先する設計になっている。
2026-08-09の検証で、NORMAL日に有効な条件(score1.5〜3.0×ratio1.4〜50.0×
前日騰落-1.2〜+9.1%×RBスコア≥5.4×連続下落≥1)を見つけたが、score1.5〜3.0は
既存の3段階のいずれでも上位に入れず、事実上発動しない。

このモジュールは戦略D(scan_morning_strategy_d_candidates.py)と同じ発想で、
戦略Aの3段階フィルタとは完全に独立した候補プールを作る。STRONG日側の
score3.0以上を優先する設計には一切触れない。

【前提】gap_pct（寄り付きギャップ）を使わないため、スキャン当日の夜間
（scan_daily.pyと同じ16:30以降）に前日分までのデータだけで全条件を確定できる。
戦略Dのような「Stage 2（当日始値待ち）」は不要。

条件（2026-08-12 大サンプル・ウォークフォワード検証済み、2年データ）:
    score       : 1.5 <= score < 3.0     （calc_score_an と同一の計算式）
    ratio       : ratio >= 1.4
    today_rise  : -2.5% <= 当日の寄り引け値幅 <= +9.1%   （2026-08-12: -2.0%→-2.5%に緩和）
    rb_score    : calc_rebound_score(hist) >= 5.4
    consec_drop : 連続下落日数 >= 1
    TP/SL       : +1.5% / -5.9%（エントリー翌営業日の寄り付きで買う想定）
    検証結果    : 9分割点(45〜85%を5%刻み)walk-forwardすべてで件数増(約+15〜30%)・
                  勝率実質維持・平均損益同水準を確認。増分部分(rise -2.5〜-2.0)単独でも
                  N=91 勝率61.5% avg+0.309%とノイズでないことを確認。ブロックブートストラップ
                  (OOS日別損益・2000回)で総損益+16%・最終赤字確率0%を確認。
                  銘柄集中度も上位5銘柄で3.8%と良好。

    2026-08-10のRISE_MIN変更(-1.2→-2.0)に伴う検証メモ:
      - ratio_minを緩める案は追加分が明確にマイナス(却下)
      - rise_hi(上限)を緩めても該当銘柄0件で無意味
      - RBスコアはRB=6を境に勝率52%台→59%台へ明確なジャンプがあり、
        現行の5.4(実質RB>=6)がノイズと信号の境界と確認、変更せず維持

    2026-08-12の追加検証: score_min/score_max/ratio_minの緩和も個別に試したが
      いずれも件数増に対し質の低下が目立つため見送り、rise_loのみ追加緩和した。

実行方法:
    python scan_morning_strategy_an_candidates.py
"""
import sys
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import db
from closing_watch import calc_rebound_score

JST = timezone(timedelta(hours=9))


def _load_name_dict():
    """code→会社名の辞書を作る（2026-08-11追加、2026-08-18にstock_masterキャッシュ
    経由に変更。J-Quants全銘柄マスタは1日1回だけ取得してDBにキャッシュされており
    (stock_master.py)、通常はここでネットワーク呼び出しは発生しない。
    取得失敗時は空dictを返し、呼び出し側でcodeそのものにフォールバックする。"""
    try:
        import stock_master
        return stock_master.get_names()
    except Exception as e:
        print(f"  ⚠️  会社名マスタ取得に失敗（コード表示にフォールバック）: {e}")
        return {}

# ── 戦略AN 判定条件（2026-08-10 検証済み。変更する場合は再検証すること）──
SCORE_MIN   = 1.5
SCORE_MAX   = 3.0
RATIO_MIN   = 1.4
RISE_MIN    = -2.5   # 2026-08-12: -2.0→-2.5（件数さらに約1.2〜1.4倍に増加、9分割点walk-forward・
                      #   増分単独検証・ブロックブートストラップ資金曲線の3手法すべてで改善確認済み）
RISE_MAX    = 9.1
RB_MIN      = 5.4
CD_MIN      = 1

TP_PCT = 0.015
SL_PCT = 0.059

MIN_VOLUME   = 50_000
MIN_TURNOVER = 50_000_000
MIN_PRICE    = 200   # simulate_precise.pyと同一（検証済み条件と揃える。scan_daily.pyのMIN_PRICE=100とは別）


def calc_score_an(hist):
    """scan_daily.py の calc_score() と同一の計算式（戦略A本体と同じscore/ratio定義を維持）。"""
    if len(hist) < 22:
        return None
    vol_series = pd.to_numeric(hist["Volume"], errors="coerce").fillna(0)
    vol_20d = vol_series.iloc[-20:].values
    vol_5d  = vol_series.iloc[-5:].values
    v1 = float(vol_series.iloc[-1])
    v3 = float(vol_series.iloc[-3])
    avg20 = vol_20d.mean()
    avg5  = vol_5d.mean()
    if avg20 == 0 or avg5 == 0 or v3 == 0:
        return None
    trend = np.polyfit(range(5), vol_5d, 1)[0] / avg5 * 100
    accel = (v1 - v3) / v3 * 100
    ratio = v1 / avg20
    score = min(trend/10, 3.0) + min(accel/100, 3.0) + min(ratio/3, 4.0)
    return round(score, 2), round(ratio, 2)


def calc_consec_drop(hist):
    """直近から遡って連続下落日数を数える（最大9日、simulate_precise.pyと同一ロジック）。"""
    n = len(hist)
    consec = 0
    for back in range(n - 1, max(0, n - 10), -1):
        if back == 0:
            break
        if float(hist.iloc[back]["Close"]) < float(hist.iloc[back - 1]["Close"]):
            consec += 1
        else:
            break
    return consec


def scan_an_candidates(as_of_date=None):
    """全銘柄に戦略AN条件を適用し、該当銘柄のリストを返す。

    as_of_date: バックテスト用。指定日までのdaily_pricesだけで判定する。
                省略時は直近データ（=本日の夜間スキャン）を使う。

    戻り値: list of dict（date, strategy='AN', code, name, score, ratio, judgment, reason）
            candidates_log の save_candidates_log_db() にそのまま渡せる形式。
            "date" は「これから取引される翌営業日」ではなく、スキャンした
            対象日（=前営業日）。実際の発注可否はエントリー当日の地合いが
            NORMALかどうかを、発注側で別途確認すること。
    """
    conn = db.get_conn()
    codes = db.get_all_codes(exclude_etf=True)
    name_dict = _load_name_dict()

    rows = []
    error_codes = []
    for code in codes:
        # 2026-08-10追加: 1銘柄の異常データ（型不整合・DB瞬断等）で
        # スキャン全体が例外停止し、その日の候補が0件になる（既に見つかった
        # 候補も失われる）のを防ぐため、銘柄単位でエラーを分離する。
        try:
            if as_of_date is not None:
                df = pd.read_sql(
                    "SELECT Date, Open, Close, Volume FROM daily_prices "
                    "WHERE code=? AND Date <= ? ORDER BY Date DESC LIMIT 26",
                    conn, params=(code, str(as_of_date)),
                )
            else:
                df = pd.read_sql(
                    "SELECT Date, Open, Close, Volume FROM daily_prices WHERE code=? ORDER BY Date DESC LIMIT 26",
                    conn, params=(code,),
                )
            if len(df) < 26:
                continue
            df = df.iloc[::-1].reset_index(drop=True)

            today = df.iloc[-1]
            if pd.isna(today["Close"]) or pd.isna(today["Open"]):
                continue
            close = float(today["Close"])
            open_ = float(today["Open"])
            volume = float(pd.to_numeric(today["Volume"], errors="coerce") or 0)
            if close < MIN_PRICE or volume < MIN_VOLUME or volume * close < MIN_TURNOVER:
                continue
            if open_ <= 0:
                continue

            sa = calc_score_an(df)
            if sa is None:
                continue
            score, ratio = sa
            if not (SCORE_MIN <= score < SCORE_MAX):
                continue
            if ratio < RATIO_MIN:
                continue

            today_rise = (close - open_) / open_ * 100
            if not (RISE_MIN <= today_rise <= RISE_MAX):
                continue

            rb_score = calc_rebound_score(df)
            if rb_score < RB_MIN:
                continue

            consec_drop = calc_consec_drop(df)
            if consec_drop < CD_MIN:
                continue

            # 候補の有効日は「データの最新日（=前営業日）」ではなく「これから取引される当日」。
            # as_of_date指定時（バックテスト用）はその日付をそのまま使う（scan_morning_strategy_d_candidates.py
            # で過去に発生した日付バグと同じ問題を避けるため、必ずJSTの実日付を使う）。
            entry_date = str(as_of_date) if as_of_date is not None else datetime.now(JST).strftime("%Y-%m-%d")
            rows.append({
                "date": entry_date,
                "strategy": "AN",
                "code": code,
                "condition": None,
                "name": name_dict.get(code, code),
                "score": score,
                "ratio": ratio,
                "judgment": "BUY",
                "reason": (f"戦略AN: score{score:.1f}×ratio{ratio:.1f}倍×寄引{today_rise:+.1f}%"
                           f"×RB{rb_score:.1f}×連続下落{consec_drop}日"
                           f" - 検証済み条件(2年OOS 勝率66.7%/平均+0.313%)"),
            })
        except Exception as e:
            error_codes.append((code, str(e)))
            continue

    conn.close()
    if error_codes:
        print(f"  ⚠️  戦略ANスキャンで{len(error_codes)}銘柄をエラーによりスキップ: "
              f"{error_codes[:5]}{'...' if len(error_codes) > 5 else ''}")
    return rows


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    result = scan_an_candidates()
    print(f"戦略AN候補: {len(result)}件")
    for r in result:
        print(f"  {r['code']}  {r['reason']}")
