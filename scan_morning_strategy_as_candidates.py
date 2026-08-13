# -*- coding: utf-8 -*-
"""
scan_morning_strategy_as_candidates.py - 戦略AS（戦略A・STRONG日 score<3.0専用）候補スキャン

戦略A(market_watch.py)は、SCORE_MIN_A(3.0)→TOP_N(30)→MAX_WATCH_A(10)という
3段階の絞り込みを持ち、いずれもscore降順で上位を優先する設計になっている。
そのためscore<3.0の銘柄は候補プールに一度も入らない。

一方、`analyze_a_strong.py`(2年・83,252件・OOS検証済み)は「STRONG日は
score0〜3が最も高パフォーマンス（勝率66.9%/平均+0.695%）」と結論しており、
戦略Aの3段階フィルタが自らの最良帯域を取りこぼしている構造だった
（戦略AN誕生の経緯と全く同じパターン）。

このモジュールは戦略AN(scan_morning_strategy_an_candidates.py)と同じ発想で、
戦略Aの3段階フィルタとは完全に独立した候補プールを作る。STRONG日側の
score3.0以上を優先する既存ロジック（戦略A本体）には一切触れない。

【前提】gap_pct（寄り付きギャップ）を使わないため、スキャン当日の夜間
（scan_daily.pyと同じ16:30以降）に前日分までのデータだけで全条件を確定できる。

条件（2026-08-10 GA探索(5000人×40世代・STRONG固定・gap無効)+ウォークフォワード
検証済み、2年データ）:
    score       : 0.0 <= score < 3.0     （calc_score_as と同一の計算式。戦略A本体の
                  SCORE_MIN_A=3.0の"外側"を狙うため、score<3.0のみに限定し重複を回避）
    ratio       : 0.0 <= ratio <= 4.0
    today_rise  : -1.8% <= 当日の寄り引け値幅 <= +7.6%
    rb_score    : calc_rebound_score(hist) >= 2.7
    price       : 200円 <= 株価 < 10,000円（2026-08-13追加。10,000円以上は勝率58.96%・
                  平均+0.579%と全価格帯中最弱、かつAS_DEFAULT_SHARES=100固定のため
                  高値株ほど1件あたりの資金インパクトも大きく除外。候補全体に占める
                  割合は約2.1%で機会損失はほぼ無し）
    TP/SL       : +4.8% / -5.6%（エントリー翌営業日の寄り付きで買う想定）
    検証結果    : 全体 N=7,028  勝率66.4%  平均+0.832%（2年、STRONG日）
                  50/60/70/80%の全分割点でwalk-forward訓練・検証とも頑健
                  （EV/日が全split点で100〜170超と大きく安定）。
                  23ヶ月中マイナス月0、銘柄重複も最大15/7,028件で偏りなし確認済み。
                  GA独立探索(196/200が安定基準クリア、うち大半がscore下限0付近)
                  でも同様の結果を再現。

実行方法:
    python scan_morning_strategy_as_candidates.py
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
    """J-Quants全銘柄マスタからcode→会社名の辞書を作る（2026-08-11追加）。
    取得失敗時は空dictを返し、呼び出し側でcodeそのものにフォールバックする。"""
    try:
        import jquantsapi
        from dotenv import load_dotenv
        load_dotenv()
        cli = jquantsapi.ClientV2(api_key=os.getenv("JQUANTS_API_KEY"))
        master = cli.get_eq_master()
        master["code4"] = master["Code"].astype(str).str[:4]
        return dict(zip(master["code4"], master["CoName"]))
    except Exception as e:
        print(f"  ⚠️  会社名マスタ取得に失敗（コード表示にフォールバック）: {e}")
        return {}

# ── 戦略AS 判定条件（2026-08-10 検証済み。変更する場合は再検証すること）──
SCORE_MIN   = 0.0
SCORE_MAX   = 3.0
RATIO_MIN   = 0.0
RATIO_MAX   = 4.0
RISE_MIN    = -1.8
RISE_MAX    = 7.6
RB_MIN      = 2.7

TP_PCT = 0.048
SL_PCT = 0.056

MIN_VOLUME   = 50_000
MIN_TURNOVER = 50_000_000
MIN_PRICE    = 200   # simulate_precise.pyと同一（検証済み条件と揃える。scan_daily.pyのMIN_PRICE=100とは別）
MAX_PRICE    = 10_000  # 2026-08-13新設: 10,000円以上は勝率58.96%・平均+0.579%と全価格帯中
                        # 最弱(sim_precise_trades.csvで検証、それ以外は平均+0.71%前後)。
                        # walk-forward4分割点中3つで同じ傾向を確認。全体候補に占める割合は
                        # 約2.1%とごく僅かで、除外による機会損失はほぼ無い（AS_DEFAULT_SHARES=100
                        # 固定のため高値株ほど1件あたりの金額も大きく、資金効率の面でも除外が妥当）。


def calc_score_as(hist):
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


def scan_as_candidates(as_of_date=None):
    """全銘柄に戦略AS条件を適用し、該当銘柄のリストを返す。

    as_of_date: バックテスト用。指定日までのdaily_pricesだけで判定する。
                省略時は直近データ（=本日の夜間スキャン）を使う。

    戻り値: list of dict（date, strategy='AS', code, name, score, ratio, judgment, reason）
            candidates_log の save_candidates_log_db() にそのまま渡せる形式。
            実際の発注可否はエントリー当日の地合いがSTRONGかどうかを、
            発注側（market_watch.py）で別途確認すること。
    """
    conn = db.get_conn()
    codes = db.get_all_codes(exclude_etf=True)
    name_dict = _load_name_dict()

    rows = []
    error_codes = []
    for code in codes:
        # 2026-08-10追加: 1銘柄の異常データでスキャン全体が例外停止し、
        # その日の候補が0件になる（既に見つかった候補も失われる）のを防ぐため、
        # 銘柄単位でエラーを分離する。
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
            if close < MIN_PRICE or close >= MAX_PRICE or volume < MIN_VOLUME or volume * close < MIN_TURNOVER:
                continue
            if open_ <= 0:
                continue

            sa = calc_score_as(df)
            if sa is None:
                continue
            score, ratio = sa
            if not (SCORE_MIN <= score < SCORE_MAX):
                continue
            if not (RATIO_MIN <= ratio <= RATIO_MAX):
                continue

            today_rise = (close - open_) / open_ * 100
            if not (RISE_MIN <= today_rise <= RISE_MAX):
                continue

            rb_score = calc_rebound_score(df)
            if rb_score < RB_MIN:
                continue

            # 候補の有効日は「データの最新日（=前営業日）」ではなく「これから取引される当日」。
            # as_of_date指定時（バックテスト用）はその日付をそのまま使う（scan_morning_strategy_d/an
            # で過去に発生した日付バグと同じ問題を避けるため、必ずJSTの実日付を使う）。
            entry_date = str(as_of_date) if as_of_date is not None else datetime.now(JST).strftime("%Y-%m-%d")
            rows.append({
                "date": entry_date,
                "strategy": "AS",
                "code": code,
                "condition": None,
                "name": name_dict.get(code, code),
                "score": score,
                "ratio": ratio,
                "judgment": "BUY",
                "reason": (f"戦略AS: score{score:.1f}×ratio{ratio:.1f}倍×寄引{today_rise:+.1f}%"
                           f"×RB{rb_score:.1f}"
                           f" - 検証済み条件(2年 勝率66.4%/平均+0.832%)"),
            })
        except Exception as e:
            error_codes.append((code, str(e)))
            continue

    conn.close()
    if error_codes:
        print(f"  ⚠️  戦略ASスキャンで{len(error_codes)}銘柄をエラーによりスキップ: "
              f"{error_codes[:5]}{'...' if len(error_codes) > 5 else ''}")
    return rows


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    result = scan_as_candidates()
    print(f"戦略AS候補: {len(result)}件")
    for r in result:
        print(f"  {r['code']}  {r['reason']}")
