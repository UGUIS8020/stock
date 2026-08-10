# -*- coding: utf-8 -*-
"""
scan_morning_strategy_d_candidates.py - 戦略D専用の候補スキャン（Stage 1）

daytime.py（戦略D・日中モメンタムシグナル）は、これまでscan_results（戦略A用の
TOP20〜30候補リスト）から候補を借りていたため、母集団が戦略Dの目的（イントラデイの
モメンタム初動銘柄を拾う）とミスマッチしていた（2026-08-04 検証で判明）。

このモジュールは、daytime.pyのエントリー条件（PREV_RISE_MIN・MA5/25乖離・出来高）の
うち、前日までのデータだけで判定できる部分を市場全体に対して事前フィルタし（Stage 1）、
戦略D専用の候補プールを作る。gap_pct（寄り付きギャップ）は当日始値が必要なため、
Stage 2（daytime.py起動時・市場開始直後）で別途確認する。

しきい値はdaytime.pyの既存定数をそのまま参照し、値を重複定義しない。
"""
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

import db
from daytime import MA5_DEV_MAX, MA25_DEV_MAX, MIN_VOLUME, PREV_RISE_MIN

JST = timezone(timedelta(hours=9))


def scan_d_prefilter(as_of_date=None):
    """前日までのdaily_pricesだけで判定できる条件（前日騰落・MA5/25乖離・出来高）を
    市場全体に適用し、戦略D候補の事前フィルタ結果を返す。

    戻り値: list of dict（date, strategy='D', code, name, score, ratio, judgment, reason）
            ※ candidates_log の save_candidates_log_db() にそのまま渡せる形式
    """
    conn = db.get_conn()
    codes = db.get_all_codes(exclude_etf=True)

    rows = []
    error_codes = []
    for code in codes:
        # 2026-08-10追加: 1銘柄の異常データでスキャン全体が例外停止し、
        # その日の候補が0件になる（既に見つかった候補も失われ、daytime.pyが
        # scan_resultsへフォールバックする）のを防ぐため、銘柄単位でエラーを分離する。
        try:
            df = pd.read_sql(
                "SELECT Date, Close, Volume FROM daily_prices WHERE code=? ORDER BY Date DESC LIMIT 26",
                conn, params=(code,),
            )
            if len(df) < 26:
                continue
            df = df.iloc[::-1].reset_index(drop=True)  # 古い順に並べ直す
            if as_of_date is not None:
                df = df[df["Date"] <= str(as_of_date)]
                if len(df) < 26:
                    continue

            closes = pd.to_numeric(df["Close"], errors="coerce").values
            volumes = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).values

            prev_close = closes[-1]
            prev2_close = closes[-2]
            if prev2_close <= 0:
                continue
            prev_rise = (prev_close - prev2_close) / prev2_close * 100
            if prev_rise < PREV_RISE_MIN:
                continue

            if volumes[-1] < MIN_VOLUME:
                continue

            ma5 = closes[-5:].mean()
            ma25 = closes[-25:].mean()
            ma5_dev = (prev_close / ma5 - 1) * 100 if ma5 > 0 else 0
            ma25_dev = (prev_close / ma25 - 1) * 100 if ma25 > 0 else 0
            if ma5_dev > MA5_DEV_MAX or ma25_dev > MA25_DEV_MAX:
                continue

            # 候補の有効日は「データの最新日（=前営業日）」ではなく「これから取引される当日」。
            # as_of_date指定時（バックテスト用）はその日付をそのまま使う。
            entry_date = str(as_of_date) if as_of_date is not None else datetime.now(JST).strftime("%Y-%m-%d")
            rows.append({
                "date": entry_date,
                "strategy": "D",
                "code": code,
                "condition": None,
                "name": code,
                "score": round(prev_rise, 2),
                "ratio": round(ma5_dev, 2),
                "judgment": "PREFILTER",
                "reason": f"prev_rise={prev_rise:+.2f}% ma5_dev={ma5_dev:+.2f}% ma25_dev={ma25_dev:+.2f}%",
            })
        except Exception as e:
            error_codes.append((code, str(e)))
            continue

    conn.close()
    if error_codes:
        print(f"  ⚠️  戦略Dスキャンで{len(error_codes)}銘柄をエラーによりスキップ: "
              f"{error_codes[:5]}{'...' if len(error_codes) > 5 else ''}")
    return rows


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    result = scan_d_prefilter()
    print(f"戦略D事前フィルタ候補: {len(result)}件")
    for r in result[:20]:
        print(f"  {r['code']}  prev_rise={r['score']:+.2f}%  ma5_dev={r['ratio']:+.2f}%")
