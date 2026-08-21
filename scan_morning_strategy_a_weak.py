# -*- coding: utf-8 -*-
"""
scan_morning_strategy_a_weak.py - 戦略A WEAK日 判定ロジック

バックテスト根拠: sim_precise_trades.csv 2年分
  WEAK×CAUTION avg+0.95% / WEAK×BUY avg+0.73%
  → score >= strategy_a_thr でCAUTION（BUYは出さない）

【注意】現在この判定結果は発注タイミングに反映されない。
market_watch.py の decide_timing() が WEAK日を judgment の値に関わらず
無条件SKIPしているため（OOS avg=-0.781%、2026-07-12検証）。
ここでのPASS/CAUTIONの閾値ロジック自体はバックテスト記録として残しているが、
実際にWEAK日で発注を再開したい場合は decide_timing() 側の修正も必要。
"""


def judge_entry_a_weak(score, ratio, today_rise, strategy_a_thr):
    """WEAK日 買い判定"""
    if score < strategy_a_thr:
        return "PASS", f"WEAK日 + スコア不十分({score:.1f} < 閾値{strategy_a_thr})"
    return "CAUTION", f"WEAK日 score{score:.1f} - スコア条件クリア・慎重にエントリー"
