# -*- coding: utf-8 -*-
"""
scan_morning_strategy_a_weak.py - 戦略A WEAK日 判定ロジック

バックテスト根拠: sim_precise_trades.csv 2年分
  WEAK×CAUTION avg+0.95% / WEAK×BUY avg+0.73%
  → score >= strategy_a_thr でCAUTION（BUYは出さない）
"""


def judge_entry_a_weak(score, ratio, today_rise, strategy_a_thr):
    """WEAK日 買い判定"""
    if score < strategy_a_thr:
        return "PASS", f"WEAK日 + スコア不十分({score:.1f} < 閾値{strategy_a_thr})"
    return "CAUTION", f"WEAK日 score{score:.1f} - スコア条件クリア・慎重にエントリー"
