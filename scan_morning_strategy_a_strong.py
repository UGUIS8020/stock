# -*- coding: utf-8 -*-
"""
scan_morning_strategy_a_strong.py - 戦略A STRONG日 判定ロジック

バックテスト根拠: sim_precise_trades.csv 2年分
  STRONG×BUY  avg-0.43% Sharpe-0.84
  STRONG×CAUTION avg+2.05% Sharpe+1.25
  → 全銘柄CAUTIONにしてdaytime.pyで+0.3%モメンタム確認後エントリー
"""


def judge_entry_a_strong(score, ratio, today_rise):
    """STRONG日 買い判定"""
    if score < 5.0:
        return "PASS", f"STRONG日 + スコア低すぎ({score:.1f})"
    if ratio >= 5.0:
        return "PASS", f"STRONG日 + 出来高過多({ratio:.1f}倍) - ratio5倍超は過熱"
    if today_rise >= 2.0:
        return "PASS", f"STRONG日 + 前日急騰({today_rise:+.1f}%) - 出尽くし"
    return "CAUTION", (f"STRONG日 score{score:.1f}×ratio{ratio:.1f}倍×前日{today_rise:+.1f}%"
                       f" - 全CAUTION（CAUTION avg+2.05% / BUY avg-0.43%）")
