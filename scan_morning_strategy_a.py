# -*- coding: utf-8 -*-
"""
scan_morning_strategy_a.py - 戦略A 判定ルーター

地合いに応じて各ファイルの判定関数に振り分ける。
  STRONG → scan_morning_strategy_a_strong.py
  NORMAL → scan_morning_strategy_a_normal.py
  WEAK   → scan_morning_strategy_a_weak.py
"""

from scan_morning_strategy_a_strong import judge_entry_a_strong
from scan_morning_strategy_a_normal import judge_entry_a_normal
from scan_morning_strategy_a_weak   import judge_entry_a_weak

# 急騰フラグ閾値
# 【根拠】score>=9.5 & ratio>=10倍: 高値+10%以上到達率10.6% / TP+8%で平均+1.02%
SURGE_SCORE_MIN = 9.5
SURGE_RATIO_MIN = 10.0


def is_surge_flag(row):
    """急騰フラグ判定: スコア9.5以上 & 出来高比率10倍以上"""
    try:
        return float(row["score"]) >= SURGE_SCORE_MIN and float(row["ratio"]) >= SURGE_RATIO_MIN
    except Exception:
        return False


def judge_entry_a(row, condition, strategy_a_thr):
    """戦略A 買い判定（地合い別ファイルへ振り分け）"""
    score      = float(row["score"])
    ratio      = float(row.get("ratio", 0))
    today_rise = float(row.get("today_rise", 0))

    if condition == "PANIC":
        return "PASS", "地合いPANIC - 全見送り"
    if condition == "STRONG":
        return judge_entry_a_strong(score, ratio, today_rise)
    if condition == "NORMAL":
        return judge_entry_a_normal(score, ratio, today_rise)
    # WEAK
    return judge_entry_a_weak(score, ratio, today_rise, strategy_a_thr)
