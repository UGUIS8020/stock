# -*- coding: utf-8 -*-
"""
scan_morning_strategy_b.py - 戦略B（リバウンド狙い）判定ロジック

[simulate_b_full.py検証結果 2025-04〜2026-03]
  閾値-4〜-5% × NORMAL日: 勝率50.5% / avg+0.219% ★
  閾値-5%以下 × NORMAL日: 勝率47.2% / avg+0.020%
  WEAK日（閾値問わず）  : 勝率30% / avg-1.26% ← 最悪
出口: 引け決済 ≒ TP+5%/SL-5% が最適
修正時は simulate_b_full.py の検証結果との整合性を確認すること。
"""


def judge_entry_b(row, condition):
    """戦略B 買い判定（リバウンド狙い）"""
    drop      = float(row["today_rise"])
    rb_score  = int(row.get("rebound_score", 0))
    rb_reason = str(row.get("rebound_reason", "指標なし"))

    if condition == "PANIC":
        return "PASS", "地合いPANIC - 逆張り非推奨（続落リスク）"

    if drop <= -20:
        return "PASS", f"暴落{drop:.1f}% + 続落リスク高 - 見送り"

    if condition == "WEAK":
        return "PASS", f"地合いWEAK + 逆張り非推奨 - WEAK日avg-1.26%（740件）"

    # NORMAL日 × RBスコア3以上 → BUY
    # evolve.py GA(20000人×100世代, 2026-05-26): TP+1.9%/SL-5.6% OOS WR67%・avg+0.276%
    if condition == "NORMAL" and rb_score >= 3:
        return "BUY", f"地合いNORMAL + リバウンド狙い({rb_score}点) - OOS WR67%・avg+0.276%（TP+1.9%/SL-5.6%推奨） / {rb_reason}"

    # STRONG日 × RB4以上 → BUY
    if condition == "STRONG" and rb_score >= 4:
        return "BUY", f"地合いSTRONG + リバウンド狙い({rb_score}点) - OOS WR67%・avg+0.276%（TP+1.9%/SL-5.6%推奨） / {rb_reason}"

    # STRONG日 × RB3 → CAUTION
    if condition == "STRONG" and rb_score == 3:
        return "CAUTION", f"地合いSTRONG + リバウンド候補({rb_score}点) - RB3はやや低め・様子見 / {rb_reason}"

    return "PASS", f"リバウンドスコア低({rb_score}点) - 見送り"
