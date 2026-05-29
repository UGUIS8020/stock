# -*- coding: utf-8 -*-
"""
scan_morning_strategy_a.py - 戦略A（モメンタム買い）判定ロジック

バックテスト根拠: sim_precise_trades.csv 2年分（約57万件）分析済み
  STRONG  全CAUTION → daytime.py で+0.3%モメンタム確認後エントリー
  NORMAL  全CAUTION → 2年データでBUYゾーンに統計的有意条件なし
           (月次で勝率36〜68%と不安定。ratio/score条件で安定したBUYゾーン確認不可)
  WEAK    score>=thr でCAUTION
修正時は simulate_ratio.py を再実行して整合性を確認すること。
"""

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
    """戦略A 買い判定（地合い別に完全分離）"""
    score      = float(row["score"])
    ratio      = float(row.get("ratio", 0))
    today_rise = float(row.get("today_rise", 0))

    if condition == "PANIC":
        return "PASS", "地合いPANIC - 全見送り"

    # ── STRONG日 ─────────────────────────────────────────
    # STRONG×BUY avg-0.43%(Sharpe-0.84) / CAUTION avg+2.05%(Sharpe+1.25)
    # → 全銘柄CAUTIONにしてdaytime.pyで0.3%上昇確認後エントリー
    if condition == "STRONG":
        if score < 5.0:
            return "PASS", f"STRONG日 + スコア低すぎ({score:.1f})"
        if ratio >= 5.0:
            return "PASS", f"STRONG日 + 出来高過多({ratio:.1f}倍) - ratio5倍超は過熱"
        if today_rise >= 2.0:
            return "PASS", f"STRONG日 + 前日急騰({today_rise:+.1f}%) - 出尽くし"
        return "CAUTION", (f"STRONG日 score{score:.1f}×ratio{ratio:.1f}倍×前日{today_rise:+.1f}% "
                           f"- 全CAUTION（CAUTION avg+2.05% / BUY avg-0.43%）")

    # ── NORMAL日 ─────────────────────────────────────────
    # analyze_a_normal.py（2年・57万件）でOOS検証通過した条件を実装
    # ※ gap_pct（寄り付きギャップ）は daytime.py 段階で gap<0 に絞ること
    #   → daytime.py: NORMAL日は GAP_MAX_PCT=0.0 に設定済み
    if condition == "NORMAL":
        if ratio >= 10.0:
            return "PASS", f"NORMAL日 + ratio過多({ratio:.1f}倍) - 10倍超は過熱"
        if today_rise > 3.0:
            return "PASS", f"NORMAL日 + 前日急騰({today_rise:+.1f}%) - 出尽くし見送り"
        if score < 5.0:
            return "PASS", f"NORMAL日 + スコア低すぎ({score:.1f})"

        # BUYゾーン① score5〜7 × ratio4〜8 × 前日-1〜+2%
        #   OOS検証: 勝率65.6% / 平均+0.372%（最優秀）
        if 5.0 <= score < 7.0 and 4.0 <= ratio < 8.0 and -1.0 <= today_rise <= 2.0:
            return "BUY", (f"NORMAL日BUY① score{score:.1f}×ratio{ratio:.1f}倍×前日{today_rise:+.1f}%"
                           f" - OOS勝率65.6%・平均+0.372%")

        # BUYゾーン② score7〜9 × ratio<6 × 前日0〜+3%
        #   OOS検証: 勝率58.9% / 平均+0.318%
        if 7.0 <= score < 9.0 and ratio < 6.0 and 0.0 <= today_rise <= 3.0:
            return "BUY", (f"NORMAL日BUY② score{score:.1f}×ratio{ratio:.1f}倍×前日{today_rise:+.1f}%"
                           f" - OOS勝率58.9%・平均+0.318%")

        # それ以外はCAUTION → daytime.py でモメンタム確認後エントリー
        return "CAUTION", (f"NORMAL日 score{score:.1f}×ratio{ratio:.1f}倍×前日{today_rise:+.1f}%"
                           f" - 条件外・様子見")

    # ── WEAK日 ───────────────────────────────────────────
    # WEAK×CAUTION avg+0.95% / BUY avg+0.73% → score>=thr でCAUTION
    if score < strategy_a_thr:
        return "PASS", f"WEAK日 + スコア不十分({score:.1f} < 閾値{strategy_a_thr})"
    return "CAUTION", f"WEAK日 score{score:.1f} - スコア条件クリア・慎重にエントリー"
