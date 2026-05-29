# -*- coding: utf-8 -*-
"""
scan_morning_strategy_a.py - 戦略A（モメンタム買い）判定ロジック

バックテスト根拠: backtest_log.csv SL=-1%/TP=+3%適用後
  STRONG  CAUTION avg+2.05% / BUY avg-0.43% → 全CAUTION
  NORMAL  BUY avg+0.81% → 有効ゾーンのみBUY
  WEAK    score>=thr でCAUTION
修正時は backtest_log.csv の分析結果との整合性を確認すること。
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
    # キーファクター: 前日-3〜+1%が対象 / ratio10倍超・前日急騰は見送り
    if condition == "NORMAL":
        if ratio >= 10.0:
            return "PASS", f"NORMAL日 + ratio過多({ratio:.1f}倍) - 10倍超は過熱"
        if today_rise > 2.0:
            return "PASS", f"NORMAL日 + 前日急騰({today_rise:+.1f}%) - 出尽くし見送り"
        if score >= 9.0:
            if ratio >= 12.0:
                return "CAUTION", (f"NORMAL日 高スコア過熱({score:.1f}) + ratio{ratio:.1f}倍 "
                                   f"- 逆行リスクあり")
            return "CAUTION", (f"NORMAL日 高スコア({score:.1f}) - score9+は"
                               f"NORMAL日WR37%・avg-0.24%のため慎重")
        # BUYゾーン: score7〜9 × ratio3〜10x × 前日-3〜+1%
        if 7.0 <= score < 9.0 and 3.0 <= ratio < 10.0 and -3.0 <= today_rise <= 1.0:
            return "BUY", (f"NORMAL日 BUYゾーン score{score:.1f}×ratio{ratio:.1f}倍×前日{today_rise:+.1f}%"
                           f" - avg+0.170%・WR54%")
        if -3.0 <= today_rise <= 1.0:
            return "CAUTION", (f"NORMAL日 + 前日{today_rise:+.1f}% score{score:.1f}×ratio{ratio:.1f}倍"
                               f" - 様子見")
        return "PASS", f"NORMAL日 + 前日{today_rise:+.1f}%（有効条件外）"

    # ── WEAK日 ───────────────────────────────────────────
    # WEAK×CAUTION avg+0.95% / BUY avg+0.73% → score>=thr でCAUTION
    if score < strategy_a_thr:
        return "PASS", f"WEAK日 + スコア不十分({score:.1f} < 閾値{strategy_a_thr})"
    return "CAUTION", f"WEAK日 score{score:.1f} - スコア条件クリア・慎重にエントリー"
