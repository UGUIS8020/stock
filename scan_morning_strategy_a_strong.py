# -*- coding: utf-8 -*-
"""
scan_morning_strategy_a_strong.py - 戦略A STRONG日 判定ロジック

バックテスト根拠: analyze_a_strong.py（2年・83,252件）OOS検証済み
  STRONG全体: 勝率65.5% / 平均+0.658%

  【発見】スコアと勝率の関係がNORMAL日と逆転している:
    score 0〜3: 勝率66.9% / 平均+0.695%  ← 低スコアでも高パフォーマンス
    score 3〜5: 勝率64.1% / 平均+0.764%
    score 5〜7: 勝率60.3% / 平均+0.665%
    score 7〜9: 勝率53.5% / 平均+0.306%
    score 9以上: 勝率39.1% / 平均-0.716%  ← 高スコアほど危険

  【PASS条件の根拠（2年データ）】
    ratio >= 5.0: 勝率45.2% / 平均-0.256% → 除外
    score >= 9.0: 勝率39.1% / 平均-0.716% → 除外（逆効果）
    score < 5.0:  勝率66.1% / 平均+0.675% → 除外不要（旧コードのミス）
    today_rise >= 2.0: 勝率61.9% / 平均+0.778% → 除外不要（旧コードのミス）

  → 上記2条件以外は全てCAUTION
  → daytime.py で +0.3% モメンタム確認後エントリー
  条件を修正する場合は analyze_a_strong.py を再実行して確認すること。
"""


def judge_entry_a_strong(score, ratio, today_rise):
    """STRONG日 買い判定"""

    # ratio >= 5.0: 勝率45.2%・平均-0.256% → 除外
    if ratio >= 5.0:
        return "PASS", f"STRONG日 + ratio過多({ratio:.1f}倍) - 5倍超は勝率45%"

    # score >= 9.0: 勝率39.1%・平均-0.716% → 逆効果のため除外
    if score >= 9.0:
        return "PASS", f"STRONG日 + スコア過熱({score:.1f}) - 9以上は勝率39%・逆効果"

    # 上記以外は全CAUTION → market_watch.py で+0.3%確認後エントリー（9:05以降）
    # ※ score<5 / today_rise>=2 は除外不要（分析で高パフォーマンス確認済み）
    return "CAUTION", (f"STRONG日 score{score:.1f}×ratio{ratio:.1f}倍×前日{today_rise:+.1f}%"
                       f" - 全CAUTION（market_watch:+0.3%確認後エントリー）")
