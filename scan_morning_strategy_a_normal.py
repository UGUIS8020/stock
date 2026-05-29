# -*- coding: utf-8 -*-
"""
scan_morning_strategy_a_normal.py - 戦略A NORMAL日 判定ロジック

バックテスト根拠: analyze_a_normal.py（2年・57万件）OOS検証通過条件
  ※ gap_pct（寄り付きギャップ）は daytime.py 側で gap<0 に絞ること
    → daytime.py: NORMAL日は gap_max=0.0 に設定済み
  条件を修正する場合は analyze_a_normal.py を再実行して確認すること。
"""


def judge_entry_a_normal(score, ratio, today_rise):
    """NORMAL日 買い判定"""
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

    return "CAUTION", (f"NORMAL日 score{score:.1f}×ratio{ratio:.1f}倍×前日{today_rise:+.1f}%"
                       f" - 条件外・様子見")
