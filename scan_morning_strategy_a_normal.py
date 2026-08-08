# -*- coding: utf-8 -*-
"""
scan_morning_strategy_a_normal.py - 戦略A NORMAL日 判定ロジック

2026-08-08: NORMAL日のBUYゾーン①②を撤去（STRONG限定相当に変更）。
  直近1年・2年のウォークフォワード検証で、BUYゾーン①(旧score5〜7×ratio4〜8×
  前日-1〜+2%)・ゾーン②(旧score7〜9×ratio<6×前日0〜+3%)とも、コード内に
  記載されていたOOS実績(勝率65.6%/58.9%)を再現できず(実測は勝率45〜48%
  ・ほぼ損益なし)、むしろSTRONG限定より一貫してわずかに成績を悪化させる
  ことが判明したため撤去。件数自体もNORMAL全体の0.5%未満とごく僅かで、
  影響は小さいがプラスにならない条件だった。
  元のバックテスト根拠(analyze_a_normal.py)がどの期間のものかは不明だが、
  少なくとも直近1〜2年では効果が確認できない。
  ※ CAUTION相当を返すことで、market_watch.py側の既存ロジック
    （NORMAL日CAUTION→見送り）がそのまま効き、実質的に発注対象外になる。
"""


def judge_entry_a_normal(score, ratio, today_rise):
    """NORMAL日 買い判定（2026-08-08〜: 常にPASS/CAUTIONのみ、BUYは返さない）"""
    if ratio >= 10.0:
        return "PASS", f"NORMAL日 + ratio過多({ratio:.1f}倍) - 10倍超は過熱"
    if today_rise > 3.0:
        return "PASS", f"NORMAL日 + 前日急騰({today_rise:+.1f}%) - 出尽くし見送り"
    if score < 5.0:
        return "PASS", f"NORMAL日 + スコア低すぎ({score:.1f})"

    return "CAUTION", (f"NORMAL日 score{score:.1f}×ratio{ratio:.1f}倍×前日{today_rise:+.1f}%"
                       f" - BUYゾーン撤去済み・様子見のみ")
