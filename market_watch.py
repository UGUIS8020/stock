"""
market_watch.py - 寄り付き後リアルタイム監視 + ルールベース発注

scan_morning.py → market_watch.py の順で自動起動される。

9:00〜9:30の間、候補銘柄の価格を15秒ごとに取得・表示し、
タイミング戦略（地合い・判定）に応じてルールベースで発注する。

実行方法:
    python market_watch.py          # 9:00まで待機して自動開始
    python market_watch.py --now    # 即時開始（テスト用）

前提:
    - scan_morning.py を先に実行済みであること
"""

import os
import re
import sys
import json
import time
import argparse
import requests
import urllib3
import pandas as pd
import yfinance as yf
import tachibana_order
import db
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from pathlib import Path
from scan_morning_strategy_a import judge_entry_a

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

_BASE_DIR            = Path(__file__).parent
CANDIDATES_LOG_CSV   = str(_BASE_DIR / "out" / "candidates_log.csv")
MORNING_LOG_CSV      = str(_BASE_DIR / "out" / "morning_log.csv")
TACHIBANA_LOGIN_FILE = str(_BASE_DIR / "tachibana_login_response.json")
OBSERVE_LOG_CSV      = str(_BASE_DIR / "out" / "observe_log.csv")
JST                = timezone(timedelta(hours=9))
TODAY              = datetime.now(JST).strftime("%Y-%m-%d")
# 監視設定
POLL_INTERVAL_SEC  = 15    # 価格取得間隔（秒）
WATCH_START_HOUR   = 9     # 監視開始時刻
WATCH_START_MIN    = 0
AI_JUDGE_MIN       = 5     # 銘柄エントリー判断のデフォルト開始時刻（9:05）
CONDITION_REFRESH_MIN = 3  # 地合い再判定タイミング（9:03）: 寄り付き初動が落ち着いた直後
WATCH_END_MIN      = 30    # 監視終了（9:30）

# TP/SL設定
TP_PCT = 0.06   # 利確 +6%（変更: 4%→6% / evolve.py GA 2026-07-18: A単独型STRONG限定 N=1528 WR=64.9% avg+0.659%
                #   10,000人×150世代・5,000人×50世代の2回の実行で一貫してTP+5.5〜+8.9%が優勢だったため採用）
SL_PCT = 0.06   # 損切 -6%（変更なし / 同GA結果でもSL-5〜-7%が中心で現行と大きくは乖離せず）

# NORMAL日ギャップダウン逆張りパス（TP3%/SL-1%）は2026-07-25に撤廃
# （ライブ勝率77.8%→0%の悪化を受け、7/12以前のシンプルなWAIT_CONFIRM判定に統一）

# STRONG日ギャップダウン逆張り TP/SL（順張りTP_PCTとは独立: 2026-07-25 分離）
# 根拠: analyze_a.py OOS検証 STRONG gap<0 score≥3 → wr=64.5% avg=+0.818% n=1,277（2026-07-12）
# 旧: 順張りTP_PCTを流用していたため、2026-07-21のTP_PCT 4%→6%変更に巻き込まれ
#     STRONG逆張り本来の根拠(TP4%)から乖離していた（force_close偏重の原因）
STRONG_GAP_TP_PCT = 0.04   # 利確 +4%
STRONG_GAP_SL_PCT = 0.06   # 損切 -6%

# 最大ポジション数（daytime.py の MAX_DAYTIME_POSITIONS と共有）
MAX_POSITIONS    = 15  # 戦略A（2026-08-14: 7→15。scoreランク別のwalk-forward検証(4分割点×
                        #   train/test)で、rank11-15がむしろrank1-10より高勝率・高平均
                        #   （8/8パターンでrank1-10以上）と判明。rank1-15を一括発注する
                        #   前提でEV/日を比較すると現行7件比で約2〜3倍（例: test70%split
                        #   EV/日 7.03→15.84）、平均損益率も同等以上（test各split
                        #   0.754/0.853/1.004/0.676 → 0.774/0.933/1.056/0.888）。
                        #   資金拘束は7×30万=210万円→15×30万=450万円に増加（他戦略との
                        #   合計は超過するが、利益率改善を優先する方針でユーザー承認済み）
MAX_POSITIONS_AN = 5   # 戦略AN（2026-08-10: 3→5。テスト運用中は建玉半分(AN_DEFAULT_SHARES等)で
                        #   資金拘束を抑えつつ枠を広げる。戦略Dの「建玉半分でテスト運用」と同じ方式）
MAX_POSITIONS_AS = 5   # 戦略AS（2026-08-14: 10→5に一時的に引き下げ。8/13の実額検証で
                        #   ASの銘柄選定ロジック自体は健全(勝率64.3%、バックテスト66.4%と
                        #   ほぼ一致)と確認済みで、質を理由にした引き下げではない。
                        #   同日にA本体をMAX_POSITIONS 7→15に引き上げ資金拘束が増えたため、
                        #   資金確保を優先して一時的に縮小。ASの実績(特にFix②導入後)が
                        #   良好であれば10に戻す想定。資金拘束は10×15万=150万円→
                        #   5×15万=75万円に半減

# 監視銘柄数の上限（絞り込み）
MAX_WATCH_A  = 15   # 戦略A: scoreで昇順（2026-08-10: 降順→昇順に修正、L638-640参照）
                     # （2026-08-14: 10→15。rank11-15を監視対象に含めるためMAX_POSITIONSと
                     #   合わせて引き上げ。詳細はMAX_POSITIONSのコメント参照）
MAX_WATCH_AN = 10   # 戦略AN（2026-08-10新設。戦略AのSCORE_MIN_A/TOP_N/MAX_WATCH_Aとは無関係の独立枠）
MAX_WATCH_AS = 20   # 戦略AS（2026-08-10新設。1日25〜500件超と候補が多いため監視上限を広めに設定）

# 戦略AN 専用TP/SL（2026-08-09検証: score1.5〜3.0×ratio1.4〜50×前日騰落-1.2〜+9.1%×
#   RBスコア≥5.4×連続下落≥1、2年OOS 検証n=149 勝率69.8% avg+0.393%）
AN_TP_PCT = 0.015
AN_SL_PCT = 0.059

# 戦略AS 専用TP/SL（2026-08-10検証: score0〜3×ratio0〜4×前日騰落-1.8〜+7.6%×RBスコア≥2.7、
#   STRONG日、2年 N=7,028 勝率66.4% avg+0.832%。GA探索(196/200が安定基準クリア)でも再現済み）
AS_TP_PCT = 0.048
AS_SL_PCT = 0.056

# 戦略AN テスト運用中の建玉サイズ（2026-08-10: 通常の半分。MAX_ORDER_AMOUNT/DEFAULT_SHARES比を
#   CHEAP_THRESHOLDと揃え、境界の断層を回避 = 150,000/100 = 1,500 = CHEAP_THRESHOLD）
AN_DEFAULT_SHARES   = 100
AN_MAX_ORDER_AMOUNT = 150_000

# 戦略AS 建玉サイズ（2026-08-10新設。実運用ゼロ日のためANと同じ半分サイズから開始）
AS_DEFAULT_SHARES   = 100
AS_MAX_ORDER_AMOUNT = 150_000

# 発注設定
AUTO_ORDER       = True     # True: 確認プロンプトなしで自動発注
LIMIT_ORDER      = False    # 成行発注（指値は受付エラー多発・モメンタム戦略には不向きのため 2026-06-25 変更）
LIMIT_WAIT_SECS  = 30       # 指値約定確認の待機秒数
DEFAULT_SHARES   = 300      # 高値株のデフォルト株数（2026-08-17: 資金1,300万円到達により200→300。
                            # A本体が実際にMAX_POSITIONS=15までどれだけ埋まるかは次のSTRONG日で
                            # 要確認、資金計画は[[project_capital_target_1300man]]参照）
CHEAP_THRESHOLD  = 1_000   # この価格未満は最大株数モード（2026-08-17: 300株化に合わせ1,500→1,000。
                            # 30万円÷300株の自然な境界は1,000円のため、戻し忘れると
                            # 999円/1,000円境界に断層が生じる点に注意）
MAX_ORDER_AMOUNT = 500_000  # 安い株の1発注上限額（2026-08-25: 逆張りが実績良好(46件+87,845円)
                            # のため30万→50万円に引き上げ。1,000円境界での300株⇔500株の
                            # 断層は許容（実際の候補が999円付近に来ることは稀なため実害小と判断）

# 戦略A本体・順張り(entry_change_pct>=0)専用の縮小建玉（2026-08-25導入）。
# 順張りn=10(STRONG regime)が平均-1.29%・-47,535円と一貫して不振なため、
# 損失を抑えつつサンプルを増やす目的で「1,000円未満は10万円まで、
# 1,000円以上は100株」に縮小。逆張り(entry_change_pct<0)は対象外
# （逆張りは順張りと異なり地合い持続に依存せず健全に推移しているため）。
FORWARD_A_DEFAULT_SHARES   = 100
FORWARD_A_MAX_ORDER_AMOUNT = 100_000


def calc_shares(price, strategy="A", is_forward=False):
    """価格に応じた発注株数を返す（100株単位）。
    CHEAP_THRESHOLD円未満の安い株はMAX_ORDER_AMOUNT以内で買えるだけ。
    strategy="AN"/"AS"はテスト運用中のため半分サイズを使う。
    is_forward=True（戦略A本体の順張りのみ）は縮小建玉を使う。"""
    if strategy == "AN":
        default_shares, max_order_amount = AN_DEFAULT_SHARES, AN_MAX_ORDER_AMOUNT
    elif strategy == "AS":
        default_shares, max_order_amount = AS_DEFAULT_SHARES, AS_MAX_ORDER_AMOUNT
    elif strategy == "A" and is_forward:
        default_shares, max_order_amount = FORWARD_A_DEFAULT_SHARES, FORWARD_A_MAX_ORDER_AMOUNT
    else:
        default_shares, max_order_amount = DEFAULT_SHARES, MAX_ORDER_AMOUNT
    if price and price < CHEAP_THRESHOLD:
        lots = int(max_order_amount / price / 100)
        return max(lots, 1) * 100
    return default_shares


# ══════════════════════════════════════════════
# エントリータイミング判断
# ══════════════════════════════════════════════
# 【戦略A】バックテスト根拠 2026-03-10〜03-25（47件・BUY判定）
# ─────────────────────────────────────────────
# WEAK日（12件）:
#   寄り付き→終値 平均 +1.49%
#   → 寄り付き成行が最適。
#
# NORMAL日（24件）:
#   寄り付き→終値 平均 -0.78%
#   → 9:05以降に上昇確認してからエントリー。
#     上昇していなければ見送りが正解。
# ─────────────────────────────────────────────

def decide_timing(condition, judgment, ai_recommendation="", strategy="A"):
    """地合い・判定からエントリータイミング戦略を返す。

    戻り値:
        {
          "style":        "OPEN_MARKET" | "WAIT_CONFIRM" | "SKIP",
          "ai_trigger_min": AI判断を起動する経過分数,
          "confirm_threshold_pct": 上昇確認に必要な前日比（%）,
          "description":  表示用テキスト,
        }
    """
    # 戦略AN（戦略A・NORMAL日専用の独立パイプライン、2026-08-10新設）:
    # 検証済み条件はNORMAL日限定・確認なしで即エントリー（confirm_threshold_pct
    # を極端に低く設定し、価格に関わらず最初のティックで発注させる）。
    # 9:00の朝予測ではなく、9:03の実測地合い再判定後の確定値で判断するため、
    # STRONG日CAUTION経路と同じCONDITION_REFRESH_MIN(9:03)まで待つ
    # （朝の予測地合いだけで即発注すると、9:03の補正で覆るリスクがあるため）。
    # STRONG/WEAK/PANICでは検証していないため見送り。
    if strategy == "AN":
        if condition == "NORMAL":
            return {
                "style":                 "WAIT_CONFIRM",
                "ai_trigger_min":        CONDITION_REFRESH_MIN,
                "confirm_threshold_pct": -99.0,
                "surge_limit_pct":       None,
                "description":           "戦略AN NORMAL日 → 9:03の地合い確定後に即エントリー",
            }
        return {
            "style":                 "SKIP",
            "ai_trigger_min":        None,
            "confirm_threshold_pct": None,
            "description":           f"戦略AN: {condition}日は未検証のため見送り",
        }

    # 戦略AS（戦略A・STRONG日 score<3.0専用の独立パイプライン、2026-08-10新設）:
    # 戦略AN(NORMAL日)と同じ設計。9:03の地合い確定後に即エントリー、STRONG以外は見送り。
    if strategy == "AS":
        if condition == "STRONG":
            return {
                "style":                 "WAIT_CONFIRM",
                "ai_trigger_min":        CONDITION_REFRESH_MIN,
                "confirm_threshold_pct": -99.0,
                "surge_limit_pct":       None,
                "description":           "戦略AS STRONG日 → 9:03の地合い確定後に即エントリー",
            }
        return {
            "style":                 "SKIP",
            "ai_trigger_min":        None,
            "confirm_threshold_pct": None,
            "description":           f"戦略AS: {condition}日は未検証のため見送り",
        }

    # 戦略A本体（AN/AS以外の全て）: 個別関数に分離（2026-08-22、純粋な切り出しのみ・ロジック変更なし）
    return decide_timing_a(condition, judgment)


def decide_timing_a(condition, judgment):
    """戦略A本体のエントリータイミング戦略を返す（decide_timing()から分離）。"""
    if condition == "PANIC":
        # PANIC日: 全見送り
        # 【根拠】PANIC日 BUY avg -0.27%、終日下落多数
        return {
            "style":                 "SKIP",
            "ai_trigger_min":        None,
            "confirm_threshold_pct": None,
            "description":           "PANIC日 → 全見送り",
        }

    # WEAK日: 戦略A全停止（OOS検証 2026-07-12: WEAK全体 avg=-0.781% n=7,289件）
    # 全gapレンジでマイナス（gap<0: avg=-0.607%, gap≥0: avg=-0.951%）
    # 【注意】judgment（scan_morning_strategy_a_weak.pyのPASS/CAUTION判定）は
    # ここで一切参照されず無条件SKIP。weak.py側のscore閾値ロジックは現状死んでいる。
    if condition == "WEAK":
        return {
            "style":                 "SKIP",
            "ai_trigger_min":        None,
            "confirm_threshold_pct": None,
            "description":           "WEAK日 → 戦略A全見送り（OOS avg=-0.781%）",
        }

    if condition in ("NORMAL", "STRONG"):
        # 上昇率上限（出尽くし除外）。STRONG: 5%超のみ除外（3〜5%は+1.38%）
        # NORMAL: 0.5%超で除外（check_surge.py 2026-07-06分析、2026-07-12導入）
        # 2026-07-25に一度撤廃したが、その根拠（7/13以降の2件のNORMAL負けトレード）は
        # 実際にはgap<0逆張りパス経由でsurge_limitとは無関係だったと判明。
        # sim_precise_trades.csv 30,473件のTP/SL込み精密シミュレーションで
        # gap帯別に検証した結果、上限が緩いほど一貫して悪化（無制限avg-0.159% → 0.5%上限avg-0.032%）
        # と確認できたため、2026-07-28に0.5%上限を再導入
        surge_limit = 0.5 if condition == "NORMAL" else 5.0

        if judgment == "CAUTION":
            # STRONG日は全銘柄CAUTION（BUYなし）。0.3%で取りこぼしを防ぐ
            #   → 9:03（地合い再判定と同タイミング / analyze_entry_timing: STRONG×9:03 avg+1.41%）
            if condition == "STRONG":
                threshold = 0.3
                trigger   = CONDITION_REFRESH_MIN
                range_desc = f"+{threshold}%〜+{surge_limit:.1f}%"
                return {
                    "style":                 "WAIT_CONFIRM",
                    "ai_trigger_min":        trigger,
                    "confirm_threshold_pct": threshold,
                    "surge_limit_pct":       surge_limit,
                    "description":           f"{condition}日CAUTION → 9:{trigger:02d}以降 前日比{range_desc}でエントリー",
                }
            # NORMAL日CAUTIONは見送り（2026-07-29: 従来threshold=0.5%とsurge_limit=0.5%が
            # 完全に一致しており、エントリー条件と出尽くし除外条件が同時に成立して
            # 実質的に一度も発注されない死んだコードパスだったため、明示的なSKIPに変更）
            return {
                "style":                 "SKIP",
                "ai_trigger_min":        None,
                "confirm_threshold_pct": None,
                "description":           "NORMAL日CAUTION → 見送り（BUY判定のみ発注対象）",
            }
        elif condition == "NORMAL":
            # NORMAL日BUY: 前日比0.0%以上〜0.5%未満でエントリー（0.5%超は出尽くし除外）
            # 9:07（analyze_entry_timing: NORMAL×9:07 avg+0.43%）
            # ※ 現状normal.pyはBUYを返さないため理論上未到達だが、将来BUYゾーンが
            #   復活した場合のために残す。
            range_desc = f"0.0%〜+{surge_limit:.1f}%" if surge_limit is not None else "0.0%以上"
            return {
                "style":                 "WAIT_CONFIRM",
                "ai_trigger_min":        7,
                "confirm_threshold_pct": 0.0,
                "surge_limit_pct":       surge_limit,
                "description":           f"NORMAL日BUY → 9:07以降 前日比{range_desc}でエントリー",
            }
        else:
            # STRONG日 + judgment=="PASS"（ratio≥5.0 or score≥9.0で除外された銘柄）。
            # 2026-08-22修正: 従来はcondition区別が無くこの分岐に落ち、NORMAL日BUY用の
            # 閾値0.0%（CAUTIONの0.3%より緩い）で発注されてしまっていた
            # （strong.pyが除外する銘柄を、除外されない銘柄より緩い条件で通す矛盾）。
            # strong.py本来の意図通りSKIPに修正。
            return {
                "style":                 "SKIP",
                "ai_trigger_min":        None,
                "confirm_threshold_pct": None,
                "description":           f"STRONG日 {judgment} → 除外対象のため見送り",
            }

    # UNKNOWN など
    return {
        "style":                 "WAIT_CONFIRM",
        "confirm_threshold_pct": 0.5,
        "description":           "地合い不明 → 9:05以降 上昇確認後エントリー",
    }


# 2026-08-05検証: ratio<=3.0の候補はn=10・勝率70.0%・平均+1.632%（ratio>3.0はn=34・勝率38.2%）。
# サンプルがまだ少ないため発注ロジック（株数・除外）は変更せず、後から集計しやすいよう
# 発注時にログへ明示するだけに留める（candidates_logと突き合わせれば集計可能）。
LOW_RATIO_THRESHOLD = 3.0


def _log_low_ratio_candidate(c):
    """低ratio候補を発注時にログへ明示する分析用マーカー（発注可否・株数には影響しない）。"""
    ratio = c.get("ratio")
    if ratio is not None and ratio <= LOW_RATIO_THRESHOLD:
        print(f"     📊 低ratio候補（ratio={ratio:.2f} ≤ {LOW_RATIO_THRESHOLD}）※分析用マーカー・検証中の指標")


# ══════════════════════════════════════════════
# 9:05 地合い再判定（立花API 実日経 + yfinance 前日終値）
# ══════════════════════════════════════════════
def _fetch_nikkei_905(url_price):
    """立花APIで code=101 (日経225) の直近約定価格(pDPP)を取得する。"""
    if not url_price:
        return None
    http = urllib3.PoolManager(cert_reqs="CERT_NONE")
    p_no_file = Path("out/last_p_no.txt")
    try:
        p_no = int(p_no_file.read_text().strip()) + 1
    except Exception:
        p_no = int(time.time())
    p_no_file.write_text(str(p_no))
    t = datetime.now(JST)
    psd = (f"{t.year}.{t.month:02}.{t.day:02}"
           f"-{t.hour:02}:{t.minute:02}:{t.second:02}"
           f".{t.microsecond // 1000:03}")
    params = (
        '{"p_no":"' + str(p_no) + '",'
        '"p_sd_date":"' + psd + '",'
        '"sCLMID":"CLMMfdsGetMarketPrice",'
        '"sTargetIssueCode":"101",'
        '"sTargetColumn":"pDPP",'
        '"sJsonOfmt":"5"}'
    )
    try:
        resp = http.request("GET", url_price + "?" + params,
                            timeout=urllib3.Timeout(connect=3, read=5))
        tachibana_order.log_api_call("market_watch.nikkei")
        result = json.loads(resp.data.decode("shift-jis", errors="ignore"))
        for item in result.get("aCLMMfdsMarketPrice", []):
            v = str(item.get("pDPP", "")).strip('"')
            if v:
                return float(v)
    except Exception:
        pass
    return None


def _get_nikkei_prev_close():
    """yfinance ^N225 で前日終値を返す。取得失敗時は None。"""
    try:
        hist = yf.Ticker("^N225").history(period="5d")
        today_str = datetime.now(JST).strftime("%Y-%m-%d")
        closes = []
        for ts, close in zip(hist.index, hist["Close"]):
            try:
                date_str = ts.tz_convert("Asia/Tokyo").strftime("%Y-%m-%d")
            except Exception:
                date_str = str(ts)[:10]
            if date_str < today_str:
                closes.append(float(close))
        return closes[-1] if closes else None
    except Exception:
        return None


def _refresh_condition_905(current_condition, market_info, url_price):
    """9:03に日経実値でLayer3スコアを再計算し、地合いが変われば新しい地合い文字列を返す。
    変化なし or 取得失敗時は None を返す。
    """
    t_label = f"9:{CONDITION_REFRESH_MIN:02d}"
    actual_nk = _fetch_nikkei_905(url_price)
    if actual_nk is None:
        print(f"  ⚠️  {t_label} 日経取得失敗 → 地合い再判定スキップ")
        return None

    prev_close = _get_nikkei_prev_close()
    if prev_close is None:
        print(f"  ⚠️  {t_label} 日経前日終値取得失敗 → 地合い再判定スキップ")
        return None

    actual_nk_chg  = (actual_nk - prev_close) / prev_close * 100
    raw_saved_nk   = market_info.get("nikkei_change")
    # 朝スキャン時にSGX/CME先物が両方取得失敗した場合、scan_morning.pyはLayer3を
    # 0点寄与として扱う（NULL保存）。ここでも同様に0点寄与として扱わないと、
    # NaN/Noneを0.0%相当のLayer3スコアとして誤って再計算に混ぜてしまう
    # （NaNはPythonで真値になるため`or 0.0`では防げない）。
    saved_nk_available = not pd.isna(raw_saved_nk)
    saved_nk_chg  = float(raw_saved_nk) if saved_nk_available else 0.0
    saved_score   = int(market_info.get("condition_score") or 0)

    def _l3(chg):
        if chg >= 1.0:  return 3
        if chg >= 0.0:  return 2
        if chg >= -0.5: return 1
        if chg >= -1.5: return 0
        return -1

    old_pts = _l3(saved_nk_chg) if saved_nk_available else 0
    new_pts = _l3(actual_nk_chg)

    print(f"\n  📊 {t_label} 日経実値: {actual_nk:,.0f}円  前日比{actual_nk_chg:+.2f}%"
          f"  (朝スキャン時: {saved_nk_chg:+.2f}%  Layer3: {old_pts}pt"
          f"{'' if saved_nk_available else '・朝は取得失敗のため0点扱い'})")

    if old_pts == new_pts:
        print(f"  ✅ Layer3スコア変化なし({old_pts}pt) → 地合い変更不要")
        # Layer3不変でも NORMAL×日経下落 → WEAK格下げチェック
        if current_condition == "NORMAL" and actual_nk_chg <= -0.3:
            new_cond = "WEAK"
            try:
                db.update_morning_log_condition(TODAY, new_cond, saved_score, actual_nk_chg)
            except Exception as e:
                print(f"  ⚠️  morning_log 更新失敗: {e}")
            print(f"  📉 NORMAL→WEAK格下げ（日経実値{actual_nk_chg:+.2f}% ≦ -0.3%）")
            return new_cond
        return None

    new_score = saved_score - old_pts + new_pts
    print(f"  📈 スコア再計算: {saved_score}pt → {new_score}pt  (Layer3: {old_pts}→{new_pts}pt)")

    dow    = float(market_info.get("dow_change")    or 0)
    nasdaq = float(market_info.get("nasdaq_change") or 0)
    sp500  = float(market_info.get("sp500_change")  or 0)
    up_count = sum(1 for v in [dow, nasdaq, sp500] if v > 0)

    # 初期判定（scan_morning.py）と統一: 2026-06-24修正済みの条件に合わせる
    # 旧: score<=3 AND nk_chg<=-1.0 → 誤PANIC多発（実績8件中1件のみ真のPANIC）
    if new_score <= 2 and actual_nk_chg <= -2.0:
        new_cond = "PANIC"
    elif new_score >= 8 and up_count >= 2:
        new_cond = "STRONG"
    elif new_score >= 4:
        new_cond = "NORMAL"
    elif new_score >= 2:
        new_cond = "WEAK"
    else:
        new_cond = "PANIC"

    # NORMAL×日経実値下落 → WEAK動的格下げ
    # OOS根拠: NORMAL日で日経<-0.3% → 騰落比0.35前後・取引avg=-0.2%台（2026-07-12）
    if new_cond == "NORMAL" and actual_nk_chg <= -0.3:
        new_cond = "WEAK"
        print(f"  📉 NORMAL→WEAK格下げ（日経実値{actual_nk_chg:+.2f}% ≦ -0.3%）")

    try:
        db.update_morning_log_condition(TODAY, new_cond, new_score, actual_nk_chg)
    except Exception as e:
        print(f"  ⚠️  morning_log 更新失敗: {e}")

    return new_cond if new_cond != current_condition else None


# ══════════════════════════════════════════════
# Tachibana API（立花証券）リアルタイム価格取得
# ══════════════════════════════════════════════
def load_tachibana_url():
    """tachibana_login_response.json から株価URLを読み込む。失敗時はNone。"""
    try:
        with open(TACHIBANA_LOGIN_FILE, encoding="utf-8") as f:
            data = json.load(f)
        url = data.get("sUrlPrice", "")
        return url if url else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def fetch_tachibana_prices(url_price, codes):
    """Tachibana APIから複数銘柄の現在値・気配値・出来高を一括取得。
    戻り値: {code: {"price", "ask", "bid", "ask_vol", "bid_vol", "prev_close", "volume"}}
    APIの上限120件/回をバッチ分割して全件取得する。
    """
    if not url_price or not codes:
        return {}
    batch_size = 120
    http = urllib3.PoolManager()
    quotes = {}
    p_no_file = Path("out/last_p_no.txt")
    try:
        p_no_base = int(p_no_file.read_text().strip()) + 1
    except Exception:
        p_no_base = int(time.time())
    for i, start in enumerate(range(0, len(codes), batch_size)):
        batch = codes[start:start + batch_size]
        code_list = ",".join(str(c) for c in batch)
        t = datetime.now(JST)
        p_sd_date = (f"{t.year}.{t.month:02}.{t.day:02}"
                     f"-{t.hour:02}:{t.minute:02}:{t.second:02}"
                     f".{t.microsecond // 1000:03}")
        params = (
            "{"
            f'"p_no":"{p_no_base + i}",'
            f'"p_sd_date":"{p_sd_date}",'
            '"sCLMID":"CLMMfdsGetMarketPrice",'
            f'"sTargetIssueCode":"{code_list}",'
            '"sTargetColumn":"pDPP,pPRP,pQAP,pQBP,pAV,pBV,pDV",'
            '"sJsonOfmt":"5"'
            "}"
        )
        try:
            p_no_file.write_text(str(p_no_base + i))
            resp = http.request("GET", url_price + "?" + params,
                                timeout=urllib3.Timeout(connect=3, read=5))
            tachibana_order.log_api_call("market_watch.fetch_prices")
            text = resp.data.decode("shift-jis", errors="ignore")
            result = json.loads(text)
            for item in result.get("aCLMMfdsMarketPrice", []):
                code = item.get("sIssueCode", "").strip('"')
                if not code or code == "stock_code":
                    continue
                def _f(key):
                    v = item.get(key, "")
                    if isinstance(v, str):
                        v = v.strip('"')
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        return None
                price = _f("pDPP") or _f("pPRP")
                quotes[code] = {
                    "price":      price,
                    "ask":        _f("pQAP"),
                    "bid":        _f("pQBP"),
                    "ask_vol":    _f("pAV"),
                    "bid_vol":    _f("pBV"),
                    "prev_close": _f("pPRP"),
                    "volume":     int(_f("pDV") or 0),
                }
        except Exception:
            pass
    return quotes


# ══════════════════════════════════════════════
# 価格取得（Yahoo Finance v8 API・フォールバック）
# ══════════════════════════════════════════════
def fetch_realtime_price(code):
    """Yahoo Finance v8 APIから直近の株価を取得する。
    遅延は約15〜30秒。立花証券API接続後はそちらに差し替え予定。
    戻り値: {"price": 現在値, "prev_close": 前日終値, "change_pct": 前日比%, "volume": 出来高}
    """
    url     = f"https://query1.finance.yahoo.com/v8/finance/chart/{code}.T"
    params  = {"interval": "1m", "range": "1d"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=8)
        if resp.status_code != 200:
            return None

        data   = resp.json()
        result = data["chart"]["result"][0]
        meta   = result["meta"]

        price      = meta.get("regularMarketPrice")
        prev_close = meta.get("previousClose") or meta.get("chartPreviousClose")

        if not price or not prev_close:
            return None

        change_pct = round((price - prev_close) / prev_close * 100, 2)

        # 直近の出来高（分足の合計）
        volumes   = result["indicators"]["quote"][0].get("volume", [])
        vol_total = sum(v for v in volumes if v) if volumes else 0

        return {
            "price":      price,
            "prev_close": prev_close,
            "change_pct": change_pct,
            "volume":     vol_total,
        }
    except Exception:
        return None


# ══════════════════════════════════════════════
# 候補銘柄の読み込み
# ══════════════════════════════════════════════
def _extract_rb_score(reason):
    """reason文字列から RBスコア（例: '(9点)'）を抽出する。"""
    m = re.search(r'\((\d+)点\)', str(reason))
    return int(m.group(1)) if m else 0


def load_candidates():
    """本日のBUY/CAUTION候補を読み込み、MAX_WATCH_A/B件に絞り込む。"""
    today_df = pd.DataFrame()
    try:
        import db as _db, sqlite3
        conn = sqlite3.connect(_db.DB_PATH)
        today_df = pd.read_sql(
            f'SELECT date,strategy,code,condition,name,score,ratio,judgment,reason '
            f'FROM candidates_log WHERE date="{TODAY}"', conn)
        conn.close()
    except Exception:
        pass

    if today_df.empty:
        print("  ⚠️  candidates_log にデータがありません。scan_morning.py を先に実行してください。")
        return [], "UNKNOWN"

    # 2026-08-10修正: candidates_logの先頭行から取るとAN/D(condition=None)の行を
    # 拾ってしまうリスクがあった（戦略Aが常に先頭に入る実装順序に暗黙依存していた）。
    # market_log→morning_logを正しく参照する既存関数に差し替え、行順に依存しないようにする。
    condition = _db.get_today_condition_db(TODAY)
    targets   = today_df[today_df["judgment"].isin(["BUY", "CAUTION"])].copy()
    targets["pending_reeval"] = False

    # 朝の予測がWEAKだった場合、実際の地合いはNORMAL/STRONGだった可能性がある
    # （daytime.py/closing_watch.py の実測補正で頻繁に上方修正されている実績あり）。
    # WEAK判定のスコア閾値(strategy_a_thr、通常8.0)でPASSになった銘柄のうち、
    # NORMAL日の最低ライン(score>=5.0)はクリアしているものを保留候補として拾っておき、
    # 9:03の実測地合い再判定時に条件が上方修正されたら再判定する。
    if condition == "WEAK":
        pending = today_df[
            (today_df["judgment"] == "PASS") &
            (today_df["strategy"] == "A") &
            (today_df["score"] >= 5.0)
        ].copy()
        if not pending.empty:
            pending["pending_reeval"] = True
            targets = pd.concat([targets, pending], ignore_index=True)
            print(f"  🔔 WEAK日PASS銘柄のうち{len(pending)}件を保留（9:03の実測補正で条件緩和した場合に再判定）")

    original  = len(targets)

    restricted = _db.get_restricted_codes(days=30)
    if restricted:
        before = len(targets)
        targets = targets[~targets["code"].astype(str).isin(restricted)]
        excluded = before - len(targets)
        if excluded > 0:
            print(f"  🚫 日計取引制限銘柄を除外: {excluded}件")

    # ── 戦略A: scoreで昇順（SCORE_MIN_Aに近い順）、上限MAX_WATCH_A件（戦略Bはclosing_watchに移管）
    # 2026-08-10変更: STRONG日はscore降順(高score優先)だと勝率46〜48%/平均ほぼ0%だったが、
    # score昇順(3.0に近い順)だと勝率62〜65%/平均+0.58〜+0.89%（4分割点全てでwalk-forward確認済み）。
    # scan_daily.pyのtop_a_low追加保存と対になる変更。WEAK日保留機構はtoday_df全体を見るため
    # 影響を受けない（MAX_WATCH_A絞り込み後のtargetsではなくtoday_dfから直接候補を拾う設計）。
    df_a = (targets[targets["strategy"] == "A"]
            .sort_values("score", ascending=True)
            .head(MAX_WATCH_A))

    # ── 戦略AN: 戦略Aとは独立した枠（SCORE_MIN_A/TOP_N/MAX_WATCH_Aの絞り込みを受けない）
    df_an = (targets[targets["strategy"] == "AN"]
             .sort_values("score", ascending=False)
             .head(MAX_WATCH_AN))

    # ── 戦略AS: 戦略Aとは独立した枠（SCORE_MIN_A/TOP_N/MAX_WATCH_Aの絞り込みを受けない）
    df_as = (targets[targets["strategy"] == "AS"]
             .sort_values("score", ascending=False)
             .head(MAX_WATCH_AS))

    targets = pd.concat([df_a, df_an, df_as], ignore_index=True).drop_duplicates(subset=["code", "strategy"])

    if len(targets) < original:
        print(f"  📌 監視対象を絞り込み: {original}件 → {len(targets)}件（A:{len(df_a)}件 / AN:{len(df_an)}件 / AS:{len(df_as)}件）")

    candidates = []
    for _, row in targets.iterrows():
        is_an = str(row["strategy"]) == "AN"
        is_as = str(row["strategy"]) == "AS"
        candidates.append({
            "code":              str(row["code"]),
            "name":              str(row["name"]),
            "strategy":          str(row["strategy"]),
            "score":             float(row["score"]),
            "ratio":             float(row["ratio"]) if "ratio" in row and pd.notna(row["ratio"]) else 0.0,
            "today_rise":        float(row["today_rise"]) if "today_rise" in row and pd.notna(row["today_rise"]) else 0.0,
            "judgment":          str(row["judgment"]),
            "reason":            str(row["reason"]),
            "ai_recommendation": str(row["ai_recommendation"]) if "ai_recommendation" in row and pd.notna(row["ai_recommendation"]) else "",
            "pending_reeval":    bool(row.get("pending_reeval", False)),
            "tp_pct":            AN_TP_PCT if is_an else (AS_TP_PCT if is_as else TP_PCT),
            "sl_pct":            AN_SL_PCT if is_an else (AS_SL_PCT if is_as else SL_PCT),
        })

    return candidates, condition


def load_market_info():
    """morning_log テーブルから本日の地合い情報を読み込む"""
    try:
        import db as _db, sqlite3
        conn = sqlite3.connect(_db.DB_PATH)
        df = pd.read_sql(
            'SELECT * FROM morning_log WHERE date=?', conn, params=[TODAY])
        conn.close()
        if not df.empty:
            return df.iloc[0].to_dict()
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════
# 異常検知（ルールベース）
# ══════════════════════════════════════════════
def check_anomaly(candidate, price=None):
    """異常な銘柄を検出してメッセージを返す。正常ならNone。
    price指定時はcandidate自身の価格ではなくこちらを使う。candidate["price"]は
    候補作成時点の値のまま更新されないため、監視中に200円を割り込んだ銘柄が
    古い価格でチェックをすり抜ける実例(2026-08-17発覚)があった。発注直前の
    最新価格(latest_px)を渡すことでこれを防ぐ。"""
    price      = price if price is not None else (candidate.get("price") or 0)
    ratio      = candidate.get("ratio") or 0
    today_rise = candidate.get("today_rise") or 0

    if 0 < price < 200:
        return f"株価{price:.0f}円（極端に安い・上場廃止リスク）"
    if ratio >= 30:
        return f"出来高比率{ratio:.0f}倍（薄商い銘柄の異常値の可能性）"
    if today_rise >= 15:
        return f"前日+{today_rise:.1f}%急騰済み（高値追いリスク）"
    if today_rise <= -15:
        return f"前日{today_rise:.1f}%暴落（業績悪化・売り継続リスク）"
    return None


def ask_claude_entry(candidate, price_history, condition, market_info, timing=None):
    """互換性のためのスタブ（AI判断は無効化済み）"""
    raise NotImplementedError("AI判断は無効化されています")

def confirm_and_order(candidate, price, url_request, condition=None, entry_change_pct=None):
    """AUTO_ORDER=True: 確認なしで自動発注。False: y/n プロンプト表示。"""
    code      = candidate["code"]
    name      = candidate["name"]
    tp_pct    = candidate.get("tp_pct", TP_PCT)
    sl_pct    = candidate.get("sl_pct", SL_PCT)
    strat     = candidate.get("strategy", "A")
    rec_price = price
    sell_p    = round(price * (1 + tp_pct)) if price else None
    stop_p    = round(price * (1 - sl_pct)) if price else None

    is_forward = strat == "A" and entry_change_pct is not None and entry_change_pct >= 0
    shares    = calc_shares(rec_price or price, strategy=strat, is_forward=is_forward)
    estimated = int((rec_price or price or 0) * shares)

    # 買い余力確認
    buying_power = None
    if url_request:
        buying_power = tachibana_order.get_buying_power(url_request)

    mode = "本番" if tachibana_order.LIVE_TRADING else "モック（実発注なし）"

    print(f"\n{'━'*56}")
    print(f"  {'🤖 自動発注' if AUTO_ORDER else '🔔🔔🔔  発注確認'}  [{code}] {name}")
    print(f"{'━'*56}")
    print(f"     現在値   : {price:>8,.0f}円" if price else "     現在値   : 取得中")
    print(f"     指値売り : {sell_p:>8,.0f}円" if sell_p else "")
    print(f"     損切り   : {stop_p:>8,.0f}円" if stop_p else "")
    print(f"     発注株数 : {shares}株")
    print(f"     概算金額 : {estimated:>8,.0f}円")
    if buying_power is not None:
        print(f"     買余力   : {buying_power:>8,.0f}円")
        if estimated > buying_power:
            print(f"  ⚠️  買余力不足の可能性があります")
    print(f"     モード   : {mode}")
    print(f"{'━'*56}")

    if AUTO_ORDER:
        # 自動発注
        if not url_request:
            print(f"  ❌ 業務URLが取得できません。再ログインしてください。")
            return
        order_type = "指値" if LIMIT_ORDER else "成行"
        print(f"  📤 自動発注中... {code} {shares}株 {order_type}買い @ {rec_price or price:.0f}円")
        _do_order(code, name, shares, rec_price or price, url_request, tp_pct=tp_pct, sl_pct=sl_pct, strategy=strat,
                  condition=condition, entry_change_pct=entry_change_pct)
        return

    # 手動確認モード
    while True:
        ans = input(">>> [y=発注 / n=見送り / 数字=株数変更] : ").strip().lower()

        if ans == "n" or ans == "":
            print(f"  ↩️  {code} 見送りました。")
            return

        if ans.isdigit():
            shares    = int(ans)
            estimated = int(rec_price * shares) if rec_price else 0
            print(f"     株数を {shares}株（概算 {estimated:,.0f}円）に変更しました。")
            print(f"  [y=発注 / n=見送り] > ", end="", flush=True)
            continue

        if ans == "y":
            if not url_request:
                print(f"  ❌ 業務URLが取得できません。再ログインしてください。")
                return
            order_type = "指値" if LIMIT_ORDER else "成行"
            print(f"  📤 発注中... {code} {shares}株 {order_type}買い @ {rec_price or price:.0f}円")
            _do_order(code, name, shares, rec_price or price, url_request, tp_pct=tp_pct, sl_pct=sl_pct, strategy=strat,
                      condition=condition, entry_change_pct=entry_change_pct)
            return


def _do_order(code, name, shares, buy_price, url_request, tp_pct=None, sl_pct=None, strategy="A",
               condition=None, entry_change_pct=None):
    """発注処理本体（自動・手動共通）。
    LIMIT_ORDER=True: 現在値で指値→LIMIT_WAIT_SECS秒後に未約定なら取消して成行。
    LIMIT_ORDER=False: 成行のみ。
    """
    mkt_code  = db.get_market_code_db(code)
    _tp       = tp_pct if tp_pct is not None else TP_PCT
    _sl       = sl_pct if sl_pct is not None else SL_PCT
    _strategy = strategy or "A"

    # 同一銘柄の重複発注防止（closing_watch.py/daytime.pyにはあるが従来market_watch.pyには無かったチェック）。
    # watch_loop内のordered辞書はプロセス単位のメモリ状態のため、
    # 何らかの理由でmarket_watch.pyが同日中に再実行されると、既に買った銘柄をまた買ってしまう
    # （2026-07-15: 463Aが想定300株の2倍600株買われた事象の原因と推定）。
    if str(code) in db.get_today_ordered_codes(TODAY):
        print(f"  ⏭️  [{code}] 本日すでに発注済み - スキップ")
        return

    if LIMIT_ORDER and buy_price:
        r = tachibana_order.place_buy_limit_with_fallback(
            url_request, code, shares, buy_price, mkt_code,
            wait_secs=LIMIT_WAIT_SECS, strategy=_strategy)
        if r["success"]:
            print(f"  ✅ {r['message']}")
            tachibana_order.save_position(code, name, r["filled"], r["fill_price"],
                                          strategy=_strategy, tp_pct=_tp, sl_pct=_sl,
                                          condition=condition, entry_change_pct=entry_change_pct,
                                          url_request=url_request, order_no=r.get("order_no"))
        else:
            print(f"  ❌ 発注失敗: {r['message']}")
    else:
        result = tachibana_order.place_buy_order(url_request, code, shares, market_code=mkt_code)
        if result["success"]:
            print(f"  ✅ {result['message']}")
            buy_px = int(buy_price or 0)
            if buy_px <= 0:
                buy_px = int(result.get("fill_price") or result.get("price") or buy_price or 0)
            if buy_px > 0:
                eigyou_day = (result.get("raw") or {}).get("sEigyouDay")
                tachibana_order.save_position(code, name, shares, buy_px,
                                              strategy=_strategy, tp_pct=_tp, sl_pct=_sl,
                                              condition=condition, entry_change_pct=entry_change_pct,
                                              url_request=url_request, order_no=result.get("order_no"),
                                              eigyou_day=eigyou_day)
            else:
                print(f"  ⚠️  買値が取得できないためDBへの記録をスキップしました（手動で register_positions.py を実行してください）")
        else:
            print(f"  ❌ 発注失敗: {result['message']}")
            print(f"     APIレスポンス: {result['raw']}")



# ══════════════════════════════════════════════
# リアルタイム監視メインループ
# ══════════════════════════════════════════════
def watch_loop(candidates, condition, market_info, start_now=False, url_price=None, url_request=None):
    """9:00〜9:30の間、価格を監視してルールベースで発注する"""

    # 銘柄ごとにタイミング戦略を決定
    timings   = {c["code"]: decide_timing(condition, c["judgment"], "", c.get("strategy", "A")) for c in candidates}
    histories = {c["code"]: [] for c in candidates}
    ordered   = {c["code"]: False for c in candidates}   # 発注済みフラグ
    results   = {c["code"]: "監視中" for c in candidates}
    condition_refreshed = False  # 9:05 地合い再判定フラグ

    # 案1+3: CMEが正の日は個別株がCMEを超過上昇していないと発注しない
    # シミュレーション結果: 現状-51.7% → CME超過フィルタ適用で-0.6%(-51.1pt改善)
    cme_chg = float(market_info.get("nikkei_change") or 0.0)

    print(f"\n{'='*60}")
    print(f"【リアルタイム監視】{len(candidates)}銘柄  地合い:{condition}")
    for c in candidates:
        t = timings[c["code"]]
        print(f"  {c['code']} {c['name']}: {t['description']}")
    print(f"{'='*60}")
    print(f"\n  {'時刻':<8}", end="")
    for c in candidates:
        print(f"  {c['code']}({c['name'][:6]})", end="")
    print()
    print("  " + "─" * (8 + len(candidates) * 18))

    while True:
        now     = datetime.now(JST)
        now_min = now.hour * 60 + now.minute

        market_start    = WATCH_START_HOUR * 60 + WATCH_START_MIN
        refresh_start   = WATCH_START_HOUR * 60 + CONDITION_REFRESH_MIN
        watch_end       = WATCH_START_HOUR * 60 + WATCH_END_MIN

        # 9:03 地合い再判定（初回のみ）: 寄り付き初動が落ち着いた直後
        if not condition_refreshed and now_min >= refresh_start:
            condition_refreshed = True
            new_cond = _refresh_condition_905(condition, market_info, url_price)
            if new_cond:
                print(f"\n  🔄 9:03 地合い再判定: {condition} → {new_cond}  ※タイミング戦略を更新")
                condition = new_cond

                # 朝WEAKでPASSだった保留銘柄を、実測地合いで再判定する
                strategy_a_thr = float(market_info.get("strategy_a_thr") or 8.0)
                for c in candidates:
                    if not c.get("pending_reeval") or ordered[c["code"]]:
                        continue
                    new_judgment, new_reason = judge_entry_a(
                        {"score": c["score"], "ratio": c["ratio"], "today_rise": c["today_rise"]},
                        condition, strategy_a_thr)
                    if new_judgment != c["judgment"]:
                        print(f"     🔁 {c['code']} {c['name']}: 保留 → {new_judgment}（{new_reason}）")
                    c["judgment"] = new_judgment
                    c["reason"]   = new_reason

                timings = {c["code"]: decide_timing(condition, c["judgment"], "", c.get("strategy", "A"))
                           for c in candidates if not ordered[c["code"]]}
                try:
                    db.update_condition_in_candidates_log(TODAY, new_cond)
                except Exception:
                    pass

        # 終了判定
        if now_min >= watch_end:
            print(f"\n  ⏰ 9:{WATCH_END_MIN:02d}になりました。監視を終了します。")
            break

        # 市場開始前は待機
        if not start_now and now_min < market_start:
            wait_sec = (market_start - now_min) * 60 - now.second
            print(f"  市場開始まで {wait_sec}秒待機中... ({now.strftime('%H:%M:%S')})")
            time.sleep(30)
            continue

        # 価格取得・表示
        time_str = now.strftime("%H:%M:%S")
        print(f"  {time_str}", end="")

        # Tachibana API で全候補を一括取得（失敗時は空dict → Yahoo Financeへ）
        tachibana_batch = {}
        if url_price:
            try:
                tachibana_batch = fetch_tachibana_prices(
                    url_price, [c["code"] for c in candidates]
                )
            except Exception:
                tachibana_batch = {}

        for c in candidates:
            code = c["code"]

            # Tachibana 優先 → Yahoo Finance フォールバック
            tq = tachibana_batch.get(str(code))
            if tq and tq.get("price"):
                prev_close = tq["prev_close"] or tq["price"]
                change_pct = (
                    round((tq["price"] - prev_close) / prev_close * 100, 2)
                    if prev_close else 0
                )
                result = {
                    "price":      tq["price"],
                    "prev_close": prev_close,
                    "change_pct": change_pct,
                    "volume":     tq.get("volume") or 0,
                    "ask":        tq.get("ask"),
                    "bid":        tq.get("bid"),
                }
            else:
                yf_result = fetch_realtime_price(code)
                result = yf_result  # None の場合もある

            if result:
                price      = result["price"]
                change_pct = result["change_pct"]
                volume     = result["volume"]

                # 勢い判定（前回からの変化）
                prev_history = histories[code]
                if prev_history:
                    prev_price = prev_history[-1]["price"]
                    momentum   = "↑↑" if price > prev_price * 1.002 else \
                                 "↑"  if price > prev_price else \
                                 "↓↓" if price < prev_price * 0.998 else \
                                 "↓"  if price < prev_price else "→"
                else:
                    momentum = "→"

                histories[code].append({
                    "time":       time_str,
                    "price":      price,
                    "prev_close": result["prev_close"],
                    "change_pct": change_pct,
                    "volume":     volume,
                    "momentum":   momentum,
                    "ask":        result.get("ask"),
                    "bid":        result.get("bid"),
                })

                chg_icon = "📈" if change_pct >= 0 else "📉"
                src_mark = "📡" if (tq and tq.get("price")) else "🌐"
                print(f"  {src_mark}{chg_icon}{price:>7,.0f}円({change_pct:>+.1f}%)", end="")
            else:
                print(f"  {'取得失敗':>14}", end="")

        print()  # 改行

        # タイミング戦略に応じてルールベースで発注判断
        # ── パス1: SKIP/OBSERVE/OPEN_MARKET を処理 ──────────────────
        wait_confirm_ready = []   # WAIT_CONFIRM で閾値超えの候補を蓄積

        for c in candidates:
            code   = c["code"]
            timing = timings[code]

            if ordered[code]:
                continue

            # SKIP: PANIC日・WEAK日など全見送り
            if timing["style"] == "SKIP":
                # 戦略AN/ASは9:03の地合い再判定より前にSKIPを確定させない。
                # 朝の予測地合いが対象条件(AN=NORMAL/AS=STRONG)と異なっていても、
                # 9:03に修正されるケースを取りこぼすため（2026-08-10発覚。
                # 9:03再判定後はtimingsが再計算されるので、そこで改めてSKIP/WAIT_CONFIRMが正しく決まる）。
                if c.get("strategy") in ("AN", "AS") and not condition_refreshed:
                    continue
                ordered[code]  = True
                results[code]  = f"見送り({condition})"
                print(f"\n  ⏭️  {code} {c['name']}: {timing['description']}")
                continue

            # OBSERVE: データ蓄積のみ（発注しない）
            if timing["style"] == "OBSERVE":
                continue

            trigger_min = timing.get("ai_trigger_min", AI_JUDGE_MIN)
            threshold   = timing.get("confirm_threshold_pct", 0.0)

            if now_min < WATCH_START_HOUR * 60 + trigger_min:
                continue
            if len(histories[code]) < 2:
                continue

            latest_chg = histories[code][-1]["change_pct"] if histories[code] else 0
            latest_px  = histories[code][-1]["price"]       if histories[code] else None

            # 異常検知 → 強制見送り（発注直前の最新価格でチェック、2026-08-17修正）
            anomaly = check_anomaly(c, price=latest_px)
            if anomaly:
                ordered[code] = True
                results[code] = f"見送り(異常検知)"
                print(f"\n  🚫 {code} {c['name']}: 異常検知 → 強制見送り（{anomaly}）")
                continue

            # OPEN_MARKET: トリガー時刻到達で即発注（変更なし）
            if timing["style"] == "OPEN_MARKET":
                strat_c  = c.get("strategy", "A")
                max_pos  = MAX_POSITIONS_AN if strat_c == "AN" else (MAX_POSITIONS_AS if strat_c == "AS" else MAX_POSITIONS)
                open_pos = db.load_open_positions(strategy=strat_c)
                if len(open_pos) >= max_pos:
                    ordered[code] = True
                    results[code] = f"見送り(上限{max_pos}件)"
                    print(f"\n  ⚠️ {code} {c['name']}: 戦略{strat_c}上限({max_pos}件)到達 → 見送り")
                    continue
                ordered[code] = True
                results[code] = "発注"
                print(f"\n  🟢 {code} {c['name']}: {timing['description']}")
                _log_low_ratio_candidate(c)
                confirm_and_order(c, latest_px, url_request, condition=condition, entry_change_pct=latest_chg)
                continue

            # STRONG日ギャップダウン逆張りパス（戦略A本体専用）
            # gap<0（寄り付きが前日終値割れ）を強い地合いで拾う
            # OOS根拠: STRONG gap<0 score≥3 → wr=64.5% avg=+0.818% n=1,277（2026-07-12）
            # このバックテストはscore≥3（戦略A本体の対象帯）に基づくもので、score<3の
            # 戦略ASでは未検証。2026-08-13: strat_c=="A"の条件が無かったため、AS候補も
            # このパスに入り込み、かつ上限チェックがstrategy="A"/MAX_POSITIONSに
            # ハードコードされていたためAS本来の上限(MAX_POSITIONS_AS)を素通りして
            # 14件保有（上限10件のはず）という事象が発生。strat_c=="A"限定に修正し、
            # ASは通常のWAIT_CONFIRM経路（上限が正しくチェックされる）に流す。
            # 専用TP/SL（STRONG_GAP_TP_PCT/SL_PCT）を適用。順張りTP_PCTとは独立（2026-07-25分離）
            # 2026-08-01: is_v_recovery()によるV字回復確認を追加していたが、正式なバックテストを
            #   経ていない変更だった（過去ログが日次で上書きされ厳密な検証ができなかったため、
            #   「実運用で効果を確認する」目的の未検証フォワードテストとして投入されていた）。
            # 2026-08-05: 全体の勝率がバックテスト実績(64.5%)を下回っていることの一因と判断し、
            #   本来のバックテスト条件（latest_chgのみ判定、V字回復確認なし）に戻した。
            if (condition == "STRONG"
                    and timing["style"] == "WAIT_CONFIRM"
                    and not ordered[code]
                    and latest_chg < 0.0
                    and c.get("strategy", "A") == "A"):
                open_pos = db.load_open_positions(strategy="A")
                if len(open_pos) >= MAX_POSITIONS:
                    ordered[code] = True
                    results[code] = f"見送り(上限{MAX_POSITIONS}件)"
                    print(f"\n  ⚠️ {code} {c['name']}: 上限到達 → 見送り")
                    continue
                ordered[code] = True
                results[code] = "発注(STRONG逆張り)"
                c_gap = {**c, "tp_pct": STRONG_GAP_TP_PCT, "sl_pct": STRONG_GAP_SL_PCT}
                print(f"\n  🔄 {code} {c['name']}: gap{latest_chg:+.1f}% STRONG逆張り"
                      f" TP={STRONG_GAP_TP_PCT*100:.0f}%/SL={STRONG_GAP_SL_PCT*100:.0f}%")
                _log_low_ratio_candidate(c)
                confirm_and_order(c_gap, latest_px, url_request, condition=condition, entry_change_pct=latest_chg)
                continue

            # WAIT_CONFIRM: 閾値超えの候補を蓄積（B案: 即発注しない）
            if timing["style"] == "WAIT_CONFIRM":
                surge_limit = timing.get("surge_limit_pct")

                # 上昇率上限チェック: 出尽くし除外（NORMAL≥0.5% / STRONG≥5%）
                if surge_limit is not None and latest_chg >= surge_limit:
                    if not ordered[code]:
                        ordered[code] = True
                        results[code] = "見送り(出尽くし)"
                        print(f"\n  ⛔ {code} {c['name']}: 前日比{latest_chg:+.1f}% ≥ 上限{surge_limit:.0f}%"
                              f" → 出尽くし見送り")
                    continue

                # CMEが正の日は個別株がCMEを超過上昇していないと発注しない（案1+3）
                eff_thr = max(threshold, cme_chg) if cme_chg > 0 else threshold
                if latest_chg >= eff_thr:
                    wait_confirm_ready.append((latest_chg, c, latest_px, eff_thr))
                continue

        # ── パス2: WAIT_CONFIRM → 上昇率が高い順に発注（B案）────────
        # 同一ティック内で閾値を超えた全候補を上昇率降順に並べ、
        # ポジション枠がある限り順番に発注する
        wait_confirm_ready.sort(key=lambda x: -x[0])
        for latest_chg, c, latest_px, threshold in wait_confirm_ready:
            code = c["code"]
            if ordered[code]:
                continue
            strat_c  = c.get("strategy", "A")
            max_pos  = MAX_POSITIONS_AN if strat_c == "AN" else (MAX_POSITIONS_AS if strat_c == "AS" else MAX_POSITIONS)
            open_pos = db.load_open_positions(strategy=strat_c)
            if len(open_pos) >= max_pos:
                ordered[code] = True
                results[code] = f"見送り(上限{max_pos}件)"
                print(f"\n  ⚠️ {code} {c['name']}: 戦略{strat_c}上限({max_pos}件)到達 → 見送り")
                continue
            ordered[code] = True
            results[code] = "発注"
            print(f"\n  🟢 {code} {c['name']}: 前日比{latest_chg:+.1f}% ≥ 閾値{threshold:+.1f}%"
                  f" → 発注（上昇率優先）")
            _log_low_ratio_candidate(c)
            confirm_and_order(c, latest_px, url_request, condition=condition, entry_change_pct=latest_chg)

        time.sleep(POLL_INTERVAL_SEC)

    # ループ終了後: 未発注のWAIT_CONFIRM銘柄をまとめて見送り処理
    for c in candidates:
        code = c["code"]
        if ordered[code]:
            continue
        timing  = timings[code]
        if timing["style"] != "WAIT_CONFIRM":
            continue
        hist        = histories.get(code, [])
        latest_chg  = hist[-1]["change_pct"] if hist else 0
        eff_thr     = timing.get("confirm_threshold_pct", 0.0)
        cme_note    = f" ※CME{cme_chg:+.1f}%超過必要" if cme_chg > eff_thr else ""
        ordered[code] = True
        results[code]  = "見送り(上昇未確認)"
        print(f"\n  🔴 {code} {c['name']}: 9:{WATCH_END_MIN:02d}まで前日比+{eff_thr:.1f}%未達"
              f"（現在{latest_chg:+.1f}%）→ 見送り{cme_note}")

    # 観察データ保存 → 最終サマリー表示
    _save_observe_log(candidates, timings, histories)
    _print_final_summary(candidates, results, histories)



def _save_observe_log(candidates, timings, histories):
    """OBSERVEスタイルの銘柄の価格推移をCSVに保存する（後日検証用）"""
    observe_candidates = [
        c for c in candidates if timings[c["code"]]["style"] == "OBSERVE"
    ]
    if not observe_candidates:
        return

    rows = []
    for c in observe_candidates:
        code = c["code"]
        for h in histories.get(code, []):
            rows.append({
                "date":       TODAY,
                "code":       code,
                "name":       c["name"],
                "condition":  "WEAK",
                "judgment":   c["judgment"],
                "score":      c["score"],
                "time":       h["time"],
                "price":      h["price"],
                "change_pct": h["change_pct"],
                "volume":     h["volume"],
                "momentum":   h["momentum"],
            })

    if not rows:
        return

    df     = pd.DataFrame(rows)
    exists = os.path.exists(OBSERVE_LOG_CSV)
    df.to_csv(OBSERVE_LOG_CSV, mode="a", header=not exists,
              index=False, encoding="utf-8-sig")
    print(f"\n  💾 観察データ保存: {OBSERVE_LOG_CSV}（{len(observe_candidates)}銘柄 × {len(rows)//max(len(observe_candidates),1)}ポイント）")


def _print_final_summary(candidates, results, histories):
    """全銘柄の最終サマリーを表示"""
    print(f"\n{'='*60}")
    print(f"【最終サマリー】")
    print(f"{'='*60}")

    for c in candidates:
        code        = c["code"]
        judgment    = results.get(code, "監視中")
        hist        = histories.get(code, [])
        first_price = hist[0]["price"]  if hist else None
        last_price  = hist[-1]["price"] if hist else None
        last_chg    = hist[-1]["change_pct"] if hist else None

        icon = "🟢" if judgment == "発注" else ("🔵" if "観察" in judgment else "🔴")
        print(f"\n  {icon} {code} {c['name']}")
        print(f"     結果    : {judgment}")
        if first_price and last_price:
            print(f"     値動き  : {first_price:,.0f}円 → {last_price:,.0f}円  ({last_chg:+.2f}%)")
        if judgment == "発注" and last_price:
            _tp = c.get("tp_pct", TP_PCT)
            _sl = c.get("sl_pct", SL_PCT)
            print(f"     TP目標  : {round(last_price*(1+_tp)):,.0f}円（+{_tp*100:.0f}%）"
                  f"  SL: {round(last_price*(1-_sl)):,.0f}円（-{_sl*100:.0f}%）")

    print(f"{'='*60}\n")


# ══════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════
def main(start_now=False):
    print(f"=== 📡 リアルタイム監視（{TODAY}）===\n")

    candidates, condition = load_candidates()
    if not candidates:
        print("  本日のBUY/CAUTION候補がありません → 監視を終了します")
        return

    # 念のため重複コードを除去
    seen_codes: set = set()
    candidates = [c for c in candidates if not (c["code"] in seen_codes or seen_codes.add(c["code"]))]

    if condition == "PANIC":
        print("  🚨 地合いPANIC → 監視を終了します")
        return

    market_info = load_market_info()

    url_price   = load_tachibana_url()
    url_request = tachibana_order.load_url_request()
    if url_price:
        print(f"  📡 Tachibana API: リアルタイム価格取得モード（遅延なし）")
    else:
        print(f"  🌐 Yahoo Finance: 遅延価格取得モード（立花APIにログインすると精度向上）")
    mode_str = "本番発注モード" if tachibana_order.LIVE_TRADING else "モックモード（実発注なし）"
    print(f"  💼 発注: {mode_str}  {CHEAP_THRESHOLD:,}円未満→最大{MAX_ORDER_AMOUNT//10000}万円分 / {CHEAP_THRESHOLD:,}円以上→{DEFAULT_SHARES}株")

    print(f"  監視対象: {len(candidates)}銘柄  地合い: {condition}")
    for c in candidates:
        print(f"    {c['judgment']:<8} {c['code']} {c['name']}（スコア{c['score']:.1f}）")

    now = datetime.now(JST)
    if not start_now and now.hour < WATCH_START_HOUR:
        target = now.replace(hour=WATCH_START_HOUR, minute=WATCH_START_MIN,
                             second=0, microsecond=0)
        wait   = (target - now).seconds
        print(f"\n  ⏰ 9:00まで {wait//60}分{wait%60}秒 待機します...")

    watch_loop(candidates, condition, market_info, start_now=start_now,
               url_price=url_price, url_request=url_request)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", action="store_true", help="即時開始（テスト用）")
    args = parser.parse_args()
    main(start_now=args.now)
