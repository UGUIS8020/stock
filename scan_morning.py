"""
scan_morning.py - 朝スキャナー（毎朝8:30〜8:50頃に実行）

前日のscan_daily.pyが生成した候補リストに対して
・米国株指数（ダウ・ナスダック・S&P500）
・ドル円
・日経先物（^N225で代用）
をもとに多層スコアリングで「買い推奨 / 見送り / 要注意」を判定して出力する。

使い方:
    python scan_morning.py
"""

import sys
import os
import json
import time
import urllib3
import pandas as pd
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

import db as _db
_db.init_db()

_BASE_DIR = Path(__file__).parent
SCAN_CSV             = str(_BASE_DIR / "out" / "scan_results.csv")
WATCHLIST_CSV        = str(_BASE_DIR / "out" / "watchlist.csv")
MORNING_LOG_CSV      = str(_BASE_DIR / "out" / "morning_log.csv")
CANDIDATES_LOG_CSV   = str(_BASE_DIR / "out" / "candidates_log.csv")
TACHIBANA_LOGIN_FILE = str(_BASE_DIR / "tachibana_login_response.json")
TACHIBANA_API_BASE   = "https://kabuka.e-shiten.jp/e_api_v4r8/"
TACHIBANA_USER_ID    = os.getenv("TACHIBANA_LOGIN_ID")
TACHIBANA_PASSWORD   = os.getenv("TACHIBANA_LOGIN_PASS")

TODAY = datetime.now().strftime("%Y-%m-%d")

# ══════════════════════════════════════════════════════
# 米国株・日経先物・ドル円 取得
# ══════════════════════════════════════════════════════
US_TICKERS = {
    "ダウ":      "^DJI",
    "ナスダック": "^IXIC",
    "S&P500":   "^GSPC",
}
USD_JPY_TICKER = "USDJPY=X"

COPPER_TICKER   = "HG=F"
CRUDE_TICKER    = "CL=F"
SEMI_TICKER     = "SOXX"

MACRO_SECTORS = [
    {
        "label":       "銅先物上昇 → 非鉄金属セクター",
        "trigger_key": "copper",
        "threshold":   0.8,
        "direction":   "up",
        "stocks": [
            ("5016", "JX金属"),
            ("5713", "住友金属鉱山"),
            ("5706", "三井金属"),
            ("5108", "三菱マテリアル"),
            ("5714", "DOWAホールディングス"),
        ],
    },
    {
        "label":       "ドル円上昇(円安) → 輸出・自動車セクター",
        "trigger_key": "usdjpy_change",
        "threshold":   0.3,
        "direction":   "up",
        "stocks": [
            ("7203", "トヨタ自動車"),
            ("7267", "本田技研工業"),
            ("7270", "SUBARU"),
            ("7201", "日産自動車"),
            ("6954", "ファナック"),
        ],
    },
    {
        "label":       "原油上昇 → エネルギー・商社セクター",
        "trigger_key": "crude",
        "threshold":   1.0,
        "direction":   "up",
        "stocks": [
            ("5020", "ENEOS"),
            ("5019", "出光興産"),
            ("8002", "丸紅"),
            ("8031", "三井物産"),
            ("8053", "住友商事"),
        ],
    },
    {
        "label":       "半導体関連(SOXX)上昇 → 半導体製造装置",
        "trigger_key": "semi",
        "threshold":   1.0,
        "direction":   "up",
        "stocks": [
            ("8035", "東京エレクトロン"),
            ("6857", "アドバンテスト"),
            ("6963", "ローム"),
            ("4063", "信越化学工業"),
            ("6146", "ディスコ"),
        ],
    },
    {
        "label":       "ドル円下落(円高) → 内需・小売セクター",
        "trigger_key": "usdjpy_change",
        "threshold":   -0.5,
        "direction":   "down",
        "stocks": [
            ("8267", "イオン"),
            ("3382", "セブン＆アイ"),
            ("9983", "ファーストリテイリング"),
            ("2502", "アサヒグループ"),
            ("2914", "日本たばこ産業"),
        ],
    },
]


def fetch_us_market():
    """米国株指数の前日比(%)を取得"""
    results = {}
    for name, ticker in US_TICKERS.items():
        try:
            data = yf.Ticker(ticker).history(period="3d")
            if len(data) >= 2:
                prev  = float(data["Close"].iloc[-2])
                close = float(data["Close"].iloc[-1])
                results[name] = {
                    "close":  close,
                    "change": round((close - prev) / prev * 100, 2),
                }
            else:
                results[name] = None
        except Exception:
            results[name] = None
    return results


# --- 設定 ---
NIKKEI_TICKER     = "NKD=F"  # CME日経先物（シカゴ）
SGX_NIKKEI_TICKER = "NK=F"   # SGX日経先物（シンガポール・東京開場直前まで動く）
MARKET_LOG_CSV    = "out/market_log.csv"

def fetch_nikkei():
    """CME先物を使用して、寄り付き前の地合いをより正確に判定"""
    try:
        data = yf.Ticker(NIKKEI_TICKER).history(period="5d")
        if len(data) >= 2:
            prev = float(data["Close"].iloc[-2])
            current = float(data["Close"].iloc[-1])
            change_pct = round((current - prev) / prev * 100, 2)
            return {"close": current, "change": change_pct}
    except Exception:
        pass
    return None


def fetch_sgx():
    """SGX日経先物を取得。
    SGXはシンガポール取引所で東京開場直前まで動くため、
    CME先物より直近の市場動向を反映している。
    3/27の誤判定例: CME -2.08%（8:30取得）→ 実際は東京開場前に急反転していた。
    """
    try:
        data = yf.Ticker(SGX_NIKKEI_TICKER).history(period="5d")
        if len(data) >= 2:
            prev    = float(data["Close"].iloc[-2])
            current = float(data["Close"].iloc[-1])
            change_pct = round((current - prev) / prev * 100, 2)
            return {"close": current, "change": change_pct}
    except Exception:
        pass
    return None


def get_prev_day_condition():
    """DBから前日（最新）の実績地合いを取得する。
    WEAK翌日はリバウンドが起きやすいパターンに対応するため使用。
    （3/12→3/13、3/17→3/18、3/26→3/27 いずれもWEAK翌日がNORMALに回復）
    """
    return _db.get_prev_day_condition_db(TODAY)


def fetch_prev_day_breadth():
    """
    前日の日本市場騰落データをDBから計算する（Layer 6用）。

    シミュレーション検証結果（2025-04〜2026-03）:
      出来高加重平均変化(vw_change)が翌日パフォーマンスと最も相関が高かった
      → vw_change >= +1.0% かつ ad_ratio >= 0.60 の日がSTRONG判定補強に有効

    戻り値:
      {
        "ad_ratio":   float,   # 騰落比（値上がり/全銘柄）
        "vw_change":  float,   # 出来高加重平均騰落(%)
        "top20":      float,   # 上位20%平均騰落(%)
        "nk_est":     float,   # 全銘柄中央値騰落(%)
        "score":      int,     # 0〜2点
        "summary":    str,     # 表示用テキスト
      }
    """
    import numpy as np

    latest = _db.get_all_latest_prices(2)
    if latest.empty:
        return None

    changes, volumes = [], []
    for _, grp in latest.groupby("code"):
        grp = grp.sort_values("Date")
        if len(grp) < 2:
            continue
        try:
            prev_c = float(grp["Close"].iloc[-2])
            last_c = float(grp["Close"].iloc[-1])
            last_v = float(grp["Volume"].iloc[-1])
            if prev_c > 0 and last_v > 0:
                chg = (last_c - prev_c) / prev_c * 100
                changes.append(chg)
                volumes.append(last_v)
        except Exception:
            continue

    if len(changes) < 100:
        return None

    changes = np.array(changes)
    volumes = np.array(volumes)

    up       = (changes > 0).sum()
    total    = len(changes)
    ad_ratio = up / total
    nk_est   = float(np.median(changes))
    top20    = float(np.percentile(changes, 80))
    vw_total = volumes.sum()
    vw_change = float((changes * volumes).sum() / vw_total) if vw_total > 0 else nk_est

    # スコア採点（0〜2点）
    # シミュレーション: composite score の ad_ratio・vw_change・top20 成分に対応
    score = 0
    if ad_ratio >= 0.60 and vw_change >= 1.0:
        score = 2
    elif ad_ratio >= 0.55 or vw_change >= 0.3:
        score = 1

    breadth_label = (
        "強い（全面高）"   if score == 2 else
        "普通"             if score == 1 else
        "弱い（全面安寄り）"
    )

    return {
        "ad_ratio":  round(ad_ratio, 4),
        "vw_change": round(vw_change, 2),
        "top20":     round(top20, 2),
        "nk_est":    round(nk_est, 2),
        "score":     score,
        "summary":   (
            f"騰落比{ad_ratio*100:.1f}% / "
            f"出来高加重{vw_change:+.2f}% / "
            f"上位20%均{top20:+.2f}%  → {score}点（{breadth_label}）"
        ),
    }


def fetch_usdjpy():
    """ドル円レートを取得"""
    try:
        data = yf.Ticker(USD_JPY_TICKER).history(period="2d")
        if not data.empty:
            return round(float(data["Close"].iloc[-1]), 2)
    except Exception:
        pass
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [3] マクロ指標取得関数（fetch_us_market() の後に追加）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
def fetch_macro_indicators(usdjpy_current):
    """
    銅先物・原油先物・半導体ETF の前日比(%)を取得。
    ドル円の変化率は既存の fetch_usdjpy() 結果から計算。
 
    戻り値:
      {
        "copper":        前日比(%) or None,
        "crude":         前日比(%) or None,
        "semi":          前日比(%) or None,
        "usdjpy_change": 前日比(%) or None,
        "details":       表示用テキスト dict,
      }
    """
    result  = {}
    details = {}
 
    for key, ticker, label in [
        ("copper", COPPER_TICKER, "銅先物(HG=F)"),
        ("crude",  CRUDE_TICKER,  "原油先物(CL=F)"),
        ("semi",   SEMI_TICKER,   "半導体ETF(SOXX)"),
    ]:
        try:
            data = yf.Ticker(ticker).history(period="5d")
            if len(data) >= 2:
                prev  = float(data["Close"].iloc[-2])
                close = float(data["Close"].iloc[-1])
                chg   = round((close - prev) / prev * 100, 2)
                # 前日比も保持（遅行反応対応: 前日に大きく動いた指標も翌日考慮）
                prev2_chg = None
                if len(data) >= 3:
                    prev2 = float(data["Close"].iloc[-3])
                    prev2_chg = round((prev - prev2) / prev2 * 100, 2)
                result[key]  = chg
                result[f"{key}_prev"] = prev2_chg   # 前日変化率
                details[key] = {"close": close, "change": chg, "prev_change": prev2_chg, "label": label}
            else:
                result[key]  = None
                result[f"{key}_prev"] = None
                details[key] = {"label": label, "error": True}
        except Exception:
            result[key]  = None
            result[f"{key}_prev"] = None
            details[key] = {"label": label, "error": True}
 
    # ドル円変化率：前日終値と比較（遅行反応対応のため前日変化率も保持）
    try:
        data = yf.Ticker(USD_JPY_TICKER).history(period="5d")
        if len(data) >= 2 and usdjpy_current:
            prev_usdjpy = float(data["Close"].iloc[-2])
            chg = round((usdjpy_current - prev_usdjpy) / prev_usdjpy * 100, 2)
            prev2_chg = None
            if len(data) >= 3:
                prev2_usdjpy = float(data["Close"].iloc[-3])
                prev2_chg = round((prev_usdjpy - prev2_usdjpy) / prev2_usdjpy * 100, 2)
            result["usdjpy_change"]       = chg
            result["usdjpy_change_prev"]  = prev2_chg
            details["usdjpy_change"] = {"close": usdjpy_current, "change": chg, "prev_change": prev2_chg, "label": "ドル円変化率"}
        else:
            result["usdjpy_change"]       = None
            result["usdjpy_change_prev"]  = None
            details["usdjpy_change"] = {"label": "ドル円変化率", "error": True}
    except Exception:
        result["usdjpy_change"]       = None
        result["usdjpy_change_prev"]  = None
        details["usdjpy_change"] = {"label": "ドル円変化率", "error": True}
 
    result["details"] = details
    return result


# ══════════════════════════════════════════════════════
# 地合い予測（多層スコアリング）
# ══════════════════════════════════════════════════════
def predict_market(us_data, nikkei, usdjpy, sgx=None, prev_condition=None, prev_breadth=None):
    """
    6つの独立したシグナルをスコア化して地合いを判定する。

    満点構成（最大13点）
    ─────────────────────────────────────────
    レイヤー1: 米3指数平均        0〜3点  (weight: 大)
    レイヤー2: ドル円             0〜2点  (weight: 中)
    レイヤー3: 日経先物           0〜3点  (SGX優先・CMEフォールバック)
    レイヤー4: 米指数のばらつき    0〜2点  (spread bonus)
    レイヤー5: 前日WEAK補正        0〜1点  (リバウンドパターン対応)
    レイヤー6: 前日日本市場騰落    0〜2点  ★新規追加
    ─────────────────────────────────────────
    合計 8〜13点 → STRONG  ※閾値引き上げ（旧7→新8）
         5〜8点  → NORMAL
         3〜4点  → WEAK
         0〜2点  → PANIC

    [シミュレーション根拠]
    2025-04〜2026-03の1年間・4432銘柄で検証。
    前日の出来高加重平均変化(vw_change)が翌日パフォーマンスと最も相関が高く、
    複合スコア>=8の日は勝率54.5%・日平均+0.170%・Sharpe3.21を達成。
    旧STRONGは勝率51.4%・日平均-0.057%と非採算だった。
    """
    score     = 0
    details   = {}
    breakdown = []   # 表示用の内訳

    # ── Layer 1: 米3指数平均 ──────────────────────── (0〜3点)
    us_changes = [d["change"] for d in us_data.values() if d]
    if us_changes:
        us_avg = sum(us_changes) / len(us_changes)
        if us_avg >= 1.0:
            pts = 3; label = "強い上昇"
        elif us_avg >= 0.0:
            pts = 2; label = "小幅上昇"
        elif us_avg >= -1.0:
            pts = 1; label = "小幅下落"
        elif us_avg >= -2.0:
            pts = 0; label = "下落"
        else:
            pts = -1; label = "急落"
        score += pts
        details["us_avg"] = round(us_avg, 2)
        breakdown.append(f"  米指数平均   : {us_avg:>+.2f}%  → {pts:>+d}点 ({label})")
    else:
        us_avg = None
        breakdown.append("  米指数平均   : 取得失敗  → 0点")

    # ── Layer 2: ドル円 ───────────────────────────── (0〜2点)
    if usdjpy:
        if usdjpy >= 155:
            pts = 2; label = "強い円安（輸出株に追い風）"
        elif usdjpy >= 148:
            pts = 1; label = "緩やかな円安"
        elif usdjpy >= 140:
            pts = 0; label = "中立圏"
        else:
            pts = -1; label = "円高（輸出株に逆風）"
        score += pts
        details["usdjpy"] = usdjpy
        breakdown.append(f"  ドル円       : {usdjpy:.2f}円  → {pts:>+d}点 ({label})")
    else:
        breakdown.append("  ドル円       : 取得失敗  → 0点")

    # ── Layer 3: 日経先物（SGX優先・CMEフォールバック）──── (0〜3点)
    # SGX日経先物は東京開場直前まで動くため、CMEより直近の動きを反映する。
    # 3/27誤判定の教訓: CME -2.08%（8:30時点）が東京開場前に急反転 → SGXで検知可能
    nk_chg = 0.0  # default（取得失敗時はPANIC判定に使わない）
    nk_data   = sgx if sgx is not None else nikkei
    nk_source = "SGX先物" if sgx is not None else "CME先物"
    if nk_data:
        nk_chg = nk_data["change"]
        if nk_chg >= 1.0:
            pts = 3; label = "強い上昇"
        elif nk_chg >= 0.0:
            pts = 2; label = "小幅上昇"
        elif nk_chg >= -0.5:
            pts = 1; label = "小幅下落"
        elif nk_chg >= -1.5:
            pts = 0; label = "下落"
        else:
            pts = -1; label = "急落"
        score += pts
        details["nikkei_change"] = nk_chg
        breakdown.append(f"  {nk_source:<10}: {nk_chg:>+.2f}%  → {pts:>+d}点 ({label})")
        # CMEとSGXが両方取得できた場合、乖離があれば参考表示
        if sgx is not None and nikkei is not None:
            cme_chg = nikkei["change"]
            if abs(sgx["change"] - cme_chg) >= 0.5:
                breakdown.append(f"  ※CME先物    : {cme_chg:>+.2f}%（SGXと{sgx['change']-cme_chg:>+.2f}%乖離）")
    else:
        breakdown.append("  日経先物     : 取得失敗  → 0点")

    # ── Layer 4: 米指数のばらつきボーナス ────────── (0〜2点)
    up_count = 0  # default（STRONG判定に使用）
    if len(us_changes) == 3:
        up_count = sum(1 for c in us_changes if c > 0)
        if up_count == 3:
            pts = 2; label = "3指数一致上昇"
        elif up_count == 2:
            pts = 1; label = "2指数上昇"
        elif up_count == 1:
            pts = 0; label = "2指数下落"
        else:
            pts = -1; label = "3指数一致下落"
        score += pts
        breakdown.append(f"  指数一致度   : {up_count}/3 上昇  → {pts:>+d}点 ({label})")
    else:
        breakdown.append("  指数一致度   : データ不足  → 0点")

    # ── Layer 5: 前日WEAK/PANICリバウンドボーナス ─── (0〜1点)
    # [検証] WEAK翌日はリバウンドが起きやすいパターンが確認されている
    # 3/12 WEAK→3/13 NORMAL(61.3%)、3/17 WEAK→3/18 NORMAL(75%)、3/26 WEAK→3/27 NORMAL(65.9%)
    # 売られすぎの翌朝は反発需要が入りやすく、CME先物の下落が過大評価されやすい
    if prev_condition in ("WEAK", "PANIC"):
        pts = 1
        score += pts
        breakdown.append(f"  前日地合い補正: {prev_condition}→リバウンド期待  → {pts:>+d}点")
    else:
        breakdown.append(f"  前日地合い補正: {prev_condition or '取得失敗'}  → +0点")

    # ── Layer 6: 前日日本市場騰落（キャッシュから計算）─── (0〜2点)
    # [シミュレーション根拠] vw_change(出来高加重平均騰落)が翌日パフォーマンスと
    # 最も相関が高い指標（corr=+0.071）。ad_ratio>=0.60 & vw>=1.0% の組み合わせで
    # 複合スコア>=8 → 勝率54.5%・日平均+0.170%・Sharpe3.21（検証期間2025-04〜2026-03）
    if prev_breadth is not None:
        pts = prev_breadth["score"]
        score += pts
        breakdown.append(f"  前日市場騰落  : {prev_breadth['summary']}  → {pts:>+d}点")
    else:
        breakdown.append("  前日市場騰落  : キャッシュ未取得  → +0点")

    # ── 総合判定 ──────────────────────────────────────
    # STRONG: 閾値を7→8に引き上げ（シミュレーションで旧7は日平均-0.057%、新8は+0.170%）
    # 米3指数のうち2つ以上上昇していることも要求（過剰判定防止）
    if score >= 8 and up_count >= 2:
        condition         = "STRONG"
        strategy_a_thr    = 7.5
        stop_loss_pct     = -5.0
    # PANICも日経先物が実際に急落していることを確認（過剰判定防止）
    elif score <= 1 and nk_chg <= -1.0:
        condition         = "PANIC"
        strategy_a_thr    = 99.0
        stop_loss_pct     = -3.0
    elif score >= 4:
        condition         = "NORMAL"
        strategy_a_thr    = 7.5
        stop_loss_pct     = -5.0
    elif score >= 2:
        condition         = "WEAK"
        strategy_a_thr    = 8.0
        stop_loss_pct     = -4.0
    else:
        condition         = "PANIC"
        strategy_a_thr    = 99.0
        stop_loss_pct     = -3.0

    details["score"]          = score
    details["strategy_a_thr"] = strategy_a_thr
    details["stop_loss_pct"]  = stop_loss_pct

    return condition, score, breakdown, details


# ══════════════════════════════════════════════════════
# 候補銘柄の買い判定
# ══════════════════════════════════════════════════════
# 急騰フラグ判定
# 【根拠】backtest_log.csv分析: score>=9.5 & ratio>=10倍の銘柄は
#         高値+10%以上到達率10.6% / TP+8%設定で平均+1.02%（通常銘柄TP+3%の3倍）
SURGE_SCORE_MIN = 9.5
SURGE_RATIO_MIN = 10.0

def is_surge_flag(row):
    """急騰フラグ判定: スコア9.5以上 & 出来高比率10倍以上"""
    try:
        return float(row["score"]) >= SURGE_SCORE_MIN and float(row["ratio"]) >= SURGE_RATIO_MIN
    except Exception:
        return False


# ══════════════════════════════════════════════════════
# 【戦略A 買い判定】地合い別に完全分離したロジック
# バックテスト根拠: backtest_log.csv SL=-1%/TP=+3%適用後
#   STRONG  BUY avg-0.43% Sharpe-0.84 / CAUTION avg+2.05% Sharpe+1.25 → 全CAUTION
#   NORMAL  BUY avg+0.81% / CAUTION avg+0.81% → 有効ゾーンBUY、それ以外CAUTION
#   WEAK    BUY avg+0.73% / CAUTION avg+0.95% → score>=thr でCАUTION
# ══════════════════════════════════════════════════════
def judge_entry_a(row, condition, strategy_a_thr):
    score      = float(row["score"])
    ratio      = float(row.get("ratio", 0))
    today_rise = float(row.get("today_rise", 0))

    # ── PANIC日 ──────────────────────────────────────────
    if condition == "PANIC":
        return "PASS", "地合いPANIC - 全見送り"

    # ── STRONG日 ─────────────────────────────────────────
    # データ: STRONG×BUY avg-0.43%(Sharpe-0.84) / CAUTION avg+2.05%(Sharpe+1.25)
    # → scoreに関わらず全銘柄CAUTIONにしてdecide_timing()で0.3%上昇確認後エントリー
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
    # キーファクター: 前日-2〜0%が一貫してプラス / ratio10倍超・前日急騰は見送り
    # 推奨出口: TP+3%/SL-1%（最適化済み）
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
        # 最良ゾーン: score5〜7 × ratio3〜10x × 前日-2〜0% → avg+0.285% / WR54%
        if 5.0 <= score < 7.0 and 3.0 <= ratio < 10.0 and -2.0 <= today_rise <= 0.0:
            return "BUY", (f"NORMAL日 最良ゾーン score{score:.1f}×ratio{ratio:.1f}倍×前日{today_rise:+.1f}%"
                           f" - avg+0.285%・WR54%")
        # 有効ゾーン: score7〜9 × ratio3〜6x × 前日-2〜0% → avg+0.170% / WR54%
        if 7.0 <= score < 9.0 and 3.0 <= ratio < 6.0 and -2.0 <= today_rise <= 0.0:
            return "CAUTION", (f"NORMAL日 有効ゾーン score{score:.1f}×ratio{ratio:.1f}倍×前日{today_rise:+.1f}%"
                               f" - avg+0.170%・WR54%")
        # 前日-2〜0% × その他 → avg+0.132% / WR50%
        if -2.0 <= today_rise <= 0.0:
            return "CAUTION", (f"NORMAL日 + 前日微下落({today_rise:+.1f}%) score{score:.1f}×ratio{ratio:.1f}倍"
                               f" - avg+0.132%・少額様子見")
        return "PASS", f"NORMAL日 + 前日{today_rise:+.1f}%（有効条件外）"

    # ── WEAK日 ───────────────────────────────────────────
    # WEAK×CAUTION avg+0.95% / BUY avg+0.73% → score>=thr でCAUTION
    if score < strategy_a_thr:
        return "PASS", f"WEAK日 + スコア不十分({score:.1f} < 閾値{strategy_a_thr})"
    return "CAUTION", f"WEAK日 score{score:.1f} - スコア条件クリア・慎重にエントリー"


def judge_entry_b(row, condition):
    drop      = float(row["today_rise"])
    rb_score  = int(row.get("rebound_score", 0))
    rb_reason = str(row.get("rebound_reason", "指標なし"))

    # PANIC日は全見送り
    if condition == "PANIC":
        return "PASS", "地合いPANIC - 逆張り非推奨（続落リスク）"

    # 暴落しすぎ銘柄は地合い問わず除外（続落リスク高）
    if drop <= -20:
        return "PASS", f"暴落{drop:.1f}% + 続落リスク高 - 見送り"

    # [simulate_b_full.py検証結果 2025-04〜2026-03]
    # ─────────────────────────────────────────────────────
    # 閾値 -4〜-5% × NORMAL日（引け決済）:
    #   全RB:   勝率50.5% / avg +0.219% / 累計+255% ← プラス ★
    #   RB3〜4: 勝率48.4% / avg +0.073% ← プラス ★
    #   RB5〜6: 勝率51.9% / avg +0.319% ← 最良 ★
    #   RB7〜9: 勝率53.3% / avg +0.425% ← 最優秀（旧「落ちナイフ」は誤り）
    # 閾値 -5%以下 × NORMAL日（引け決済）:
    #   全RB:   勝率47.2% / avg +0.020% ← ほぼゼロ
    # WEAK日（閾値問わず）: 勝率30% / avg -1.26% ← 最悪・変わらず
    # 出口戦略: 引け決済 ≒ TP+5%/SL-5% > TP+3%/SL-3%
    # ─────────────────────────────────────────────────────

    # WEAK日は全見送り
    if condition == "WEAK":
        return "PASS", f"地合いWEAK + 逆張り非推奨 - WEAK日avg-1.26%（740件）"

    # NORMAL日 × RBスコア3以上 → BUY
    # 【根拠】evolve_b.py GA(20000人×100世代, 2026-05-26): TP+3.2%/SL-6.2%が最優秀
    #         N=371件 / OOS WR=53.1% / avg=+0.130% / Sharpe=+1.054
    if condition == "NORMAL" and rb_score >= 3:
        return "BUY", f"地合いNORMAL + リバウンド狙い({rb_score}点) - OOS WR53%・avg+0.13%（TP+3.2%/SL-6.2%推奨） / {rb_reason}"

    # STRONG日 × RB4以上 → BUY
    # 【根拠】evolve_b.py GA(20000人×100世代, 2026-05-26): NORMAL+STRONG条件でOOS通過率76.5%
    if condition == "STRONG" and rb_score >= 4:
        return "BUY", f"地合いSTRONG + リバウンド狙い({rb_score}点) - OOS WR53%・avg+0.13%（TP+3.2%/SL-6.2%推奨） / {rb_reason}"

    # STRONG日 × RB3 → CAUTION（スコアやや低め）
    if condition == "STRONG" and rb_score == 3:
        return "CAUTION", f"地合いSTRONG + リバウンド候補({rb_score}点) - RB3はやや低め・様子見 / {rb_reason}"

    return "PASS", f"リバウンドスコア低({rb_score}点) - 見送り"
    

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [4] 戦略D 判定・表示関数（judge_entry_b() の後に追加）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
def scan_strategy_d(macro, condition, tachibana_prices=None):
    """
    マクロ指標連動の大型株をスコアリング付きで提示する。
    各銘柄に対してキャッシュまたはyfinanceから履歴を取得しスコア計算する。
    """
    print(f"\n{'='*60}")
    print(f"【戦略D】大型株マクロ連動候補（スコアリング付き）")
    print(f"  参考: 銅/原油/ドル円/半導体の動きから恩恵銘柄を提示")
    print(f"{'='*60}")

    if condition == "PANIC":
        print(f"  🚨 地合いPANIC - 戦略D も全見送り推奨")
        return

    details = macro.get("details", {})

    # マクロ指標サマリー表示
    print(f"\n  📊 マクロ指標（前日比）:")
    for key, info in details.items():
        if isinstance(info, dict) and "change" in info:
            icon      = "📈" if info["change"] >= 0 else "📉"
            label     = info["label"]
            chg       = info["change"]
            close     = info.get("close", "")
            close_str = f"  ({close:,.2f})" if close else ""
            prev_chg  = info.get("prev_change")
            prev_str  = f"  前日:{prev_chg:+.2f}%" if prev_chg is not None else ""
            print(f"    {icon} {label:<20}: {chg:>+.2f}%{close_str}{prev_str}")
        else:
            label = info.get("label", key) if isinstance(info, dict) else key
            print(f"    ❓ {label:<20}: 取得失敗")

    # セクタートリガー判定（当日 OR 前日の遅行反応も考慮）
    triggered = []
    for sector in MACRO_SECTORS:
        key       = sector["trigger_key"]
        threshold = sector["threshold"]
        direction = sector["direction"]
        val       = macro.get(key)          # 当日変化率
        val_prev  = macro.get(f"{key}_prev")  # 前日変化率

        if val is None:
            continue

        # 当日トリガー
        hit_today = (direction == "up"   and val >= threshold) or \
                    (direction == "down" and val <= threshold)
        # 前日トリガー（遅行反応: 前日に閾値の1.5倍以上動いた場合も対象）
        hit_prev = False
        lag_note = ""
        if val_prev is not None:
            lag_threshold = threshold * 1.5
            hit_prev = (direction == "up"   and val_prev >= lag_threshold) or \
                       (direction == "down" and val_prev <= -lag_threshold)
            if hit_prev and not hit_today:
                lag_note = f"（前日{val_prev:+.2f}%の遅行反応）"

        if hit_today or hit_prev:
            triggered.append((sector, val, val_prev, lag_note))

    if not triggered:
        print(f"\n  本日はマクロ連動トリガーなし（全指標が閾値未満）")
        print(f"  → 戦略A/B の候補を優先してください")
        return

    print(f"\n  {'─'*64}")
    for sector, val, val_prev, lag_note in triggered:
        sign  = "+" if val >= 0 else ""
        label = sector["label"]
        lag_str = f"  ⏰ {lag_note}" if lag_note else ""
        print(f"\n  🎯 {label}  ({sign}{val:.2f}%){lag_str}")
        print(f"  {'コード':<6} {'銘柄名':<18} {'現在値':>8} {'スコア':>6} {'比率':>6}  判定")
        print(f"  {'─'*62}")

        scored = []
        for code, name in sector["stocks"]:
            rt = _tprice(tachibana_prices, code)
            hist, price, source = fetch_stock_hist(code, realtime_price=rt)
            s = calc_score_d(hist) if hist is not None else None
            scored.append((code, name, price, s, source))

        scored.sort(key=lambda x: x[3]["score"] if x[3] else -1, reverse=True)

        for code, name, price, s, source in scored:
            price_str = f"{price:>8,.0f}円" if price else f"{'取得失敗':>8}"
            if s:
                score = s["score"]
                ratio = s["ratio"]
                if condition == "WEAK":
                    judge = "⚠️ 少額検討" if score >= 7.5 else "❌ 見送り"
                else:
                    if score >= 8.0:
                        judge = "🏆 最優先"
                    elif score >= 7.5:
                        judge = "✅ 買い検討"
                    elif score >= 6.0:
                        judge = "⚠️ 様子見"
                    else:
                        judge = "❌ 見送り"
                src_mark = {"tachibana": "📡", "cache": "📁", "yfinance": "🌐"}.get(source, "🌐")
                print(f"  {code:<6} {name:<18} {price_str} {score:>6.2f} {ratio:>+5.1f}%  {judge} {src_mark}")
            else:
                print(f"  {code:<6} {name:<18} {price_str} {'--':>6} {'--':>6}  ❓ データ不足")

    print(f"\n  ⚠️  注意事項:")
    print(f"  □ 📡=Tachibana実況  📁=キャッシュ使用  🌐=yfinance取得（前日データ）")
    print(f"  □ 寄り付き後の値動きで追従確認してからエントリー")
    print(f"  □ 損切りは寄り付きから-2〜-3%（小型株より引き締め）")
    if condition == "WEAK":
        print(f"  □ 地合いWEAK中 - 少額のみ")


def calc_stop_loss(buy_price, pct=-5.0):
    return round(buy_price * (1 + pct / 100))


def calc_score_d(hist):
    """
    大型株向けスコアリング（戦略D専用）
    出来高急増ではなく価格トレンドで評価する。

    採点基準（最大10点）:
      直近5日騰落トレンド : 0〜4点（連続上昇傾向）
      MA5 vs MA25乖離    : 0〜3点（短期MAが長期MAを上回っているか）
      直近1日騰落        : 0〜3点（前日の動きがプラスか）
    """
    import numpy as np
    if len(hist) < 26:
        return None

    closes = hist["Close"].values
    score  = 0

    # ── 直近5日の価格トレンド（0〜4点）
    last5     = closes[-5:]
    slope     = np.polyfit(range(5), last5, 1)[0]
    slope_pct = slope / last5[0] * 100

    if slope_pct >= 1.0:
        score += 4
    elif slope_pct >= 0.3:
        score += 3
    elif slope_pct >= 0.0:
        score += 2
    elif slope_pct >= -0.3:
        score += 1

    # ── MA5 vs MA25 乖離（0〜3点）
    ma5     = closes[-5:].mean()
    ma25    = closes[-25:].mean()
    ma_diff = (ma5 / ma25 - 1) * 100

    if ma_diff >= 2.0:
        score += 3
    elif ma_diff >= 0.5:
        score += 2
    elif ma_diff >= 0.0:
        score += 1

    # ── 直近1日騰落（0〜3点）
    prev    = float(closes[-2])
    today   = float(closes[-1])
    day_chg = (today - prev) / prev * 100

    if day_chg >= 2.0:
        score += 3
    elif day_chg >= 0.5:
        score += 2
    elif day_chg >= 0.0:
        score += 1

    return {
        "score": round(score, 2),
        "ratio": round(ma_diff, 2),
    }


def load_tachibana_url():
    """tachibana_login_response.json から株価URLを読み込む。失敗時はNone。"""
    try:
        with open(TACHIBANA_LOGIN_FILE, encoding="utf-8") as f:
            data = json.load(f)
        url = data.get("sUrlPrice", "")
        return url if url else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _url_encode(s):
    """パスワード等に含まれる記号をURLエンコードする"""
    table = {
        ' ': '%20', '!': '%21', '"': '%22', '#': '%23', '$': '%24',
        '%': '%25', '&': '%26', "'": '%27', '(': '%28', ')': '%29',
        '*': '%2A', '+': '%2B', ',': '%2C', '/': '%2F', ':': '%3A',
        ';': '%3B', '<': '%3C', '=': '%3D', '>': '%3E', '?': '%3F',
        '@': '%40', '[': '%5B', ']': '%5D', '^': '%5E', '`': '%60',
        '{': '%7B', '|': '%7C', '}': '%7D', '~': '%7E',
    }
    return ''.join(table[c] if c in table else c for c in s)


def _check_tachibana_session(url_price):
    """株価URLに試験リクエストを送り、セッションが有効かどうか確認する。"""
    if not url_price:
        return False
    try:
        t = datetime.now()
        p_sd = (f"{t.year}.{t.month:02}.{t.day:02}"
                f"-{t.hour:02}:{t.minute:02}:{t.second:02}"
                f".{t.microsecond // 1000:03}")
        params = (
            '{'
            '"p_no":"0",'
            f'"p_sd_date":"{p_sd}",'
            '"sCLMID":"CLMMfdsGetMarketPrice",'
            '"sTargetIssueCode":"7203",'
            '"sTargetColumn":"pDPP,pPRP",'
            '"sJsonOfmt":"5"'
            '}'
        )
        http = urllib3.PoolManager()
        resp = http.request("GET", url_price + "?" + params,
                            timeout=urllib3.Timeout(connect=3, read=5))
        text = resp.data.decode("shift-jis", errors="ignore")
        result = json.loads(text)
        p_errno = result.get("p_errno", "")
        # p_errno が "0" 以外（"2", "-2", 空文字, 未知コード）は無効
        if p_errno != "0":
            return False
        # 期限切れ時はリストが空になる（p_errno="0" でも無効）
        items = result.get("aCLMMfdsMarketPrice", [])
        return len(items) > 0
    except Exception:
        return False


def _tachibana_login():
    """CLMAuthLoginRequest を送信し、仮想URLをJSONに保存して url_price を返す。"""
    t = datetime.now()
    p_sd = (f"{t.year}.{t.month:02}.{t.day:02}"
            f"-{t.hour:02}:{t.minute:02}:{t.second:02}"
            f".{t.microsecond // 1000:03}")
    params = (
        '{'
        '"p_no":"1",'
        f'"p_sd_date":"{p_sd}",'
        '"sCLMID":"CLMAuthLoginRequest",'
        f'"sUserId":"{TACHIBANA_USER_ID}",'
        f'"sPassword":"{_url_encode(TACHIBANA_PASSWORD)}",'
        '"sJsonOfmt":"5"'
        '}'
    )
    url = TACHIBANA_API_BASE + "auth/?" + params
    try:
        http = urllib3.PoolManager()
        resp = http.request("GET", url, timeout=urllib3.Timeout(connect=10, read=15))
        text = resp.data.decode("shift-jis", errors="ignore")
        result = json.loads(text)
    except Exception as e:
        print(f"  ❌ 通信エラー: {e}")
        return None

    result_code = result.get("sResultCode", "")
    if result_code != "0":
        err = result.get("sResultText", "不明")
        print(f"  ❌ ログイン失敗（コード: {result_code}、内容: {err}）")
        if result_code == "10088":
            print("  → 電話認証が完了していない可能性があります。再度お電話ください。")
        return None

    login_data = {
        "sResultCode":        result_code,
        "sUrlRequest":        result.get("sUrlRequest", ""),
        "sUrlPrice":          result.get("sUrlPrice", ""),
        "sUrlMaster":         result.get("sUrlMaster", ""),
        "sUrlEvent":          result.get("sUrlEvent", ""),
        "sUrlEventWebSocket": result.get("sUrlEventWebSocket", ""),
        "sZyoutoekiKazeiC":   result.get("sZyoutoekiKazeiC", ""),
        "sLastLoginDate":     result.get("sLastLoginDate", ""),
    }
    with open(TACHIBANA_LOGIN_FILE, "w", encoding="utf-8") as f:
        json.dump(login_data, f, ensure_ascii=False, indent=2)

    url_price = login_data["sUrlPrice"]
    print(f"  ✅ 接続成功（{TACHIBANA_LOGIN_FILE} 保存済）")
    return url_price if url_price else None


def ensure_tachibana_session():
    """Tachibanaセッションを確保する。期限切れなら電話認証フローを案内する。"""
    if not TACHIBANA_USER_ID or not TACHIBANA_PASSWORD:
        print("  ⚠️  .env に TACHIBANA_LOGIN_ID / TACHIBANA_LOGIN_PASS が未設定です")
        print("     → Tachibana API なしで続行します")
        return None

    import time as _time
    today_str = datetime.now().strftime("%Y%m%d")

    # ログインJSONから今日の日付で発行されたセッションか確認
    session_is_today = False
    try:
        with open(TACHIBANA_LOGIN_FILE, encoding="utf-8") as f:
            _d = json.load(f)
        last_login = _d.get("sLastLoginDate", "")  # "20260526090701"
        session_is_today = last_login.startswith(today_str)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass

    # ファイルが30分以内 かつ 今日のセッション → スキップ（再起動時の多重認証防止）
    if session_is_today:
        try:
            mtime = os.path.getmtime(TACHIBANA_LOGIN_FILE)
            if _time.time() - mtime < 1800:
                existing_url = load_tachibana_url()
                if existing_url:
                    print("  ✅ Tachibana API セッション有効（30分以内のログイン）")
                    return existing_url
        except FileNotFoundError:
            pass
        # 今日ログイン済みだが30分超 → そのまま使える（注文URLも有効）
        existing_url = load_tachibana_url()
        if existing_url:
            print(f"  ✅ Tachibana API セッション有効（本日 {last_login[8:10]}:{last_login[10:12]} ログイン済み）")
            return existing_url

    # 昨日以前のセッション or ファイルなし → 電話認証フロー
    existing_url = load_tachibana_url()
    if existing_url:
        print("  ⚠️  Tachibana API セッションが前日のものです（毎朝の電話認証が必要）")
    else:
        print("  ℹ️  Tachibana API 未接続（初回認証が必要）")
        print("  ℹ️  ※ 毎朝 python tachibana_login.py を先に実行することを推奨します")

    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  【立花証券e支店API 電話認証】                    │")
    print("  │                                                   │")
    print("  │  登録済み電話番号から今すぐ発信してください:       │")
    print("  │    📞 0120-28-6592（無料）                        │")
    print("  │    📞 050-3102-6575（通話料有料）                 │")
    print("  │                                                   │")
    print("  │  ※ 非通知・184発信は認証されません               │")
    print("  │  ※ 音声が流れ、自動的に切れたらOKです            │")
    print("  └─────────────────────────────────────────────────┘")
    print()
    input("  電話が切れたら Enter キーを押してください...")
    print()
    print("  ログインリクエスト送信中...")

    url_price = _tachibana_login()
    if not url_price:
        print("  ⚠️  認証に失敗しました。Tachibana API なしで続行します。")
    return url_price


def fetch_tachibana_prices(url_price, codes):
    """Tachibana APIから複数銘柄の気配値・現在値を一括取得。
    戻り値: {code: {"price", "ask", "bid", "ask_vol", "bid_vol", "prev_close"}}
    APIの上限120件/回をバッチ分割して全件取得する。"""
    if not url_price or not codes:
        return {}
    batch_size = 120
    http = urllib3.PoolManager()
    quotes = {}
    p_no_base = int(time.time()) % 100000
    for i, start in enumerate(range(0, len(codes), batch_size)):
        batch = codes[start:start + batch_size]
        code_list = ",".join(str(c) for c in batch)
        t = datetime.now()
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
            resp = http.request("GET", url_price + "?" + params,
                                timeout=urllib3.Timeout(connect=3, read=5))
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


def _tprice(quotes, code):
    """tachibana_prices辞書から現在値floatを取得するヘルパー。"""
    q = (quotes or {}).get(str(code))
    return q["price"] if q else None


def _tquote(quotes, code):
    """tachibana_prices辞書から気配dict全体を取得するヘルパー。"""
    return (quotes or {}).get(str(code))


def _quote_summary(q):
    """気配dictから表示用の (売気配str, 買気配str, 需給比str) を返す。"""
    if not q:
        return "    ---", "    ---", " ---"
    ask = q.get("ask")
    bid = q.get("bid")
    av  = q.get("ask_vol") or 0
    bv  = q.get("bid_vol") or 0
    ask_str = f"{ask:>7,.0f}" if ask else "    ---"
    bid_str = f"{bid:>7,.0f}" if bid else "    ---"
    total = av + bv
    ratio_str = f"{bv/total*100:>4.0f}%" if total > 0 else " ---"
    return ask_str, bid_str, ratio_str


def fetch_stock_hist(code, realtime_price=None):
    """
    銘柄の株価履歴を取得する。
    優先順: ① Tachibana API（realtime_price引数） → ② SQLite DB → ③ yfinance
    """
    hist = _db.get_stock_history(code)
    if len(hist) >= 22:
        if realtime_price is not None:
            return hist, realtime_price, "tachibana"
        return hist, float(hist["Close"].iloc[-1]), "cache"

    try:
        ticker = yf.Ticker(f"{code}.T")
        hist   = ticker.history(period="3mo")
        if len(hist) >= 22:
            hist = hist.reset_index()
            hist["Date"] = pd.to_datetime(hist["Date"])
            if realtime_price is not None:
                return hist, realtime_price, "tachibana"
            return hist, float(hist["Close"].iloc[-1]), "yfinance"
    except Exception:
        pass

    if realtime_price is not None:
        return None, realtime_price, "tachibana"
    return None, None, None


# ══════════════════════════════════════════════════════
# 表示ヘルパー
# ══════════════════════════════════════════════════════
def judge_icon(judgment):
    return {"BUY": "✅ 買い", "CAUTION": "⚠️ 要注意", "PASS": "❌ 見送り"}.get(judgment, "  ")

def condition_icon(cond):
    return {"STRONG": "🚀", "NORMAL": "✅", "WEAK": "⚠️ ", "PANIC": "🚨", "UNKNOWN": "❓"}.get(cond, "")


# ══════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════
def main():
    print(f"=== 🌅 朝スキャナー（{TODAY} 寄り付き前）===\n")

    candidate_rows = []

    # ── 0. DBの鮮度確認 → 古ければS3から取得、今日更新済みならスキップ ──
    import time as _time
    _db_path = Path(__file__).parent / "out" / "stock.db"
    _db_age_hours = (_time.time() - _db_path.stat().st_mtime) / 3600 if _db_path.exists() else 999
    if _db_age_hours > 20:  # 20時間以上古い（=昨日以前）→ S3から取得
        print(f"  DBが{_db_age_hours:.0f}時間前のものです。S3から最新版を取得します...")
        try:
            import download_db
            download_db.run(db_only=True)
            _db.init_db()
        except SystemExit:
            print("  ⚠️  S3ダウンロード失敗 - ローカルDBで続行します")
        except Exception as e:
            print(f"  ⚠️  S3ダウンロードエラー: {e} - ローカルDBで続行します")
    else:
        print(f"  ✅ ローカルDB使用（{_db_age_hours:.1f}時間前に更新済み）")

    # ── Tachibana API セッション確保（期限切れなら電話認証フローを案内）──
    print("  Tachibana API セッション確認中...")
    tachibana_url = ensure_tachibana_session()

    # ── 1. データ取得 ──
    print("  海外市場データ取得中...")
    us_data        = fetch_us_market()
    nikkei         = fetch_nikkei()
    sgx            = fetch_sgx()
    usdjpy         = fetch_usdjpy()
    prev_condition = get_prev_day_condition()
    print("  前日市場騰落データ計算中...")
    prev_breadth   = fetch_prev_day_breadth()

    # ── Tachibana から候補銘柄の現在値を一括取得 ──
    tachibana_prices = {}
    if tachibana_url:
        all_codes = set()
        try:
            df_tmp = _db.get_scan_results()
            if not df_tmp.empty:
                latest = df_tmp["scan_date"].max()
                for c in df_tmp[df_tmp["scan_date"] == latest]["code"].astype(str):
                    all_codes.add(c)
        except Exception:
            pass
        try:
            df_tmp2 = _db.get_watchlist()
            if not df_tmp2.empty:
                for c in df_tmp2["code"].astype(str):
                    all_codes.add(c)
        except Exception:
            pass
        for sector in MACRO_SECTORS:
            for code, _ in sector.get("stocks", []):
                all_codes.add(str(code))
        if all_codes:
            print(f"  Tachibana API: {len(all_codes)}銘柄の現在値取得中...")
            tachibana_prices = fetch_tachibana_prices(tachibana_url, list(all_codes))
            ok = len([v for v in tachibana_prices.values() if v and v.get("price")])
            print(f"  → {ok}/{len(all_codes)}銘柄 取得成功")
    else:
        print("  ⚠️  Tachibana API未接続（tachibana_login_response.json なし）- キャッシュ使用")

    print(f"\n{'='*60}")
    print(f"【海外市場】（前日終値ベース）")
    print(f"{'='*60}")
    for name, d in us_data.items():
        if d:
            icon = "📈" if d["change"] >= 0 else "📉"
            print(f"  {icon} {name:<12}: {d['close']:>10,.2f}  ({d['change']:>+.2f}%)")
        else:
            print(f"  ❓ {name:<12}: 取得失敗")

    # SGXとCMEの両方を表示（乖離がある場合に視覚的に確認できる）
    if sgx:
        print(f"  🌐 日経先物(SGX): {sgx['close']:>10,.0f}  ({sgx['change']:>+.2f}%)  ← 直近値")
    else:
        print(f"  ❓ 日経先物(SGX): 取得失敗")
    if nikkei:
        print(f"  🌐 日経先物(CME): {nikkei['close']:>10,.0f}  ({nikkei['change']:>+.2f}%)")
    else:
        print(f"  ❓ 日経先物(CME): 取得失敗")

    if usdjpy:
        print(f"  💴 ドル円      : {usdjpy:>10.2f} 円")

    if prev_condition:
        print(f"  📅 前日地合い  : {prev_condition}")
    else:
        print(f"  📅 前日地合い  : 取得失敗（market_log.csvなし）")

    # ── 2. 地合い予測（多層スコアリング）──
    condition, score, breakdown, market_details = predict_market(
        us_data, nikkei, usdjpy, sgx=sgx, prev_condition=prev_condition,
        prev_breadth=prev_breadth
    )
    strategy_a_thr = market_details["strategy_a_thr"]
    stop_loss_pct  = market_details["stop_loss_pct"]
    cond_icon      = condition_icon(condition)

    print(f"\n{'='*60}")
    print(f"【今日の地合い予測】（多層スコアリング）")
    print(f"{'='*60}")
    print(f"\n  📊 スコア内訳:")
    for line in breakdown:
        print(line)

    max_score = 10
    bar_filled = max(0, min(score, max_score))
    bar = "█" * bar_filled + "░" * (max_score - bar_filled)
    print(f"\n  総合スコア   : {score:>+d} / {max_score}点  [{bar}]")
    print(f"  判定         : {cond_icon} {condition}")
    print()

    if condition == "STRONG":
        print(f"  戦略A      : 積極エントリー可")
        print(f"  推奨条件   : score5〜8 / ratio<5倍 / 前日比<+2% / 上位3件")
        print(f"  TP/SL      : +3%利確 / -1%損切り")
        print(f"  ※スイープ検証: Sharpe4.21 / 年率+10.8% / DD-1.6%（アウトオブサンプル確認済み）")
    elif condition == "NORMAL":
        print(f"  戦略A      : ✅ 有望条件のみ（前日微下落 × スコア3〜6 × ratio<3倍）")
        print(f"  closing_watch: ✅ NORMAL日も戦略B対象（14:30〜自動起動）")
    elif condition == "WEAK":
        print(f"  戦略A      : ⚠️  スコア{strategy_a_thr}以上のみ検討")
        print(f"  closing_watch: ⚠️  WEAK日は見送り")
    elif condition == "PANIC":
        print(f"  戦略A      : ❌ 全見送り推奨")
        print(f"  closing_watch: ❌ 全見送り推奨")
        print(f"  → ノーポジが最強戦略です")
    else:
        print(f"  予測       : ❓ データ取得失敗（慎重に判断してください）")

    print(f"{'='*60}")

    # ── 3. 戦略A候補の判定 ──
    print(f"\n{'='*60}")
    print(f"【戦略A】順張り候補 - 本日エントリー判定")
    print(f"{'='*60}")

    df_a = _db.get_scan_results()
    if df_a.empty:
        print("  ⚠️  戦略Aデータが見つかりません")
        print("  → 前日にscan_daily.pyを実行してください")
    else:
        latest_date = df_a["scan_date"].max()
        candidates_a = df_a[df_a["scan_date"] == latest_date].copy()
        candidates_a = candidates_a.sort_values(["score", "ratio"], ascending=[False, False])
        print(f"  スキャン日: {latest_date}  候補数: {len(candidates_a)}件\n")

        buy_a = caution_a = pass_a = 0
        print(f"  {'判定':<10} {'コード':<6} {'銘柄名':<16} {'現在値':>8} {'売気配':>7} {'買気配':>7} {'需給':>5} {'スコア':>6} {'比率':>5}  理由")
        print("  " + "─" * 95)

        surge_candidates = []
        for _, row in candidates_a.iterrows():
            judgment, reason = judge_entry_a(row, condition, strategy_a_thr)
            surge = is_surge_flag(row)
            candidate_rows.append({
                "date": TODAY, "strategy": "A",
                "condition": condition,
                "code": row["code"], "name": row["name"],
                "score": row["score"], "ratio": row["ratio"],
                "judgment": judgment, "reason": reason,
            })
            icon = judge_icon(judgment)
            surge_mark = " ⚡" if surge else ""
            if judgment == "BUY":      buy_a    += 1
            elif judgment == "CAUTION": caution_a += 1
            else:                       pass_a    += 1
            q = _tquote(tachibana_prices, row["code"])
            rt = q["price"] if q else None
            price_str = f"{rt:>8,.0f}円" if rt else f"{'---':>8}  "
            ask_s, bid_s, ratio_s = _quote_summary(q)
            print(f"  {icon:<10} {str(row['code']):<6} {str(row['name'])[:14]:<16} "
                  f"{price_str} {ask_s} {bid_s} {ratio_s} "
                  f"{float(row['score']):>6.2f} {float(row['ratio']):>5.1f}倍{surge_mark}  {reason}")
            if surge:
                surge_candidates.append(row)

        print(f"\n  【集計】 ✅買い:{buy_a}件  ⚠️要注意:{caution_a}件  ❌見送り:{pass_a}件")

        # 急騰フラグ銘柄の専用表示
        if surge_candidates:
            print(f"\n  {'='*56}")
            print(f"  ⚡【急騰候補】score>={SURGE_SCORE_MIN} & ratio>={SURGE_RATIO_MIN:.0f}倍以上  {len(surge_candidates)}件")
            print(f"  {'='*56}")
            print(f"  過去実績: 高値+10%以上到達率10.6% / TP+8%設定で平均+1.02%")
            print(f"  推奨: TP+8% / SL-1%（引け決済は不可 → 高値で反落するケース多）")
            print(f"  {'─'*56}")
            for row in surge_candidates:
                jdg, _ = judge_entry_a(row, condition, strategy_a_thr)
                icon = judge_icon(jdg)
                note = "→ 地合い不問で注目" if jdg != "BUY" else "→ BUY推奨"
                print(f"  {icon} [{row['code']}]{str(row['name'])[:16]}  "
                      f"score:{float(row['score']):.1f}  ratio:{float(row['ratio']):.1f}倍  {note}")
                print(f"    推奨注文: 寄付き成行 / TP+8% / SL-1%")

    # ── 4. 前日closing_watch 仕込み済み確認 ──
    # 戦略B（逆張り）はclosing_watchに統合済み。
    # 前日14:58に引け前で仕込んだポジションの現在値を参考表示する。
    print(f"\n{'='*60}")
    print(f"【前日closing_watch 仕込み済みポジション確認】")
    print(f"  ※ 逆張り買いは前日引け前（closing_watch）で実施済み")
    print(f"  ※ 本日の決済目安: TP+3.2% / SL-6.2%（GA最適化結果 2026-05-26）")
    print(f"{'='*60}")

    df_b = _db.get_watchlist()
    if df_b.empty:
        print("  前日のclosing_watchデータが見つかりません")
    else:
        latest_date_b = df_b["buy_date"].max()
        candidates_b  = df_b[
            (df_b["buy_date"] == latest_date_b) & df_b["next_rise"].isna()
        ].copy()

        if candidates_b.empty:
            print(f"  前日（{latest_date_b}）の仕込み銘柄なし（STRONG地合い以外 or 条件該当なし）")
        else:
            print(f"  仕込み日: {latest_date_b}  銘柄数: {len(candidates_b)}件\n")
            print(f"  {'コード':<6} {'銘柄名':<16} {'現在値':>8} {'売気配':>7} {'買気配':>7} {'需給':>5} {'仕込み騰落':>10} {'TP目安':>8} {'SL目安':>8} {'RB':>4}")
            print("  " + "─" * 100)

            for _, row in candidates_b.iterrows():
                buy_p    = float(row["buy_price"])
                tp_price = round(buy_p * 1.03)
                sl_price = round(buy_p * 0.94)
                drop_str = f"{float(row['today_rise']):>+.1f}%"
                rb_score = int(row.get("rebound_score", 0))
                q = _tquote(tachibana_prices, row["code"])
                rt = q["price"] if q else None
                price_str = f"{rt:>8,.0f}円" if rt else f"{'---':>8}  "
                ask_s, bid_s, ratio_s = _quote_summary(q)
                print(f"  {str(row['code']):<6} {str(row['name'])[:14]:<16} "
                      f"{price_str} {ask_s} {bid_s} {ratio_s} {drop_str:>10} "
                      f"{tp_price:>8,}円 {sl_price:>8,}円 {rb_score:>4}点")
                
    # ── 4.5 戦略D: 大型株マクロ連動スキャン ──
    macro = fetch_macro_indicators(usdjpy)
    scan_strategy_d(macro, condition, tachibana_prices=tachibana_prices)

    # ── 5. 候補銘柄ログ保存 ──
    if candidate_rows:
        _db.save_candidates_log_db(candidate_rows)

     # ── 6. 本日のアクションプラン ──
    print(f"\n{'='*60}")
    print(f"【本日のアクションプラン】")
    print(f"{'='*60}")

    if condition == "PANIC":
        print(f"  🚨 全ポジション見送り。ノーポジが最強戦略です。")
        print(f"\n  ⏰ 本日のチェックリスト")
        print(f"  □ 新規エントリーは一切しない")
        print(f"  □ 保有中のポジションがあれば損切りを検討")
        print(f"  □ 米国先物・ニュースの動向を引き続き監視")
        print(f"  □ 地合い回復のシグナル（騰落比40%超）を待つ")
    elif condition == "WEAK":
        print(f"  ⚠️  スコア{strategy_a_thr}以上の戦略A候補のみ、少額で検討。")
        print(f"  ⚠️  損切りを{stop_loss_pct:.0f}%に設定（通常より引き締め）。")
        print(f"  ❌ closing_watch本日は見送り（WEAK地合いは逆張り非推奨）")
        print(f"\n  ⏰ チェックリスト（8:50まで）")
        print(f"  □ 候補銘柄のチャートを確認（前日夜〜今朝のPTS動向）")
        print(f"  □ 候補銘柄の最新ニュース・IR確認")
        print(f"  □ 損切りライン: 戦略A=寄付きから-1%")
        print(f"  □ 前日closing_watch仕込み済みポジションは早めの損切り検討")
        print(f"  □ 1銘柄あたりの投資額を確認（リスク管理）")
    elif condition == "STRONG":
        print(f"  🚀 地合い良好。戦略Aの優良銘柄を優先。")
        print(f"  ✅ closing_watch 14:30〜 自動起動（scan_morningから連続実行）")
        print(f"\n  ⏰ チェックリスト（8:50まで）")
        print(f"  □ 候補銘柄のチャートを確認（前日夜〜今朝のPTS動向）")
        print(f"  □ 候補銘柄の最新ニュース・IR確認")
        print(f"  □ 損切りライン: 戦略A=寄付きから-1%")
        print(f"  □ 1銘柄あたりの投資額を確認（リスク管理）")
    else:  # NORMAL
        print(f"  ✅ 通常通りスキャン結果に従ってエントリー。")
        print(f"  📌 9:00直後の値動きで方向感を確認してから判断もOK。")
        print(f"  ℹ️  closing_watchは地合いが改善しSTRONGになれば本日起動可")
        print(f"\n  ⏰ チェックリスト（8:50まで）")
        print(f"  □ 候補銘柄のチャートを確認（前日夜〜今朝のPTS動向）")
        print(f"  □ 候補銘柄の最新ニュース・IR確認")
        print(f"  □ 損切りライン: 戦略A=寄付きから-1%")
        print(f"  □ 1銘柄あたりの投資額を確認（リスク管理）")

    # ── 7. 朝判定ログ保存 ──
    log_row = {
        "date":               TODAY,
        "condition_forecast": condition,
        "condition_score":    score,
        "us_avg_change":      market_details.get("us_avg"),
        "dow_change":         us_data.get("ダウ", {}).get("change")     if us_data.get("ダウ")      else None,
        "nasdaq_change":      us_data.get("ナスダック", {}).get("change") if us_data.get("ナスダック") else None,
        "sp500_change":       us_data.get("S&P500", {}).get("change")   if us_data.get("S&P500")   else None,
        "nikkei_change":      nikkei["change"] if nikkei else None,
        "usdjpy":             usdjpy,
        "strategy_a_thr":     strategy_a_thr,
        "stop_loss_pct":      stop_loss_pct,
    }
    _db.save_morning_log_db(log_row)

    print(f"\n✅ 朝スキャン完了（DB記録）")
    print(f"   候補銘柄ログ: {len(candidate_rows)}件記録")
    print(f"{'='*60}\n")

    import subprocess, sys as _sys

    # ── 戦略A：朝に BUY/CAUTION 候補がある場合のみ寄り付き監視・発注 ──
    has_buy = any(r["judgment"] in ("BUY", "CAUTION") for r in candidate_rows)
    if has_buy and condition != "PANIC":
        try:
            import market_watch
            market_watch.main()
        except Exception as e:
            print(f"⚠️  リアルタイム監視でエラーが発生しました: {e}")
            print("   → python market_watch.py を手動で実行してください")

    # ── PANIC以外は常に position_monitor・closing_watch を起動 ──
    # 朝の地合いに関わらず午後に地合いが変化する可能性があるため常時起動する
    # closing_watch が14:30に実際のAD比率を計測してSTRONG以外なら自動終了する
    if condition != "PANIC":
        try:
            subprocess.Popen([_sys.executable, str(_BASE_DIR / "position_monitor.py")])
            print("  📊 position_monitor をバックグラウンドで起動しました（15:28まで監視）")
        except Exception as e:
            print(f"⚠️  position_monitor の起動に失敗しました: {e}")
            print("   → python position_monitor.py を手動で実行してください")

        try:
            subprocess.Popen([_sys.executable, str(_BASE_DIR / "daytime.py")])
            print("  📈 daytime をバックグラウンドで起動しました（9:00〜14:00 自動売買）")
        except Exception as e:
            print(f"⚠️  daytime の起動に失敗しました: {e}")

        try:
            import closing_watch
            closing_watch.main()
        except Exception as e:
            print(f"⚠️  引け前スキャンでエラーが発生しました: {e}")
            print("   → python closing_watch.py を手動で実行してください")


if __name__ == "__main__":
    main()


