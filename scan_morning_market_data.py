# -*- coding: utf-8 -*-
"""
scan_morning_market_data.py - 外部マーケットデータ取得モジュール

scan_morning.py から分割。今後ほぼ変更不要な安定ロジック。
取得対象: 米国3指数・日経先物(CME/SGX)・ドル円・銅/原油/半導体ETF・前日騰落
"""

import numpy as np
import yfinance as yf

import db as _db

from datetime import datetime
from pathlib import Path

TODAY = datetime.now().strftime("%Y-%m-%d")

# ── 米国株指数 ──────────────────────────────────────
US_TICKERS = {
    "ダウ":      "^DJI",
    "ナスダック": "^IXIC",
    "S&P500":   "^GSPC",
}

# ── ドル円 ─────────────────────────────────────────
USD_JPY_TICKER = "USDJPY=X"

# ── マクロ指標 ─────────────────────────────────────
COPPER_TICKER = "HG=F"
CRUDE_TICKER  = "CL=F"
SEMI_TICKER   = "SOXX"

# ── 日経先物 ───────────────────────────────────────
NIKKEI_TICKER     = "NKD=F"   # CME日経先物（シカゴ）
SGX_NIKKEI_TICKER = "NK=F"    # SGX日経先物（シンガポール・東京開場直前まで動く）

# ── マクロ連動セクター定義 ─────────────────────────
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


def fetch_nikkei():
    """CME先物を使用して寄り付き前の地合いをより正確に判定"""
    try:
        data = yf.Ticker(NIKKEI_TICKER).history(period="5d")
        if len(data) >= 2:
            prev    = float(data["Close"].iloc[-2])
            current = float(data["Close"].iloc[-1])
            return {"close": current, "change": round((current - prev) / prev * 100, 2)}
    except Exception:
        pass
    return None


def fetch_sgx():
    """SGX日経先物を取得。東京開場直前まで動くためCMEより直近の動向を反映する。
    3/27の誤判定例: CME -2.08%（8:30取得）→ 実際は東京開場前に急反転していた。
    """
    try:
        data = yf.Ticker(SGX_NIKKEI_TICKER).history(period="5d")
        if len(data) >= 2:
            prev    = float(data["Close"].iloc[-2])
            current = float(data["Close"].iloc[-1])
            return {"close": current, "change": round((current - prev) / prev * 100, 2)}
    except Exception:
        pass
    return None


def fetch_usdjpy():
    """ドル円レートを取得"""
    try:
        data = yf.Ticker(USD_JPY_TICKER).history(period="2d")
        if not data.empty:
            return round(float(data["Close"].iloc[-1]), 2)
    except Exception:
        pass
    return None


def fetch_macro_indicators(usdjpy_current):
    """銅先物・原油先物・半導体ETFの前日比(%)と前日変化率を取得。

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
                prev2_chg = None
                if len(data) >= 3:
                    prev2 = float(data["Close"].iloc[-3])
                    prev2_chg = round((prev - prev2) / prev2 * 100, 2)
                result[key]            = chg
                result[f"{key}_prev"]  = prev2_chg
                details[key] = {"close": close, "change": chg, "prev_change": prev2_chg, "label": label}
            else:
                result[key] = result[f"{key}_prev"] = None
                details[key] = {"label": label, "error": True}
        except Exception:
            result[key] = result[f"{key}_prev"] = None
            details[key] = {"label": label, "error": True}

    # ドル円変化率
    try:
        data = yf.Ticker(USD_JPY_TICKER).history(period="5d")
        if len(data) >= 2 and usdjpy_current:
            prev_usdjpy = float(data["Close"].iloc[-2])
            chg = round((usdjpy_current - prev_usdjpy) / prev_usdjpy * 100, 2)
            prev2_chg = None
            if len(data) >= 3:
                prev2_usdjpy = float(data["Close"].iloc[-3])
                prev2_chg = round((prev_usdjpy - prev2_usdjpy) / prev2_usdjpy * 100, 2)
            result["usdjpy_change"]      = chg
            result["usdjpy_change_prev"] = prev2_chg
            details["usdjpy_change"] = {"close": usdjpy_current, "change": chg, "prev_change": prev2_chg, "label": "ドル円変化率"}
        else:
            result["usdjpy_change"] = result["usdjpy_change_prev"] = None
            details["usdjpy_change"] = {"label": "ドル円変化率", "error": True}
    except Exception:
        result["usdjpy_change"] = result["usdjpy_change_prev"] = None
        details["usdjpy_change"] = {"label": "ドル円変化率", "error": True}

    result["details"] = details
    return result


def get_prev_day_condition():
    """DBから前日の実績地合いを取得する。
    WEAK翌日のリバウンドパターン補正（Layer 5）に使用。
    """
    return _db.get_prev_day_condition_db(TODAY)


def fetch_prev_day_breadth():
    """前日の日本市場騰落データをDBから計算する（地合い判定 Layer 6 用）。

    シミュレーション検証結果（2025-04〜2026-03）:
      出来高加重平均変化(vw_change)が翌日パフォーマンスと最も相関が高かった
      → vw_change >= +1.0% かつ ad_ratio >= 0.60 の日がSTRONG判定補強に有効

    戻り値:
      {"ad_ratio", "vw_change", "top20", "nk_est", "score", "summary"}
    """
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
                changes.append((last_c - prev_c) / prev_c * 100)
                volumes.append(last_v)
        except Exception:
            continue

    if len(changes) < 100:
        return None

    changes  = np.array(changes)
    volumes  = np.array(volumes)
    up       = (changes > 0).sum()
    total    = len(changes)
    ad_ratio = up / total
    nk_est   = float(np.median(changes))
    top20    = float(np.percentile(changes, 80))
    vw_total = volumes.sum()
    vw_change = float((changes * volumes).sum() / vw_total) if vw_total > 0 else nk_est

    # スコア採点（0〜2点）
    if ad_ratio >= 0.60 and vw_change >= 1.0:
        score = 2
    elif ad_ratio >= 0.55 or vw_change >= 0.3:
        score = 1
    else:
        score = 0

    breadth_label = (
        "強い（全面高）"    if score == 2 else
        "普通"              if score == 1 else
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
