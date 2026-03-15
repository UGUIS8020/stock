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

import os
import pandas as pd
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

SCAN_CSV           = "out/scan_results.csv"
WATCHLIST_CSV      = "out/watchlist.csv"
MORNING_LOG_CSV    = "out/morning_log.csv"
CANDIDATES_LOG_CSV = "out/candidates_log.csv"

TODAY = datetime.now().strftime("%Y-%m-%d")

# ══════════════════════════════════════════════════════
# 米国株・日経先物・ドル円 取得
# ══════════════════════════════════════════════════════
US_TICKERS = {
    "ダウ":      "^DJI",
    "ナスダック": "^IXIC",
    "S&P500":   "^GSPC",
}
NIKKEI_TICKER  = "^N225"
USD_JPY_TICKER = "USDJPY=X"

COPPER_TICKER   = "HG=F"
CRUDE_TICKER    = "CL=F"
SEMI_TICKER     = "SOXX"

MACRO_SECTORS = [
    {
        "label":       "銅先物上昇 → 非鉄金属セクター",
        "trigger_key": "copper",
        "threshold":   0.8,      # 銅先物 +0.8%以上
        "direction":   "up",
        "stocks": [
            ("5016", "JX金属",        "4,000円台"),
            ("5713", "住友金属鉱山",   "4,000円台"),
            ("5706", "三井金属",       "3,000円台"),
            ("5108", "三菱マテリアル", "2,000円台"),
            ("5714", "DOWAホールディングス", "4,000円台"),
        ],
    },
    {
        "label":       "ドル円上昇(円安) → 輸出・自動車セクター",
        "trigger_key": "usdjpy_change",
        "threshold":   0.3,      # ドル円 +0.3%以上（≒+0.5円程度）
        "direction":   "up",
        "stocks": [
            ("7203", "トヨタ自動車",   "3,000円台"),
            ("7267", "本田技研工業",   "1,500円台"),
            ("7270", "SUBARU",        "3,000円台"),
            ("7201", "日産自動車",     "400円台"),
            ("6954", "ファナック",     "4,000円台"),
        ],
    },
    {
        "label":       "原油上昇 → エネルギー・商社セクター",
        "trigger_key": "crude",
        "threshold":   1.0,      # 原油 +1.0%以上
        "direction":   "up",
        "stocks": [
            ("5020", "ENEOS",         "800円台"),
            ("5019", "出光興産",       "900円台"),
            ("8002", "丸紅",           "2,500円台"),
            ("8031", "三井物産",       "3,000円台"),
            ("8053", "住友商事",       "3,000円台"),
        ],
    },
    {
        "label":       "半導体関連(SOXX)上昇 → 半導体製造装置",
        "trigger_key": "semi",
        "threshold":   1.0,      # SOXX +1.0%以上
        "direction":   "up",
        "stocks": [
            ("8035", "東京エレクトロン", "25,000円台"),
            ("6857", "アドバンテスト",   "6,000円台"),
            ("6963", "ローム",           "2,000円台"),
            ("4063", "信越化学工業",     "5,000円台"),
            ("6146", "ディスコ",         "40,000円台"),
        ],
    },
    {
        "label":       "ドル円下落(円高) → 内需・小売セクター",
        "trigger_key": "usdjpy_change",
        "threshold":   -0.5,     # ドル円 -0.5%以下（円高）
        "direction":   "down",
        "stocks": [
            ("8267", "イオン",         "3,000円台"),
            ("3382", "セブン＆アイ",   "2,000円台"),
            ("9983", "ファーストリテイリング", "50,000円台"),
            ("2502", "アサヒグループ", "2,000円台"),
            ("2914", "日本たばこ産業", "4,000円台"),
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
    """日経平均の前日比(%)を取得（先物代用）"""
    try:
        data = yf.Ticker(NIKKEI_TICKER).history(period="3d")
        if len(data) >= 2:
            prev  = float(data["Close"].iloc[-2])
            close = float(data["Close"].iloc[-1])
            return {"close": close, "change": round((close - prev) / prev * 100, 2)}
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
            data = yf.Ticker(ticker).history(period="3d")
            if len(data) >= 2:
                prev  = float(data["Close"].iloc[-2])
                close = float(data["Close"].iloc[-1])
                chg   = round((close - prev) / prev * 100, 2)
                result[key]  = chg
                details[key] = {"close": close, "change": chg, "label": label}
            else:
                result[key]  = None
                details[key] = {"label": label, "error": True}
        except Exception:
            result[key]  = None
            details[key] = {"label": label, "error": True}
 
    # ドル円変化率：前日終値と比較
    try:
        data = yf.Ticker(USD_JPY_TICKER).history(period="3d")
        if len(data) >= 2 and usdjpy_current:
            prev_usdjpy = float(data["Close"].iloc[-2])
            chg = round((usdjpy_current - prev_usdjpy) / prev_usdjpy * 100, 2)
            result["usdjpy_change"]  = chg
            details["usdjpy_change"] = {"close": usdjpy_current, "change": chg, "label": "ドル円変化率"}
        else:
            result["usdjpy_change"]  = None
            details["usdjpy_change"] = {"label": "ドル円変化率", "error": True}
    except Exception:
        result["usdjpy_change"]  = None
        details["usdjpy_change"] = {"label": "ドル円変化率", "error": True}
 
    result["details"] = details
    return result


# ══════════════════════════════════════════════════════
# 地合い予測（多層スコアリング）
# ══════════════════════════════════════════════════════
def predict_market(us_data, nikkei, usdjpy):
    """
    4つの独立したシグナルをスコア化して地合いを判定する。

    満点構成（最大10点）
    ─────────────────────────────────────────
    レイヤー1: 米3指数平均        0〜3点  (weight: 大)
    レイヤー2: ドル円             0〜2点  (weight: 中)
    レイヤー3: 日経先物           0〜3点  (weight: 大)
    レイヤー4: 米指数のばらつき    0〜2点  (spread bonus)
    ─────────────────────────────────────────
    合計 8〜10点 → STRONG
         5〜7点  → NORMAL
         3〜4点  → WEAK
         0〜2点  → PANIC
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

    # ── Layer 3: 日経先物 ─────────────────────────── (0〜3点)
    if nikkei:
        nk_chg = nikkei["change"]
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
        breakdown.append(f"  日経先物     : {nk_chg:>+.2f}%  → {pts:>+d}点 ({label})")
    else:
        breakdown.append("  日経先物     : 取得失敗  → 0点")

    # ── Layer 4: 米指数のばらつきボーナス ────────── (0〜2点)
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

     # ── 総合判定 ──────────────────────────────────────
    if score >= 7:
        condition         = "STRONG"
        strategy_a_thr    = 7.5
        stop_loss_pct     = -5.0
    elif score >= 4:
        condition         = "NORMAL"
        strategy_a_thr    = 7.5
        stop_loss_pct     = -5.0
    elif score >= 3:          # ← 1 を 3 に変更
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
def judge_entry_a(row, condition, strategy_a_thr):
    score = float(row["score"])

    if condition == "PANIC":
        return "PASS", "地合いPANIC - 全見送り"

    if score < strategy_a_thr:
        return "PASS", f"地合い{condition} + スコア不十分(閾値{strategy_a_thr})"

    if score >= 9.0:
        return "CAUTION", "スコア過熱(9.0↑) - 逆行リスクあり"

    if condition == "WEAK":
        return "CAUTION", "地合い軟調 - スコア高いが慎重に"

    return "BUY", "スコア優良 + 地合い良好"


def judge_entry_b(row, condition, stop_loss_pct):
    drop      = float(row["today_rise"])
    rb_score  = int(row.get("rebound_score", 0))
    rb_reason = str(row.get("rebound_reason", "指標なし"))

    if condition == "PANIC":
        return "PASS", "地合いPANIC - 逆張り非推奨（続落リスク）"

    if condition == "WEAK":
        if rb_score >= 7:
            return "CAUTION", f"地合い軟調だがリバウンド高({rb_score}点) 少額のみ / {rb_reason}"
        return "PASS", f"地合い軟調 + リバウンドスコア{rb_score}点 - 見送り"

    if drop <= -20:
        if rb_score >= 7:
            return "CAUTION", f"暴落-20%超 材料確認必須({rb_score}点) / {rb_reason}"
        return "PASS", f"暴落-20%超 + リバウンドスコア{rb_score}点 - 見送り"

    if rb_score >= 7:
        return "BUY",     f"リバウンド期待大({rb_score}点) / {rb_reason}"
    elif rb_score >= 5:
        return "CAUTION", f"リバウンド中程度({rb_score}点) / {rb_reason}"
    else:
        return "PASS",    f"リバウンドスコア低({rb_score}点) - 見送り"
    

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [4] 戦略D 判定・表示関数（judge_entry_b() の後に追加）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
def scan_strategy_d(macro, condition):
    """
    マクロ指標を元に「今日+2%が狙える大型株」を提示する。
 
    - condition が PANIC の場合は全見送り
    - 各セクターのトリガーを確認し、閾値を超えたものだけ表示
    - 銘柄ごとの個別判定は行わず、「セクター追い風」を伝えるにとどめる
      （エントリー判断はユーザーが最終確認）
    """
    print(f"\n{'='*60}")
    print(f"【戦略D】大型株マクロ連動候補")
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
            icon  = "📈" if info["change"] >= 0 else "📉"
            label = info["label"]
            chg   = info["change"]
            close = info.get("close", "")
            close_str = f"  ({close:,.2f})" if close else ""
            print(f"    {icon} {label:<20}: {chg:>+.2f}%{close_str}")
        else:
            label = info.get("label", key) if isinstance(info, dict) else key
            print(f"    ❓ {label:<20}: 取得失敗")
 
    # セクタートリガー判定
    triggered = []
    for sector in MACRO_SECTORS:
        key       = sector["trigger_key"]
        threshold = sector["threshold"]
        direction = sector["direction"]
        val       = macro.get(key)
 
        if val is None:
            continue
 
        hit = (direction == "up"   and val >= threshold) or \
              (direction == "down" and val <= threshold)
 
        if hit:
            triggered.append((sector, val))
 
    if not triggered:
        print(f"\n  本日はマクロ連動トリガーなし（全指標が閾値未満）")
        print(f"  → 戦略A/B の候補を優先してください")
        return
 
    print(f"\n  {'─'*56}")
    for sector, val in triggered:
        sign  = "+" if val >= 0 else ""
        label = sector["label"]
        print(f"\n  🎯 {label}  ({sign}{val:.2f}%)")
        print(f"  {'コード':<6} {'銘柄名':<18} {'株価帯':>10}  目安")
        print(f"  {'─'*50}")
 
        for code, name, price_range in sector["stocks"]:
            caution = ""
            if condition == "WEAK":
                caution = " ⚠️少額"
            print(f"  {code:<6} {name:<18} {price_range:>10}  +2%狙い{caution}")
 
    print(f"\n  ⚠️  注意事項:")
    print(f"  □ 寄り付き後の実際の値動きで追従を確認してからエントリー")
    print(f"  □ 大型株は小型株より動きが鈍い場合あり（+1%で満足も可）")
    print(f"  □ 損切りは寄り付きから-2〜-3%（小型株より引き締め）")


def calc_stop_loss(buy_price, pct=-5.0):
    return round(buy_price * (1 + pct / 100))


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

    # ── 1. データ取得 ──
    print("  海外市場データ取得中...")
    us_data = fetch_us_market()
    nikkei  = fetch_nikkei()
    usdjpy  = fetch_usdjpy()

    print(f"\n{'='*60}")
    print(f"【海外市場】（前日終値ベース）")
    print(f"{'='*60}")
    for name, d in us_data.items():
        if d:
            icon = "📈" if d["change"] >= 0 else "📉"
            print(f"  {icon} {name:<12}: {d['close']:>10,.2f}  ({d['change']:>+.2f}%)")
        else:
            print(f"  ❓ {name:<12}: 取得失敗")

    if nikkei:
        icon = "📈" if nikkei["change"] >= 0 else "📉"
        print(f"  {icon} 日経225     : {nikkei['close']:>10,.2f}  ({nikkei['change']:>+.2f}%)")
    else:
        print(f"  ❓ 日経225     : 取得失敗")

    if usdjpy:
        print(f"  💴 ドル円      : {usdjpy:>10.2f} 円")

    # ── 2. 地合い予測（多層スコアリング）──
    condition, score, breakdown, market_details = predict_market(us_data, nikkei, usdjpy)
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
        print(f"  戦略A      : 🚀 積極エントリー可（閾値 ≥{strategy_a_thr}）")
        print(f"  戦略B      : ✅ 通常通り（損切り{stop_loss_pct:.0f}%）")
    elif condition == "NORMAL":
        print(f"  戦略A      : ✅ 通常通り（閾値 ≥{strategy_a_thr}）")
        print(f"  戦略B      : ✅ 通常通り（損切り{stop_loss_pct:.0f}%）")
    elif condition == "WEAK":
        print(f"  戦略A      : ⚠️  スコア{strategy_a_thr}以上のみ検討")
        print(f"  戦略B      : ⚠️  損切り{stop_loss_pct:.0f}%に引き締め")
    elif condition == "PANIC":
        print(f"  戦略A      : ❌ 全見送り推奨")
        print(f"  戦略B      : ❌ 全見送り推奨")
        print(f"  → ノーポジが最強戦略です")
    else:
        print(f"  予測       : ❓ データ取得失敗（慎重に判断してください）")

    print(f"{'='*60}")

    # ── 3. 戦略A候補の判定 ──
    print(f"\n{'='*60}")
    print(f"【戦略A】順張り候補 - 本日エントリー判定")
    print(f"{'='*60}")

    if not os.path.exists(SCAN_CSV):
        print("  ⚠️  out/scan_results.csv が見つかりません")
        print("  → 前日にscan_daily.pyを実行してください")
    else:
        df_a = pd.read_csv(SCAN_CSV, encoding="utf-8-sig")
        latest_date = df_a["scan_date"].max()
        candidates_a = df_a[df_a["scan_date"] == latest_date].copy()
        print(f"  スキャン日: {latest_date}  候補数: {len(candidates_a)}件\n")

        buy_a = caution_a = pass_a = 0
        print(f"  {'判定':<10} {'コード':<6} {'銘柄名':<18} {'スコア':>6} {'比率':>6}  理由")
        print("  " + "─" * 70)

        for _, row in candidates_a.iterrows():
            judgment, reason = judge_entry_a(row, condition, strategy_a_thr)
            candidate_rows.append({
                "date": TODAY, "strategy": "A",
                "code": row["code"], "name": row["name"],
                "score": row["score"], "ratio": row["ratio"],
                "judgment": judgment, "reason": reason,
            })
            icon = judge_icon(judgment)
            if judgment == "BUY":      buy_a    += 1
            elif judgment == "CAUTION": caution_a += 1
            else:                       pass_a    += 1
            print(f"  {icon:<10} {str(row['code']):<6} {str(row['name'])[:16]:<18} "
                  f"{float(row['score']):>6.2f} {float(row['ratio']):>5.1f}倍  {reason}")

        print(f"\n  【集計】 ✅買い:{buy_a}件  ⚠️要注意:{caution_a}件  ❌見送り:{pass_a}件")

    # ── 4. 戦略B候補の判定 ──
    print(f"\n{'='*60}")
    print(f"【戦略B】逆張り候補 - 本日エントリー判定")
    print(f"  寄付き買い → 引け売り  |  損切り: 寄付きから{stop_loss_pct:.0f}%")
    print(f"{'='*60}")

    if not os.path.exists(WATCHLIST_CSV):
        print("  ⚠️  out/watchlist.csv が見つかりません")
        print("  → 前日にscan_daily.pyを実行してください")
    else:
        df_b = pd.read_csv(WATCHLIST_CSV, encoding="utf-8-sig")
        latest_date_b = df_b["buy_date"].max()
        candidates_b  = df_b[
            (df_b["buy_date"] == latest_date_b) & df_b["next_rise"].isna()
        ].copy()

        if candidates_b.empty:
            print(f"  本日の逆張り候補なし")
            print(f"  （直近のwatchlist更新日: {latest_date_b} ※条件該当銘柄なし）")
        else:
            print(f"  スキャン日: {latest_date_b}  候補数: {len(candidates_b)}件\n")
            print(f"  {'判定':<10} {'コード':<6} {'銘柄名':<18} {'前日騰落':>8} {'終値':>7} {'損切り':>7} {'RBスコア':>8}  理由")
            print("  " + "─" * 85)

            for _, row in candidates_b.iterrows():
                judgment, reason = judge_entry_b(row, condition, stop_loss_pct)
                candidate_rows.append({
                    "date": TODAY, "strategy": "B",
                    "code": row["code"], "name": row["name"],
                    "score": row["score"], "ratio": row.get("today_rise"),
                    "judgment": judgment, "reason": reason,
                })
                icon     = judge_icon(judgment)
                stop     = calc_stop_loss(float(row["buy_price"]), stop_loss_pct)
                drop_str = f"{float(row['today_rise']):>+.1f}%"
                rb_score = int(row.get("rebound_score", 0))
                print(f"  {icon:<10} {str(row['code']):<6} {str(row['name'])[:16]:<18} "
                      f"{drop_str:>8} {float(row['buy_price']):>7.0f}円 "
                      f"{stop:>7}円 {rb_score:>6}点  {reason}")
                
    # ── 4.5 戦略D: 大型株マクロ連動スキャン ──
    macro = fetch_macro_indicators(usdjpy)
    scan_strategy_d(macro, condition)

    # ── 5. 候補銘柄ログ保存 ──
    if candidate_rows:
        new_candidates = pd.DataFrame(candidate_rows)
        if os.path.exists(CANDIDATES_LOG_CSV):
            ex_c = pd.read_csv(CANDIDATES_LOG_CSV, encoding="utf-8-sig")
            ex_c = ex_c[ex_c["date"] != TODAY]
            new_candidates = pd.concat([ex_c, new_candidates], ignore_index=True)
        new_candidates.to_csv(CANDIDATES_LOG_CSV, index=False, encoding="utf-8-sig")

    # ── 6. 本日のアクションプラン ──
    print(f"\n{'='*60}")
    print(f"【本日のアクションプラン】")
    print(f"{'='*60}")

    if condition == "PANIC":
        print(f"  🚨 全ポジション見送り。ニュースと先物を引き続き監視。")
    elif condition == "WEAK":
        print(f"  ⚠️  スコア{strategy_a_thr}以上の戦略A候補のみ、少額で検討。")
        print(f"  ⚠️  損切りを{stop_loss_pct:.0f}%に設定（通常より引き締め）。")
    elif condition == "STRONG":
        print(f"  🚀 地合い良好。戦略Aの優良銘柄を優先。")
        print(f"  ✅ 戦略Bも通常通りエントリー可。")
    else:
        print(f"  ✅ 通常通りスキャン結果に従ってエントリー。")
        print(f"  📌 9:00直後の値動きで方向感を確認してから判断もOK。")

    print(f"\n  ⏰ チェックリスト（8:50まで）")
    print(f"  □ 候補銘柄のチャートを確認（前日夜〜今朝のPTS動向）")
    print(f"  □ 候補銘柄の最新ニュース・IR確認")
    print(f"  □ 損切りライン（寄付き{stop_loss_pct:.0f}%）を注文画面で設定")
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
    new_log = pd.DataFrame([log_row])
    if os.path.exists(MORNING_LOG_CSV):
        ex = pd.read_csv(MORNING_LOG_CSV, encoding="utf-8-sig")
        ex = ex[ex["date"] != TODAY]
        new_log = pd.concat([ex, new_log], ignore_index=True)
    new_log.to_csv(MORNING_LOG_CSV, index=False, encoding="utf-8-sig")

    print(f"\n✅ 朝スキャン完了（ログ: {MORNING_LOG_CSV}）")
    print(f"   候補銘柄ログ: {CANDIDATES_LOG_CSV}（{len(candidate_rows)}件記録）")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


