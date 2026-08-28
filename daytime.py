# -*- coding: utf-8 -*-
"""
daytime.py - 日中シグナル検知（9:00〜14:30）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【統計分析（analyze_a.py, 2026-05-26）に基づく改訂版】

  旧: ブレイクアウト + 出来高急増 + モメンタム のうち2つ以上
      → 全期間でマイナス、GAで安定戦略ゼロ件

  新: 以下の5条件をすべて満たすときのみ発報（check_signal）
    1. STRONG地合いのみ発注（REQUIRE_STRONG=True: 2026-07-24〜 NORMALは監視のみ）
    2. 寄り付きギャップ -5〜+2%（急落・急騰は除外）
    3. 前日終値プラス（前日も上昇していた銘柄）
    4. MA過熱圏でない（5日MA+3%超 or 25日MA+6%超 は除外）
    5. モメンタム +0.3% 以上（前日終値比で上昇中）

  月曜フィルタ（analyze_a Section 8 / Sharpe=2.1 と構造的に弱い）:
    - MAX_POSITIONS=1 / gap≥0% / score<3 に制限

  発注優先順位（同ポーリング内で複数シグナル時）:
    adj_score = (base_score + 1) × 価格帯重み × 業種重み × 上ひげ重み
    → 金融・不動産 × 2000〜5000円 × 上ひげ大 が最優先

  根拠（analyze_a.py, OOS検証済み Sharpe=6.684）:
    - STRONG地合い: 全体WR=65.5%（NORMAL=48.7%）
    - gap -2〜0% × STRONG: WR=74.5%〜81.2% ★
    - MA過熱（5日+3%超）: WR=61.6% に急落 → ハードフィルタ

使い方:
    python daytime.py        # 9:00まで待機して自動開始
    python daytime.py --now  # 即時開始（テスト用）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import json
import time
import argparse
import urllib3
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()
import db
import tachibana_order
import closing_watch

JST      = timezone(timedelta(hours=9))
TODAY    = datetime.now(JST).strftime("%Y-%m-%d")
_BASE_DIR = Path(__file__).parent

# ── 時間設定 ──────────────────────────────────────
WATCH_START_HOUR = 9
WATCH_START_MIN  = 30  # market_watch.py（9:00-9:30）終了後から開始
WATCH_END_HOUR   = 14
WATCH_END_MIN    = 30
POLL_INTERVAL    = 30   # 秒（2026-08-10: 15→30。立花証券よりAPI高負荷の指摘を受け、
                         # gap絞り込みと合わせてTachibana API呼び出し回数を削減するため）

# ── TP / SL ───────────────────────────────────────
TP_PCT = 0.03   # +3%（STRONG地合い × gap小 の平均高値 +2.39% に基づく）
SL_PCT = 0.05   # -5%（analyze_a TP/SL最適化結果: Sharpe=6.558, WR=68.5%）

# ── 候補銘柄ソース ────────────────────────────────
# True: scan_morning_strategy_d_candidates.py の戦略D専用候補（candidates_log, strategy='D'）を使う
# False: 従来通りscan_results（戦略A用TOP20〜30リスト）を使う
# 2026-08-04: scan_resultsは母集団ミスマッチと判明したため新設。テスト運用のため
# 2026-08-05からTrueに切り替え、建玉サイズを半分にして様子を見る（下記参照）。
USE_D_CANDIDATE_SCAN  = True

# ── 発注設定 ──────────────────────────────────────
DAYTIME_TRADING       = True      # False にすると監視のみ（発注なし）
# 2026-08-05: 新候補ソース(USE_D_CANDIDATE_SCAN)のテスト運用のため、
# 検証が終わるまで戦略Dのみ建玉サイズを半分にしていた（20万円→10万円 / 200株→100株）。
# 2026-08-06: 初稼働日(8/5)の実績が良好だったため戦略Bと同水準（30万円）に引き上げ。
# 株数は300ではなく200のまま（資金所要額を抑制）。CHEAP_THRESHOLDも1,000→1,500円へ
# 引き上げ、30万円÷200株の自然な境界と一致させ閾値の断層を解消。
MAX_ORDER_AMOUNT      = 300_000   # 安い株の1発注上限額（戦略Bと同水準）
DEFAULT_SHARES        = 200       # 高値株のデフォルト株数（戦略Bと同水準）
CHEAP_THRESHOLD       = 1_500     # この価格未満は金額ベースで株数計算（戦略Bと同水準）
HIGH_PRICE_CAP_AMOUNT = 1_000_000  # 2026-08-28新設: 高値株の建玉青天井問題を受けて再導入。
                    # ニッポン高度紙工業(6,735円→135万円)・芝浦メカトロニクス(4,130円→82.6万円)
                    # のような高値株で建玉が過大化していた。200株×価格が100万円を超える場合のみ、
                    # 金額ベースで株数を縮小する（200株換算で約5,000円が実質的な切替点）。
MAX_DAYTIME_POSITIONS = 5         # 日中最大ポジション数（戦略A とは別カウント。2026-08-05: 保有過多につき10→5）
MIN_VOLUME            = 50_000    # 前日出来高フィルター（5万株未満は除外）
LIMIT_ORDER           = False     # 成行発注（指値は受付エラー多発・モメンタム戦略には不向きのため 2026-06-25 変更）
LIMIT_WAIT_SECS       = 30        # 指値約定確認の待機秒数

# ── シグナル条件（analyze_a.py 統計分析結果に基づく）──
# 2026-07-24: evolve_d.py GA検証で上位戦略が軒並みSTRONG限定だった一方、
#   実際のライブ発注はほぼ全てNORMAL地合いに偏っており（18件中16件）、
#   NORMAL側はforce_close負けが大半（当日高値でもTP+3%に一度も届かず）だったため、
#   STRONG限定に変更（NORMALは監視のみで発注しない）
REQUIRE_STRONG  = True   # True=STRONG地合いのみ発注（NORMAL/WEAKはcan_orderで除外・監視は継続）
ALLOWED_ORDER_CONDITIONS = ("STRONG",) if REQUIRE_STRONG else ("STRONG", "NORMAL")
GAP_MAX_PCT     = 2.0    # 寄り付きギャップ上限（+2%超のギャップアップは除外）
GAP_MIN_PCT     = -5.0   # 寄り付きギャップ下限（-5%超の急落も除外）
PREV_RISE_MIN   = -0.5   # 前日騰落率の下限（OOS検証: -0.5〜0%帯も avg+0.139% wr55.3% と同品質・2026-07-12）
MOMENTUM_PCT    = 0.3    # 前日比がこれ以上で「上昇中」と判定（発報トリガー）

# ── MA乖離率フィルタ（analyze_a.py Section 11: 過熱圏を除外）──
# 5日MA乖離率が +3% 超: WR=61.6% / Sharpe=3.7（他帯比 -11%）
# 25日MA乖離率が +6% 超: WR=64.0% / Sharpe=4.1（他帯比 -8%）
MA5_DEV_MAX  = 3.0   # 5日MA乖離率の上限（%）。超えたら発注見送り
MA25_DEV_MAX = 6.0   # 25日MA乖離率の上限（%）。超えたら発注見送り

# ── 月曜フィルタ（analyze_a.py Section 8: 月曜は構造的に弱い）──
# 月曜の Sharpe=2.1、火〜金=8〜9.5。見送りではなく制限付き発注
# 最善条件: gap 0〜+2% / score 0〜3 → Sharpe=2.447
MON_MAX_POSITIONS = 1      # 月曜の最大ポジション数（通常3 → 1に制限）
MON_GAP_MIN_PCT   = 0.0    # 月曜のギャップ下限（マイナスギャップは除外）
MON_SCORE_MAX     = 3      # 月曜はスコア低め銘柄のみ（0〜3が最良）

# ── 属性重み（analyze_a.py Section 9 結果に基づく）──────
# 複数シグナルが同時に出た場合、adj_score 順に発注優先度を決める
# adj_score = (base_score + 1) × price_weight × sector_weight × shadow_weight
#
# 価格帯重み（Sharpe比: ④2000~5000円=7.2 が最良、①~500円=4.2 が最弱）
_PRICE_WEIGHTS = {
    "①~500円":      0.80,
    "②500~1000円":  0.95,
    "③1000~2000円": 1.10,
    "④2000~5000円": 1.10,
    "⑤5000円~":     0.95,
}
# 業種重み（Sharpe比: 金融・不動産=9.3 が最良、食品・繊維=4.3 が最弱）
_SECTOR_WEIGHTS = {
    "金融・不動産": 1.20,
    "農林・鉱業":   1.20,
    "機械・電機":   1.10,
    "非鉄・窯業":   1.10,
    "輸送・精密":   1.05,
    "サービス・IT": 1.00,
    "薬品・IT":     1.00,
    "化学・ゴム":   0.95,
    "食品・繊維":   0.85,
}
_SECTOR_MAP = [
    (20,  "農林・鉱業"),  (30, "食品・繊維"),  (40, "化学・ゴム"),
    (50,  "薬品・IT"),    (60, "非鉄・窯業"),  (70, "機械・電機"),
    (80,  "輸送・精密"),  (90, "金融・不動産"), (100, "サービス・IT"),
]

def _code_to_sector(code):
    try:
        c = int(code) // 100
        for threshold, name in _SECTOR_MAP:
            if c < threshold:
                return name
        return "サービス・IT"
    except Exception:
        return None

def _price_label(close):
    if not close or close <= 0:
        return None
    if close < 500:   return "①~500円"
    if close < 1000:  return "②500~1000円"
    if close < 2000:  return "③1000~2000円"
    if close < 5000:  return "④2000~5000円"
    return "⑤5000円~"

def calc_attr_weight(code, prev_close, upper_shadow_ratio=None):
    """銘柄コードと前日終値・上ひげ比率から属性重みを計算して返す。

    adj_score = base_score × price_weight × sector_weight × shadow_weight
    shadow_weight: usr=0（高値引け）→×1.0、usr=1.0（安値引け・大上ひげ）→×1.15
    根拠: analyze_a Section 10 — usr高いほど WR+6.9%（単調増加）
    """
    pw  = _PRICE_WEIGHTS.get(_price_label(prev_close), 1.0)
    sw  = _SECTOR_WEIGHTS.get(_code_to_sector(code), 1.0)
    usr = upper_shadow_ratio if upper_shadow_ratio is not None else 0.5
    shw = 1.0 + 0.15 * max(0.0, min(1.0, usr))
    return round(pw * sw * shw, 4)

# ── 設定 ─────────────────────────────────────────
TACHIBANA_LOGIN_FILE = str(_BASE_DIR / "tachibana_login_response.json")


# ══════════════════════════════════════════════════
# データ読み込み
# ══════════════════════════════════════════════════

def load_candidates():
    """戦略D専用候補（candidates_log, strategy='D'）を優先して読み込み、
    未生成の場合はscan_results（戦略A用TOP20〜30リスト）にフォールバックする。

    2026-08-04の検証で、scan_resultsは戦略A用に絞られた狭い候補プール（1日20〜30銘柄）
    であり、戦略Dのモメンタム条件（市場全体では1日150〜200銘柄が該当）とは母集団が
    ミスマッチしていることが判明。scan_morning_strategy_d_candidates.py の事前フィルタ
    結果（USE_D_CANDIDATE_SCAN=Trueの場合のみ有効）に切り替えられるようにする。
    """
    conn = db.get_conn()
    try:
        if USE_D_CANDIDATE_SCAN:
            df = pd.read_sql(
                "SELECT code, name, score FROM candidates_log WHERE strategy='D' AND date=?",
                conn, params=(TODAY,)
            )
            if not df.empty:
                print(f"  候補銘柄: {len(df)}件（戦略D専用スキャン・{TODAY}）")
                source = "d_scan"
            else:
                df = None
        else:
            df = None

        if df is None:
            row = conn.execute("SELECT MAX(scan_date) FROM scan_results").fetchone()
            if not row or not row[0]:
                return []
            latest_date = row[0]
            df = pd.read_sql(
                "SELECT * FROM scan_results WHERE scan_date=?", conn,
                params=(latest_date,)
            )
            source = "scan_results"
    finally:
        conn.close()

    if df.empty:
        return []

    restricted = db.get_restricted_codes(days=30)
    if restricted:
        before = len(df)
        df = df[~df["code"].astype(str).isin(restricted)]
        excluded = before - len(df)
        if excluded > 0:
            print(f"  🚫 日計取引制限銘柄を除外: {excluded}件 {restricted & set(df['code'].astype(str).tolist()) or restricted}")

    if source == "scan_results":
        print(f"  候補銘柄: {len(df)}件（スキャン日: {latest_date}）")
    else:
        print(f"  候補銘柄: {len(df)}件（除外後）")
    return df.to_dict("records")


def _ma_dev(closes, n, ref_close):
    """closes リストの先頭 n 件の単純平均と ref_close の乖離率（%）を返す。"""
    if len(closes) < n or not ref_close:
        return None
    ma = sum(closes[:n]) / n
    return (ref_close - ma) / ma * 100 if ma > 0 else None


def load_prev_ohlcv(codes):
    """daily_prices から各銘柄の直近26営業日の OHLCV を一括取得する。
    戻り値: {code: {"high", "low", "close", "volume", "date",
                    "prev_rise", "upper_shadow_ratio", "ma5_dev", "ma25_dev"}}
      prev_rise          : 前日終値の騰落率（前日 vs 2日前）
      upper_shadow_ratio : 前日の上ひげ比率（analyze_a Section 10）
      ma5_dev / ma25_dev : 5日・25日MA乖離率（analyze_a Section 11）
    """
    if not codes:
        return {}

    str_codes    = [str(c) for c in codes]
    placeholders = ",".join(["?"] * len(str_codes))

    conn = db.get_conn()
    try:
        all_rows = pd.read_sql(
            f"SELECT code, High, Low, Close, Volume, Date FROM daily_prices "
            f"WHERE code IN ({placeholders}) AND Date < ? ORDER BY code, Date DESC",
            conn, params=str_codes + [TODAY]
        )
    finally:
        conn.close()

    all_rows["code"] = all_rows["code"].astype(str)
    result = {}

    for code in str_codes:
        rows = all_rows[all_rows["code"] == code].head(26).reset_index(drop=True)
        if rows.empty:
            continue

        r          = rows.iloc[0]
        prev_close = float(r["Close"]) if r["Close"] else None
        high       = float(r["High"])  if r["High"]  else None
        low        = float(r["Low"])   if r["Low"]   else None

        # 前日騰落率（前日 vs 2日前）
        prev_rise = None
        if len(rows) >= 2 and prev_close:
            close_2d = float(rows.iloc[1]["Close"])
            if close_2d > 0:
                prev_rise = (prev_close - close_2d) / close_2d * 100

        # 上ひげ比率: (High - Close) / (High - Low)
        rng = (high - low) if (high and low) else None
        usr = ((high - prev_close) / rng
               if (high and prev_close and rng and rng > 0) else None)

        # MA乖離率（5日・25日）
        closes   = pd.to_numeric(rows["Close"], errors="coerce").dropna().tolist()
        ma5_dev  = _ma_dev(closes, 5,  prev_close)
        ma25_dev = _ma_dev(closes, 25, prev_close)

        result[code] = {
            "high":               high,
            "low":                low,
            "close":              prev_close,
            "volume":             float(r["Volume"]) if r["Volume"] else None,
            "date":               r["Date"],
            "prev_rise":          prev_rise,
            "upper_shadow_ratio": usr,
            "ma5_dev":            ma5_dev,
            "ma25_dev":           ma25_dev,
        }

    return result


# ══════════════════════════════════════════════════
# Tachibana API
# ══════════════════════════════════════════════════

def load_tachibana_url():
    try:
        with open(TACHIBANA_LOGIN_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("sUrlPrice") or None
    except Exception:
        return None


def _parse_price_field(item, k):
    """Tachibana APIレスポンスの1フィールドを float に変換して返す。"""
    v = item.get(k, "")
    if isinstance(v, str):
        v = v.strip('"')
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def fetch_prices(url_price, codes):
    """Tachibana API で複数銘柄の現在値・出来高を一括取得。
    戻り値: {code: {"price": ..., "prev_close": ..., "volume": ...}}
    """
    if not url_price or not codes:
        return {}

    http    = urllib3.PoolManager()
    result  = {}
    batch   = 100

    for i in range(0, len(codes), batch):
        chunk  = codes[i:i + batch]
        t      = datetime.now(JST)
        p_sd   = (f"{t.year}.{t.month:02}.{t.day:02}"
                  f"-{t.hour:02}:{t.minute:02}:{t.second:02}"
                  f".{t.microsecond // 1000:03}")
        params = (
            "{"
            f'"p_no":"{tachibana_order._next_p_no()}",'
            f'"p_sd_date":"{p_sd}",'
            '"sCLMID":"CLMMfdsGetMarketPrice",'
            f'"sTargetIssueCode":"{",".join(str(c) for c in chunk)}",'
            '"sTargetColumn":"pDPP,pPRP,pOP,pDV",'
            '"sJsonOfmt":"5"'
            "}"
        )
        try:
            resp = http.request("GET", url_price + "?" + params,
                                timeout=urllib3.Timeout(connect=3, read=8))
            tachibana_order.log_api_call("daytime.fetch_prices")
            data = json.loads(resp.data.decode("shift-jis", errors="ignore"))
            for item in data.get("aCLMMfdsMarketPrice", []):
                code = item.get("sIssueCode", "").strip('"')
                if not code:
                    continue
                price  = _parse_price_field(item, "pDPP") or _parse_price_field(item, "pPRP")
                prev   = _parse_price_field(item, "pPRP")
                open_p = _parse_price_field(item, "pOP")
                vol    = int(_parse_price_field(item, "pDV") or 0)
                if price:
                    result[code] = {"price": price, "prev_close": prev,
                                    "open_price": open_p, "volume": vol}
        except Exception as e:
            print(f"  ⚠️  価格取得エラー: {e}")

    return result


# ══════════════════════════════════════════════════
# シグナル判定
# ══════════════════════════════════════════════════


def check_signal(code, quote, prev_ohlcv, first_prices, score=None,
                 gap_min=None, gap_max=None, score_max=None):
    """シグナル条件を判定して結果を返す（analyze_a.py 統計分析結果に基づく改訂版）。

    フィルター条件（すべて満たす必要あり）:
        1. 寄り付きギャップ: gap_min 〜 gap_max の範囲（デフォルト: GAP_MIN_PCT/GAP_MAX_PCT）
        2. 前日騰落プラス  : 前日終値 > 2日前終値（PREV_RISE_MIN 以上）
        3. スコア上限      : score_max 指定時は score がそれ未満の銘柄のみ（月曜フィルタ）
        4. MA乖離率フィルタ: 5日MA乖離 > MA5_DEV_MAX or 25日MA乖離 > MA25_DEV_MAX は除外
        5. 上昇モメンタム  : 現在値 > 前日終値 × (1 + MOMENTUM_PCT/100)

    戻り値: dict（条件を満たさない場合は None）
    """
    price      = quote.get("price")
    prev_close = quote.get("prev_close")

    if not price or not prev_close or prev_close <= 0:
        return None

    change_pct = (price - prev_close) / prev_close * 100
    prev       = prev_ohlcv or {}
    prev_high  = prev.get("high")
    prev_rise  = prev.get("prev_rise")  # 前日の騰落率（前日 vs 2日前）

    _gap_min = GAP_MIN_PCT if gap_min is None else gap_min
    _gap_max = GAP_MAX_PCT if gap_max is None else gap_max

    # ── 寄り付きギャップ ─────────────────────────
    # 起動時に pOP（始値）から計算済み。未取得の銘柄のみ初回観測値で代用。
    if code not in first_prices:
        first_prices[code] = change_pct   # フォールバック（始値取得失敗時）
    gap_pct = first_prices[code]

    # フィルター1: ギャップ範囲（月曜は gap_min=0 でマイナスギャップを除外）
    if not (_gap_min <= gap_pct <= _gap_max):
        return None

    # フィルター2: 前日がプラスでない銘柄は除外
    if prev_rise is None or prev_rise < PREV_RISE_MIN:
        return None

    # フィルター3: スコア上限（月曜は低スコア銘柄のみ）
    if score_max is not None and score is not None and score >= score_max:
        return None

    # フィルター4: MA乖離率の過熱圏を除外（analyze_a Section 11）
    ma5_dev  = prev.get("ma5_dev")
    ma25_dev = prev.get("ma25_dev")
    if ma5_dev  is not None and ma5_dev  > MA5_DEV_MAX:
        return None
    if ma25_dev is not None and ma25_dev > MA25_DEV_MAX:
        return None

    # フィルター5: 前日出来高（小型株による市場インパクト回避）
    vol = prev.get("volume")
    if vol is not None and vol < MIN_VOLUME:
        return None

    # シグナルトリガー: 前日終値比でモメンタム確認
    if change_pct < MOMENTUM_PCT:
        return None

    return {
        "price":      price,
        "prev_high":  prev_high,
        "change_pct": round(change_pct, 2),
        "gap_pct":    round(gap_pct, 2),
        "prev_rise":  round(prev_rise, 2) if prev_rise is not None else None,
    }


# ══════════════════════════════════════════════════
# ログ保存
# ══════════════════════════════════════════════════

def log_noon_ad_observation(time_label, ad_ratio, nikkei_change, scanned_cond, condition):
    """2026-08-14追加: 13:00再判定の30分バッファ(12:30後場再開→13:00)が妥当か検証するため、
    12:40時点のAD比率を発注判断に使わず観測目的のみでlogs/に記録する。
    dt_live.txt(当日上書き)と違い、複数日分を蓄積して後日比較できるようappendする。"""
    try:
        log_dir = _BASE_DIR / "logs"
        log_dir.mkdir(exist_ok=True)
        path = log_dir / "noon_ad_observation.log"
        nikkei_str = f"{nikkei_change:.2f}" if nikkei_change is not None else "NA"
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{TODAY},{time_label},{ad_ratio:.3f},{nikkei_str},"
                    f"{scanned_cond},{condition}\n")
    except Exception:
        pass


def save_signal(code, name, sig):
    """シグナルを DB（daytime_signals テーブル）に保存する。"""
    db.save_daytime_signal({
        "date":           TODAY,
        "time":           datetime.now(JST).strftime("%H:%M:%S"),
        "code":           code,
        "name":           name,
        "price":          sig["price"],
        "change_pct":     sig["change_pct"],
        "prev_high":      sig["prev_high"],
        "conditions_met": 1,
        "breakout":       0,
        "volume_surge":   0,
        "momentum":       1,
        "volume_pace_ratio": 0,
        "tp_price":       round(sig["price"] * (1 + TP_PCT)),
        "sl_price":       round(sig["price"] * (1 - SL_PCT)),
    })


# ══════════════════════════════════════════════════
# メインループ
# ══════════════════════════════════════════════════

def watch_loop(candidates, url_price, url_request, condition, start_now=False):
    codes         = [str(c["code"]) for c in candidates]
    names         = {str(c["code"]): c.get("name", c["code"]) for c in candidates}
    scores        = {str(c["code"]): c.get("score", 0) for c in candidates}
    signaled      = set()    # 発注完了 or 永続失敗（このセッションでチェックしない）
    alerted       = set()    # シグナル表示済み（重複表示防止のみ）
    ordered_today = set(db.get_today_ordered_codes(TODAY))
    signal_count  = 0
    ROUTINE_LOG_INTERVAL = 300   # 「監視中」の定型メッセージは5分に1回だけ出力（ノイズ削減）
    last_routine_print   = None
    last_lunch_print      = None

    # 曜日別パラメータ上書き（月曜は構造的に弱いため制限付き発注）
    is_monday     = (datetime.now(JST).weekday() == 0)
    max_positions = MON_MAX_POSITIONS if is_monday else MAX_DAYTIME_POSITIONS
    gap_min       = MON_GAP_MIN_PCT   if is_monday else GAP_MIN_PCT
    score_max     = MON_SCORE_MAX     if is_monday else None

    # 地合い別 gap_max 調整
    # NORMAL日は gap < 0（マイナスギャップ）のみ通過
    # 根拠: analyze_a_normal.py OOS検証結果
    #   gap 0〜+2%: 勝率47.1% / 平均+0.017% → ランダム以下
    #   gap  < 0%:  勝率53〜57% / 平均+0.13〜0.37% → 有効
    gap_max = 0.0 if condition == "NORMAL" else GAP_MAX_PCT

    # 月曜×NORMAL: gap条件が相互矛盾（月曜gap≥0% vs NORMAL gap<0%）のため全見送り
    # 根拠: ga_monday_normal.py GA最適化（2024-06-01〜, N=16,669件）
    #   月曜×NORMAL OOS: WR=39.2% / avg=-0.273% / Sharpe=-2.795 → 全gap範囲で有害
    mon_normal_skip = is_monday and condition == "NORMAL"

    print(f"\n  監視開始: {WATCH_START_HOUR}:{WATCH_START_MIN:02d} 〜 "
          f"{WATCH_END_HOUR}:{WATCH_END_MIN:02d}  対象: {len(codes)}銘柄")
    print(f"  地合い: {condition}  （{'発注有効' if condition in ALLOWED_ORDER_CONDITIONS else '監視のみ（NORMAL/WEAK/PANIC）' if REQUIRE_STRONG else '監視のみ（WEAK/PANIC）'}）")
    if mon_normal_skip:
        print(f"  ⚠️  月曜×NORMAL: 発注見送り（OOS Sharpe=-2.795, WR=39.2%）")
    elif is_monday:
        print(f"  ⚠️  月曜モード: 最大ポジション={max_positions} / gap≥{gap_min:+.0f}% / score<{score_max}")
    if condition == "NORMAL" and not mon_normal_skip:
        print(f"  ⚠️  NORMAL日: gap < 0%（マイナスギャップのみ）に絞って発注")
    print(f"  条件: {condition}地合い × gap{gap_min:+}〜{gap_max:+}% × 前日プラス × モメンタム{MOMENTUM_PCT:+.1f}%以上")
    print(f"  TP: +{TP_PCT*100:.0f}%  SL: -{SL_PCT*100:.0f}%\n")

    # 前日OHLCV 読み込み（2日分 → 前日騰落率も取得）
    print("  前日OHLCV 読み込み中...")
    prev_ohlcv = load_prev_ohlcv(codes)
    print(f"  読み込み完了: {len(prev_ohlcv)}銘柄\n")

    # 寄り付き価格（pOP）でギャップを初期化
    # バックテスト(analyze_a.py)が使う「9:00始値ベースのgap%」と一致させる
    first_prices = {}
    if url_price:
        print("  寄り付き価格（始値）取得中...")
        init_q = fetch_prices(url_price, codes)
        for _code, q in init_q.items():
            op   = q.get("open_price")
            prev = q.get("prev_close")
            if op and op > 0 and prev and prev > 0:
                first_prices[_code] = round((op - prev) / prev * 100, 2)
        got = len(first_prices)
        print(f"  → {got}/{len(codes)}銘柄の始値gap取得完了"
              f"  (残{len(codes) - got}銘柄は初回観測値で代用)\n")

    watch_end_min   = WATCH_END_HOUR * 60 + WATCH_END_MIN
    start_min       = WATCH_START_HOUR * 60 + WATCH_START_MIN
    NOON_RESCAN_MIN = 13 * 60   # 後場再開(12:30)から30分経過した13:00に実測地合いを取り直す
                                # ※12:00は前場引け(11:30)〜後場再開(12:30)の昼休み中で値が動かないため不適
    LUNCH_START_MIN = 11 * 60 + 30   # 前場引け
    LUNCH_END_MIN   = 12 * 60 + 30   # 後場再開
    opening_rescanned = False  # 監視開始(9:30)時点の実測補正フラグ
    noon_rescanned     = False
    OBSERVE_MIN  = 12 * 60 + 40   # 2026-08-14追加: 13:00再判定の30分バッファが妥当か
                                   # 検証するため、12:40時点のAD比率を発注判断に使わず
                                   # 観測目的のみで記録する（[[entry_timing_review_plan]]と同じ考え方）
    observed_1240 = False

    while True:
        now     = datetime.now(JST)
        now_min = now.hour * 60 + now.minute
        now_str = now.strftime("%H:%M:%S")

        if now_min >= watch_end_min:
            print(f"\n  ⏰ {WATCH_END_HOUR}:{WATCH_END_MIN:02d} 到達 → 監視終了")
            break

        # 2026-08-10追加: 前場引け(11:30)〜後場再開(12:30)の昼休み中は市場が
        # 閉まっており値が動かないため、Tachibana APIへの問い合わせを完全に休止する
        # （立花証券のAPI高負荷指摘を受けた対応の一環。この時間帯のシグナル発生は
        # 実績上ゼロで、機会損失にはならない）。
        if LUNCH_START_MIN <= now_min < LUNCH_END_MIN:
            if last_lunch_print is None or (now - last_lunch_print).total_seconds() >= ROUTINE_LOG_INTERVAL:
                last_lunch_print = now
                print(f"  [{now_str}] 昼休み中（11:30〜12:30）→ 市場休場のためAPI呼び出しを休止")
            time.sleep(60)
            continue

        # 市場開始前は待機（このプロセス自体は9:00頃から起動しているため、
        # 実測補正スキャンも9:30に到達するまではここで待つ）
        if not start_now and now_min < start_min:
            wait = (start_min - now_min) * 60 - now.second
            print(f"  開始まで {wait//60}分{wait%60}秒待機中... ({now_str})")
            time.sleep(30)
            continue

        # 監視開始(9:30)時点の地合い再チェック（朝予測は前夜の海外指標のみのため外れることがある）
        if not opening_rescanned and url_price:
            opening_rescanned = True
            try:
                print(f"\n  [{now_str}] 🔔 開始時点の地合い再チェック中（全銘柄スキャン）...")
                ad_ratio, nikkei_change, scanned_cond, _ = closing_watch.scan_dropping_stocks(url_price)
                if scanned_cond != condition:
                    print(f"  ⚠️  地合い変化: {condition}（朝予測）→ {scanned_cond}（実測 AD{ad_ratio:.2f}）")
                    condition       = scanned_cond
                    gap_max         = 0.0 if condition == "NORMAL" else GAP_MAX_PCT
                    mon_normal_skip = is_monday and condition == "NORMAL"
                    print(f"  地合い（開始時補正後）: {condition}  "
                          f"（{'発注有効' if condition in ALLOWED_ORDER_CONDITIONS else '監視のみ（NORMAL/WEAK/PANIC）' if REQUIRE_STRONG else '監視のみ（WEAK/PANIC）'}）")
                else:
                    print(f"  地合い変化なし: {condition}（AD{ad_ratio:.2f}）")
            except Exception as e:
                print(f"  ⚠️  開始時点の地合い再チェックに失敗（現在の判定のまま継続）: {e}")

        # 2026-08-14追加: 12:40時点のAD比率を観測目的のみで記録（発注判断には使わない）。
        # 13:00再判定の30分バッファが妥当か、後日ログを見て検証するための布石。
        if not observed_1240 and now_min >= OBSERVE_MIN and url_price:
            observed_1240 = True
            try:
                ad_ratio, nikkei_change, scanned_cond, _ = closing_watch.scan_dropping_stocks(url_price)
                log_noon_ad_observation("12:40", ad_ratio, nikkei_change, scanned_cond, condition)
                print(f"  [{now_str}] 📝 12:40観測記録: AD{ad_ratio:.2f} → {scanned_cond}"
                      f"（現行条件{condition}、発注判断には未使用）")
            except Exception as e:
                print(f"  ⚠️  12:40観測の記録に失敗（無視して継続）: {e}")

        # 昼(13:00)の地合い再チェック（朝9:30の1回だけだと午後の反転を拾えないため）
        if not noon_rescanned and now_min >= NOON_RESCAN_MIN and url_price:
            noon_rescanned = True
            try:
                print(f"\n  [{now_str}] 🕐 後場の地合い再チェック中（全銘柄スキャン）...")
                ad_ratio, nikkei_change, scanned_cond, _ = closing_watch.scan_dropping_stocks(url_price)
                log_noon_ad_observation("13:00", ad_ratio, nikkei_change, scanned_cond, condition)
                if scanned_cond != condition:
                    print(f"  ⚠️  地合い変化: {condition}（直前）→ {scanned_cond}（昼実測 AD{ad_ratio:.2f}）")
                    condition       = scanned_cond
                    gap_max         = 0.0 if condition == "NORMAL" else GAP_MAX_PCT
                    mon_normal_skip = is_monday and condition == "NORMAL"
                    print(f"  地合い（昼補正後）: {condition}  "
                          f"（{'発注有効' if condition in ALLOWED_ORDER_CONDITIONS else '監視のみ（NORMAL/WEAK/PANIC）' if REQUIRE_STRONG else '監視のみ（WEAK/PANIC）'}）")
                else:
                    print(f"  地合い変化なし: {condition}（AD{ad_ratio:.2f}）")
            except Exception as e:
                print(f"  ⚠️  昼の地合い再チェックに失敗（現在の判定のまま継続）: {e}")

        # 2026-08-10追加: gap%(first_prices)は寄り付き後は変わらないため、
        # gap_min〜gap_max（Filter1、check_signal内と同一条件）を満たす銘柄だけに
        # 問い合わせ対象を絞り込む。まだfirst_pricesが無い銘柄（初回取得失敗）は
        # フォールバック確定のため対象に残す。これにより1日あたりのTachibana API
        # 呼び出し回数を候補全数(1日数百〜900件超)から大幅に削減する
        # （立花証券より2026-08-10にAPI高負荷の指摘を受けて対応）。
        watch_codes = [
            c for c in codes
            if c not in first_prices or (gap_min <= first_prices[c] <= gap_max)
        ]
        quotes = fetch_prices(url_price, watch_codes) if url_price else {}

        # ── Step1: このポーリングで出たシグナルを全件収集 ──
        new_signals = []
        for code in codes:
            if code in signaled:   # 発注完了 or 永続失敗 → スキップ
                continue
            quote = quotes.get(code)
            if not quote:
                continue
            sig = check_signal(code, quote, prev_ohlcv.get(code), first_prices,
                               score=scores.get(code),
                               gap_min=gap_min, gap_max=gap_max,
                               score_max=score_max)
            if sig is None:
                continue

            # adj_score = (base_score + 1) × 価格帯重み × 業種重み × 上ひげ重み
            # +1 オフセット: score=0 でも attr_w がランキングに反映されるようにする
            ohlcv      = prev_ohlcv.get(code) or {}
            prev_close = ohlcv.get("close")
            usr        = ohlcv.get("upper_shadow_ratio")
            attr_w     = calc_attr_weight(code, prev_close, upper_shadow_ratio=usr)
            adj_score  = (float(scores.get(code) or 0) + 1.0) * attr_w

            name = names.get(code, code)
            is_retry = code in alerted   # 既に表示済み = 発注失敗後の再試行
            new_signals.append((code, name, sig, adj_score, attr_w, is_retry))
            if not is_retry:
                alerted.add(code)
                signal_count += 1
                save_signal(code, name, sig)

        # ── Step2: adj_score 降順でソートして優先発注 ────────
        new_signals.sort(key=lambda x: -x[3])

        can_order = (DAYTIME_TRADING and
                     condition in ALLOWED_ORDER_CONDITIONS and
                     not mon_normal_skip and
                     bool(url_request))

        for code, name, sig, adj_score, attr_w, is_retry in new_signals:
            tp = round(sig["price"] * (1 + TP_PCT))
            sl = round(sig["price"] * (1 - SL_PCT))
            if is_retry:
                print(f"\n  🔄 [{now_str}] 再試行: {code} {name}"
                      f"  adj_score={adj_score:.2f}（w={attr_w:.2f}）")
            else:
                print(f"\n  🔔 [{now_str}] シグナル: {code} {name}"
                      f"  adj_score={adj_score:.2f}（w={attr_w:.2f}）")
                print(f"     前日比: {sig['change_pct']:+.2f}%  寄付ギャップ: {sig['gap_pct']:+.2f}%"
                      f"  前日騰落: {sig['prev_rise']:+.2f}%")
                print(f"     現在値: {sig['price']:,.0f}円  "
                      f"TP: {tp:,.0f}円（+{TP_PCT*100:.0f}%）  SL: {sl:,.0f}円（-{SL_PCT*100:.0f}%）")

            if not can_order:
                reason = ("DAYTIME_TRADING=False" if not DAYTIME_TRADING
                          else f"地合い={condition}")
                print(f"     [{reason}] 発注スキップ")
                signaled.add(code)   # 発注条件外は再試行しても意味がないのでスキップ
                continue
            if code in ordered_today:
                print(f"     ℹ️  本日発注済み → スキップ")
                signaled.add(code)
                continue
            open_pos = db.load_open_positions(strategy="D")
            if len(open_pos) >= max_positions:
                print(f"     ⚠️  戦略D上限({max_positions})到達 → 発注スキップ")
                signaled.add(code)
                continue
            if sig["price"] < CHEAP_THRESHOLD:
                shares = max(100, int(MAX_ORDER_AMOUNT / sig["price"] / 100) * 100)
            elif sig["price"] * DEFAULT_SHARES > HIGH_PRICE_CAP_AMOUNT:
                shares = max(100, int(HIGH_PRICE_CAP_AMOUNT / sig["price"] / 100) * 100)
            else:
                shares = DEFAULT_SHARES
            mkt_code = db.get_market_code_db(code)
            order_type = "指値" if LIMIT_ORDER else "成行"
            print(f"     発注方式: {order_type} @ {sig['price']:,.0f}円")
            if LIMIT_ORDER:
                r = tachibana_order.place_buy_limit_with_fallback(
                    url_request, code, shares, sig["price"], mkt_code,
                    wait_secs=LIMIT_WAIT_SECS, strategy="D")
                if r["success"]:
                    tachibana_order.save_position(
                        code, name, r["filled"], r["fill_price"], "D",
                        tp_pct=TP_PCT, sl_pct=SL_PCT,
                        entry_change_pct=sig["change_pct"],
                        url_request=url_request, order_no=r.get("order_no"),
                    )
                    ordered_today.add(code)
                    signaled.add(code)
                    print(f"     ✅ 買い発注完了: {r['filled']}株  {r['message']}")
                else:
                    msg = r["message"]
                    print(f"     ❌ 発注失敗: {msg}")
                    if "規制銘柄" in msg:
                        signaled.add(code)
                        print(f"     ⛔ 規制銘柄のため以降スキップ")
            else:
                result = tachibana_order.place_buy_order(url_request, code, shares, market_code=mkt_code)
                if result["success"]:
                    eigyou_day = (result.get("raw") or {}).get("sEigyouDay")
                    tachibana_order.save_position(
                        code, name, shares, sig["price"], "D",
                        tp_pct=TP_PCT, sl_pct=SL_PCT,
                        entry_change_pct=sig["change_pct"],
                        url_request=url_request, order_no=result.get("order_no"),
                        eigyou_day=eigyou_day,
                    )
                    ordered_today.add(code)
                    signaled.add(code)
                    print(f"     ✅ 買い発注完了: {shares}株  {result['message']}")
                else:
                    msg = result["message"]
                    print(f"     ❌ 発注失敗: {msg}")
                    if "規制銘柄" in msg:
                        signaled.add(code)
                        print(f"     ⛔ 規制銘柄のため以降スキップ")

        if not new_signals:
            if last_routine_print is None or (now - last_routine_print).total_seconds() >= ROUTINE_LOG_INTERVAL:
                last_routine_print = now
                active = len(codes) - len(signaled)
                print(f"  [{now_str}] 監視中: {active}銘柄  発報済み: {len(alerted)}件")

        time.sleep(POLL_INTERVAL)

    return signal_count


# ══════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════

def main(start_now=False):
    print(f"=== 📈 日中シグナル検知（{TODAY}）===\n")
    db.init_db()

    # 候補銘柄読み込み
    print("  候補銘柄を読み込み中...")
    candidates = load_candidates()
    if not candidates:
        print("  ❌ scan_results にデータがありません。scan_daily.py を先に実行してください。")
        return

    # Tachibana API
    url_price   = load_tachibana_url()
    url_request = tachibana_order.load_url_request()
    mode_str    = "本番" if tachibana_order.LIVE_TRADING else "モック"
    if url_price:
        print(f"  📡 Tachibana API: リアルタイム価格取得モード  [{mode_str}]")
    else:
        print(f"  ⚠️  Tachibana API 未ログイン → 価格取得できません。")
        print(f"      scan_morning.py を先に実行してログインしてください。")
        return

    # 地合い取得（market_log → morning_log の順で検索）
    # ※ market_logは前営業日16:45のscan_daily.pyでしか書かれないため、
    #   実質ここで得られるのは前夜の海外指標のみで作った「朝予測」であり、
    #   寄り付き後の実際の値動きは反映されていない（2026-07-14: WEAK予測→実際はSTRONGで機会損失）。
    # ※ このプロセス自体はscan_morning.pyがmarket_watch.pyと並行して9:00頃に起動するため、
    #   ここで実測補正すると9:00直後（ノイズが大きい・market_watch.pyとAPI負荷が競合）に
    #   実行されてしまう。実測補正は watch_loop() 側で9:30到達を待ってから行う。
    condition = db.get_today_condition_db(TODAY)
    print(f"\n  地合い（朝予測）: {condition}")

    if condition == "UNKNOWN":
        print(f"  ⚠️  本日の地合いが取得できません（scan_morning.py を先に実行してください）")
        print(f"     地合い不明のため監視のみで続行します")

    if REQUIRE_STRONG and condition not in ("STRONG", "UNKNOWN"):
        print(f"\n  ⚠️  本日の地合い: {condition}")
        print(f"     STRONG地合い以外は発注しません（監視は継続）")

    signal_count = watch_loop(candidates, url_price, url_request, condition, start_now)

    print(f"\n{'='*60}")
    print(f"  日中シグナル検知 完了")
    print(f"  本日の発報件数: {signal_count}件")
    if signal_count > 0:
        print(f"  ログ: out/stock.db（daytime_signals テーブル）")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="日中シグナル検知")
    parser.add_argument("--now", action="store_true", help="即時開始（テスト用）")
    args = parser.parse_args()
    main(start_now=args.now)
