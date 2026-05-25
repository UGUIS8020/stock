# -*- coding: utf-8 -*-
"""
daytime.py - 日中シグナル検知（9:30〜14:00）

scan_daily.py の候補リストを使い、日中の出来高急増・価格ブレイクアウトを検知して自動発注する。

使い方:
    python daytime.py        # 9:00まで待機して自動開始
    python daytime.py --now  # 即時開始（テスト用）

シグナル条件（以下のうち2つ以上で発報）:
    1. 価格ブレイクアウト : 現在値 > 前日高値
    2. 出来高ペース急増   : 現在の出来高ペース > 前日出来高 × VOLUME_PACE_RATIO
    3. モメンタム         : 前日比 > MOMENTUM_PCT %
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

JST      = timezone(timedelta(hours=9))
TODAY    = datetime.now(JST).strftime("%Y-%m-%d")
_BASE_DIR = Path(__file__).parent

# ── 時間設定 ──────────────────────────────────────
WATCH_START_HOUR = 9
WATCH_START_MIN  = 0
WATCH_END_HOUR   = 14
WATCH_END_MIN    = 30
POLL_INTERVAL    = 15   # 秒

# ── TP / SL ───────────────────────────────────────
TP_PCT = 0.03   # +3%
SL_PCT = 0.015  # -1.5%（シミュレーション最適値 SL-1%〜2% に合わせて調整）

# ── 発注設定 ──────────────────────────────────────
DAYTIME_TRADING      = True      # False にすると監視のみ（発注なし）
ORDER_AMOUNT         = 300_000   # 1発注あたりの目安金額（30万円）
MAX_DAYTIME_POSITIONS = 3        # 日中最大ポジション数

# ── シグナル条件（チューニングポイント）────────────
BREAKOUT_BUFFER_PCT    = 0.0   # 前日高値 × (1 + これ) を超えたらブレイクアウト
MOMENTUM_PCT           = 1.0   # 前日比がこの%以上でモメンタムあり
VOLUME_PACE_RATIO      = 1.5   # 出来高ペースが前日出来高のこの倍以上で急増
SIGNAL_MIN_CONDITIONS  = 2     # 3条件のうちこの数以上で発報

# ── 設定 ─────────────────────────────────────────
TACHIBANA_LOGIN_FILE = str(_BASE_DIR / "tachibana_login_response.json")

# 東京市場の取引時間（分換算）
MARKET_OPEN_MIN  = 9 * 60       # 9:00
MARKET_CLOSE_MIN = 15 * 60 + 30 # 15:30


# ══════════════════════════════════════════════════
# データ読み込み
# ══════════════════════════════════════════════════

def load_candidates():
    """scan_results から直近スキャン日の候補銘柄を取得する。"""
    conn = db.get_conn()
    try:
        df = pd.read_sql(
            "SELECT * FROM scan_results ORDER BY scan_date DESC LIMIT 200", conn
        )
    finally:
        conn.close()

    if df.empty:
        return []

    latest_date = df["scan_date"].iloc[0]
    df = df[df["scan_date"] == latest_date]
    print(f"  候補銘柄: {len(df)}件（スキャン日: {latest_date}）")
    return df.to_dict("records")


def load_prev_ohlcv(codes):
    """daily_prices から各銘柄の直近営業日の OHLCV を取得する。
    戻り値: {code: {"high": ..., "volume": ..., "close": ...}}
    """
    conn = db.get_conn()
    result = {}
    try:
        for code in codes:
            rows = pd.read_sql(
                "SELECT High, Low, Close, Volume, Date FROM daily_prices "
                "WHERE code=? ORDER BY Date DESC LIMIT 2",
                conn, params=(str(code),)
            )
            if len(rows) >= 1:
                r = rows.iloc[0]
                result[str(code)] = {
                    "high":   float(r["High"])   if r["High"]   else None,
                    "low":    float(r["Low"])    if r["Low"]    else None,
                    "close":  float(r["Close"])  if r["Close"]  else None,
                    "volume": float(r["Volume"]) if r["Volume"] else None,
                    "date":   r["Date"],
                }
    finally:
        conn.close()
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
            '"sTargetColumn":"pDPP,pPRP,pDV",'
            '"sJsonOfmt":"5"'
            "}"
        )
        try:
            resp = http.request("GET", url_price + "?" + params,
                                timeout=urllib3.Timeout(connect=3, read=8))
            data = json.loads(resp.data.decode("shift-jis", errors="ignore"))
            for item in data.get("aCLMMfdsMarketPrice", []):
                code = item.get("sIssueCode", "").strip('"')
                if not code:
                    continue
                def _f(k):
                    v = item.get(k, "")
                    if isinstance(v, str):
                        v = v.strip('"')
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        return None
                price = _f("pDPP") or _f("pPRP")
                prev  = _f("pPRP")
                vol   = int(_f("pDV") or 0)
                if price:
                    result[code] = {"price": price, "prev_close": prev, "volume": vol}
        except Exception as e:
            print(f"  ⚠️  価格取得エラー: {e}")

    return result


# ══════════════════════════════════════════════════
# シグナル判定
# ══════════════════════════════════════════════════

def elapsed_market_minutes():
    """本日の市場開始からの経過分数を返す（前場・後場の休憩を考慮）。"""
    now     = datetime.now(JST)
    now_min = now.hour * 60 + now.minute

    if now_min < MARKET_OPEN_MIN:
        return 0
    # 前場: 9:00〜11:30（90分）
    if now_min <= 11 * 60 + 30:
        return now_min - MARKET_OPEN_MIN
    # 後場: 12:30〜15:30（180分）
    if now_min < 12 * 60 + 30:
        return 90  # 休憩中は前場終了時点の分数
    return 90 + (now_min - (12 * 60 + 30))


def check_signal(code, quote, prev_ohlcv):
    """シグナル条件を判定して結果を返す。

    戻り値:
        {
            "conditions_met": int,          # 満たした条件数
            "breakout":       bool,
            "volume_surge":   bool,
            "momentum":       bool,
            "change_pct":     float,
            "volume_pace_ratio": float,
        }
    """
    price      = quote.get("price")
    prev_close = quote.get("prev_close")
    volume_now = quote.get("volume", 0)

    if not price or not prev_close or prev_close <= 0:
        return None

    change_pct = (price - prev_close) / prev_close * 100
    prev_high  = (prev_ohlcv or {}).get("high")
    prev_vol   = (prev_ohlcv or {}).get("volume")

    # ── 条件1: 価格ブレイクアウト ─────────────────
    breakout = bool(
        prev_high and price > prev_high * (1 + BREAKOUT_BUFFER_PCT)
    )

    # ── 条件2: 出来高ペース急増 ───────────────────
    # 現在の出来高ペース = 今日の累積出来高 ÷ 経過市場分数 × 全営業時間(270分)
    elapsed = max(elapsed_market_minutes(), 1)
    projected_volume = volume_now / elapsed * 270
    volume_pace_ratio = (projected_volume / prev_vol) if prev_vol and prev_vol > 0 else 0
    volume_surge = volume_pace_ratio >= VOLUME_PACE_RATIO

    # ── 条件3: モメンタム ─────────────────────────
    momentum = change_pct >= MOMENTUM_PCT

    conditions_met = sum([breakout, volume_surge, momentum])

    return {
        "conditions_met":    conditions_met,
        "breakout":          breakout,
        "volume_surge":      volume_surge,
        "momentum":          momentum,
        "change_pct":        round(change_pct, 2),
        "volume_pace_ratio": round(volume_pace_ratio, 2),
        "price":             price,
        "prev_high":         prev_high,
    }


# ══════════════════════════════════════════════════
# ログ保存
# ══════════════════════════════════════════════════

def save_signal(code, name, sig):
    """シグナルを DB（daytime_signals テーブル）に保存する。"""
    db.save_daytime_signal({
        "date":              TODAY,
        "time":              datetime.now(JST).strftime("%H:%M:%S"),
        "code":              code,
        "name":              name,
        "price":             sig["price"],
        "change_pct":        sig["change_pct"],
        "prev_high":         sig["prev_high"],
        "conditions_met":    sig["conditions_met"],
        "breakout":          int(sig["breakout"]),
        "volume_surge":      int(sig["volume_surge"]),
        "momentum":          int(sig["momentum"]),
        "volume_pace_ratio": sig["volume_pace_ratio"],
        "tp_price":          round(sig["price"] * (1 + TP_PCT)),
        "sl_price":          round(sig["price"] * (1 - SL_PCT)),
    })


# ══════════════════════════════════════════════════
# メインループ
# ══════════════════════════════════════════════════

def watch_loop(candidates, url_price, url_request, start_now=False):
    codes     = [str(c["code"]) for c in candidates]
    names     = {str(c["code"]): c.get("name", c["code"]) for c in candidates}
    signaled  = set()   # 本日発報済みコード
    ordered_today = set(db.get_today_ordered_codes(TODAY))  # 本日発注済みコード
    signal_count = 0

    print(f"\n  監視開始: {WATCH_START_HOUR}:{WATCH_START_MIN:02d} 〜 "
          f"{WATCH_END_HOUR}:{WATCH_END_MIN:02d}  対象: {len(codes)}銘柄")
    print(f"  シグナル条件: ブレイクアウト + 出来高ペース({VOLUME_PACE_RATIO}倍) + "
          f"モメンタム(+{MOMENTUM_PCT}%) のうち{SIGNAL_MIN_CONDITIONS}つ以上")
    print(f"  TP: +{TP_PCT*100:.0f}%  SL: -{SL_PCT*100:.0f}%\n")

    # 前日OHLCV 読み込み
    print("  前日OHLCV 読み込み中...")
    prev_ohlcv = load_prev_ohlcv(codes)
    print(f"  読み込み完了: {len(prev_ohlcv)}銘柄\n")

    watch_end_min = WATCH_END_HOUR * 60 + WATCH_END_MIN

    while True:
        now     = datetime.now(JST)
        now_min = now.hour * 60 + now.minute
        now_str = now.strftime("%H:%M:%S")

        # 終了判定
        if now_min >= watch_end_min:
            print(f"\n  ⏰ {WATCH_END_HOUR}:{WATCH_END_MIN:02d} 到達 → 監視終了")
            break

        # 開始待機
        start_min = WATCH_START_HOUR * 60 + WATCH_START_MIN
        if not start_now and now_min < start_min:
            wait = (start_min - now_min) * 60 - now.second
            print(f"  開始まで {wait//60}分{wait%60}秒待機中... ({now_str})", end="\r")
            time.sleep(10)
            continue

        # 価格取得
        quotes = fetch_prices(url_price, codes) if url_price else {}

        # シグナル判定
        new_signals = []
        for code in codes:
            if code in signaled:
                continue
            quote = quotes.get(code)
            if not quote:
                continue
            sig = check_signal(code, quote, prev_ohlcv.get(code))
            if sig and sig["conditions_met"] >= SIGNAL_MIN_CONDITIONS:
                name = names.get(code, code)
                new_signals.append((code, name, sig))
                signaled.add(code)
                signal_count += 1
                save_signal(code, name, sig)

                # ── 買い発注 ────────────────────────────────
                if not DAYTIME_TRADING:
                    print(f"     [監視モード] 発注スキップ（DAYTIME_TRADING=False）")
                    continue
                if url_request and code not in ordered_today:
                    open_pos = db.load_open_positions()
                    if len(open_pos) >= MAX_DAYTIME_POSITIONS:
                        print(f"     ⚠️  最大ポジション数({MAX_DAYTIME_POSITIONS})到達 → 発注スキップ")
                    else:
                        shares = max(100, int(ORDER_AMOUNT // sig["price"] // 100) * 100)
                        result = tachibana_order.place_buy_order(url_request, code, shares)
                        if result["success"]:
                            tachibana_order.save_position(
                                code, name, shares, sig["price"], "D",
                                tp_pct=TP_PCT, sl_pct=SL_PCT,
                                entry_change_pct=sig["change_pct"],
                            )
                            ordered_today.add(code)
                            print(f"     ✅ 買い発注完了: {shares}株  {result['message']}")
                        else:
                            print(f"     ❌ 発注失敗: {result['message']}")

        # シグナル表示
        for code, name, sig in new_signals:
            flags = []
            if sig["breakout"]:      flags.append("ブレイクアウト")
            if sig["volume_surge"]:  flags.append(f"出来高{sig['volume_pace_ratio']:.1f}倍ペース")
            if sig["momentum"]:      flags.append(f"前日比+{sig['change_pct']:.1f}%")
            tp = round(sig["price"] * (1 + TP_PCT))
            sl = round(sig["price"] * (1 - SL_PCT))
            print(f"\n  🔔 [{now_str}] シグナル発報: {code} {name}")
            print(f"     条件: {' / '.join(flags)}")
            print(f"     現在値: {sig['price']:,.0f}円  前日高値: {sig['prev_high']:,.0f}円" if sig['prev_high'] else
                  f"     現在値: {sig['price']:,.0f}円")
            print(f"     TP目安: {tp:,.0f}円（+{TP_PCT*100:.0f}%）  SL目安: {sl:,.0f}円（-{SL_PCT*100:.0f}%）")

        # 定期ステータス表示（シグナルがない時間帯）
        if not new_signals:
            active = len(codes) - len(signaled)
            print(f"  [{now_str}] 監視中: {active}銘柄  発報済み: {len(signaled)}件", end="\r")

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

    signal_count = watch_loop(candidates, url_price, url_request, start_now)

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
