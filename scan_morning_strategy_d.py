# -*- coding: utf-8 -*-
"""
scan_morning_strategy_d.py - 戦略D（大型株マクロ連動）スキャン

scan_morning.py から分割。今後ほぼ変更不要な安定ロジック。
銅/原油/ドル円/半導体ETFの動きから恩恵を受ける大型株を提示する。
"""

import numpy as np

from scan_morning_market_data import MACRO_SECTORS
from scan_morning_tachibana_session import _tprice, fetch_stock_hist


def calc_stop_loss(buy_price, pct=-5.0):
    """ストップロス価格を計算する。"""
    return round(buy_price * (1 + pct / 100))


def calc_score_d(hist):
    """大型株向けスコアリング（戦略D専用）。

    採点基準（最大10点）:
      直近5日騰落トレンド : 0〜4点（連続上昇傾向）
      MA5 vs MA25乖離    : 0〜3点（短期MAが長期MAを上回っているか）
      直近1日騰落        : 0〜3点（前日の動きがプラスか）
    """
    if len(hist) < 26:
        return None

    closes = hist["Close"].values
    score  = 0

    # 直近5日の価格トレンド（0〜4点）
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

    # MA5 vs MA25 乖離（0〜3点）
    ma5     = closes[-5:].mean()
    ma25    = closes[-25:].mean()
    ma_diff = (ma5 / ma25 - 1) * 100

    if ma_diff >= 2.0:
        score += 3
    elif ma_diff >= 0.5:
        score += 2
    elif ma_diff >= 0.0:
        score += 1

    # 直近1日騰落（0〜3点）
    prev    = float(closes[-2])
    today   = float(closes[-1])
    day_chg = (today - prev) / prev * 100

    if day_chg >= 2.0:
        score += 3
    elif day_chg >= 0.5:
        score += 2
    elif day_chg >= 0.0:
        score += 1

    return {"score": round(score, 2), "ratio": round(ma_diff, 2)}


def scan_strategy_d(macro, condition, tachibana_prices=None):
    """マクロ指標連動の大型株をスコアリング付きで提示する。"""
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
        val       = macro.get(key)
        val_prev  = macro.get(f"{key}_prev")

        if val is None:
            continue

        hit_today = (direction == "up"   and val >= threshold) or \
                    (direction == "down" and val <= threshold)
        hit_prev  = False
        lag_note  = ""
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
        sign    = "+" if val >= 0 else ""
        label   = sector["label"]
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
