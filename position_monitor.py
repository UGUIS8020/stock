# -*- coding: utf-8 -*-
"""
position_monitor.py - ポジション監視・自動売り

out/positions.csv の "open" 行を15秒ごとに監視し、
TP または SL に達したら自動で成行売りを発注する。
15:25 になるか全ポジションがクローズされたら終了。
"""

import sys
import csv
import json
import os
import time
import urllib3
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
import tachibana_order

POSITIONS_CSV      = "out/positions.csv"
POLL_INTERVAL      = 15
EXIT_HOUR          = 15
EXIT_MIN           = 28
JST                = timezone(timedelta(hours=9))
FIELDNAMES         = tachibana_order._POSITIONS_FIELDS
GAP_SELL_THRESHOLD = 1.0   # 翌朝始値が買値より+1%超 → 即売り
GAP_CHECK_END_MIN  = 9 * 60 + 15  # 9:15までギャップチェック


def load_positions():
    if not os.path.exists(POSITIONS_CSV):
        return []
    with open(POSITIONS_CSV, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row.get("status") == "open"]


def save_all_positions(updated_list):
    """updated_list にあるポジションを positions.csv 内の該当行へ上書き保存する。"""
    if not os.path.exists(POSITIONS_CSV):
        return
    with open(POSITIONS_CSV, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    lookup = {(p["date"], p["code"]): p for p in updated_list}
    result = []
    for row in all_rows:
        key = (row["date"], row["code"])
        result.append(lookup.get(key, row))

    with open(POSITIONS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(result)


def fetch_prices(url_price, codes):
    """Tachibana API で複数銘柄の現在値・始値を一括取得。
    戻り値: {code: {"price": 現在値, "open": 始値orNone}}
    """
    if not url_price or not codes:
        return {}
    http    = urllib3.PoolManager()
    p_base  = int(time.time()) % 100000
    prices  = {}
    batch   = 50
    for i in range(0, len(codes), batch):
        chunk = codes[i:i + batch]
        t     = datetime.now(JST)
        p_sd  = (f"{t.year}.{t.month:02}.{t.day:02}"
                 f"-{t.hour:02}:{t.minute:02}:{t.second:02}"
                 f".{t.microsecond // 1000:03}")
        params = (
            "{"
            f'"p_no":"{p_base + i}",'
            f'"p_sd_date":"{p_sd}",'
            '"sCLMID":"CLMMfdsGetMarketPrice",'
            f'"sTargetIssueCode":"{",".join(chunk)}",'
            '"sTargetColumn":"pDPP,pPRP,pOP",'
            '"sJsonOfmt":"5"'
            "}"
        )
        try:
            resp   = http.request("GET", url_price + "?" + params,
                                  timeout=urllib3.Timeout(connect=5, read=10))
            result = json.loads(resp.data.decode("shift-jis", errors="ignore"))
            for item in result.get("aCLMMfdsMarketPrice", []):
                code = item.get("sIssueCode", "").strip('"')
                def _f(k):
                    v = item.get(k, "")
                    try:
                        return float(str(v).strip('"'))
                    except (ValueError, TypeError):
                        return None
                px   = _f("pDPP") or _f("pPRP")
                open_px = _f("pOP")
                if px and code:
                    prices[code] = {"price": px, "open": open_px}
        except Exception as e:
            print(f"  ⚠️ 価格取得エラー: {e}")
    return prices


def load_url_price():
    try:
        with open("tachibana_login_response.json", encoding="utf-8") as f:
            return json.load(f).get("sUrlPrice") or None
    except Exception:
        return None


def main():
    print(f"\n{'═'*56}")
    print(f"  📊 position_monitor 起動")
    print(f"     監視開始: {datetime.now(JST).strftime('%H:%M:%S')}")
    print(f"     終了予定: {EXIT_HOUR:02}:{EXIT_MIN:02}")
    print(f"{'═'*56}")

    url_price   = load_url_price()
    url_request = tachibana_order.load_url_request()
    mode_str    = "本番" if tachibana_order.LIVE_TRADING else "モック"
    print(f"  モード: {mode_str}  |  価格取得: {'Tachibana' if url_price else '価格なし（要ログイン）'}")

    if not url_request:
        print("  ⚠️ 業務URLが取得できません。タスクを終了します。")
        return

    gap_checked = set()  # 当日ギャップチェック済みのcode

    while True:
        now     = datetime.now(JST)
        now_min = now.hour * 60 + now.minute
        if now_min >= EXIT_HOUR * 60 + EXIT_MIN:
            print(f"\n  ⏰ {EXIT_HOUR}:{EXIT_MIN:02} 到達 → 監視終了")
            break

        positions = load_positions()
        if not positions:
            # 15:25以降はclosing_watchの発注も終わっているので終了
            if now_min >= 15 * 60 + 25:
                print(f"  [{now.strftime('%H:%M:%S')}] ポジションなし・発注締め切り済み → 終了")
                break
            # それ以前はclosing_watchが買う可能性があるため待機継続
            time.sleep(60)
            continue

        codes       = [p["code"] for p in positions]
        prices_data = fetch_prices(url_price, codes)

        changed = False
        for pos in positions:
            code    = pos["code"]
            name    = pos["name"]
            buy_px  = float(pos["buy_price"])
            tp_px   = float(pos["tp_price"])
            sl_px   = float(pos["sl_price"])
            strategy = pos.get("strategy", "")
            data    = prices_data.get(code)

            if data is None:
                print(f"  [{now.strftime('%H:%M:%S')}] {code} {name}: 価格取得できず")
                continue

            current = data["price"]
            open_px = data.get("open")

            # ── ギャップ売り判定（9:00〜9:15 / 戦略B / 未チェック） ──
            if (strategy == "B"
                    and code not in gap_checked
                    and now_min < GAP_CHECK_END_MIN
                    and open_px and open_px > 0):
                gap_checked.add(code)
                gap_pct = (open_px - buy_px) / buy_px * 100
                if gap_pct > GAP_SELL_THRESHOLD:
                    print(f"  🚀 {code} {name}: 翌朝ギャップ+{gap_pct:.2f}% → 即売り（シミュ検証済み）")
                    result = tachibana_order.place_sell_order(url_request, code, int(pos["shares"]))
                    if result["success"]:
                        pos["status"]     = "closed"
                        pos["sell_price"] = str(open_px)
                        pos["sell_time"]  = now.strftime("%H:%M:%S")
                        pos["pnl_pct"]    = str(round(gap_pct, 2))
                        changed = True
                        print(f"  ✅ {result['message']}  損益: {gap_pct:+.2f}% (ギャップ売り)")
                    else:
                        print(f"  ❌ ギャップ売り失敗: {result['message']}")
                    continue
                else:
                    print(f"  [{now.strftime('%H:%M:%S')}] {code} {name}: "
                          f"ギャップ{gap_pct:+.2f}% → 閾値以下、通常監視へ")

            chg    = (current - buy_px) / buy_px * 100
            hit_tp = current >= tp_px
            hit_sl = current <= sl_px

            if hit_tp or hit_sl:
                reason = f"TP({tp_px:.0f}円)" if hit_tp else f"SL({sl_px:.0f}円)"
                print(f"  {'🎯' if hit_tp else '🛑'} {code} {name}: "
                      f"{reason} 到達 ({current}円 {chg:+.2f}%) → 売り発注")
                result = tachibana_order.place_sell_order(url_request, code, int(pos["shares"]))
                if result["success"]:
                    pos["status"]     = "closed"
                    pos["sell_price"] = str(current)
                    pos["sell_time"]  = now.strftime("%H:%M:%S")
                    pos["pnl_pct"]    = str(round(chg, 2))
                    changed = True
                    emoji = "✅" if hit_tp else "🛑"
                    print(f"  {emoji} {result['message']}  損益: {chg:+.2f}%")
                else:
                    print(f"  ❌ 売り失敗: {result['message']}")
            else:
                bar = "▲" if chg >= 0 else "▼"
                print(f"  [{now.strftime('%H:%M:%S')}] {code} {name}: "
                      f"{current}円 {bar}{abs(chg):.2f}%  "
                      f"TP:{tp_px:.0f} SL:{sl_px:.0f}")

        if changed:
            save_all_positions(positions)

        remaining_sec = (EXIT_HOUR * 60 + EXIT_MIN - now_min) * 60 - now.second
        time.sleep(min(POLL_INTERVAL, max(1, remaining_sec)))

    print(f"  📊 position_monitor 終了: {datetime.now(JST).strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
