# -*- coding: utf-8 -*-
"""
position_monitor.py - ポジション監視・自動売り

out/positions.csv の "open" 行を15秒ごとに監視し、
TP または SL に達したら自動で成行売りを発注する。
15:00 に前日持越しポジションを強制決済、15:28 に終了。
（2026-08-14: 15:20→15:00に変更。詳細はFORCE_CLOSE_MINのコメント参照）
"""

import sys
import json
import os
import time
import urllib3
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import tachibana_order
import db

_BASE_DIR = Path(__file__).parent


def _is_trading_day(d):
    """土日・祝日・東証の年末年始休場を除く営業日か判定する（run_daily.pyのis_trading_dayと同ロジック）。"""
    if d.weekday() >= 5:
        return False
    if (d.month == 12 and d.day == 31) or (d.month == 1 and d.day in (2, 3)):
        return False
    try:
        import jpholiday
        if jpholiday.is_holiday(d):
            return False
    except ImportError:
        pass
    return True


def _prev_trading_day(today_str):
    """TODAY（"YYYY-MM-DD"）の直前の営業日を"YYYY-MM-DD"で返す。"""
    d = datetime.strptime(today_str, "%Y-%m-%d").date()
    d -= timedelta(days=1)
    while not _is_trading_day(d):
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")

POLL_INTERVAL      = 15
EXIT_HOUR          = 15
EXIT_MIN           = 28
JST                = timezone(timedelta(hours=9))
GAP_SELL_THRESHOLD = 1.0   # 翌朝始値が買値より+1%超 → 即売り
GAP_CHECK_END_MIN  = 9 * 60 + 15   # 9:15までギャップチェック
# 一回限りの手動早期決済リスト（2026-09-03: 戦略Bが 9983 ファストリ 100株@¥68,000=約¥680万を
# 建玉。週末を跨いで¥680万を保有するのは不安、とユーザー判断。通常の2泊満了(月曜)を待たず
# FORCE_CLOSE_MIN(15:00)で引け成行決済する。約定確認後、このセットは空に戻すこと）。
MANUAL_CLOSE_CODES = {"9983"}
FORCE_CLOSE_MIN    = 15 * 60 + 0   # 2026-08-14: 15:20→15:00に変更。戦略B(引け前15:05スキャン
                                   # →15:13-15:20発注)が、A/AN/AS/D当日分の強制決済(同じ仕組み
                                   # で処理される2泊満了のB自身の決済も含む)と資金を取り合わない
                                   # よう、Bの発注ウィンドウより前に確実に資金を解放する。
                                   # intraday_pricesの実測(A/AN/AS/D強制決済92件中31件でデータ
                                   # 照合: 15:00頃→15:20頃の変化率は平均-0.046%・標準偏差0.346%
                                   # とごく僅かで、20分早めることによる価格面の悪影響は限定的
                                   # と判断（B戦略自体はN=2と少なく未検証だが、同じ市場性質が
                                   # 働くと推測）。
                                   # ※15:25は東証がオークションモードに切り替わり成行注文不可
                                   # (11481エラー)のため、15:00なら十分な安全マージンがある。
DASHBOARD_SYNC_INTERVAL_SEC = 600  # ダッシュボード向けに10分おきintraday_prices保存+S3同期


def log_exit_detection(code, name, strategy, reason_type, detect_price, target_price, now):
    """2026-08-24追加: TP/SL条件を検知した瞬間(発注前)の時刻・価格を、
    実際の約定時刻(positions.sell_time)と突き合わせてスリッページの原因
    (検知の遅れか、発注〜約定の執行ラグか)を後日切り分けられるよう、
    複数日分蓄積してlogs/に記録する。"""
    try:
        log_dir = _BASE_DIR / "logs"
        log_dir.mkdir(exist_ok=True)
        path = log_dir / "exit_detection.log"
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{date_str},{time_str},{code},{name},{strategy},"
                    f"{reason_type},{detect_price},{target_price}\n")
    except Exception:
        pass


def load_positions():
    db.init_db()
    return db.load_open_positions()


def save_all_positions(updated_list):
    """updated_list にあるポジションを DB へ上書き保存する。"""
    for p in updated_list:
        sell_price = p.get("sell_price")
        pnl_pct    = p.get("pnl_pct")
        db.update_position(
            p["date"], p["code"],
            status     = p.get("status", "open"),
            sell_price = float(sell_price) if sell_price not in ("", None) else None,
            sell_time  = p.get("sell_time") or None,
            pnl_pct    = float(pnl_pct)    if pnl_pct    not in ("", None) else None,
            exit_reason= p.get("exit_reason", ""),
        )


def fetch_prices(url_price, codes):
    """Tachibana API で複数銘柄の現在値・始値を一括取得。
    戻り値: {code: {"price": 現在値, "open": 始値orNone}}
    """
    if not url_price or not codes:
        return {}
    http    = urllib3.PoolManager()
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
            f'"p_no":"{tachibana_order._next_p_no()}",'
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
            tachibana_order.log_api_call("position_monitor.fetch_prices")
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


def _confirm_sell_price(url_request, pos, result, buy_px, now):
    """売り注文成功後、実際の約定単価で pos["sell_price"]/["pnl_pct"] を補正する（2026-08-15、Fix③）。

    CLMOrderListDetailを発注直後に照会するため数秒かかる。取得できない場合は
    気配値のままフォールバックする（呼び出し元のpos内容は変更しない）。
    DBへの保存は呼び出し元の既存の save_all_positions() 経路に任せる。
    """
    order_no = result.get("order_no")
    if not order_no:
        return
    eigyou_day = (result.get("raw") or {}).get("sEigyouDay") or now.strftime("%Y%m%d")
    true_price = tachibana_order.get_true_execution_price(url_request, order_no, eigyou_day)
    if true_price and buy_px:
        true_chg = (true_price - buy_px) / buy_px * 100
        old_price = pos["sell_price"]
        pos["sell_price"] = str(true_price)
        pos["pnl_pct"]    = str(round(true_chg, 2))
        print(f"  🔧 [{pos['code']}] {pos['name']}: 約定単価を補正: {float(old_price):.1f}円 → "
              f"{true_price:.1f}円  損益: {true_chg:+.2f}%")


def load_url_price():
    try:
        from pathlib import Path
        login_file = Path(__file__).parent / "tachibana_login_response.json"
        with open(login_file, encoding="utf-8") as f:
            return json.load(f).get("sUrlPrice") or None
    except Exception:
        return None


def fetch_api_holdings(url_request):
    """Tachibana APIから実保有株（現物＋信用）を取得して返す。
    戻り値: {code: {...}} or None（API呼び出し失敗時 — 照合スキップ用）
    どちらか一方でもAPIエラーになった場合は None を返す（誤closeを防ぐため）。
    """
    import urllib3, json as _json
    http = urllib3.PoolManager()
    holdings = {}

    for clm_id, list_key, shares_key, price_key, acct in [
        ("CLMKabuZanList",     "aCLMKabuZanList",     "sZansuuKabu",       "sHiritsuTanka",   "genbutsu"),
        ("CLMShinyouZanList",  "aCLMShinyouZanList",  "sZandakaZansuuKabu","sTategyokuTanka", "shinyou"),
    ]:
        time.sleep(0.6)   # p_sd_date競合（p_errno=6/8）を避けるため他プロセスと時間をずらす
        t = datetime.now(JST)
        p_sd = (f"{t.year}.{t.month:02}.{t.day:02}"
                f"-{t.hour:02}:{t.minute:02}:{t.second:02}"
                f".{t.microsecond // 1000:03}")
        params = (
            "{"
            f'"p_no":"{tachibana_order._next_p_no()}",'
            f'"p_sd_date":"{p_sd}",'
            f'"sCLMID":"{clm_id}",'
            '"sJsonOfmt":"5"'
            "}"
        )
        try:
            resp   = http.request("GET", url_request + "?" + params,
                                  timeout=urllib3.Timeout(connect=5, read=10))
            tachibana_order.log_api_call("position_monitor.fetch_api_holdings")
            result = _json.loads(resp.data.decode("shift-jis", errors="ignore"))
            if result.get("p_errno", "0") != "0":
                print(f"  ⚠️ {clm_id} APIエラー p_errno={result.get('p_errno')} → 照合スキップ")
                return None   # エラー時は None を返して誤close防止
            for item in result.get(list_key, []):
                def _f(k):
                    v = item.get(k, "")
                    if isinstance(v, str): v = v.strip('"')
                    try: return float(v)
                    except: return None
                code   = str(item.get("sIssueCode","")).strip('"')
                name   = str(item.get("sIssueName","")).strip('"')
                shares = int(_f(shares_key) or 0)
                price  = _f(price_key)
                if code and shares > 0:
                    holdings[code] = {"name": name, "shares": shares,
                                      "buy_price": price, "account_type": acct}
        except Exception as e:
            print(f"  ⚠️ {clm_id} 通信エラー: {e} → 照合スキップ")
            return None   # 通信エラーも None で照合スキップ
    return holdings


def reconcile_db_with_api(url_request):
    """起動時にAPIの実保有株とDBを照合し、ズレがあれば修正する。"""
    print(f"\n  🔍 API保有株照合中...")
    api_holdings = fetch_api_holdings(url_request)

    if api_holdings is None:
        print(f"  ⚠️ API照合スキップ（セッション未確立）")
        return

    db_positions = db.load_open_positions()
    db_codes     = {p["code"] for p in db_positions}
    api_codes    = set(api_holdings.keys())

    # DBにあってAPIにない → 実際には売済み → DBをclosedに更新
    stale = db_codes - api_codes
    if stale:
        print(f"  ⚠️ DBのみ存在（実際は未保有）: {stale} → DBをcloseします")
        for p in db_positions:
            if p["code"] in stale:
                db.update_position(
                    p["date"], p["code"],
                    status      = "closed",
                    sell_price  = float(p["buy_price"]),
                    sell_time   = datetime.now(JST).strftime("%H:%M:%S"),
                    pnl_pct     = 0.0,
                    exit_reason = "api_reconcile",
                )
                print(f"     [{p['code']}] {p['name']} → closed（API不一致）")

    # APIにあってDBにない → DB未登録の保有株 → 警告のみ
    new_in_api = api_codes - db_codes
    if new_in_api:
        print(f"  ⚠️ APIのみ存在（DB未登録）: {new_in_api}")
        for code in new_in_api:
            h = api_holdings[code]
            print(f"     [{code}] {h['name']} {h['shares']}株 "
                  f"取得単価:{h['buy_price']}円 [{h['account_type']}]"
                  f" ← position_monitor監視対象外")

    # 一致している銘柄
    matched = db_codes & api_codes
    if matched:
        print(f"  ✅ DB/API一致: {len(matched)}件 "
              f"({', '.join(sorted(matched))})")

        # 2026-08-11追加: 取得単価(sHiritsuTanka、分割約定も加味した真の平均取得価格)
        # とDBのbuy_priceにズレがあれば補正する。成行注文は発注直前の気配値を
        # 暫定的にbuy_priceとして記録しているため、約定までのタイムラグや
        # 分割約定でズレることがある（ナガホリ8139の実例で発見。約20円/約1%）。
        # TP/SLもbuy_priceからの比率を保ったまま再計算する。
        corrected = 0
        for p in db_positions:
            code = p["code"]
            if code not in matched:
                continue
            h = api_holdings[code]
            true_price = h.get("buy_price")
            old_price  = float(p["buy_price"]) if p.get("buy_price") not in (None, "") else 0
            if not true_price or old_price <= 0:
                continue
            if abs(true_price - old_price) < 0.5:   # 誤差50銭未満は無視
                continue
            update_fields = {"buy_price": true_price}
            if p.get("tp_price") not in (None, ""):
                tp_pct = float(p["tp_price"]) / old_price - 1
                update_fields["tp_price"] = round(true_price * (1 + tp_pct))
            if p.get("sl_price") not in (None, ""):
                sl_pct = 1 - float(p["sl_price"]) / old_price
                update_fields["sl_price"] = round(true_price * (1 - sl_pct))
            if h.get("name"):
                update_fields["name"] = h["name"]
            db.update_position(p["date"], code, **update_fields)
            corrected += 1
            print(f"     🔧 [{code}] {h.get('name') or p['name']}: "
                  f"buy_price {old_price:.1f}円 → {true_price:.1f}円 に補正（TP/SLも再計算）")
        if corrected:
            print(f"  🔧 取得単価の補正: {corrected}件")

    if not stale and not new_in_api:
        print(f"  ✅ DB/API完全一致（{len(matched)}件）")


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

    reconcile_db_with_api(url_request)

    TODAY       = datetime.now(JST).strftime("%Y-%m-%d")
    gap_checked  = set()   # 当日ギャップチェック済みのcode
    force_closed = False   # 15:25 引け決済実施フラグ
    ROUTINE_LOG_INTERVAL   = 300   # 通常ステータス（価格・TP/SL・価格取得失敗）は5分に1回だけ出力（ノイズ削減）
    last_routine_print     = {}    # code -> 直近出力時刻（監視自体は15秒間隔のまま変更なし）
    last_no_position_print = None  # 「ポジションなし」待機メッセージの直近出力時刻
    last_dashboard_sync    = None  # ダッシュボード向けintraday_prices保存+S3同期の直近実施時刻
    while True:
        now     = datetime.now(JST)
        now_min = now.hour * 60 + now.minute

        # ── 15:00: 引け成行で強制決済（2026-08-14: 15:20→15:00、詳細はFORCE_CLOSE_MIN参照） ──
        # ①前々日以前に買ったポジション（戦略B最適化2026-06-23: 2泊保有のため「前日」→「前々日以前」に変更）
        # ②当日買った戦略A/D/AN/ASのポジション（2026-07-28追加、2026-08-10にAN・ASを追加）
        #   戦略A/D/AN/ASは simulate_precise.py 系のバックテストが「買った当日1日分の値動きのみ」で
        #   TP/SLを検証しており、翌日以降への持ち越しはバックテスト対象外（想定外のオーバーナイトリスク）。
        #   従来はTODAY分が対象外で、TP/SL未達のまま翌営業日の本ブロックまで無自覚に持ち越されていた。
        #   戦略B（引け前逆張り）は2泊保有が設計通りのため対象外のまま。
        # 2026-08-05: 上記コメント通りの「前々日以前」に実装されておらず、
        #   実際には p["date"] < TODAY （前日を含む）のままだったため、戦略Bが
        #   意図した2泊ではなく1泊で強制決済されていたバグを修正（2216/3407が該当）。
        if not force_closed and now_min >= FORCE_CLOSE_MIN:
            force_closed = True
            prev_trading_day = _prev_trading_day(TODAY)
            prev_positions = [
                p for p in load_positions()
                if p["date"] < prev_trading_day
                or (p["date"] == TODAY and p.get("strategy") in ("A", "D", "AN", "AS"))
                or p["code"] in MANUAL_CLOSE_CODES
            ]
            if prev_positions:
                print(f"\n  ⏰ 15:00 引け決済開始（{len(prev_positions)}件）")
                updated = []
                to_confirm = []   # (pos, result, buy_px) 発注完了後にまとめて約定単価を確認する（Fix③）
                for pos in prev_positions:
                    code  = pos["code"]
                    name  = pos["name"]
                    shares = int(pos["shares"])
                    acct = pos.get("account_type") or "genbutsu"
                    zc   = pos.get("zyoutoeki_c") or None
                    tag  = "当日分" if pos["date"] == TODAY else "前日以前"
                    print(f"  📤 引け成行売り: [{code}] {name} {shares}株 [{acct}]（{tag}）")
                    mkt_code = db.get_market_code_db(code)
                    result = tachibana_order.place_sell_order(url_request, code, shares, market_code=mkt_code, account_type=acct, zyoutoeki_c=zc)
                    if result["success"]:
                        data    = fetch_prices(url_price, [code])
                        current = (data.get(code) or {}).get("price") or float(pos["buy_price"])
                        chg     = (current - float(pos["buy_price"])) / float(pos["buy_price"]) * 100
                        pos["status"]      = "closed"
                        pos["sell_price"]  = str(current)
                        pos["sell_time"]   = now.strftime("%H:%M:%S")
                        pos["pnl_pct"]     = str(round(chg, 2))
                        pos["exit_reason"] = "force_close"
                        print(f"  ✅ {result['message']}  損益: {chg:+.2f}%（引け決済）")
                        to_confirm.append((pos, result, float(pos["buy_price"])))
                    else:
                        print(f"  ❌ 引け決済失敗: {result['message']}")
                        if "11482" in result["message"]:
                            db.add_restriction(code, "11482")
                            print(f"  🚫 {code} {name}: 日計取引制限（11482）→ ブラックリスト登録（30日間）")
                    updated.append(pos)

                # 全銘柄の発注完了後にまとめて約定単価を確認・補正する（Fix③、2026-08-15追加）。
                # 15:25のオークションモード切替までの発注ウィンドウを優先し、
                # 1件ずつ発注の合間に確認を挟んで遅延させることはしない。
                if to_confirm:
                    print(f"  🔧 約定単価の確認・補正を開始（{len(to_confirm)}件）")
                    for pos, result, buy_px in to_confirm:
                        _confirm_sell_price(url_request, pos, result, buy_px, now)
                save_all_positions(updated)

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
            if last_no_position_print is None or (now - last_no_position_print).total_seconds() >= ROUTINE_LOG_INTERVAL:
                last_no_position_print = now
                print(f"  [{now.strftime('%H:%M:%S')}] ポジションなし → 60秒待機（closing_watch発注待ち）")
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
                last_t = last_routine_print.get(code)
                if last_t is None or (now - last_t).total_seconds() >= ROUTINE_LOG_INTERVAL:
                    last_routine_print[code] = now
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
                    mkt_code = db.get_market_code_db(code)
                    acct = pos.get("account_type") or "genbutsu"
                    zc   = pos.get("zyoutoeki_c") or None
                    result = tachibana_order.place_sell_order(url_request, code, int(pos["shares"]), market_code=mkt_code, account_type=acct, zyoutoeki_c=zc)
                    if result["success"]:
                        pos["status"]     = "closed"
                        pos["sell_price"] = str(open_px)
                        pos["sell_time"]  = now.strftime("%H:%M:%S")
                        pos["pnl_pct"]    = str(round(gap_pct, 2))
                        pos["exit_reason"] = "gap_sell"
                        changed = True
                        print(f"  ✅ {result['message']}  損益: {gap_pct:+.2f}% (ギャップ売り)")
                        _confirm_sell_price(url_request, pos, result, buy_px, now)
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
                log_exit_detection(code, name, strategy, "TP" if hit_tp else "SL",
                                    current, tp_px if hit_tp else sl_px, now)
                mkt_code = db.get_market_code_db(code)
                acct = pos.get("account_type") or "genbutsu"
                zc   = pos.get("zyoutoeki_c") or None
                result = tachibana_order.place_sell_order(url_request, code, int(pos["shares"]), market_code=mkt_code, account_type=acct, zyoutoeki_c=zc)
                if result["success"]:
                    pos["status"]      = "closed"
                    pos["sell_price"]  = str(current)
                    pos["sell_time"]   = now.strftime("%H:%M:%S")
                    pos["pnl_pct"]     = str(round(chg, 2))
                    pos["exit_reason"] = "TP" if hit_tp else "SL"
                    changed = True
                    emoji = "✅" if hit_tp else "🛑"
                    print(f"  {emoji} {result['message']}  損益: {chg:+.2f}%")
                    _confirm_sell_price(url_request, pos, result, buy_px, now)
                else:
                    print(f"  ❌ 売り失敗: {result['message']}")
                    if "11482" in result["message"]:
                        db.add_restriction(code, "11482")
                        print(f"  🚫 {code} {name}: 日計取引制限（11482）→ ブラックリスト登録（30日間）")
            else:
                last_t = last_routine_print.get(code)
                if last_t is None or (now - last_t).total_seconds() >= ROUTINE_LOG_INTERVAL:
                    last_routine_print[code] = now
                    bar = "▲" if chg >= 0 else "▼"
                    print(f"  [{now.strftime('%H:%M:%S')}] {code} {name}: "
                          f"{current}円 {bar}{abs(chg):.2f}%  "
                          f"TP:{tp_px:.0f} SL:{sl_px:.0f}")

        if changed:
            save_all_positions(positions)

        # ── ダッシュボード向け: 10分おきに現在値をintraday_pricesへ保存しS3同期 ──
        if (last_dashboard_sync is None
                or (now - last_dashboard_sync).total_seconds() >= DASHBOARD_SYNC_INTERVAL_SEC):
            last_dashboard_sync = now
            snap_time = now.strftime("%H:%M:%S")
            for pos in positions:
                data = prices_data.get(pos["code"])
                if data and data.get("price"):
                    try:
                        db.save_intraday_snapshot(pos["code"], TODAY, snap_time, data["price"])
                    except Exception as e:
                        print(f"  ⚠️  ダッシュボード用スナップショット保存失敗 [{pos['code']}]: {e}")
            if tachibana_order.LIVE_TRADING:
                try:
                    import boto3 as _boto3
                    _s3 = _boto3.client("s3")
                    _db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "stock.db")
                    _s3.upload_file(_db_path, os.getenv("S3_BUCKET", "shibuya8020"), "stock-db/stock.db")
                    print(f"  ☁️  ダッシュボード向けS3同期完了（{snap_time}）")
                except Exception as e:
                    print(f"  ⚠️  ダッシュボード向けS3同期失敗: {e}")

        remaining_sec = (EXIT_HOUR * 60 + EXIT_MIN - now_min) * 60 - now.second
        time.sleep(min(POLL_INTERVAL, max(1, remaining_sec)))

    print(f"  📊 position_monitor 終了: {datetime.now(JST).strftime('%H:%M:%S')}")

    # ── S3にDBを即時バックアップ ──
    # 16:45のscan_daily.py自動実行はJ-Quantsの当日データ配信待ちのためのタイミングで、
    # バックアップとは無関係。run_daily.pyのターミナルが16:45より前に閉じられると
    # 本日分の建玉データが一度もS3に上がらず、翌朝のdownload_db.pyで消えてしまうため、
    # position_monitor終了（ほぼ確実に完走する15:28前後）の直後にも独立してバックアップする。
    # モックモード（LIVE_TRADING=False）ではモックポジションで共有S3のDBを汚染しないようアップロードしない。
    if not tachibana_order.LIVE_TRADING:
        print(f"  🔧 [モックモード] S3バックアップはスキップします（LIVE_TRADING=False）")
    else:
        try:
            import boto3 as _boto3
            _s3 = _boto3.client("s3")
            _db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "stock.db")
            _s3.upload_file(_db_path, os.getenv("S3_BUCKET", "shibuya8020"), "stock-db/stock.db")
            print(f"  ☁️  S3バックアップ完了（引け後・建玉データ保護）")
        except Exception as e:
            print(f"  ⚠️  S3バックアップ失敗（16:45の自動アップロードに期待）: {e}")

    # 16:30まで待機してyfinanceで当日シグナルを検証
    now = datetime.now(JST)
    verify_time = now.replace(hour=16, minute=30, second=0, microsecond=0)
    wait_sec = (verify_time - now).total_seconds()
    if wait_sec > 0:
        print(f"\n  ⏳ 16:30に当日シグナル検証を開始します（{int(wait_sec//60)}分{int(wait_sec%60)}秒後）")
        time.sleep(wait_sec)
    print(f"\n  📋 当日シグナル検証開始...")
    try:
        import verify_signals
        verify_signals.verify(live=True)
    except Exception as e:
        print(f"  ⚠️  検証エラー: {e}")


if __name__ == "__main__":
    main()
