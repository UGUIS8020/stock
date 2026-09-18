"""
strategy_n.py - 戦略N（ナンピン戦略）パイロット: 日本精工・中部電力の2銘柄限定

【方式】(analyze/screen_nanpin_candidates.py でバックテスト・検証済み)
    20日移動平均を終値が下回っている間、暦週(ISO週)に1回100株ずつ買い増しし、
    平均取得単価×(1+TP_PCT%)まで戻ったら保有全株を成行で一括売却する。
    損切りルールなし（意図的な設計。投入資金の上限=MAX_CAMPAIGN_CAPITALのみが
    歯止め）。既存の戦略A/AN/AS/B/Dとは別の独立したDBテーブル
    (nanpin_campaigns/nanpin_buys)で管理し、既存のpositionsテーブル
    (PRIMARY KEY (date, code)、単発売買前提)には一切触れない。

【既知の制約（パイロット段階での意図的な簡略化）】
    売却判定は1日1回・実行時点の現在値のみで行う。バックテストは「日中の
    高値がTP到達したら約定」を前提にしているため、日中に一瞬TPへ到達して
    その後押し戻された場合はここでは検知できない（保守的な簡略化）。

実行方法:
    python strategy_n.py        # 15:00まで待機して自動開始
    python strategy_n.py --now  # 即時開始（テスト・手動確認用）

前提:
    - tachibana_login_response.json が存在すること（scan_morning.py 実行後）
    - パイロット期間中は scan_morning.py からは起動しない。手動実行専用。
"""

import sys
import json
import time
import argparse
import urllib3
import tachibana_order
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()
import db

JST = timezone(timedelta(hours=9))
_BASE_DIR = Path(__file__).parent
TACHIBANA_LOGIN_FILE = str(_BASE_DIR / "tachibana_login_response.json")

# ── パイロット対象銘柄（2026-09-18時点、固定） ──
CANDIDATES = [("6471", "日本精工"), ("9502", "中部電力")]

TREND_N   = 20          # 移動平均日数（バックテストと同じ）
TP_PCT    = 2.0          # 平均取得単価+2.0%で全売却（バックテスト中間値、調整可）
BUY_SHARES = 100
MAX_CAMPAIGN_CAPITAL = 2_000_000   # 1銘柄あたりの投入上限(円)。ユーザー確定値
MAX_BUYS_HARD_CAP    = 20          # 資金上限とは独立のバックストップ

ACT_HOUR, ACT_MIN = 15, 0   # closing_watch.pyのSCAN_STARTと同じタイミングに合わせる
SELL_POLL_INTERVAL_SEC = 900  # 2026-09-18追加: 15:00までの間、この間隔で売却判定だけ繰り返す
                               # （買い判定は引き続き15:00頃の1回のみ。日中の一瞬のTP到達を
                               # 拾い漏らす既知の制約を緩和するための追加、position_monitor.py
                               # と同様のポーリング方式。15分間隔）


def load_tachibana_url():
    try:
        with open(TACHIBANA_LOGIN_FILE, encoding="utf-8") as f:
            data = json.load(f)
        url = data.get("sUrlPrice", "")
        return url if url else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def fetch_prices(url_price, codes):
    """closing_watch.py の _fetch_price_batch と同じ方式で現在値を一括取得する。"""
    http = urllib3.PoolManager(cert_reqs="CERT_NONE")
    code_list = ",".join(str(c) for c in codes)
    t = datetime.now(JST)
    p_sd_date = (f"{t.year}.{t.month:02}.{t.day:02}"
                 f"-{t.hour:02}:{t.minute:02}:{t.second:02}"
                 f".{t.microsecond // 1000:03}")
    params = (
        "{"
        f'"p_no":"{tachibana_order._next_p_no()}",'
        f'"p_sd_date":"{p_sd_date}",'
        '"sCLMID":"CLMMfdsGetMarketPrice",'
        f'"sTargetIssueCode":"{code_list}",'
        '"sTargetColumn":"pDPP,pPRP",'
        '"sJsonOfmt":"5"'
        "}"
    )
    try:
        resp = http.request("GET", url_price + "?" + params,
                            timeout=urllib3.Timeout(connect=3, read=8))
        tachibana_order.log_api_call("strategy_n.fetch_prices")
        result = json.loads(resp.data.decode("shift-jis", errors="ignore"))
        out = {}
        for item in result.get("aCLMMfdsMarketPrice", []):
            code = item.get("sIssueCode", "").strip('"')
            if not code:
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
            if price:
                out[code] = price
        return out
    except Exception:
        return {}


def calc_ma20(code, current_price):
    """直近19日分の終値(daily_prices)＋当日の現在値の20点で移動平均を出す。
    データ不足時はNoneを返す。"""
    hist = db.get_stock_history(code, days=TREND_N - 1 + 5)
    if hist is None or len(hist) < TREND_N - 1:
        return None
    closes = hist["Close"].dropna().tolist()[-(TREND_N - 1):]
    if len(closes) < TREND_N - 1:
        return None
    return (sum(closes) + current_price) / TREND_N


def try_sell(code, name, current_price, url_request):
    """保有中キャンペーンがあり、TP到達していれば全株成行売却する。"""
    campaign = db.get_open_nanpin_campaign(code)
    if not campaign or campaign["shares_held"] <= 0:
        return False

    avg_cost = campaign["total_cost"] / campaign["shares_held"]
    sell_target = avg_cost * (1 + TP_PCT / 100)
    if current_price < sell_target:
        print(f"  {code} {name}: 保有中 {campaign['shares_held']}株  "
              f"平均取得単価{avg_cost:,.1f}円  目標{sell_target:,.1f}円  "
              f"現在値{current_price:,.1f}円 → 未到達")
        return False

    shares = campaign["shares_held"]
    mkt_code = db.get_market_code_db(code)
    mode_str = "本番" if tachibana_order.LIVE_TRADING else "モック"
    print(f"  🟢 {code} {name}: 目標{sell_target:,.1f}円到達（現在値{current_price:,.1f}円）"
          f" → 全株成行売却 [{mode_str}]")
    result = tachibana_order.place_sell_order(
        url_request, code, shares, market_code=mkt_code,
        account_type=campaign.get("account_type", "genbutsu"),
        zyoutoeki_c=campaign.get("zyoutoeki_c", "1"),
    )
    if not result["success"]:
        print(f"  ❌ 売却失敗: {result['message']}")
        return False

    pnl_yen = current_price * shares - campaign["total_cost"]
    pnl_pct = pnl_yen / campaign["total_cost"] * 100
    now_str = datetime.now(JST).strftime("%H:%M:%S")
    db.close_nanpin_campaign(
        campaign["campaign_id"], sell_price=current_price, sell_time=now_str,
        sell_order_no=result.get("order_no"),
        realized_pnl_yen=pnl_yen, realized_pnl_pct=pnl_pct,
    )
    return True


def try_buy(code, name, current_price, url_request):
    """下降トレンド中かつ当該ISO週未購入・資金上限内であれば100株買い増しする。"""
    ma20 = calc_ma20(code, current_price)
    if ma20 is None:
        print(f"  {code} {name}: 移動平均データ不足のためスキップ")
        return False

    downtrend = current_price < ma20
    print(f"  {code} {name}: 現在値{current_price:,.1f}円  20日MA{ma20:,.1f}円  "
          f"乖離{(current_price - ma20) / ma20 * 100:+.1f}%  "
          f"{'下降トレンド' if downtrend else 'トレンドなし'}")
    if not downtrend:
        return False

    today = datetime.now(JST)
    iso_year, iso_week, _ = today.isocalendar()

    campaign = db.get_open_nanpin_campaign(code)
    if campaign is None:
        campaign_id = db.create_nanpin_campaign(code, name, opened_date=today.strftime("%Y-%m-%d"))
        campaign = {"campaign_id": campaign_id, "total_cost": 0.0}
        print(f"  📂 {code} {name}: 新規キャンペーン開始 (campaign_id={campaign_id})")
    else:
        campaign_id = campaign["campaign_id"]

    if db.has_bought_this_iso_week(campaign_id, iso_year, iso_week):
        print(f"  {code} {name}: 今週(ISO{iso_year}-W{iso_week:02d})はすでに買い増し済み → スキップ")
        return False

    if db.count_nanpin_buys(campaign_id) >= MAX_BUYS_HARD_CAP:
        print(f"  ⚠️ {code} {name}: 買い増し回数上限({MAX_BUYS_HARD_CAP}回)到達 → スキップ")
        return False

    prospective_cost = campaign["total_cost"] + current_price * BUY_SHARES
    if prospective_cost > MAX_CAMPAIGN_CAPITAL:
        print(f"  ⚠️ {code} {name}: 投入上限({MAX_CAMPAIGN_CAPITAL:,}円)到達見込み → 買い増し停止")
        return False

    mkt_code = db.get_market_code_db(code)
    mode_str = "本番" if tachibana_order.LIVE_TRADING else "モック"
    print(f"  🔵 {code} {name}: {BUY_SHARES}株 成行買い増し [{mode_str}]")
    result = tachibana_order.place_buy_order(url_request, code, BUY_SHARES, market_code=mkt_code)
    if not result["success"]:
        print(f"  ❌ 買い増し失敗: {result['message']}")
        return False

    now_str = today.strftime("%H:%M:%S")
    try:
        db.record_nanpin_buy(
            campaign_id, code, buy_date=today.strftime("%Y-%m-%d"), buy_time=now_str,
            iso_year=iso_year, iso_week=iso_week, shares=BUY_SHARES,
            price=current_price, order_no=result.get("order_no"),
        )
    except Exception as e:
        print(f"  ⚠️ DB記録に失敗（注文自体は成立済み・要手動確認）: {e}")
        return False
    return True


def main(start_now=False):
    db.init_db()
    mode_str = "本番発注モード" if tachibana_order.LIVE_TRADING else "モックモード（実発注なし）"
    print(f"=== 戦略N ナンピン戦略パイロット（{datetime.now(JST).strftime('%Y-%m-%d')}）===")
    print(f"  対象: {', '.join(f'{c}{n}' for c, n in CANDIDATES)}")
    print(f"  発注: {mode_str}  TP+{TP_PCT}%  1銘柄上限{MAX_CAMPAIGN_CAPITAL:,}円\n")

    url_price = load_tachibana_url()
    url_request = tachibana_order.load_url_request()
    if not url_price or not url_request:
        print("❌ Tachibana APIにログインしていません。scan_morning.py を先に実行してください。")
        return

    codes = [c for c, _ in CANDIDATES]
    now = datetime.now(JST)

    if not start_now and (now.hour < ACT_HOUR or (now.hour == ACT_HOUR and now.minute < ACT_MIN)):
        target = now.replace(hour=ACT_HOUR, minute=ACT_MIN, second=0, microsecond=0)
        wait = int((target - now).total_seconds())
        print(f"  ⏰ {ACT_HOUR}:{ACT_MIN:02d}まで {wait // 60}分{wait % 60}秒、"
              f"{SELL_POLL_INTERVAL_SEC}秒間隔で売却判定のみ実施します...\n")
        while True:
            now = datetime.now(JST)
            if now.hour > ACT_HOUR or (now.hour == ACT_HOUR and now.minute >= ACT_MIN):
                break
            quotes = fetch_prices(url_price, codes)
            for code, name in CANDIDATES:
                price = quotes.get(code)
                if price:
                    try_sell(code, name, price, url_request)
            time.sleep(SELL_POLL_INTERVAL_SEC)
        print()

    # ── 15:00頃の最終判定（売却チェック＋買い判定） ──
    quotes = fetch_prices(url_price, codes)
    for code, name in CANDIDATES:
        price = quotes.get(code)
        if not price:
            print(f"  ⚠️ {code} {name}: 現在値取得失敗 → スキップ")
            continue
        sold = try_sell(code, name, price, url_request)
        if not sold:
            try_buy(code, name, price, url_request)

    print("\n=== 戦略N 処理完了 ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="戦略N ナンピン戦略パイロット")
    parser.add_argument("--now", action="store_true", help="即時開始（テスト・手動確認用）")
    args = parser.parse_args()
    main(start_now=args.now)
