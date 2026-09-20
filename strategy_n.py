"""
strategy_n.py - 戦略N（ナンピン戦略）パイロット: 資金枠2枠(slot A/B)限定

【方式】(analyze/screen_nanpin_candidates.py でバックテスト・検証済み)
    20日移動平均を終値が下回っている間、暦週(ISO週)に1回買い増しし、
    平均取得単価×(1+TP%)まで戻ったら保有全株を成行で一括売却する。
    損切りルールなし（意図的な設計。銘柄ごとの投入上限=max_campaign_capitalのみが
    歯止め）。既存の戦略A/AN/AS/B/Dとは別の独立したDBテーブル
    (nanpin_campaigns/nanpin_buys)で管理し、既存のpositionsテーブル
    (PRIMARY KEY (date, code)、単発売買前提)には一切触れない。

【2026-09-20: 銘柄の自動ローテーション(stock_usa/strategy_n_us.pyと同じ設計)】
    銘柄を固定せず、SLOT_CONFIGSで定義した2つの資金枠(slot A/B)だけを固定し、
    実際にどの銘柄を保有するかはanalyze/screen_nanpin_candidates.pyが毎月生成する
    ランキング(out/nanpin_candidates_ranking_latest.json)から自動選定する。既に
    open中のキャンペーンは、開始時に自分のcode/shares_per_buy/max_campaign_capital/
    tp_pctをDBに保存しているため、月次ランキングが変わっても待機し続け、TP到達で
    自然に決済されるまでは影響を受けない。空いた資金枠のみ、次回実行時にその時点の
    ランキング上位(他スロットで保有中でない銘柄・株価と資金枠のバランスが妥当な銘柄、
    その中でも下降トレンド中の銘柄を優先)から新規に選ばれる。パイロットはこの機能の
    導入時点(2026-09-20)でまだ建玉ゼロだったため、当初固定していた日本精工(6471)・
    中部電力(9502)は初期値としては使わず、ランキング不在時の緊急フォールバックとしてのみ残す。

【既知の制約（パイロット段階での意図的な簡略化）】
    売却判定は市場が開いている間、SELL_POLL_INTERVAL_SEC(15分)間隔で繰り返す。
    バックテストは「日中の高値がTP到達したら約定」を前提にしているため、
    ポーリング間隔の間に一瞬TPへ到達してその後押し戻された場合は検知できない
    （常時監視ではないための保守的な簡略化）。買い判定は終値ベースのため
    15:00頃の1回のみ。

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
RANKING_PATH = _BASE_DIR / "out" / "nanpin_candidates_ranking_latest.json"

# 2026-09-20: 銘柄固定のCANDIDATESから、資金枠(スロット)固定・銘柄は月次ランキングで
# 自動ローテーションする方式に変更(stock_usa側と同じ設計)。target_buy_yenは旧
# BUY_SHARES=100株の実績値(日本精工1,082円なら約108,200円/回)を踏襲。
SLOT_CONFIGS = [
    {"slot": "A", "target_buy_yen": 100_000, "max_campaign_capital": 2_000_000},
    {"slot": "B", "target_buy_yen": 100_000, "max_campaign_capital": 2_000_000},
]
# ランキングJSON不在時(初回・cron未実行等)の緊急フォールバック
FALLBACK_CANDIDATES = {"A": ("6471", "日本精工"), "B": ("9502", "中部電力")}

TREND_N   = 20          # 移動平均日数（バックテストと同じ）
TP_PCT    = 3.0          # 平均取得単価+3.0%で全売却（2026-09-20: J-Quantsを5年分に
                        # バックフィル後、fold1/fold2(各約1.5年)で再検証。TP+2%比で
                        # 平均スコア8.4%→9.5%、15銘柄中11銘柄で改善(USほど強い
                        # シグナルではないため、USの新規分のみ適用という慎重な方式
                        # ではなく、まだ建玉が無い[パイロット未開始]このタイミングで
                        # シンプルに全体を3%へ変更した)。新規キャンペーンは開始時に
                        # この値をtp_pctとしてスナップショット保存する。
MIN_UNIT_SHARES = 100   # 単元株数。shares_per_buyは常にこの倍数に丸める
MAX_BUYS_HARD_CAP    = 20          # 資金上限とは独立のバックストップ
MIN_BUYS_PER_CAMPAIGN = 10  # 投入上限までに最低これだけ買い増しできることを求める
                            # (stock_usa側と同じ、株価が高すぎて割に合わない銘柄を除外)

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
    データ不足時はNoneを返す。
    2026-09-18修正: 16:45のscan_daily.py実行後(または翌日以降)に本スクリプトを
    再実行すると、当日の終値がdaily_pricesに既に記録済みのため、それを履歴
    としてもcurrent_priceとしても二重にカウントしてしまうバグがあった。
    当日日付の行を明示的に除外することで、実行タイミングに依存せず正しい
    20日移動平均になるよう修正。"""
    hist = db.get_stock_history(code, days=TREND_N + 5)
    if hist is None or hist.empty:
        return None
    today_str = datetime.now(JST).strftime("%Y-%m-%d")
    hist = hist[hist["Date"] < today_str]
    closes = hist["Close"].dropna().tolist()[-(TREND_N - 1):]
    if len(closes) < TREND_N - 1:
        return None
    return (sum(closes) + current_price) / TREND_N


def load_ranking():
    """analyze/screen_nanpin_candidates.pyが毎月生成する機械可読ランキングを読み込む。
    無ければ空リストを返す(呼び出し側でフォールバックする)。"""
    if not RANKING_PATH.exists():
        return []
    try:
        return json.loads(RANKING_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ⚠️ ランキングJSON読み込み失敗、フォールバックを使用: {e}")
        return []


def get_price_and_trend(url_price, code):
    """指定銘柄の現在値と、その場での下降/上昇トレンド判定(20日MA基準)を
    まとめて返す。ランキングJSONのtrend値は作成時点のスナップショットで
    古くなり得るため、実際の選定時にはここで都度最新値から判定し直す
    (stock_usa側と同じ設計)。取得失敗時はNoneを返す。"""
    quotes = fetch_prices(url_price, [code])
    price = quotes.get(code)
    if not price:
        return None
    ma20 = calc_ma20(code, price)
    is_downtrend = ma20 is not None and price < ma20
    return price, is_downtrend


def _pick_candidate_for_slot(sc, ranking, held_codes, get_price_and_trend_fn):
    """ランキング上位から順に試し、株価と資金枠(max_campaign_capital)のバランスが
    妥当な(=投入上限までにMIN_BUYS_PER_CAMPAIGN回以上買い増しできる)銘柄の中から、
    その時点で下降トレンド中の銘柄を優先する。条件を満たす銘柄が一つも無い場合は、
    資金バランスより「候補が居ないよりはまし」を優先し最上位にフォールバックする。"""
    best_fit = None
    for r in ranking:
        code = r["code"]
        if code in held_codes:
            continue
        info = get_price_and_trend_fn(code)
        if not info:
            continue
        price, is_downtrend = info
        if not price:
            continue
        shares_per_buy = max(MIN_UNIT_SHARES,
                              round(sc["target_buy_yen"] / price / MIN_UNIT_SHARES) * MIN_UNIT_SHARES)
        implied_buys = sc["max_campaign_capital"] / (shares_per_buy * price)
        if implied_buys < MIN_BUYS_PER_CAMPAIGN:
            print(f"  slot {sc['slot']}: {code} {r['name']} 株価{price:,.1f}円 → "
                  f"{shares_per_buy}株/回、上限までの想定買い増し回数{implied_buys:.1f}回"
                  f"(<{MIN_BUYS_PER_CAMPAIGN}回) → 見送り、次点を検討")
            continue

        trend_str = "下降トレンド" if is_downtrend else "上昇トレンド"
        candidate = {"code": code, "name": r["name"], "shares_per_buy": shares_per_buy}
        if best_fit is None:
            best_fit = candidate
        if is_downtrend:
            print(f"  slot {sc['slot']}: {code} {r['name']} 株価{price:,.1f}円 → "
                  f"{shares_per_buy}株/回、想定買い増し回数{implied_buys:.1f}回、{trend_str} "
                  f"→ 採用(下降トレンド優先)")
            return candidate
        print(f"  slot {sc['slot']}: {code} {r['name']} 株価{price:,.1f}円 → "
              f"{shares_per_buy}株/回、想定買い増し回数{implied_buys:.1f}回、{trend_str} "
              f"→ 条件は満たすが下降トレンドの候補を優先したいため保留、次点を確認")
    if best_fit:
        print(f"  slot {sc['slot']}: 下降トレンド中の適格候補が無いため、"
              f"条件を満たす最上位({best_fit['code']} {best_fit['name']})を採用")
    return best_fit


def resolve_slot_candidates(get_price_and_trend_fn):
    """今回の実行で対象にする銘柄リストを組み立てる。open中のキャンペーンがある
    スロットは、そのキャンペーンが保存しているcode/shares_per_buy/max_campaign_capital/
    tp_pctをそのまま使う(=待機、ローテーション対象外)。openなキャンペーンが無い
    空きスロットのみ、月次ランキング上位から新規候補を選ぶ(_pick_candidate_for_slot参照)。"""
    held_codes = set()
    open_by_slot = {}
    for sc in SLOT_CONFIGS:
        oc = db.get_open_nanpin_campaign_by_slot(sc["slot"])
        if oc:
            open_by_slot[sc["slot"]] = oc
            held_codes.add(oc["code"])

    ranking = load_ranking()
    candidates = []
    for sc in SLOT_CONFIGS:
        slot = sc["slot"]
        oc = open_by_slot.get(slot)
        if oc:
            candidates.append({
                "code": oc["code"], "name": oc["name"], "slot": slot,
                "shares_per_buy": oc["shares_per_buy"] or MIN_UNIT_SHARES,
                "max_campaign_capital": oc["max_campaign_capital"] or sc["max_campaign_capital"],
                "tp_pct": oc["tp_pct"] or TP_PCT,
            })
            continue

        chosen = _pick_candidate_for_slot(sc, ranking, held_codes, get_price_and_trend_fn)
        if chosen is None:
            code, name = FALLBACK_CANDIDATES[slot]
            print(f"  ⚠️ slot {slot}: 月次ランキングから選定不可のためフォールバック({code} {name})を使用")
            info = get_price_and_trend_fn(code)
            price = info[0] if info else None
            shares_per_buy = (max(MIN_UNIT_SHARES,
                                   round(sc["target_buy_yen"] / price / MIN_UNIT_SHARES) * MIN_UNIT_SHARES)
                               if price else MIN_UNIT_SHARES)
            chosen = {"code": code, "name": name, "shares_per_buy": shares_per_buy}
        if chosen["code"] in held_codes:
            print(f"  ⚠️ slot {slot}: 候補({chosen['code']} {chosen['name']})が他スロットで保有中と重複のためスキップ")
            continue
        held_codes.add(chosen["code"])

        candidates.append({
            "code": chosen["code"], "name": chosen["name"], "slot": slot,
            "shares_per_buy": chosen["shares_per_buy"],
            "max_campaign_capital": sc["max_campaign_capital"],
            "tp_pct": TP_PCT,
        })
    return candidates


def try_sell(code, name, current_price, campaign, url_request):
    """保有中キャンペーンがあり、TP到達していれば全株成行売却する。
    TP%はcampaign自身のtp_pct(開始時点のスナップショット)を使う(無ければTP_PCTに
    フォールバック、stock_usa側と同じ設計)。"""
    if not campaign or campaign["shares_held"] <= 0:
        return False

    tp_pct = campaign.get("tp_pct") or TP_PCT
    avg_cost = campaign["total_cost"] / campaign["shares_held"]
    sell_target = avg_cost * (1 + tp_pct / 100)
    if current_price < sell_target:
        print(f"  {code} {name}: 保有中 {campaign['shares_held']}株  "
              f"平均取得単価{avg_cost:,.1f}円  目標{sell_target:,.1f}円(TP+{tp_pct}%)  "
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


def try_buy(cand, current_price, campaign, url_request):
    """下降トレンド中かつ当該ISO週未購入・資金上限内であれば買い増しする。
    株数・資金上限・TP%はcand辞書(スロットの設定 or 既存キャンペーンのスナップショット)
    から取得する(stock_usa側と同じ設計)。"""
    code, name = cand["code"], cand["name"]
    shares_per_buy = cand["shares_per_buy"]
    max_capital = cand["max_campaign_capital"]

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

    # 2026-09-20: キャンペーン行はプレビュー時には作らず、実際に発注する直前
    # (下記のexecute成功後)まで遅らせる(stock_usa側と同じ設計)。
    if campaign is not None:
        campaign_id = campaign["campaign_id"]
        total_cost_so_far = campaign["total_cost"]
        if db.has_bought_this_iso_week(campaign_id, iso_year, iso_week):
            print(f"  {code} {name}: 今週(ISO{iso_year}-W{iso_week:02d})はすでに買い増し済み → スキップ")
            return False
        if db.count_nanpin_buys(campaign_id) >= MAX_BUYS_HARD_CAP:
            print(f"  ⚠️ {code} {name}: 買い増し回数上限({MAX_BUYS_HARD_CAP}回)到達 → スキップ")
            return False
    else:
        campaign_id = None
        total_cost_so_far = 0.0

    prospective_cost = total_cost_so_far + current_price * shares_per_buy
    if prospective_cost > max_capital:
        print(f"  ⚠️ {code} {name}: 投入上限({max_capital:,.0f}円)到達見込み → 買い増し停止")
        return False

    mkt_code = db.get_market_code_db(code)
    mode_str = "本番" if tachibana_order.LIVE_TRADING else "モック"
    print(f"  🔵 {code} {name}: {shares_per_buy}株 成行買い増し [{mode_str}]")
    result = tachibana_order.place_buy_order(url_request, code, shares_per_buy, market_code=mkt_code)
    if not result["success"]:
        print(f"  ❌ 買い増し失敗: {result['message']}")
        return False

    if campaign_id is None:
        campaign_id = db.create_nanpin_campaign(
            code, name, opened_date=today.strftime("%Y-%m-%d"), slot=cand.get("slot"),
            shares_per_buy=shares_per_buy, max_campaign_capital=max_capital,
            tp_pct=cand.get("tp_pct", TP_PCT),
        )
        print(f"  📂 {code} {name}: 新規キャンペーン開始 (campaign_id={campaign_id}, slot={cand.get('slot')})")

    now_str = today.strftime("%H:%M:%S")
    try:
        db.record_nanpin_buy(
            campaign_id, code, buy_date=today.strftime("%Y-%m-%d"), buy_time=now_str,
            iso_year=iso_year, iso_week=iso_week, shares=shares_per_buy,
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
    print(f"  発注: {mode_str}  新規TP+{TP_PCT}%\n")

    # 2026-09-20追加: slot列を追加した際、それ以前に作られたキャンペーン行のslotを
    # 自動補完し忘れると、ローテーションロジックが「空きスロット」と誤認し重複して
    # 新規建玉を開始するおそれがある(stock_usa側の教訓と同じガード)。
    orphaned = [c for c in db.get_all_open_nanpin_campaigns() if not c.get("slot")]
    if orphaned:
        codes_str = ", ".join(f"{c['code']}{c['name']}(campaign_id={c['campaign_id']})" for c in orphaned)
        print(f"  🛑 異常検知: slot情報の無いopenキャンペーンがあります({codes_str})。"
              f"このまま進めるとローテーションロジックが「空きスロット」と誤認し、"
              f"重複して新規建玉を開始するおそれがあるため、安全のため処理を中断します。"
              f"nanpin_campaignsのslot/shares_per_buy/max_campaign_capital/tp_pctを"
              f"手動で補完してから再実行してください。")
        return

    url_price = load_tachibana_url()
    url_request = tachibana_order.load_url_request()
    if not url_price or not url_request:
        print("❌ Tachibana APIにログインしていません。scan_morning.py を先に実行してください。")
        return

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
            # 2026-09-20: 日中ポーリングは新規候補の解決(月次ランキング参照・価格取得)を
            # 行わず、現在openなキャンペーンだけを直接見る(売却判定にランキングは不要なため、
            # stock_usa側と同じ設計)。
            open_campaigns = db.get_all_open_nanpin_campaigns()
            if open_campaigns:
                quotes = fetch_prices(url_price, [c["code"] for c in open_campaigns])
                for oc in open_campaigns:
                    price = quotes.get(oc["code"])
                    if price:
                        try_sell(oc["code"], oc["name"], price, oc, url_request)
            time.sleep(SELL_POLL_INTERVAL_SEC)
        print()

    # ── 15:00頃の最終判定（売却チェック＋買い判定、銘柄はresolve_slot_candidatesで決定） ──
    def _get_price_and_trend(code):
        return get_price_and_trend(url_price, code)

    candidates = resolve_slot_candidates(_get_price_and_trend)
    candidates_desc = ", ".join(f"{c['code']}{c['name']}(slot{c['slot']})" for c in candidates)
    print(f"  対象: {candidates_desc}\n")

    quotes = fetch_prices(url_price, [c["code"] for c in candidates])
    for cand in candidates:
        code, name = cand["code"], cand["name"]
        price = quotes.get(code)
        if not price:
            print(f"  ⚠️ {code} {name}: 現在値取得失敗 → スキップ")
            continue
        campaign = db.get_open_nanpin_campaign(code)
        sold = try_sell(code, name, price, campaign, url_request)
        if not sold:
            try_buy(cand, price, campaign, url_request)

    print("\n=== 戦略N 処理完了 ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="戦略N ナンピン戦略パイロット")
    parser.add_argument("--now", action="store_true", help="即時開始（テスト・手動確認用）")
    args = parser.parse_args()
    main(start_now=args.now)
