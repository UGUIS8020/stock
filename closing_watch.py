"""
closing_watch.py - 引け前（15:05〜15:20）戦略B下落銘柄スキャン + 買い判断

【パラメータ根拠】
  GA最適化（20,000人×100世代, 2026-05-26）OOS検証済み:
    drop -5.0〜-3.1% / RB>=4 / STRONG+NORMAL地合い
    OOS Sharpe=+3.502
  TP+1.9% / SL-5.6%  ← 2026-05-26更新（旧: TP+5.4% / SL-0.5%）
  ※フィルタ: 株価200円以上・出来高50,000株以上・±15%超除外
  ※地合い: NORMAL+STRONG（旧: STRONGのみ）

実行方法:
    python closing_watch.py       # 15:05まで待機して自動開始
    python closing_watch.py --now # 即時開始（テスト用）

前提:
    - tachibana_login_response.json が存在すること（scan_morning.py 実行後）
    - STRONG/NORMAL地合いの日のみ有効（WEAK/PANICは自動終了）
"""

import os
import sys
import json
import time
import argparse
import urllib3
import pandas as pd
import numpy as np
import tachibana_order
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()
import db

JST      = timezone(timedelta(hours=9))
TODAY    = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
_BASE_DIR = Path(__file__).parent

CANDIDATES_LOG_CSV   = str(_BASE_DIR / "out" / "candidates_log.csv")
MORNING_LOG_CSV      = str(_BASE_DIR / "out" / "morning_log.csv")
MARKET_LOG_CSV       = str(_BASE_DIR / "out" / "market_log.csv")
CLOSING_LOG_CSV      = str(_BASE_DIR / "out" / "closing_log.csv")
TACHIBANA_LOGIN_FILE = str(_BASE_DIR / "tachibana_login_response.json")

SCAN_START_HOUR    = 15
SCAN_START_MIN     = 0     # 2026-08-18: 15:05→15:00（下落候補が多い日にRBスコア計算＋
                           # J-Quants銘柄名取得の処理時間が伸び、15:20の発注締切に間に合わない
                           # リスクが判明。処理時間の余裕を15分→20分に拡大。発注タイミング
                           # (ORDER_WAIT_MIN/ORDER_DEADLINE_MIN)自体は変更しない。
                           # FORCE_CLOSE_MIN(15:00, position_monitor.py)と同時刻になるが、
                           # p_noはファイルロックで排他制御済み(tachibana_order._next_p_no)
                           # のため衝突せず、Bのスキャン自体は資金を使わないため資金競合もない。
ORDER_WAIT_MIN     = 15 * 60 + 13  # 15:13まで待機してから発注（引けギリギリ約15:15執行）
ORDER_DEADLINE_MIN = 15 * 60 + 20  # 15:20 以降は発注しない（引けオークション開始前）

DROP_LO      = -5.0   # 下落下限（GA最適化2026-05-26 20000人・OOS Sharpe=+3.502）
DROP_HI      = -3.1   # 下落上限（GA最適化2026-05-26 20000人・OOS Sharpe=+3.502）
RB_MIN       = 4      # RBスコア最小値（GA最適化2026-05-26 20000人・OOS Sharpe=+3.502）
CD_MIN       = 3      # 連続下落日数最小値（cd≥3: 前々日も下落していること）

# 超過下落フィルター（2026-08-01追加）
# 根拠: sim_precise_trades.csv 15,304件をrb_score>=4・consec_drop>=3・today_rise<0で検証したところ、
#   「その日の下落幅」-「市場全体の平均リターン」（超過下落幅）が-8%を下回るゾーンだけ
#   勝率41.7%・平均pnl-0.226%と明確に悪化（それ以外は-8%〜0%でなだらかにWR47〜51%）。
#   個別の悪材料で市場より突出して売られた銘柄は、市場に連動しただけの下落銘柄より反発しにくい。
# 除外ラインは-5%（母数の88.7%を維持しつつ明確な崖を除外）。7/30の実トレード5件では
# 超過下落幅-2.8%〜-3.8%と全てこのライン内で、遡って適用しても結果は変わらないことを確認済み。
EXCESS_DECLINE_MIN = -5.0

# 「落ちるナイフ」除外フィルター（2026-08-19追加）
# 根拠: 2026-08-18購入のサン電子(RB7.0で通過)が翌朝SL、直近6営業日で-29.99%という
# 通常の押し目を超えた暴落の途中だったと判明。sim_b_2night_trades.csv(N=1,499、
# MIN_PRICE=1,000円適用済み)で「エントリー時点の価格が6営業日前終値比でどれだけ
# 下がっていたか」を検証したところ、-30%以下で平均pnlがマイナス転落(N=47・
# 平均-0.09%・勝率34.0%)、-28%以下でもまだ明確に弱い(N=49・平均+0.12%・
# 勝率36.7%、それ以外はN=1450・平均+0.6%台・勝率55〜60%台)。walk-forward4分割
# 全てで直近ほど悪化と再現性を確認。既存のconsec_drop(連続下落日数、直近2-3日
# しか見ない)とは相関なし(両グループとも平均3.0日)で、独立した情報。
# RBスコアが高くても救えない(むしろ極端下落グループの平均RBスコアはより高い、
# 下落幅が大きいほどRBスコアが上がりやすい性質のため)。
# 閾値は-30%(全体平均が赤字転換する境界)ではなく-28%を採用: サン電子の実例
# (-29.99%)を確実に捕捉するため、追加除外コストはわずか2件（47→49件）。
PRIOR_6D_MAX_DECLINE = -28.0
MIN_VOLUME          = 50_000
MAX_POSITIONS_PER_DAY = 6      # 1日の最大自動発注件数（安全装置）。2026-08-13: 5→7へ変更した
                                # 直後、2泊保有の重なり(最大14件)×300株×MIN_PRICE導入後の
                                # 平均株価上昇を合わせた資金試算で、全戦略合計が資金(1,000万円)を
                                # 大きく超過(約2,300万円)すると判明したため、5へ差し戻し。
                                # 2026-08-14: STRONG日はそもそも候補が少なく(中央値1件、5件超は
                                # 5.8%の日のみ)上限がほぼ効かない一方、NORMAL日は候補が豊富
                                # (5件超が37.7%の日)で資金にも余裕があるため、7より安全側の
                                # 6へ引き上げ（2泊重なり最大12件、200株×2,609円平均で
                                # 約626万円。7の場合(14件・約731万円)より抑制）。

DEFAULT_SHARES   = 300      # これまで200↔300を複数回往復（詳細はコミット履歴参照）。
                            # 2026-08-17: 資金1,300万円到達を受け300へ再変更。今回は事前に
                            # 「株数拡大が先、件数上限(MAX_POSITIONS_PER_DAY)拡大は後」という
                            # 順序で合意した上での変更（RBスコア並び順の課題[[strategy_b_rb_sort_order_issue]]
                            # 自体は未解決だが、株数拡大は並び順に依存しない確度の高い変更と判断）。
                            # 金額ベースの分岐は廃止済みのため常に固定株数。
MIN_PRICE = 1_000  # 2026-08-13新設: 1,000円未満は勝率43.9%・平均-0.12%〜+0.05%と明確に不振
                    # (2泊分の正確なシミュレーションsimulate_b_2night.pyで検証、1,000円以上は
                    # 平均+0.47%・勝率53.4%)。walk-forwardでも境界として最も切れ味が良く、
                    # 1,000円未満を除外することにした。これに伴いCHEAP_THRESHOLD/
                    # MAX_ORDER_AMOUNTによる金額ベースの株数分岐は廃止。

TP_PCT = 0.05    # +5.0%（2026-06-23最適化: 2泊×TP5%×SL4% 合計+16.59% / 旧: TP+1.9% 合計-55.50%）
SL_PCT = 0.04    # -4.0%（2026-06-23最適化: 旧: SL-5.6%）

LIMIT_ORDER     = False  # 成行発注（指値はsResultCode=11010エラー多発のため 2026-06-25 変更）
LIMIT_WAIT_SECS = 30     # 指値約定確認の待機秒数


def calc_shares(price):
    """発注株数を返す。MIN_PRICE未満は候補から除外済みのため、常に固定株数。"""
    return DEFAULT_SHARES

PANIC_NIKKEI  = -2.0;  PANIC_AD  = 0.20
WEAK_NIKKEI   = -1.0;  WEAK_AD   = 0.35
STRONG_NIKKEI =  0.5;  STRONG_AD = 0.60


# ══════════════════════════════════════════════
# テクニカル指標
# ══════════════════════════════════════════════
def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    d  = np.diff(closes)
    ag = np.where(d > 0, d, 0.0)[-period:].mean()
    al = np.where(d < 0, -d, 0.0)[-period:].mean()
    return round(100 - 100 / (1 + ag / al), 1) if al > 0 else 100.0


def calc_rebound_score(hist):
    if len(hist) < 26:
        return 0
    score   = 0
    closes  = pd.to_numeric(hist["Close"], errors="coerce").fillna(0).values
    vol_ser = pd.to_numeric(hist["Volume"], errors="coerce").fillna(0)

    rsi = calc_rsi(closes, 14)
    if rsi is not None:
        if   rsi < 20: score += 4
        elif rsi < 30: score += 3
        elif rsi < 40: score += 2
        elif rsi < 50: score += 1

    if len(hist) >= 21:
        avg20 = vol_ser.iloc[-21:-1].mean()
        vol   = float(vol_ser.iloc[-1])
        if avg20 > 0:
            r = vol / avg20
            if   r >= 3.0: score += 3
            elif r >= 2.0: score += 2
            elif r >= 1.5: score += 1

    ma25 = pd.to_numeric(hist["Close"], errors="coerce").iloc[-25:].mean()
    dev  = (float(hist["Close"].iloc[-1]) / ma25 - 1) * 100 if ma25 > 0 else 0
    if   dev < -15: score += 3
    elif dev < -10: score += 2
    elif dev <  -5: score += 1

    return score


# ══════════════════════════════════════════════
# Tachibana API
# ══════════════════════════════════════════════
def load_tachibana_url():
    try:
        with open(TACHIBANA_LOGIN_FILE, encoding="utf-8") as f:
            data = json.load(f)
        url = data.get("sUrlPrice", "")
        return url if url else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _fetch_price_batch(url_price, codes, http):
    """120件以内のコードリストを一括取得して {code: quote_dict} を返す"""
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
        '"sTargetColumn":"pDPP,pPRP,pDV",'
        '"sJsonOfmt":"5"'
        "}"
    )
    try:
        resp = http.request("GET", url_price + "?" + params,
                            timeout=urllib3.Timeout(connect=3, read=8))
        tachibana_order.log_api_call("closing_watch.fetch_prices")
        text   = resp.data.decode("shift-jis", errors="ignore")
        result = json.loads(text)
        out    = {}
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
            prev  = _f("pPRP")
            vol   = int(_f("pDV") or 0)
            if price:
                out[code] = {"price": price, "prev_close": prev, "volume": vol}
        return out
    except Exception:
        return {}


# ══════════════════════════════════════════════
# 地合い判定
# ══════════════════════════════════════════════
def get_today_condition():
    """今日の地合いを取得。DB（market_log → morning_log）の順で試みる。"""
    return db.get_today_condition_db(TODAY)


def log_condition_change(morning_cond, scanned_cond, ad_ratio, nikkei_change):
    """2026-08-24追加: 15:00時点の地合い実測(朝予測との差)を、closing_watch_live.txt
    (当日上書き)とは別に複数日分蓄積してlogs/に記録する。daytime.pyのnoon_ad_observation.log
    と同じ方式(朝STRONG→日中軟化パターンの後日検証用)。"""
    try:
        log_dir = _BASE_DIR / "logs"
        log_dir.mkdir(exist_ok=True)
        path = log_dir / "condition_1500_observation.log"
        nikkei_str = f"{nikkei_change:.2f}" if nikkei_change is not None else "NA"
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{TODAY},{morning_cond},{scanned_cond},{ad_ratio:.3f},{nikkei_str}\n")
    except Exception:
        pass


def _fetch_nikkei_change(url_price, http):
    """日経225の当日騰落率(%)を取得。取得失敗時は None を返す。"""
    NIKKEI_CODE = "101"   # 立花API: 998407は無効。101が正しい日経225コード
    quotes = _fetch_price_batch(url_price, [NIKKEI_CODE], http)
    q = quotes.get(NIKKEI_CODE)
    if not q or not q.get("price") or not q.get("prev_close") or q["prev_close"] <= 0:
        return None
    return (q["price"] - q["prev_close"]) / q["prev_close"] * 100


def _classify_condition(ad_ratio, nikkei_change=None):
    """AD比率 + 日経騰落率で地合いを分類。

    STRONG: AD >= 0.60 かつ 日経 >= +0.5%（大型株も堅調）
    PANIC:  AD <= 0.20 かつ 日経 <= -2.0%（全面安）
    nikkei_change=None の場合はAD比率のみで判定。
    """
    if ad_ratio <= PANIC_AD:
        if nikkei_change is None or nikkei_change <= PANIC_NIKKEI:
            return "PANIC"
        return "WEAK"  # 小型株は崩れているが日経はPANICほどでない

    if ad_ratio >= STRONG_AD:
        if nikkei_change is None or nikkei_change >= STRONG_NIKKEI:
            return "STRONG"
        return "NORMAL"  # 小型株は堅調だが大型株（日経）が弱い

    if ad_ratio <= WEAK_AD:
        return "WEAK"
    return "NORMAL"


# ══════════════════════════════════════════════
# 全銘柄スキャン
# ══════════════════════════════════════════════
def scan_dropping_stocks(url_price):
    """
    全銘柄を Tachibana API でスキャンし、
    ・市場全体の AD比率 + 日経騰落率（地合い計算用）
    ・-5.0〜-6.3% 下落中の銘柄リスト（市場平均比の超過下落幅がEXCESS_DECLINE_MIN以上のもののみ）
    を返す。戻り値: (ad_ratio, nikkei_change, scanned_cond, dropping)
    """
    http    = urllib3.PoolManager()
    codes   = db.get_all_codes(exclude_etf=True)   # ETF/ETN(market_code=0109)を除外
    total   = len(codes)
    batch   = 120
    n_batch = (total + batch - 1) // batch

    print(f"  全{total:,}銘柄をスキャン中（{n_batch}バッチ）...", flush=True)

    # 日経225取得（先に取得しておく）
    nikkei_change = _fetch_nikkei_change(url_price, http)
    if nikkei_change is not None:
        print(f"  日経225: {nikkei_change:+.2f}%", flush=True)
    else:
        print(f"  日経225: 取得失敗（AD比率のみで判定）", flush=True)

    up_count = down_count = 0
    chg_sum  = 0.0
    chg_n    = 0
    dropping = []

    for i in range(0, total, batch):
        chunk  = codes[i:i + batch]
        quotes = _fetch_price_batch(url_price, chunk, http)

        for code, q in quotes.items():
            price = q.get("price")
            prev  = q.get("prev_close")
            vol   = q.get("volume", 0)
            if not price or not prev or prev <= 0:
                continue

            chg = (price - prev) / prev * 100
            if chg > 0:
                up_count += 1
            else:
                down_count += 1
            chg_sum += chg
            chg_n   += 1

            if DROP_LO <= chg <= DROP_HI and vol >= MIN_VOLUME:
                dropping.append({
                    "code":       code,
                    "price":      price,
                    "prev_close": prev,
                    "change_pct": round(chg, 2),
                    "volume":     vol,
                    "rb_score":   0,  # 後で計算
                })

        done = min(i + batch, total)
        if done % (batch * 8) == 0 or done == total:
            print(f"  {done:,}/{total:,}件 完了... 候補{len(dropping)}件", flush=True)

    tot          = up_count + down_count
    ad_ratio     = up_count / tot if tot > 0 else 0.5
    scanned_cond = _classify_condition(ad_ratio, nikkei_change)
    market_avg_ret = chg_sum / chg_n if chg_n > 0 else 0.0

    # 超過下落フィルター: 市場平均より突出して下落した銘柄（個別要因の急落）を除外
    before_excess = len(dropping)
    for c in dropping:
        c["excess_decline"] = round(c["change_pct"] - market_avg_ret, 2)
    dropping = [c for c in dropping if c["excess_decline"] >= EXCESS_DECLINE_MIN]
    excess_excluded = before_excess - len(dropping)
    if excess_excluded > 0:
        print(f"  超過下落フィルター（市場平均{market_avg_ret:+.2f}%比 <{EXCESS_DECLINE_MIN}%）: "
              f"{excess_excluded}件除外", flush=True)

    nikkei_str = f"  日経{nikkei_change:+.2f}%" if nikkei_change is not None else ""
    print(f"  スキャン完了: 上昇{up_count:,}件 / 下落{down_count:,}件  AD比率{ad_ratio:.2f}{nikkei_str} → {scanned_cond}")
    print(f"  {DROP_HI}〜{DROP_LO}%下落候補: {len(dropping)}件")
    return ad_ratio, nikkei_change, scanned_cond, dropping


def enrich_with_rb(candidates):
    """候補銘柄のRBスコア・連続下落日数・6営業日前比をSQLiteの履歴から計算して追加"""
    print(f"  RBスコア＋連続下落チェック中（{len(candidates)}件）...", flush=True)
    skip_cd = 0
    for c in candidates:
        try:
            hist = db.get_stock_history(c["code"])
            c["rb_score"]    = calc_rebound_score(hist) if len(hist) >= 26 else 0
            # 連続下落チェック（cd≥3: 前日・前々日も下落していること）
            if len(hist) >= 3:
                y_close  = float(hist["Close"].iloc[-1])   # 前日終値
                d2_close = float(hist["Close"].iloc[-2])   # 2日前終値
                d3_close = float(hist["Close"].iloc[-3])   # 3日前終値
                c["consec_drop_ok"] = y_close < d2_close and d2_close < d3_close
            else:
                c["consec_drop_ok"] = False
            # 6営業日前比（「落ちるナイフ」除外用、2026-08-19追加）
            if len(hist) >= 6 and c.get("price"):
                close_6ago = float(hist["Close"].iloc[-6])
                c["prior_6d_pct"] = (c["price"] - close_6ago) / close_6ago * 100 if close_6ago else None
            else:
                c["prior_6d_pct"] = None
        except Exception:
            c["rb_score"]       = 0
            c["consec_drop_ok"] = False
            c["prior_6d_pct"]   = None

    before = len(candidates)
    result = [c for c in candidates
              if c["rb_score"] >= RB_MIN and c.get("consec_drop_ok", False)]
    skip_cd = before - len([c for c in candidates if c["rb_score"] >= RB_MIN])
    skip_cd2 = len([c for c in candidates if c["rb_score"] >= RB_MIN]) - len(result)
    if skip_cd2 > 0:
        print(f"  cd≥{CD_MIN}フィルタ: {skip_cd2}件除外（前日も下落していない）", flush=True)

    # 「落ちるナイフ」除外（6営業日前比がPRIOR_6D_MAX_DECLINEを超えて下落している銘柄）
    # 履歴不足でprior_6d_pctがNoneの場合はデータ不足で除外せず通過させる（安全側）
    before_knife = len(result)
    result = [c for c in result
              if c.get("prior_6d_pct") is None or c["prior_6d_pct"] > PRIOR_6D_MAX_DECLINE]
    skip_knife = before_knife - len(result)
    if skip_knife > 0:
        print(f"  落ちるナイフ除外: {skip_knife}件除外（6営業日前比{PRIOR_6D_MAX_DECLINE}%以下）", flush=True)

    return result



# ══════════════════════════════════════════════
# 自動発注
# ══════════════════════════════════════════════
def auto_order(candidate, url_request, ordered_today, condition="STRONG"):
    """完全自動発注（確認なし）。成功したら True を返す。"""
    code      = candidate["code"]
    name      = candidate.get("name", code)
    price     = candidate["price"]
    shares    = calc_shares(price)
    estimated = int(price * shares)
    tp_price  = round(price * (1 + TP_PCT))
    sl_price  = round(price * (1 - SL_PCT))

    # 重複発注防止
    if str(code) in ordered_today:
        print(f"  ⏭️  [{code}] 本日すでに発注済み - スキップ")
        return False

    # 買付余力チェック
    if url_request:
        buying_power = tachibana_order.get_buying_power(url_request)
        if buying_power is not None and estimated > buying_power:
            print(f"  ❌ [{code}] 買付余力不足（必要{estimated:,}円 / 余力{buying_power:,}円）- スキップ")
            return False

    mode = "本番" if tachibana_order.LIVE_TRADING else "モック"
    print(f"\n  📤 自動発注 [{mode}]: [{code}] {name}  "
          f"{price:,.0f}円 × {shares}株 = {estimated:,}円")
    print(f"     TP目安: {tp_price:,}円 (+{TP_PCT*100:.1f}%)  SL目安: {sl_price:,}円 (-{SL_PCT*100:.1f}%)")

    mkt_code = db.get_market_code_db(code)
    order_type = "指値" if LIMIT_ORDER else "成行"
    print(f"     発注方式: {order_type} @ {price:,.0f}円")

    if LIMIT_ORDER:
        r = tachibana_order.place_buy_limit_with_fallback(
            url_request, code, shares, price, mkt_code,
            wait_secs=LIMIT_WAIT_SECS, strategy="B")
        if r["success"]:
            print(f"  ✅ {r['message']}")
            tachibana_order.save_position(
                code, name, r["filled"], r["fill_price"],
                strategy="B", tp_pct=TP_PCT, sl_pct=SL_PCT,
                entry_change_pct=candidate.get("change_pct"),
                rb_score=candidate.get("rb_score"),
                condition=condition,
                url_request=url_request, order_no=r.get("order_no"),
            )
            ordered_today.add(str(code))
            return True
        else:
            print(f"  ❌ 発注失敗: {r['message']}")
            return False
    else:
        result = tachibana_order.place_buy_order(url_request, code, shares, market_code=mkt_code)
        if result["success"]:
            print(f"  ✅ {result['message']}")
            eigyou_day = (result.get("raw") or {}).get("sEigyouDay")
            tachibana_order.save_position(
                code, name, shares, price,
                strategy="B", tp_pct=TP_PCT, sl_pct=SL_PCT,
                entry_change_pct=candidate.get("change_pct"),
                rb_score=candidate.get("rb_score"),
                condition=condition,
                url_request=url_request, order_no=result.get("order_no"),
                eigyou_day=eigyou_day,
            )
            ordered_today.add(str(code))
            return True
        else:
            print(f"  ❌ 発注失敗: {result['message']}")
            return False


# ══════════════════════════════════════════════
# 結果保存
# ══════════════════════════════════════════════
def save_closing_log(candidates):
    """候補銘柄を closing_log テーブルに保存"""
    rows = [{
        "date":       TODAY,
        "code":       c["code"],
        "name":       c.get("name", ""),
        "change_pct": c["change_pct"],
        "rb_score":   c["rb_score"],
        "price":      c["price"],
        "tp_price":   round(c["price"] * (1 + TP_PCT)),
        "sl_price":   round(c["price"] * (1 - SL_PCT)),
    } for c in candidates]
    db.save_closing_log_db(rows)
    print(f"\n  💾 ログ保存: closing_log（{len(rows)}件）")


# ══════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════
def main(start_now=False, manual=False):
    db.init_db()
    mode_label = "【手動確認モード】" if manual else "【全自動モード】"
    print(f"=== 📉 引け前スキャン（{TODAY}）{mode_label}===\n")

    # 金曜日チェック（データ検証済み：金曜引け買いは週末リスクで負け越し）
    if datetime.now(JST).weekday() == 4:  # 0=月 … 4=金
        print("  ⛔  本日は金曜日のため引け前買いを見送ります。")
        print("     （理由：翌月曜のギャップアップ後下落が多く、勝率44.7% / 平均-0.21%）")
        return

    # Tachibana API 確認
    url_price   = load_tachibana_url()
    url_request = tachibana_order.load_url_request()
    if not url_price:
        print("❌ Tachibana APIにログインしていません。scan_morning.py を先に実行してください。")
        return
    print(f"  📡 Tachibana API: リアルタイム価格取得モード")
    mode_str = "本番発注モード" if tachibana_order.LIVE_TRADING else "モックモード（実発注なし）"
    print(f"  💼 発注: {mode_str}  {MIN_PRICE:,}円未満は除外 / {MIN_PRICE:,}円以上→{DEFAULT_SHARES}株固定\n")

    # 開始待機
    now = datetime.now(JST)
    if not start_now and (now.hour < SCAN_START_HOUR or
                          (now.hour == SCAN_START_HOUR and now.minute < SCAN_START_MIN)):
        target = now.replace(hour=SCAN_START_HOUR, minute=SCAN_START_MIN, second=0, microsecond=0)
        wait   = int((target - now).total_seconds())
        print(f"  ⏰ {SCAN_START_HOUR}:{SCAN_START_MIN:02d}まで {wait//60}分{wait%60}秒 待機します...")
        while True:
            now = datetime.now(JST)
            if now.hour > SCAN_START_HOUR or (now.hour == SCAN_START_HOUR and now.minute >= SCAN_START_MIN):
                break
            time.sleep(10)

    # 地合い確認（CSVから読む → スキャン後に上書き）
    condition = get_today_condition()
    print(f"  地合い（事前）: {condition}")

    # ── 全銘柄スキャン ──
    print(f"\n【ステップ1】全銘柄スキャン開始...")
    ad_ratio, nikkei_change, scanned_cond, raw_candidates = scan_dropping_stocks(url_price)

    # スキャン結果で地合いを補正（実際のAD比率+日経を優先）
    # 朝の予測よりリアルタイムの指標の方が正確なため常に上書き
    if condition != scanned_cond:
        print(f"  ⚠️  地合い変化: {condition}（朝予測）→ {scanned_cond}（実測AD+日経）")
    log_condition_change(condition, scanned_cond, ad_ratio, nikkei_change)
    condition = scanned_cond
    nikkei_info = f"  日経{nikkei_change:+.2f}%" if nikkei_change is not None else ""
    print(f"  地合い（スキャン補正後）: {condition}  AD比率: {ad_ratio:.2f}{nikkei_info}")

    # WEAK/PANIC は終了（NORMAL+STRONG のみ有効）
    if condition not in ("STRONG", "NORMAL"):
        print(f"\n  ⚠️  本日は{condition}地合いです。")
        print(f"     戦略B引け前買いはSTRONG/NORMAL地合いのみ有効のため終了します。")
        now_str = f"AD{ad_ratio:.2f}{nikkei_info}"
        print(f"     （現在: {now_str}）")
        return

    if not raw_candidates:
        print(f"\n  該当銘柄なし（{DROP_HI}〜{DROP_LO}%下落 × 出来高{MIN_VOLUME:,}株以上）")
        return

    # ── RBスコア計算 ──
    print(f"\n【ステップ2】RBスコア計算...")
    candidates = enrich_with_rb(raw_candidates)

    if not candidates:
        print(f"  RBスコア{RB_MIN}以上×連続下落{CD_MIN}日以上を両方満たす銘柄なし")
        return

    # RBスコア降順 → 下落幅降順でソート
    candidates.sort(key=lambda c: (-c["rb_score"], c["change_pct"]))

    # 銘柄名を取得（可能な場合）
    # scan_resultsは戦略AのTOP20候補のみを含むため、そこにない銘柄（戦略B候補は
    # 全銘柄が対象）はstock_masterキャッシュ（J-Quants全銘柄マスタを1日1回取得
    # したもの、2026-08-18追加）で補う。ここで毎回J-Quantsをライブ取得すると
    # ネットワーク遅延で15:20の発注締切に迫るリスクがあったため、ライブ取得は
    # キャッシュが空の場合のフォールバックのみに限定している。
    try:
        conn = db.get_conn()
        code_names = dict(pd.read_sql(
            "SELECT DISTINCT code, name FROM scan_results WHERE name != ''", conn
        ).values.tolist())
        conn.close()
    except Exception:
        code_names = {}

    missing_codes = [c["code"] for c in candidates if c["code"] not in code_names]
    if missing_codes:
        try:
            import stock_master
            code_names.update(stock_master.get_names())
        except Exception as e:
            print(f"  ⚠️  銘柄マスタ取得失敗（コード表示にフォールバック）: {e}")

    for c in candidates:
        c["name"] = code_names.get(c["code"], c.get("name") or c["code"])

    # ── 異常フラグ除外 + 低位株除外（コードで処理）──
    before = len(candidates)
    candidates = [c for c in candidates if c["change_pct"] > -15 and c["price"] >= MIN_PRICE]
    removed = before - len(candidates)
    if removed > 0:
        print(f"  異常フラグ除外: {removed}件（-15%超暴落 or 株価{MIN_PRICE:,}円未満）")

    if not candidates:
        print(f"  異常フラグ除外後、該当銘柄なし")
        return

    print(f"\n{'='*62}")
    print(f"【引け前候補】STRONG/NORMAL地合い × {DROP_HI}〜{DROP_LO}% × RB>={RB_MIN} × cd≥{CD_MIN}  {len(candidates)}件")
    print(f"{'='*62}")
    print(f"  {'コード':<7} {'銘柄名':<14} {'下落率':>7} {'現在値':>8} {'RB':>4} {'TP目安':>8} {'SL目安':>8} {'出来高':>10}")
    print(f"  {'─'*72}")
    for c in candidates:
        tp = round(c["price"] * (1 + TP_PCT))
        sl = round(c["price"] * (1 - SL_PCT))
        print(f"  {c['code']:<7} {c.get('name',''):<14} {c['change_pct']:>+6.2f}% "
              f"{c['price']:>8,.0f}円  {c['rb_score']:>3}点  {tp:>8,}円 {sl:>8,}円  {c['volume']:>10,}株")

    # ── 本日すでに発注済みの銘柄を取得（再実行時の重複防止）──
    ordered_today = db.get_today_ordered_codes(TODAY)

    # ── 発注待機（引けギリギリ約15:15執行）──
    now = datetime.now(JST)
    now_min = now.hour * 60 + now.minute
    if now_min < ORDER_WAIT_MIN:
        wait_sec = (ORDER_WAIT_MIN - now_min) * 60 - now.second
        print(f"\n  ⏰ {ORDER_WAIT_MIN//60}:{ORDER_WAIT_MIN%60:02d}まで {wait_sec//60}分{wait_sec%60}秒 待機中（引けギリギリ発注）...")
        while True:
            now = datetime.now(JST)
            if now.hour * 60 + now.minute >= ORDER_WAIT_MIN:
                break
            time.sleep(5)
        print(f"  → {ORDER_WAIT_MIN//60}:{ORDER_WAIT_MIN%60:02d} 発注開始")

    # ── 発注（自動 or 手動確認）──
    label = "手動確認" if manual else f"自動発注（上限{MAX_POSITIONS_PER_DAY}件）"
    print(f"\n【ステップ3】{label}...")
    order_count = 0
    for c in candidates:
        if not manual and order_count >= MAX_POSITIONS_PER_DAY:
            print(f"  ⚠️  本日の上限{MAX_POSITIONS_PER_DAY}件に達しました。残り銘柄はスキップします。")
            break
        now_min = datetime.now(JST)
        now_min = now_min.hour * 60 + now_min.minute
        if now_min >= ORDER_DEADLINE_MIN:
            print(f"  ⏰ 15:15を過ぎました。発注を中止します。")
            break

        if manual:
            # 手動確認モード: y/n で1件ずつ確認
            code  = c["code"]
            name  = c.get("name", code)
            price = c["price"]
            tp    = round(price * (1 + TP_PCT))
            sl    = round(price * (1 - SL_PCT))
            print(f"\n  [{code}] {name}  {price:,.0f}円  {c['change_pct']:+.2f}%  "
                  f"RB{c['rb_score']}点  TP:{tp:,}円(+{TP_PCT*100:.1f}%)  SL:{sl:,}円(-{SL_PCT*100:.1f}%)")
            ans = input("  >>> 発注しますか？ [y=発注 / n=見送り / q=終了] : ").strip().lower()
            if ans == "q":
                print("  手動モード終了。")
                break
            elif ans == "y":
                if auto_order(c, url_request, ordered_today, condition):
                    order_count += 1
            else:
                print(f"  ↩️  {code} 見送り。")
        else:
            if auto_order(c, url_request, ordered_today, condition):
                order_count += 1

    print(f"\n  本日の発注件数: {order_count}件")

    # ── ログ保存 ──
    save_closing_log(candidates)

    print(f"\n{'='*62}")
    print(f"  引け前スキャン完了  {datetime.now(JST).strftime('%H:%M:%S')}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="引け前下落銘柄スキャン + 買い判断")
    parser.add_argument("--now",    action="store_true", help="即時開始（テスト用）")
    parser.add_argument("--manual", action="store_true", help="手動確認モード（1件ずつ y/n で確認）")
    args = parser.parse_args()
    main(start_now=args.now, manual=args.manual)
