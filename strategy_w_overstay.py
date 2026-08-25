# -*- coding: utf-8 -*-
"""
strategy_w_overstay.py - 戦略W_OVERSTAY（WEAK日の翌営業日エントリー・2泊保有の逆張り、2026-08-14新設）

【命名について】当初「戦略W」として開発したが、実際の狙い(WEAK当日、他戦略が
休止している間の空き資金を使う)とは異なり、エントリーは「WEAKの翌営業日」に
なる(引け確定データで判定→翌朝発注という仕組み上、当日中の判定・発注はできない)。
本来のWEAK当日発動型の戦略は別途「戦略W」として改めて開発する予定のため、
本ファイル・戦略名は区別のため W_OVERSTAY とする。

戦略B(closing_watch.py)はNORMAL/STRONG日限定で、WEAK日(2年間で全体の
約16%)は完全に休止している。戦略W_OVERSTAYはこの空白の翌営業日を狙う独立戦略。

【他戦略との関係】
    - 発動条件（トリガー）はWEAK限定だが、実際の発注はWEAK当日ではなく
      翌営業日のため、A/AN/AS/B/Dと資金・APIタイミングが重なる可能性がある
      （詳細はStage2の地合い実測ガード参照）
    - コード・ロジックは closing_watch.py から完全に独立（import・共有なし。
      calc_rebound_score/calc_rsiもこのファイル内に複製し、依存を持たない）
    - position_monitor.py の変更は不要（15:20強制決済は「strategy in
      (A,D,AN,AS)以外は当日は決済しない」設計のため、W_OVERSTAYは自動的に
      B同様の2泊保有として扱われる）

【条件（2026-08-14 GA探索(20,000人×100世代)+walk-forward検証済み、2年データ）】
    today_rise  : -4.2% < 下落率 <= -3.0%
    rb_score    : calc_rebound_score(hist) >= 4
    lower_shadow_ratio : >= 0.35（下ヒゲの深さ）
    gap_pct     : 翌朝の寄り付きギャップ <= +4.4%（Stage2で確認）
    price       : 1,000円以上（戦略Bと同一）
    TP/SL       : +5.2% / -5.2%
    検証結果    : N=215、勝率60.5%・平均+0.695%（2年）
                  walk-forward 4分割点×train/testの8パターン全てで勝率55%超
                  ブロックブートストラップ2000回で赤字確率9.1%

【2段階方式（戦略Dと同じ発想）】
    Stage1（前日夜、daily_pricesのDBデータのみで完結・API呼び出しなし）:
        WEAK地合い判定 + 下落幅・RBスコア・下ヒゲ条件で候補を絞り込み、
        out/w_overstay_stage1_candidates.csv に保存
    Stage2（翌朝寄り付き、ライブAPIで始値を取得）:
        Stage1候補の始値を取得しgap_pct条件を確認、当日朝の地合いを実測して発注

実行方法:
    python strategy_w_overstay.py --stage1          # 前日夜（scan_daily.py等から呼ぶ想定）
    python strategy_w_overstay.py --stage2           # 翌朝寄り付き
"""
import sys
import os
import json
import time
import argparse
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import urllib3
import jpholiday

sys.path.insert(0, os.path.dirname(__file__))
import db
import tachibana_order

JST = timezone(timedelta(hours=9))


def _is_trading_day(d=None):
    """東証の営業日判定（deploy/run_daily_cron.shと同一ロジックを複製）。
    土日・祝日に加え、祝日扱いでない年末年始休場(12/31, 1/2, 1/3)も除外する。
    独立したcrontabエントリで起動するため、休日判定もこのファイル内で完結させる。
    """
    d = d or datetime.now(JST).date()
    tse_year_end_new_year = (d.month == 12 and d.day == 31) or (d.month == 1 and d.day in (2, 3))
    return d.weekday() < 5 and not jpholiday.is_holiday(d) and not tse_year_end_new_year


_BASE_DIR = os.path.dirname(__file__)
STAGE1_CSV = os.path.join(_BASE_DIR, "out", "w_overstay_stage1_candidates.csv")
TACHIBANA_LOGIN_FILE = os.path.join(_BASE_DIR, "tachibana_login_response.json")

# ── 地合い判定用（closing_watch.pyと同一値だが独立管理） ──
PANIC_NIKKEI, PANIC_AD   = -2.0, 0.20
WEAK_NIKKEI,  WEAK_AD    = -1.0, 0.35
STRONG_NIKKEI, STRONG_AD =  0.5, 0.60

# ── 戦略W_OVERSTAY 判定条件（2026-08-14 GA探索+walk-forward検証済み。変更する場合は再検証すること）──
DROP_LO      = -4.2
DROP_HI      = -3.0
RB_MIN       = 4
LOWER_SHADOW_MIN = 0.35
GAP_MAX      = 4.4
MIN_VOLUME   = 50_000
MIN_PRICE    = 1_000
MAX_PRICE    = 2_500     # 2026-08-14新設: 試験運用中は建玉サイズ(100株固定)を10万円前後に
                          # 抑えるための上限。株価帯検証でも1,000〜2,500円帯(N=109)の方が
                          # 2,500円以上(N=106)より好成績(勝率63.3%/avg+0.875% vs
                          # 57.5%/avg+0.492%)で、walk-forward4分割点でも逆転なし一貫して
                          # 55%超だったため、資金抑制と成績の両面で妥当と判断

TP_PCT = 0.052
SL_PCT = 0.052

MAX_POSITIONS_W  = 10    # 2026-08-14: 3→10。2年間で候補が10件を超えた日は3日のみだが、
                          # その3日(2025-03-04/03-11、2026-07-17)は勝率86〜96%・平均+2.4〜2.9%
                          # と全体平均(勝率60.5%・avg+0.695%)を大きく上回っていた（候補が
                          # 多い日=市場全体が大きく崩れた日ほどV字回復が強く効くと推測）。
                          # 上限3件のままだと好機会の大半を取りこぼすため10に引き上げ。
                          # ただしサンプルはわずか3日分のため、上限なしにはせず抑制的な値に留めた。
DEFAULT_SHARES_W = 100   # 戦略Bの半分（試験運用）

STAGE1_MAX_AGE_HOURS = 20   # Stage1(前日夜)→Stage2(翌朝)の想定間隔より少し余裕を持たせた値。
                             # これを超えて古い場合、Stage1がその夜実行されなかった（cron失敗等）
                             # 可能性が高く、古い候補で誤発注するのを防ぐため見送る。


# ══════════════════════════════════════════════
# 指標計算（closing_watch.pyから複製・独立化）
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


def _classify_condition(ad_ratio, nikkei_change=None):
    if ad_ratio <= PANIC_AD:
        if nikkei_change is None or nikkei_change <= PANIC_NIKKEI:
            return "PANIC"
        return "WEAK"
    if ad_ratio >= STRONG_AD:
        if nikkei_change is None or nikkei_change >= STRONG_NIKKEI:
            return "STRONG"
        return "NORMAL"
    if ad_ratio <= WEAK_AD:
        return "WEAK"
    return "NORMAL"


# ══════════════════════════════════════════════
# Stage1: 前日夜、DBデータのみで候補を確定
# ══════════════════════════════════════════════
def stage1_scan(as_of_date=None):
    """daily_pricesのデータだけで当日の地合い判定＋候補抽出を行う。
    API呼び出しは一切発生しない（AN/AS/D候補スキャンと同じ設計思想）。
    """
    conn = db.get_conn()
    codes = db.get_all_codes(exclude_etf=True)

    if as_of_date is None:
        row = conn.execute("SELECT MAX(Date) FROM daily_prices").fetchone()
        scan_date = row[0] if row else None
    else:
        scan_date = str(as_of_date)

    if not scan_date:
        print("  ⚠️  daily_pricesにデータがありません")
        return []

    day_df = pd.read_sql(
        "SELECT code, Open, High, Low, Close, Volume, prev_close FROM daily_prices WHERE Date=?",
        conn, params=(scan_date,)
    )
    day_df = day_df[day_df["code"].isin(codes)]
    valid = day_df[day_df["prev_close"] > 0].copy()
    valid["chg"] = (valid["Close"] - valid["prev_close"]) / valid["prev_close"] * 100

    if len(valid) == 0:
        print(f"  ⚠️  {scan_date}: 有効なデータがありません")
        return []

    up_ratio = (valid["chg"] > 0).mean()
    condition = _classify_condition(up_ratio, nikkei_change=None)
    print(f"  {scan_date}  AD比率{up_ratio:.2f} → {condition}")

    if condition != "WEAK":
        print(f"  地合い{condition} → 戦略W_OVERSTAY見送り（WEAK限定）")
        _save_stage1_csv([], scan_date)
        return []

    cand = valid[(valid["chg"] > DROP_LO) & (valid["chg"] <= DROP_HI)
                 & (valid["Volume"] >= MIN_VOLUME)
                 & (valid["Close"] >= MIN_PRICE) & (valid["Close"] < MAX_PRICE)]

    rows = []
    for _, r in cand.iterrows():
        code = r["code"]
        hist = pd.read_sql(
            "SELECT Date, Open, High, Low, Close, Volume FROM daily_prices "
            "WHERE code=? AND Date <= ? ORDER BY Date DESC LIMIT 26",
            conn, params=(code, scan_date)
        )
        if len(hist) < 26:
            continue
        hist = hist.iloc[::-1].reset_index(drop=True)

        rb_score = calc_rebound_score(hist)
        if rb_score < RB_MIN:
            continue

        t_o, t_h, t_l, t_c = r["Open"], r["High"], r["Low"], r["Close"]
        day_range = t_h - t_l
        lower_shadow = min(t_o, t_c) - t_l
        lower_shadow_ratio = round(lower_shadow / day_range, 3) if day_range > 0 else 0.0
        if lower_shadow_ratio < LOWER_SHADOW_MIN:
            continue

        rows.append({
            "scan_date": scan_date, "code": code, "close": t_c,
            "today_rise": round(r["chg"], 2), "rb_score": rb_score,
            "lower_shadow_ratio": lower_shadow_ratio,
        })

    conn.close()
    print(f"  戦略W_OVERSTAY候補: {len(rows)}件")
    _save_stage1_csv(rows, scan_date)
    return rows


def _save_stage1_csv(rows, scan_date):
    os.makedirs(os.path.join(_BASE_DIR, "out"), exist_ok=True)
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["scan_date", "code", "close", "today_rise", "rb_score", "lower_shadow_ratio"])
    df.to_csv(STAGE1_CSV, index=False, encoding="utf-8-sig")


# ══════════════════════════════════════════════
# Stage2: 翌朝寄り付き、ギャップ確認して発注
# ══════════════════════════════════════════════
def _load_url_price():
    try:
        with open(TACHIBANA_LOGIN_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("sUrlPrice") or None
    except Exception:
        return None


def _fetch_open_prices(url_price, codes):
    """始値取得（closing_watch._fetch_price_batchと同じAPI・独立実装）"""
    if not codes:
        return {}
    http = urllib3.PoolManager()
    out = {}
    for i in range(0, len(codes), 120):
        chunk = codes[i:i+120]
        code_list = ",".join(str(c) for c in chunk)
        time.sleep(0.6)   # p_sd_date競合（p_errno=6/8）を避けるため他プロセスと時間をずらす
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
            '"sTargetColumn":"pDOP,pDPP,pPRP",'
            '"sJsonOfmt":"5"'
            "}"
        )
        try:
            resp = http.request("GET", url_price + "?" + params,
                                timeout=urllib3.Timeout(connect=3, read=8))
            tachibana_order.log_api_call("strategy_w_overstay.fetch_open_prices")
            result = json.loads(resp.data.decode("shift-jis", errors="ignore"))
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
                open_p = _f("pDOP") or _f("pDPP") or _f("pPRP")
                if open_p:
                    out[code] = open_p
        except Exception:
            continue
    return out


def _check_entry_day_condition(url_price):
    """当日朝の始値ベースでAD比率を計算し、STRONG/NORMALかどうかを返す。

    2026-08-14追加: WEAK当日に候補を抽出しても、実際に買うのは翌営業日の
    ため、その日の地合いがSTRONG/NORMALに転じているとは限らない
    （WEAKが連続する・PANICに転じるケースもある）。検証の結果、
    エントリー日がSTRONG/NORMALの時はwalk-forwardでも一貫してプラス
    （N=93 勝率65.6% avg+1.106%）だが、WEAK継続/PANICの時は明確に
    マイナス（N=16 avg-0.469%）だったため、発注直前に実測して見送る。
    全銘柄スキャンのため戦略Bのscan_dropping_stocks相当のAPI負荷が
    かかるが、Wが実際に候補を持つ日（月2回程度）のみ発生するため軽微。
    """
    codes = db.get_all_codes(exclude_etf=True)
    http = urllib3.PoolManager()
    up = down = 0
    for i in range(0, len(codes), 120):
        chunk = codes[i:i+120]
        code_list = ",".join(str(c) for c in chunk)
        time.sleep(0.6)   # p_sd_date競合（p_errno=6/8）を避けるため他プロセスと時間をずらす
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
            '"sTargetColumn":"pDOP,pPRP",'
            '"sJsonOfmt":"5"'
            "}"
        )
        try:
            resp = http.request("GET", url_price + "?" + params,
                                timeout=urllib3.Timeout(connect=3, read=8))
            tachibana_order.log_api_call("strategy_w_overstay.check_entry_day_condition")
            result = json.loads(resp.data.decode("shift-jis", errors="ignore"))
            for item in result.get("aCLMMfdsMarketPrice", []):
                def _f(key):
                    v = item.get(key, "")
                    if isinstance(v, str):
                        v = v.strip('"')
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        return None
                open_p = _f("pDOP")
                prev   = _f("pPRP")
                if open_p and prev and prev > 0:
                    if open_p > prev:
                        up += 1
                    else:
                        down += 1
        except Exception:
            continue

    total = up + down
    ad_ratio = up / total if total > 0 else 0.5
    condition = _classify_condition(ad_ratio, nikkei_change=None)
    print(f"  当日朝の地合い実測: AD比率{ad_ratio:.2f}（上昇{up}/下落{down}） → {condition}")
    return condition


def stage2_order(url_price=None, url_request=None):
    """Stage1候補の始値を取得し、gap_pct条件を満たす銘柄を発注する。"""
    if not os.path.exists(STAGE1_CSV):
        print("  ⚠️  Stage1候補ファイルがありません（前日夜のstage1_scan未実行）")
        return

    age_hours = (datetime.now(JST).timestamp() - os.path.getmtime(STAGE1_CSV)) / 3600
    if age_hours > STAGE1_MAX_AGE_HOURS:
        print(f"  ⚠️  Stage1候補ファイルが古すぎます（{age_hours:.1f}時間前）"
              f"→ 前日夜のstage1_scanが未実行の疑いがあるため見送り")
        return

    df = pd.read_csv(STAGE1_CSV, encoding="utf-8-sig")
    if df.empty:
        print("  戦略W_OVERSTAY: 前日はWEAK日でないか候補0件でした")
        return

    url_price   = url_price   or _load_url_price()
    url_request = url_request or tachibana_order.load_url_request()
    if not url_price or not url_request:
        print("  ⚠️  APIセッションが取得できません")
        return

    entry_condition = _check_entry_day_condition(url_price)
    if entry_condition not in ("STRONG", "NORMAL"):
        print(f"  戦略W_OVERSTAY: 当日の地合いが{entry_condition}（WEAK継続 or PANIC）のため発注見送り")
        return

    codes = df["code"].astype(str).tolist()
    open_prices = _fetch_open_prices(url_price, codes)

    open_pos = db.load_open_positions(strategy="W_OVERSTAY")
    order_count = len(open_pos)

    for _, r in df.iterrows():
        if order_count >= MAX_POSITIONS_W:
            print(f"  戦略W_OVERSTAY: 上限{MAX_POSITIONS_W}件に到達、以降スキップ")
            break
        code = str(r["code"])
        open_p = open_prices.get(code)
        if not open_p or open_p <= 0:
            continue
        gap_pct = (open_p - r["close"]) / r["close"] * 100
        if gap_pct > GAP_MAX:
            print(f"  [{code}] ギャップ+{gap_pct:.2f}% > 上限{GAP_MAX}% → 見送り")
            continue

        mkt_code = db.get_market_code_db(code)
        result = tachibana_order.place_buy_order(url_request, code, DEFAULT_SHARES_W, market_code=mkt_code)
        if result["success"]:
            print(f"  ✅ [{code}] 買い成功: {DEFAULT_SHARES_W}株 gap{gap_pct:+.2f}%  {result['message']}")
            eigyou_day = (result.get("raw") or {}).get("sEigyouDay")
            tachibana_order.save_position(
                code, code, DEFAULT_SHARES_W, open_p, strategy="W_OVERSTAY",
                tp_pct=TP_PCT, sl_pct=SL_PCT,
                entry_change_pct=r["today_rise"], rb_score=r["rb_score"],
                condition="WEAK", url_request=url_request,
                order_no=result.get("order_no"), eigyou_day=eigyou_day,
            )
            order_count += 1
        else:
            print(f"  ❌ [{code}] 発注失敗: {result['message']}")


def main():
    parser = argparse.ArgumentParser(description="戦略W_OVERSTAY（WEAK日の翌営業日エントリー）")
    parser.add_argument("--stage1", action="store_true", help="前日夜: 候補抽出")
    parser.add_argument("--stage2", action="store_true", help="翌朝: ギャップ確認・発注")
    parser.add_argument("--as-of-date", type=str, default=None, help="stage1のバックテスト用日付指定")
    args = parser.parse_args()

    if args.stage1:
        if args.as_of_date is None and not _is_trading_day():
            print("  休日（土日・祝日・年末年始休場）のためstage1をスキップ")
            return
        stage1_scan(as_of_date=args.as_of_date)
    elif args.stage2:
        if not _is_trading_day():
            print("  休日（土日・祝日・年末年始休場）のためstage2をスキップ")
            return
        stage2_order()
    else:
        print("  --stage1 または --stage2 を指定してください")


if __name__ == "__main__":
    main()
