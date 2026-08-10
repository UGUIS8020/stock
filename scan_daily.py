"""
scan_daily.py - 統合スキャナー（毎日16:30以降に実行）
【戦略A】順張り: 出来高急増スコア上位銘柄
【戦略C】暴落逆行高: 地合いPANIC日に上昇した銘柄（材料株分析用）

実行方法:
    python scan_daily.py              # 本日データで実行
    python scan_daily.py --date 20260522  # 指定日データで実行（週末に前営業日分を処理する場合など）
"""
import sys
import argparse
import jquantsapi
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from datetime import datetime
import re
import requests
import time
from bs4 import BeautifulSoup
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
cli = jquantsapi.ClientV2(api_key=os.getenv("JQUANTS_API_KEY"))

import db as _db
_db.init_db()

import tachibana_order
from scan_morning_strategy_c import scan_strategy_c, verify_strategy_c

_BASE_DIR = Path(__file__).parent
CACHE_DIR        = str(_BASE_DIR / "out" / "cache")
SCAN_CSV         = str(_BASE_DIR / "out" / "scan_results.csv")
MARKET_LOG_CSV   = str(_BASE_DIR / "out" / "market_log.csv")
STRATEGY_C_CSV   = str(_BASE_DIR / "out" / "strategy_c_log.csv")
BACKTEST_LOG_CSV    = str(_BASE_DIR / "out" / "backtest_log.csv")
CANDIDATES_LOG_CSV  = str(_BASE_DIR / "out" / "candidates_log.csv")
BANK_RESULTS_CSV    = str(_BASE_DIR / "out" / "bank_results.csv")
TOP_N            = 30
TOP_N_LOW        = 30   # 2026-08-10追加: score降順TOP_N=30とは別に、SCORE_MIN_Aに近い
                         # （昇順）候補も別途保存する。STRONG日の実際の売買選定
                         # (market_watch.pyのMAX_WATCH_A)はこちらを優先する設計に変更したため。
                         # 根拠: STRONG日はscore降順(高score優先)だと勝率46〜48%/平均ほぼ0%だが、
                         # score昇順(3.0に近い順)だと勝率62〜65%/平均+0.58〜+0.89%
                         # （4分割点全てでwalk-forward確認済み、2026-08-10）。
                         # 既存のTOP_N=30(降順)はWEAK日保留機構(score>=5.0が必要)向けに
                         # そのまま残し、追加する形にして既存ロジックへの影響を避けている。
SCORE_MIN_A      = 3.0
MIN_PRICE        = 100
MIN_VOLUME       = 50_000
MIN_TURNOVER     = 50_000_000

# ★地合い判定閾値
PANIC_NIKKEI_THRESHOLD  = -2.0
PANIC_AD_THRESHOLD      = 0.20
WEAK_NIKKEI_THRESHOLD   = -1.0
WEAK_AD_THRESHOLD       = 0.35
STRONG_NIKKEI_THRESHOLD = 0.5
STRONG_AD_THRESHOLD     = 0.60

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(str(_BASE_DIR / "out"), exist_ok=True)

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--date", default=None, help="処理対象日 YYYYMMDD（省略時=本日）")
_args, _ = _parser.parse_known_args()

if _args.date:
    TODAY_STR = _args.date
    TODAY     = f"{_args.date[:4]}-{_args.date[4:6]}-{_args.date[6:]}"
else:
    TODAY     = datetime.now().strftime("%Y-%m-%d")
    TODAY_STR = datetime.now().strftime("%Y%m%d")


# ══════════════════════════════════════════════
# ★ 地合い判定
# ══════════════════════════════════════════════
def judge_market_condition(df_today):
    up_count   = (df_today["today_rise"] > 0).sum()
    down_count = (df_today["today_rise"] < 0).sum()
    flat_count = (df_today["today_rise"] == 0).sum()
    total      = up_count + down_count + flat_count
    ad_ratio   = up_count / total if total > 0 else 0.5

    nikkei_change = None
    try:
        nk = df_today[df_today["code4"] == "0000"]
        if nk.empty:
            nikkei_change = float(df_today["today_rise"].median())
        else:
            nikkei_change = float(nk.iloc[0]["today_rise"])
    except Exception:
        nikkei_change = float(df_today["today_rise"].median())

    is_panic = (
        (nikkei_change is not None and nikkei_change <= PANIC_NIKKEI_THRESHOLD) or
        ad_ratio <= PANIC_AD_THRESHOLD
    )
    is_weak = (
        (nikkei_change is not None and nikkei_change <= WEAK_NIKKEI_THRESHOLD) or
        ad_ratio <= WEAK_AD_THRESHOLD
    )

    is_strong = (
        (nikkei_change is not None and nikkei_change >= STRONG_NIKKEI_THRESHOLD) and
        ad_ratio >= STRONG_AD_THRESHOLD
    )

    if is_panic:
        condition = "PANIC"
    elif is_weak:
        condition = "WEAK"
    elif is_strong:
        condition = "STRONG"
    else:
        condition = "NORMAL"

    return {
        "condition": condition,
        "ad_ratio": round(ad_ratio, 4),
        "nikkei_change": round(nikkei_change, 2) if nikkei_change is not None else None,
        "up_count": int(up_count),
        "down_count": int(down_count),
        "flat_count": int(flat_count),
        "total": int(total),
    }


def print_market_condition(mc):
    cond   = mc["condition"]
    ad_pct = mc["ad_ratio"] * 100
    nk     = mc["nikkei_change"]
    nk_str = f"{nk:+.2f}%" if nk is not None else "取得不可"

    print(f"\n{'='*60}")
    print(f"【地合い判定】")
    print(f"  騰落比     : {ad_pct:.1f}%"
          f"（値上がり{mc['up_count']}件 / 値下がり{mc['down_count']}件）")
    print(f"  日経推定   : {nk_str}")

    if cond == "PANIC":
        print(f"  判定       : 🚨 PANIC（市場大暴落）")
        print(f"")
        print(f"  ⛔  戦略A 本日エントリー非推奨")
        print(f"  ⛔  スキャン結果は参考表示のみ。ノーポジが最強戦略です。")
        print(f"  📊 戦略C（逆行高銘柄）のみ参考にしてください")
    elif cond == "WEAK":
        print(f"  判定       : ⚠️  WEAK（地合い軟調）")
        print(f"")
        print(f"  ⚠️  戦略A は見送り推奨")
        print(f"  ⚠️  closing_watch は見送り")
    elif cond == "STRONG":
        print(f"  判定       : 🚀 STRONG（強気相場）")
        print(f"")
        print(f"  🚀  戦略A 積極エントリー推奨")
        print(f"  ✅  closing_watch 14:43に起動推奨")
    else:
        print(f"  判定       : ✅ NORMAL（通常相場）")
    print(f"{'='*60}")

    return cond


def save_market_log(mc, strategy_a_success_rate=None):
    log_row = {
        "date":                    TODAY,
        "condition":               mc["condition"],
        "ad_ratio":                mc["ad_ratio"],
        "nikkei_change":           mc["nikkei_change"],
        "up_count":                mc["up_count"],
        "down_count":              mc["down_count"],
        "strategy_a_success_rate": strategy_a_success_rate,
    }
    _db.save_market_log_db(log_row)


# ══════════════════════════════════════════════
# ★ 戦略C: 暴落逆行高銘柄
# ══════════════════════════════════════════════
# → scan_morning_strategy_c.py に移動済み


# ══════════════════════════════════════════════
# スコアリング関数
# ══════════════════════════════════════════════
def calc_score(hist):
    if len(hist) < 22:
        return None
    vol_series = pd.to_numeric(hist["Volume"], errors="coerce").fillna(0)
    vol_20d = vol_series.iloc[-20:].values
    vol_5d  = vol_series.iloc[-5:].values
    v1 = float(vol_series.iloc[-1])
    v3 = float(vol_series.iloc[-3])
    avg20 = vol_20d.mean()
    avg5  = vol_5d.mean()
    if avg20 == 0 or avg5 == 0 or v3 == 0:
        return None
    trend = np.polyfit(range(5), vol_5d, 1)[0] / avg5 * 100
    accel = (v1 - v3) / v3 * 100
    ratio = v1 / avg20
    total = min(trend/10, 3.0) + min(accel/100, 3.0) + min(ratio/3, 4.0)
    return {"trend": round(trend,2), "accel": round(accel,2),
            "ratio": round(ratio,2), "score": round(total,2)}


def update_cache(code4, row):
    _db.upsert_daily_price(code4, row)




# verify_strategy_c → scan_morning_strategy_c.py に移動済み


# ══════════════════════════════════════════════
# TDnet 適時開示取得
# ══════════════════════════════════════════════
TDNET_BASE = "https://www.release.tdnet.info/inbs/"
TDNET_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36"
}

# 好材料キーワード（タイトルに含まれる場合 → 格上げ）
POSITIVE_KEYWORDS = [
    "資本業務提携", "業務提携", "資本提携", "合併", "買収", "TOB",
    "自社株買い", "自己株式取得", "増配", "特別配当", "復配",
    "黒字転換", "上方修正", "業績予想の修正", "純利益", "営業利益.*上方",
    "新製品", "新サービス", "特許取得", "受注", "大型契約",
    "MBO", "上場廃止勧告", "公開買付",
]
# 悪材料キーワード（タイトルに含まれる場合 → 格下げ）
NEGATIVE_KEYWORDS = [
    "下方修正", "業績予想の修正.*減", "特別損失", "損失計上",
    "不祥事", "不正", "訴訟", "行政処分", "上場廃止",
    "債務超過", "破産", "民事再生", "希望退職",
]


def fetch_tdnet(date_str, target_codes):
    """
    TDnetから指定日の適時開示を取得し、対象銘柄コードでフィルタリングする。
    date_str: YYYYMMDD形式
    target_codes: set of str（4桁コード）
    戻り値: {code: [{"time": "16:30", "title": "...", "sentiment": "好材料/悪材料/中立"}]}
    """
    results = {}
    page = 1
    while True:
        url = f"{TDNET_BASE}I_list_{page:03d}_{date_str}.html"
        try:
            res = requests.get(url, headers=TDNET_HEADERS, timeout=10)
            if res.status_code != 200:
                break
            res.encoding = "utf-8"
            soup = BeautifulSoup(res.content, "html.parser")
            rows = soup.select("table#main-list-table tr")
            if not rows:
                break
            for tr in rows:
                tds = tr.find_all("td")
                rec = {}
                for td in tds:
                    classes = td.get("class") or []
                    classes = classes if isinstance(classes, list) else [classes]
                    if "kjTime"  in classes: rec["time"]  = td.text.strip()
                    if "kjCode"  in classes: rec["code"]  = td.text.strip()
                    if "kjTitle" in classes: rec["title"] = td.text.strip()
                if not rec.get("code") or rec["code"] not in target_codes:
                    continue
                title = rec.get("title", "")
                if any(re.search(kw, title) for kw in POSITIVE_KEYWORDS):
                    sentiment = "好材料"
                elif any(re.search(kw, title) for kw in NEGATIVE_KEYWORDS):
                    sentiment = "悪材料"
                else:
                    sentiment = "中立"
                entry = {"time": rec.get("time", ""), "title": title, "sentiment": sentiment}
                results.setdefault(rec["code"], []).append(entry)
            page += 1
            time.sleep(0.5)
        except Exception:
            break
    return results




def verify_candidates_log(df_today, name_dict):
    """scan_morning.py の BUY/CAUTION 判定を当日終値で検証し candidates_log DB に書き戻す"""
    cl = _db.get_candidates_log(date=TODAY)
    if cl.empty:
        return

    # 未検証（result_pct が NaN）の BUY/CAUTION のみ対象
    targets = cl[cl["judgment"].isin(["BUY", "CAUTION"]) & cl["result_pct"].isna()]
    if targets.empty:
        return

    print(f"\n=== 📋 【朝スキャン検証】本日 scan_morning.py 判定の結果 ===")
    tp_pct = 3.0
    sl_pct = -1.0  # market_watch.py の SL_PCT と統一
    updated = 0
    win = loss = tp_hit = sl_hit = 0
    today_results = []  # 当日分の結果を集計用に保持

    for _, row in targets.iterrows():
        code4 = str(row["code"])
        t = df_today[df_today["code4"] == code4]
        if t.empty or float(t.iloc[0]["Open"]) <= 0:
            continue
        t = t.iloc[0]
        open_p  = float(t["Open"])
        close_p = float(t["Close"])
        high_p  = float(t["High"])
        low_p   = float(t["Low"])

        ret_close = round((close_p - open_p) / open_p * 100, 2)
        ret_high  = round((high_p  - open_p) / open_p * 100, 2)
        ret_low   = round((low_p   - open_p) / open_p * 100, 2)

        if ret_high >= tp_pct and ret_low > sl_pct:
            grade = f"TP+{tp_pct:.0f}%"; result_pct = tp_pct; tp_hit += 1; win += 1
        elif ret_low <= sl_pct and ret_high < tp_pct:
            grade = f"SL{sl_pct:.0f}%"; result_pct = sl_pct; sl_hit += 1; loss += 1
        elif ret_close >= 0:
            grade = "引け+"; result_pct = ret_close; win += 1
        else:
            grade = "引け-"; result_pct = ret_close; loss += 1

        _db.update_candidates_result(
            row["date"], row["strategy"], code4,
            open_p, close_p, high_p, low_p, result_pct, grade)

        today_results.append({"judgment": row["judgment"], "result_pct": result_pct})
        jdg  = str(row.get("judgment", ""))
        strat = str(row.get("strategy", ""))
        name  = name_dict.get(code4, str(row.get("name", "")))
        icon  = "✅" if grade.startswith("TP") else ("❌" if grade.startswith("SL") else ("⚠️" if ret_close >= 0 else "❌"))
        print(f"  {icon} [{code4}]{name}  {jdg}({strat})  "
              f"寄:{open_p:.0f}→終:{close_p:.0f}  実損益:{result_pct:+.1f}%  引け:{ret_close:+.1f}%  [{grade}]")
        updated += 1

    if updated > 0:
        total = win + loss
        wr    = win / total * 100 if total > 0 else 0
        avg   = sum(r["result_pct"] for r in today_results) / len(today_results)
        print(f"  ── 本日集計: {total}件  勝率{wr:.0f}%  TP到達:{tp_hit}件  SL到達:{sl_hit}件  平均{avg:+.2f}%")

        # 累積集計（DB全件から再取得）
        all_cl = _db.get_candidates_log()
        verified = all_cl[all_cl["result_pct"].notna()].copy()
        buy_v = verified[verified["judgment"] == "BUY"]
        if len(buy_v) > 0:
            bwr  = (buy_v["result_pct"] >= 0).mean() * 100
            bavg = buy_v["result_pct"].mean()
            print(f"  ── BUY累積: {len(buy_v)}件  勝率{bwr:.0f}%  平均{bavg:+.2f}%")


def verify_scan_a(name_dict, df_today):
    df = _db.get_scan_results()
    if df.empty:
        return None
    unverified = df[(df["scan_date"] == df["scan_date"].max()) & df["actual_top20"].isna()]
    if unverified.empty:
        return None

    print(f"\n=== 📋 【戦略A】前日順張り候補の検証 ===")
    tp_hit = near_hit = miss = down = 0

    backtest_rows = []
    scan_date = df["scan_date"].max()

    morning_judgments = {}
    try:
        # scan_dateの翌営業日を計算（土日をスキップ）
        next_date_ts = pd.Timestamp(scan_date) + pd.Timedelta(days=1)
        if next_date_ts.weekday() == 5:
            next_date_ts += pd.Timedelta(days=2)
        elif next_date_ts.weekday() == 6:
            next_date_ts += pd.Timedelta(days=1)
        next_date = next_date_ts.strftime("%Y-%m-%d")
        cl_prev = _db.get_candidates_log(date=next_date)
        morning_judgments = dict(zip(cl_prev["code"].astype(str), cl_prev["judgment"]))
    except Exception:
        pass

    for _, row in unverified.iterrows():
        code4 = str(row["code"])
        name  = name_dict.get(code4, "")
        t        = df_today[df_today["code4"] == code4]

        if not t.empty and t.iloc[0]["Open"] > 0:
            t_row    = t.iloc[0]
            open_p   = float(t_row["Open"])
            close_p  = float(t_row["Close"])
            high_p   = float(t_row["High"])                          # ← 追加
            rise     = round((close_p - open_p) / open_p * 100, 2)  # 終値ベース（実損益）
            max_rise = round((high_p  - open_p) / open_p * 100, 2)  # ← 追加（高値ベース）
            rise_str = f"{rise:>+.1f}% (高値:{max_rise:>+.1f}%)"    # ← 変更
        else:
            open_p = close_p = high_p = rise = max_rise = None
            rise_str = "  N/A"

        # ── グレード判定（高値ベース、戦略A利確目標+3%基準）──────────────────
        is_hit = max_rise is not None and max_rise >= 3.0
        if is_hit:
            status = "✅ TP到達(高値+3%超)"; grade = "TP";   tp_hit   += 1
        elif max_rise is not None and max_rise >= 2.0:
            status = "✨ 惜しい(高値+2%台)"; grade = "NEAR"; near_hit += 1
        elif max_rise is not None and max_rise >= 0:
            status = "📈 小幅プラス";         grade = "SMALL"; miss    += 1
        elif max_rise is not None and max_rise < 0:
            status = "❌ 逆行";               grade = "FAIL";  down    += 1
        else:
            status = "➖ データなし";          grade = "N/A";   miss    += 1

        df.loc[(df["scan_date"] == row["scan_date"]) &
               (df["code"] == row["code"]), "actual_top20"] = int(is_hit)
        print(f"  {status} [{code4}]{name}  {rise_str}  スコア:{row['score']:.2f}")

        backtest_rows.append({
            "verify_date":      TODAY,
            "scan_date":        scan_date,
            "code":             code4,
            "name":             name,
            "score":            row["score"],
            "ratio":            row.get("ratio"),
            "morning_judgment": morning_judgments.get(code4, "UNKNOWN"),
            "open_price":       open_p,
            "close_price":      close_p,
            "high_price":       high_p,        # ← 追加
            "result_pct":       rise,          # 終値ベース（実損益）
            "max_rise_pct":     max_rise,      # ← 追加（高値ベース）
            "grade":            grade,
            "is_top20":         int(is_hit),
            "market_condition": row.get("market_condition", "UNKNOWN"),
        })

    total = len(unverified)
    success_rate = None
    if total > 0:
        success_rate = round(tp_hit / total * 100, 1)
        print(f"\n  【集計】")
        print(f"  ✅ TP到達(高値+3%超): {tp_hit}件")
        print(f"  ✨ 惜しい(高値+2%台): {near_hit}件")
        print(f"  📈 小幅プラス       : {miss}件")
        print(f"  ❌ 逆行             : {down}件")
        print(f"  TP到達率(高値+3%以上): {tp_hit}/{total}件 ({success_rate}%)")

    _db.save_scan_results(df)

    if backtest_rows:
        new_bt = pd.DataFrame(backtest_rows)
        if os.path.exists(BACKTEST_LOG_CSV):
            ex_bt = pd.read_csv(BACKTEST_LOG_CSV, encoding="utf-8-sig")
            ex_bt = ex_bt[ex_bt["scan_date"] != scan_date]
            new_bt = pd.concat([ex_bt, new_bt], ignore_index=True)
        new_bt.to_csv(BACKTEST_LOG_CSV, index=False, encoding="utf-8-sig")
        print(f"\n  ✅ バックテストログ保存: {BACKTEST_LOG_CSV}（{len(backtest_rows)}件）")

    return success_rate


def show_market_log_summary():
    df = _db.get_market_log()
    df = df[df["strategy_a_success_rate"].notna()]
    if len(df) < 3:
        return

    print(f"\n{'='*60}")
    print(f"【地合い×戦略A 成功率 累積分析】（{len(df)}日分）")
    print(f"{'='*60}")
    for cond in ["STRONG", "NORMAL", "WEAK", "PANIC"]:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        avg = sub["strategy_a_success_rate"].mean()
        print(f"  {cond:<8}: {len(sub):>3}日  平均成功率 {avg:>5.1f}%")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════
# 分足データ収集（scan_daily.py末尾に追加）
# ══════════════════════════════════════════════════════

INTRADAY_DIR = str(_BASE_DIR / "out" / "intraday")

# 戦略D固定銘柄
STRATEGY_D_CODES = [
    "5016",  # JX金属
    "5713",  # 住友金属鉱山
    "5706",  # 三井金属
    "7203",  # トヨタ
    "8031",  # 三井物産
    "8035",  # 東京エレクトロン
    "6857",  # アドバンテスト
    "5020",  # ENEOS
    "4063",  # 信越化学工業
]

def save_intraday(scan_a_codes):
    """
    候補銘柄と固定銘柄の1分足データを保存する。
    yfinanceは直近7日分のみ取得可能なので毎日実行が必須。
    """
    import yfinance as yf
    os.makedirs(INTRADAY_DIR, exist_ok=True)

    # 保存対象をまとめる（重複除去）
    all_codes = set(scan_a_codes) | set(STRATEGY_D_CODES)

    print(f"\n{'='*60}")
    print(f"【分足データ収集】{len(all_codes)}銘柄")
    print(f"{'='*60}")

    saved = 0
    skipped = 0

    for code in sorted(all_codes):
        if _db.intraday_exists(code, TODAY):
            skipped += 1
            continue

        try:
            ticker = yf.Ticker(f"{code}.T")
            hist   = ticker.history(period="2d", interval="1m")

            if hist.empty:
                continue

            hist = hist.reset_index()
            hist.columns = [str(c) for c in hist.columns]

            # datetime列を特定
            dt_col = "Datetime" if "Datetime" in hist.columns else hist.columns[0]
            hist[dt_col] = pd.to_datetime(hist[dt_col])

            # タイムゾーン変換
            if hist[dt_col].dt.tz is not None:
                hist[dt_col] = hist[dt_col].dt.tz_convert("Asia/Tokyo")

            # 当日分のみ抽出
            hist["date_only"] = hist[dt_col].dt.strftime("%Y-%m-%d")
            today_hist = hist[hist["date_only"] == TODAY].copy()

            if today_hist.empty:
                continue

            today_hist["time"] = today_hist[dt_col].dt.strftime("%H:%M")

            # 保存列を整理
            save_df = today_hist.rename(columns={
                dt_col: "datetime",
                "Open":   "open",
                "High":   "high",
                "Low":    "low",
                "Close":  "close",
                "Volume": "volume",
            })

            cols = ["datetime", "time", "open", "high", "low", "close", "volume"]
            cols = [c for c in cols if c in save_df.columns]
            _db.save_intraday_df(code, TODAY, save_df[cols])
            saved += 1

        except Exception as e:
            print(f"  ⚠️ {code}: {e}")

    print(f"  ✅ 保存: {saved}銘柄  スキップ(既存): {skipped}銘柄")
    print(f"  📁 保存先: {INTRADAY_DIR}/")


# ══════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════
def main():
    print(f"=== 🔍 統合スキャナー（{TODAY}）===")
    print(f"  当日株価データを取得中...")

    try:
        df_today = cli.get_eq_bars_daily(date_yyyymmdd=TODAY_STR)
    except Exception as e:
        print(f"  エラー: {e}")
        return

    if df_today.empty:
        print("  データなし（市場休業日 or 16:30前）")
        return

    df_today = df_today.rename(columns={"O":"Open","H":"High","L":"Low","C":"Close","Vo":"Volume"})
    df_today["code4"] = df_today["Code"].astype(str).str[:4]
    df_today = df_today[
        df_today["code4"].str.match(r"^\d{4}$") |
        df_today["code4"].str.match(r"^\d{3}[A-Z]$")
    ]
    df_today = df_today[(df_today["Volume"] > 0) & (df_today["Close"] >= 50)].copy()
    df_today["today_rise"] = (df_today["Close"] - df_today["Open"]) / df_today["Open"] * 100
    df_today["turnover"]   = df_today["Volume"] * df_today["Close"]
    print(f"  {len(df_today)}銘柄のデータ取得完了")

    top20_codes = set(df_today.nlargest(20, "today_rise")["code4"].tolist())
    top_min = df_today.nlargest(20, "today_rise")["today_rise"].min()
    top_max = df_today.nlargest(20, "today_rise")["today_rise"].max()
    print(f"  本日TOP20: {top_min:.1f}%〜{top_max:.1f}%")

    try:
        master = cli.get_eq_master()
        master["code4"] = master["Code"].astype(str).str[:4]
        name_dict = dict(zip(master["code4"], master["CoName"]))
        exclude_keywords = ["ＥＴＦ", "ＥＴＮ", "投信", "ファンド", "リート", "ＲＥＩＴ",
                            "インデックス", "ｉシェアーズ", "ＮＥＸＴ", "上場投資"]
        def is_etf(row):
            name = str(row.get("CoName", ""))
            s17  = str(row.get("S17", ""))
            return s17 == "99" or any(kw in name for kw in exclude_keywords)
        exclude_codes = set(master[master.apply(is_etf, axis=1)]["code4"].tolist())
        print(f"  ETF/ETN/REIT除外: {len(exclude_codes)}銘柄")
    except Exception:
        name_dict = {}
        exclude_codes = set()

    # 増担保規制銘柄を除外（JPXサイトから自動取得）
    try:
        from fetch_kisoku_list import fetch_margin_restricted_codes
        margin_codes = fetch_margin_restricted_codes()
        if margin_codes:
            exclude_codes |= margin_codes
            print(f"  増担保規制除外: {len(margin_codes)}銘柄 {sorted(margin_codes)}")
        else:
            print("  増担保規制リスト: 取得失敗（スキップ）")
    except Exception as e:
        print(f"  増担保規制リスト: 取得エラー（{e}）→ スキップ")

    # 市場コードを JQuants から取得して DB に保存（sResultCode=11008 対策）
    try:
        df_list = cli.get_list()
        if not df_list.empty and "Mkt" in df_list.columns:
            df_list["code4"] = df_list["Code"].astype(str).str[:4]
            code_mkt = dict(zip(df_list["code4"], df_list["Mkt"].astype(str)))
            _db.upsert_market_codes(code_mkt)
            print(f"  市場コード更新: {len(code_mkt)}銘柄")
    except Exception as e:
        print(f"  市場コード取得スキップ: {e}")

    mc = judge_market_condition(df_today)
    condition = print_market_condition(mc)

    verify_strategy_c(df_today, name_dict)
    verify_candidates_log(df_today, name_dict)
    success_rate = verify_scan_a(name_dict, df_today)

    save_market_log(mc, success_rate)
    show_market_log_summary()

    # ── 全銘柄スコアリング ──
    print(f"\n  全銘柄スコアリング中...")
    results_a = []
    cached_codes = _db.get_all_codes()

    for code4 in cached_codes:
        today_row = df_today[df_today["code4"] == code4]
        if not today_row.empty:
            update_cache(code4, today_row.iloc[0])

        hist = _db.get_stock_history(code4)
        s = calc_score(hist)
        if s is None:
            continue

        name = name_dict.get(code4, "")
        if code4 in exclude_codes:
            continue
        if today_row.empty:
            continue

        t = today_row.iloc[0]
        vol      = float(t["Volume"])
        turnover = float(t["turnover"])
        if vol < MIN_VOLUME or turnover < MIN_TURNOVER:
            continue

        today_rise = float(t["today_rise"])
        close      = float(t["Close"])

        # 戦略A: 当日 -5%以上の暴落銘柄を除外（急落中の銘柄は順張り対象外）
        if today_rise > -5.0:
            results_a.append({
                "code": code4,
                "name": name,
                "today_rise": round(today_rise, 2),
                "close": close,
                **s
            })

    # ── 戦略A表示 ──
    df_a = pd.DataFrame(results_a).sort_values("score", ascending=False)
    today_surge = dict(zip(df_today["code4"], df_today["today_rise"].astype(float)))
    df_a["today_rise"] = df_a["code"].map(today_surge).fillna(0)
    # ratio >= 5.0 は翌朝 judge_entry_a_strong でPASS（勝率45%）のため先に除外
    # これにより翌朝の候補がSTRONG翌日でも有効な銘柄で埋まる
    RATIO_LIMIT_A = 5.0
    bank_mask = df_a["name"].str.contains("銀行|フィナンシャルグループ", na=False, regex=True)
    if bank_mask.sum() > 0:
        print(f"  🏦 銀行系除外: {bank_mask.sum()}件 {df_a[bank_mask]['name'].tolist()}")
    low_price_mask = df_a["close"] < MIN_PRICE
    if low_price_mask.sum() > 0:
        print(f"  💴 低位株除外: {low_price_mask.sum()}件 {df_a[low_price_mask]['name'].tolist()}")
    a_pool = df_a[
        (df_a["score"] >= SCORE_MIN_A) &
        (df_a["ratio"] < RATIO_LIMIT_A) &
        (~bank_mask) &
        (~low_price_mask)
    ]
    top_a = a_pool.head(TOP_N)

    # 2026-08-10追加: score降順とは別に、SCORE_MIN_Aに近い（昇順）候補も保存。
    # STRONG日の実際の売買選定に使う（TOP_Nコメント参照）。
    top_a_low = a_pool.sort_values("score", ascending=True).head(TOP_N_LOW)

    # 戦略F: 銀行候補を別ファイルに保存（翌朝WEAK地合い時に market_watch.py が活用）
    bank_f = df_a[
        bank_mask &
        (df_a["score"] >= SCORE_MIN_A) &
        (df_a["ratio"] < RATIO_LIMIT_A) &
        (~low_price_mask)
    ].copy()
    bank_f.to_csv(BANK_RESULTS_CSV, index=False, encoding="utf-8-sig")
    if len(bank_f) > 0:
        print(f"  🏦 戦略F候補: {len(bank_f)}件保存 → {', '.join(bank_f['name'].tolist())}")

    def get_rank(row):
        s = row["score"]
        surge = row.get("today_rise", 0)
        if 8.0 <= s <= 8.5:
            return "🏆S"
        elif 6.0 <= s < 8.0 and 0 <= surge <= 3:
            return "⭐A"
        elif s >= 9.0:
            return "⚠️ "
        else:
            return "  "

    SURGE_SCORE_MIN = 9.5
    SURGE_RATIO_MIN = 10.0

    print(f"\n{'='*60}")
    print(f"【戦略A】順張り候補 TOP{len(top_a)}（翌朝急騰狙い）")
    print(f"  🏆S=最優先(スコア8.0〜8.5)  ⭐A=初動狙い  ⚠️=過熱注意(9.0以上)  ⚡=急騰候補")

    if condition == "PANIC":
        print(f"  🚨 本日地合いPANIC: 参考表示のみ、エントリー非推奨")
    elif condition == "WEAK":
        print(f"  ⚠️  本日地合い軟調: 戦略Aは見送り推奨")

    print(f"{'='*60}")
    print(f"  {'ランク':<4} {'コード':<6} {'銘柄名':<16} {'スコア':>6} {'比率':>6} {'当日':>6} {'加速度':>8}")
    print("  " + "─" * 62)

    surge_list = []
    for _, r in top_a.iterrows():
        rank = get_rank(r)
        surge_str = f"{float(r['today_rise']):>+5.1f}%"
        is_surge = float(r["score"]) >= SURGE_SCORE_MIN and float(r["ratio"]) >= SURGE_RATIO_MIN
        surge_mark = " ⚡" if is_surge else ""
        print(f"  {rank:<4} {r['code']:<6} {r['name']:<16} {r['score']:>6.2f} "
              f"{r['ratio']:>5.1f}倍 {surge_str} {r['accel']:>+8.0f}%{surge_mark}")
        if is_surge:
            surge_list.append(r)

    if surge_list:
        print(f"\n  ⚡【急騰候補】翌朝は TP+8% / SL-1% 推奨（引け決済不可）")
        for r in surge_list:
            print(f"     [{r['code']}]{r['name']}  score:{r['score']:.1f}  ratio:{r['ratio']:.1f}倍"
                  f"  過去実績: 高値+10%到達率10.6% / avg+1.02%")

    # top_a(降順・WEAK保留機構等の従来用途向け)とtop_a_low(昇順・STRONG日の
    # 実際の売買選定向け、2026-08-10追加)を統合して保存。重複はcodeで除去。
    save_pool = pd.concat([top_a, top_a_low]).drop_duplicates(subset=["code"])
    save_a = [{"scan_date": TODAY, "code": r["code"], "name": r["name"],
               "score": r["score"], "trend": r["trend"], "accel": r["accel"],
               "ratio": r["ratio"], "today_rise": r.get("today_rise", 0),
               "actual_top20": pd.NA, "market_condition": condition}
              for _, r in save_pool.iterrows()]
    new_a = pd.DataFrame(save_a)
    _db.save_scan_results(new_a)

    scan_strategy_c(df_today, name_dict, exclude_codes, mc)

    # ── 分足データ収集 ──
    scan_a_codes = [r["code"] for r in save_a]
    save_intraday(scan_a_codes)

    print(f"\n✅ 完了")

    # ── S3にDBを自動アップロード ──
    # モックモード（LIVE_TRADING=False）ではモックポジションで共有S3のDBを汚染しないようアップロードしない。
    if not tachibana_order.LIVE_TRADING:
        print(f"🔧 [モックモード] S3アップロードはスキップします（LIVE_TRADING=False）")
    else:
        import boto3 as _boto3
        s3_bucket = os.getenv("S3_BUCKET", "shibuya8020")
        s3_key    = "stock-db/stock.db"
        db_path   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "stock.db")
        try:
            _s3 = _boto3.client("s3")
            _s3.upload_file(db_path, s3_bucket, s3_key)
            print(f"☁️  S3アップロード完了（s3://{s3_bucket}/{s3_key}）")
        except Exception as e:
            print(f"⚠️  S3アップロードエラー: {e}")


if __name__ == "__main__":
    main()
