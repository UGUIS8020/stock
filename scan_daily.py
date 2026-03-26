"""
scan_daily.py - 統合スキャナー（毎日16:30以降に実行）
【戦略A】順張り: 出来高急増スコア上位銘柄
【戦略B】逆張り: 当日-5%以下+リバウンドスコア3点以上（翌朝寄付き買い）
【戦略C】暴落逆行高: 地合いPANIC日に上昇した銘柄（材料株分析用）
"""
import jquantsapi
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from datetime import datetime

load_dotenv()
cli = jquantsapi.ClientV2(api_key=os.getenv("JQUANTS_API_KEY"))

CACHE_DIR        = "out/cache"
SCAN_CSV         = "out/scan_results.csv"
WATCHLIST_CSV    = "out/watchlist.csv"
MARKET_LOG_CSV   = "out/market_log.csv"
STRATEGY_C_CSV   = "out/strategy_c_log.csv"
BACKTEST_LOG_CSV = "out/backtest_log.csv"
TOP_N            = 20
SCORE_MIN_A      = 3.0
SCORE_MIN_B      = 3.0
SCORE_MAX_B      = 6.0
PRICE_DROP_B     = -10.0
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
os.makedirs("out", exist_ok=True)

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
        print(f"  ⛔  戦略A / 戦略B ともに本日エントリー非推奨")
        print(f"  ⛔  スキャン結果は参考表示のみ。ノーポジが最強戦略です。")
        print(f"  📊 戦略C（逆行高銘柄）のみ参考にしてください")
    elif cond == "WEAK":
        print(f"  判定       : ⚠️  WEAK（地合い軟調）")
        print(f"")
        print(f"  ⚠️  戦略A は見送り推奨")
        print(f"  ✅  戦略B は慎重に検討（損切り厳守）")
    elif cond == "STRONG":
        print(f"  判定       : 🚀 STRONG（強気相場）")
        print(f"")
        print(f"  🚀  戦略A 積極エントリー推奨")
        print(f"  ✅  戦略B も通常通り検討可")
    else:
        print(f"  判定       : ✅ NORMAL（通常相場）")
    print(f"{'='*60}")

    return cond


def save_market_log(mc, strategy_a_success_rate=None):
    log_row = {
        "date": TODAY,
        "condition": mc["condition"],
        "ad_ratio": mc["ad_ratio"],
        "nikkei_change": mc["nikkei_change"],
        "up_count": mc["up_count"],
        "down_count": mc["down_count"],
        "strategy_a_success_rate": strategy_a_success_rate,
    }
    new_df = pd.DataFrame([log_row])
    if os.path.exists(MARKET_LOG_CSV):
        ex = pd.read_csv(MARKET_LOG_CSV, encoding="utf-8-sig")
        ex = ex[ex["date"] != TODAY]
        new_df = pd.concat([ex, new_df], ignore_index=True)
    new_df.to_csv(MARKET_LOG_CSV, index=False, encoding="utf-8-sig")


# ══════════════════════════════════════════════
# ★ 戦略C: 暴落逆行高銘柄
# ══════════════════════════════════════════════
def scan_strategy_c(df_today, name_dict, exclude_codes, mc):
    if mc["condition"] != "PANIC":
        return

    print(f"\n{'='*60}")
    print(f"【戦略C】暴落逆行高銘柄（地合いPANIC日専用）")
    print(f"  条件: 当日+3%以上 かつ 出来高急増")
    print(f"  目的: 材料・業種パターンの蓄積（次回PANIC日の参考）")
    print(f"{'='*60}")

    df_c = df_today[
        (df_today["today_rise"] >= 3.0) &
        (df_today["Volume"] >= MIN_VOLUME) &
        (df_today["turnover"] >= MIN_TURNOVER) &
        (~df_today["code4"].isin(exclude_codes))
    ].copy()

    df_c = df_c.sort_values("today_rise", ascending=False).head(20)

    if df_c.empty:
        print("  本日は該当銘柄なし（全銘柄が下落）")
        return

    print(f"  {'コード':<6} {'銘柄名':<20} {'当日騰落':>8} {'終値':>8} {'出来高':>10}")
    print("  " + "─" * 60)

    save_rows = []
    for _, r in df_c.iterrows():
        code4 = r["code4"]
        name  = name_dict.get(code4, "")
        rise  = r["today_rise"]
        icon  = "🚀" if rise >= 10 else ("📈" if rise >= 5 else "↗️")
        print(f"  {code4:<6} {name:<20} {icon}{rise:>+5.1f}% "
              f"{r['Close']:>8.0f}円 {r['Volume']:>10,.0f}")
        save_rows.append({
            "date": TODAY,
            "code": code4,
            "name": name,
            "today_rise": round(rise, 2),
            "close": r["Close"],
            "volume": r["Volume"],
            "nikkei_change": mc["nikkei_change"],
            "ad_ratio": mc["ad_ratio"],
        })

    print(f"\n  ※ チャート・ニュースで材料を確認し、業種傾向を記録しておきましょう")

    new_c = pd.DataFrame(save_rows)
    if os.path.exists(STRATEGY_C_CSV):
        ex = pd.read_csv(STRATEGY_C_CSV, encoding="utf-8-sig")
        ex = ex[ex["date"] != TODAY]
        new_c = pd.concat([ex, new_c], ignore_index=True)
    new_c.to_csv(STRATEGY_C_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  ✅ 戦略Cログ保存: {STRATEGY_C_CSV}")


# ══════════════════════════════════════════════
# スコアリング関数
# ══════════════════════════════════════════════
def calc_score(hist):
    if len(hist) < 22:
        return None
    vol_20d = hist["Volume"].iloc[-20:].values
    vol_5d  = hist["Volume"].iloc[-5:].values
    v1 = float(hist["Volume"].iloc[-1])
    v3 = float(hist["Volume"].iloc[-3])
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


# ══════════════════════════════════════════════
# ★ リバウンドスコア計算（戦略B用）
# ══════════════════════════════════════════════
def calc_rsi(prices, period=14):
    """RSI(14)を計算して返す"""
    if len(prices) < period + 1:
        return None
    deltas = prices.diff().dropna()
    gain = deltas.clip(lower=0).rolling(period).mean()
    loss = (-deltas.clip(upper=0)).rolling(period).mean()
    rs   = gain / loss.replace(0, 1e-9)
    rsi  = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 1)


def calc_rebound_score(code4):
    """
    リバウンドスコアを計算（0〜10点）
    RSI / 出来高倍率 / MA25乖離 の3指標

    採点基準:
      RSI       : <20→4点 / <30→3点 / <40→2点 / <50→1点
      出来高倍率 : 3倍超→3点 / 2倍超→2点 / 1.5倍超→1点
      MA25乖離   : <-15%→3点 / <-10%→2点 / <-5%→1点
    """
    path = f"{CACHE_DIR}/{code4}.csv"
    if not os.path.exists(path):
        return 0, []

    hist = pd.read_csv(path)
    hist["Date"] = pd.to_datetime(hist["Date"])
    hist = hist.sort_values("Date")

    if len(hist) < 26:
        return 0, []

    score   = 0
    reasons = []

    # ── RSI（0〜4点）
    rsi = calc_rsi(hist["Close"], 14)
    if rsi is not None:
        if rsi < 20:
            score += 4; reasons.append(f"RSI極度売られ過ぎ({rsi:.0f})")
        elif rsi < 30:
            score += 3; reasons.append(f"RSI売られ過ぎ({rsi:.0f})")
        elif rsi < 40:
            score += 2; reasons.append(f"RSIやや低め({rsi:.0f})")
        elif rsi < 50:
            score += 1; reasons.append(f"RSI中立({rsi:.0f})")

    # ── 出来高倍率（0〜3点）
    if len(hist) >= 21:
        avg20     = hist["Volume"].iloc[-21:-1].mean()
        vol_today = hist["Volume"].iloc[-1]
        if avg20 > 0:
            vol_ratio = vol_today / avg20
            if vol_ratio >= 3.0:
                score += 3; reasons.append(f"出来高急増({vol_ratio:.1f}倍)売り枯れ示唆")
            elif vol_ratio >= 2.0:
                score += 2; reasons.append(f"出来高増加({vol_ratio:.1f}倍)")
            elif vol_ratio >= 1.5:
                score += 1; reasons.append(f"出来高やや増({vol_ratio:.1f}倍)")

    # ── MA25乖離（0〜3点）
    if len(hist) >= 25:
        ma25      = hist["Close"].iloc[-25:].mean()
        deviation = (hist["Close"].iloc[-1] / ma25 - 1) * 100
        if deviation < -15:
            score += 3; reasons.append(f"MA25大幅下離れ({deviation:.1f}%)")
        elif deviation < -10:
            score += 2; reasons.append(f"MA25下離れ({deviation:.1f}%)")
        elif deviation < -5:
            score += 1; reasons.append(f"MA25やや下離れ({deviation:.1f}%)")

    return score, reasons


def update_cache(code4, row):
    path = f"{CACHE_DIR}/{code4}.csv"
    if not os.path.exists(path):
        return
    hist = pd.read_csv(path)
    hist["Date"] = pd.to_datetime(hist["Date"])
    if pd.Timestamp(TODAY) not in hist["Date"].values:
        new = pd.DataFrame([{
            "Date": pd.Timestamp(TODAY),
            "Open": row["Open"], "High": row["High"],
            "Low":  row["Low"],  "Close": row["Close"],
            "Volume": row["Volume"],
            "prev_close": hist["Close"].iloc[-1],
        }])
        pd.concat([hist, new], ignore_index=True).to_csv(path, index=False)


def verify_watchlist(df_today, name_dict):
    if not os.path.exists(WATCHLIST_CSV):
        return
    wl = pd.read_csv(WATCHLIST_CSV, encoding="utf-8-sig")
    unverified = wl[(wl["buy_date"] == wl["buy_date"].max()) & wl["next_rise"].isna()]
    if unverified.empty:
        return
    print(f"\n=== 📋 【戦略B】前日逆張り候補の検証 ===")
    updated = 0
    for idx, row in unverified.iterrows():
        code4 = str(row["code"])
        t = df_today[df_today["code4"] == code4]
        if t.empty or t.iloc[0]["Open"] <= 0:
            continue
        t = t.iloc[0]
        rise = round((t["Close"] - t["Open"]) / t["Open"] * 100, 2)
        wl.at[idx, "next_rise"]  = rise
        wl.at[idx, "next_open"]  = t["Open"]
        wl.at[idx, "next_close"] = t["Close"]
        status = "✅" if rise >= 5 else ("⚠️" if rise >= 0 else "❌")
        name = name_dict.get(code4, "")
        print(f"  {status} [{code4}]{name}  寄付き:{t['Open']:.0f}円→終値:{t['Close']:.0f}円  {rise:+.1f}%")
        updated += 1
    if updated > 0:
        wl.to_csv(WATCHLIST_CSV, index=False, encoding="utf-8-sig")
        verified = wl[wl["next_rise"].notna()]
        hit = (verified["next_rise"] >= 5).sum()
        avg = verified["next_rise"].mean()
        print(f"  累積: {len(verified)}件  +5%達成:{hit}件({hit/len(verified)*100:.1f}%)  平均:{avg:+.2f}%")


def verify_strategy_c(df_today, name_dict):
    """戦略C（PANIC日逆行高銘柄）の翌日成績を検証する"""
    if not os.path.exists(STRATEGY_C_CSV):
        return
    sc = pd.read_csv(STRATEGY_C_CSV, encoding="utf-8-sig")

    # 翌日検証列がなければ追加
    for col in ["next_open", "next_close", "next_rise"]:
        if col not in sc.columns:
            sc[col] = None

    # 直近日付の未検証データを対象
    latest_date = sc["date"].max()
    unverified  = sc[(sc["date"] == latest_date) & sc["next_rise"].isna()]
    if unverified.empty:
        return

    print(f"\n=== 📋 【戦略C】前日PANIC逆行高銘柄の翌日検証 ===")
    updated = 0

    for idx, row in unverified.iterrows():
        code4 = str(row["code"])
        t     = df_today[df_today["code4"] == code4]
        if t.empty or t.iloc[0]["Open"] <= 0:
            continue
        t      = t.iloc[0]
        rise   = round((t["Close"] - t["Open"]) / t["Open"] * 100, 2)
        sc.at[idx, "next_rise"]  = rise
        sc.at[idx, "next_open"]  = t["Open"]
        sc.at[idx, "next_close"] = t["Close"]

        status = "✅" if rise >= 5 else ("⚠️" if rise >= 0 else "❌")
        name   = name_dict.get(code4, row["name"])
        print(f"  {status} [{code4}]{name}  "
              f"寄付き:{t['Open']:.0f}円→終値:{t['Close']:.0f}円  {rise:+.1f}%")
        updated += 1

    if updated > 0:
        sc.to_csv(STRATEGY_C_CSV, index=False, encoding="utf-8-sig")
        verified = sc[sc["next_rise"].notna()]
        hit = (verified["next_rise"] >= 5).sum()
        avg = verified["next_rise"].mean()
        print(f"  累積: {len(verified)}件  "
              f"+5%達成:{hit}件({hit/len(verified)*100:.1f}%)  "
              f"平均:{avg:+.2f}%")


def verify_scan_a(top20_codes, name_dict, df_today):
    if not os.path.exists(SCAN_CSV):
        return None
    df = pd.read_csv(SCAN_CSV, encoding="utf-8-sig")
    unverified = df[(df["scan_date"] == df["scan_date"].max()) & df["actual_top20"].isna()]
    if unverified.empty:
        return None

    print(f"\n=== 📋 【戦略A】前日順張り候補の検証 ===")
    s_hit = a_hit = b_hit = miss = down = 0

    backtest_rows = []
    scan_date = df["scan_date"].max()

    morning_judgments = {}
    if os.path.exists("out/candidates_log.csv"):
        cl = pd.read_csv("out/candidates_log.csv", encoding="utf-8-sig")

        # scan_dateの翌営業日を計算（土日をスキップ）
        next_date_ts = pd.Timestamp(scan_date) + pd.Timedelta(days=1)
        if next_date_ts.weekday() == 5:    # 土曜 → 月曜
            next_date_ts += pd.Timedelta(days=2)
        elif next_date_ts.weekday() == 6:  # 日曜 → 月曜
            next_date_ts += pd.Timedelta(days=1)
        next_date = next_date_ts.strftime("%Y-%m-%d")

        cl_prev = cl[cl["date"] == next_date]
        morning_judgments = dict(zip(cl_prev["code"].astype(str), cl_prev["judgment"]))

    for _, row in unverified.iterrows():
        code4    = str(row["code"])
        is_top20 = code4 in top20_codes
        name     = name_dict.get(code4, "")
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

        # ── グレード判定を高値ベースに変更 ──────────────────
        if is_top20:
            status = "✅ S級(TOP20)"; grade = "S"; s_hit += 1
        elif max_rise is not None and max_rise >= 5.0:
            status = "✨ A級(高値+5%超)"; grade = "A"; a_hit += 1
        elif max_rise is not None and max_rise >= 2.0:
            status = "📈 B級(高値+2%超)"; grade = "B"; b_hit += 1
        elif max_rise is not None and max_rise < 0:
            status = "❌ 逆行";           grade = "FAIL"; down += 1
        else:
            status = "➖ 横ばい";         grade = "FLAT"; miss += 1

        df.loc[(df["scan_date"] == row["scan_date"]) &
               (df["code"] == row["code"]), "actual_top20"] = int(is_top20)
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
            "is_top20":         int(is_top20),
            "market_condition": row.get("market_condition", "UNKNOWN"),
        })

    total = len(unverified)
    success_rate = None
    if total > 0:
        meaningful   = s_hit + a_hit + b_hit
        success_rate = round(meaningful / total * 100, 1)
        print(f"\n  【集計】")
        print(f"  ✅ S級(TOP20)      : {s_hit}件")
        print(f"  ✨ A級(高値+5%超)  : {a_hit}件")
        print(f"  📈 B級(高値+2%超)  : {b_hit}件")
        print(f"  ➖ 横ばい          : {miss}件")
        print(f"  ❌ 逆行            : {down}件")
        print(f"  利確チャンスあり(高値+2%以上): {meaningful}/{total}件 ({success_rate}%)")

    df.to_csv(SCAN_CSV, index=False, encoding="utf-8-sig")

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
    if not os.path.exists(MARKET_LOG_CSV):
        return
    df = pd.read_csv(MARKET_LOG_CSV, encoding="utf-8-sig")
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

INTRADAY_DIR = "out/intraday"

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

def save_intraday(scan_a_codes, scan_b_codes):
    """
    候補銘柄と固定銘柄の1分足データを保存する。
    yfinanceは直近7日分のみ取得可能なので毎日実行が必須。
    """
    import yfinance as yf
    os.makedirs(INTRADAY_DIR, exist_ok=True)

    # 保存対象をまとめる（重複除去）
    all_codes = set(scan_a_codes) | set(scan_b_codes) | set(STRATEGY_D_CODES)

    print(f"\n{'='*60}")
    print(f"【分足データ収集】{len(all_codes)}銘柄")
    print(f"{'='*60}")

    saved = 0
    skipped = 0

    for code in sorted(all_codes):
        save_path = f"{INTRADAY_DIR}/{code}_{TODAY}.csv"

        # 既に保存済みならスキップ
        if os.path.exists(save_path):
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
            save_df[cols].to_csv(save_path, index=False, encoding="utf-8-sig")
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

    mc = judge_market_condition(df_today)
    condition = print_market_condition(mc)

    verify_watchlist(df_today, name_dict)
    verify_strategy_c(df_today, name_dict)
    success_rate = verify_scan_a(top20_codes, name_dict, df_today)

    save_market_log(mc, success_rate)
    show_market_log_summary()

    # ── 全銘柄スコアリング ──
    print(f"\n  全銘柄スコアリング中...")
    results_a = []
    results_b = []
    cached_codes = [f.replace(".csv","") for f in os.listdir(CACHE_DIR) if f.endswith(".csv")]

    for code4 in cached_codes:
        today_row = df_today[df_today["code4"] == code4]
        if not today_row.empty:
            update_cache(code4, today_row.iloc[0])

        cache_path = f"{CACHE_DIR}/{code4}.csv"
        hist = pd.read_csv(cache_path)
        hist["Date"] = pd.to_datetime(hist["Date"])
        hist = hist.sort_values("Date")
        s = calc_score(hist[hist["Date"] < pd.Timestamp(TODAY)])
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

        # 戦略A: 当日 -5%超の銘柄のみ
        if today_rise > -5.0:
            results_a.append({
                "code": code4,
                "name": name,
                "today_rise": round(today_rise, 2),
                "close": close,
                **s
            })

        # 戦略B: 当日 -5%以下の銘柄をリバウンド候補として抽出
        if today_rise <= -5.0:
            rb_score, rb_reasons = calc_rebound_score(code4)
            if rb_score >= 3:
                results_b.append({
                    "code": code4,
                    "name": name,
                    "today_rise": round(today_rise, 2),
                    "close": close,
                    "rebound_score": rb_score,
                    "rebound_reason": " / ".join(rb_reasons),
                    **s
                })

    # ── 戦略A表示 ──
    df_a = pd.DataFrame(results_a).sort_values("score", ascending=False)
    today_surge = dict(zip(df_today["code4"], df_today["today_rise"].astype(float)))
    df_a["today_rise"] = df_a["code"].map(today_surge).fillna(0)
    top_a = df_a[df_a["score"] >= SCORE_MIN_A].head(TOP_N)

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

    print(f"\n{'='*60}")
    print(f"【戦略A】順張り候補 TOP{len(top_a)}（翌朝急騰狙い）")
    print(f"  🏆S=最優先(スコア8.0〜8.5)  ⭐A=初動狙い  ⚠️=過熱注意(9.0以上)")

    if condition == "PANIC":
        print(f"  🚨 本日地合いPANIC: 参考表示のみ、エントリー非推奨")
    elif condition == "WEAK":
        print(f"  ⚠️  本日地合い軟調: 戦略Aは見送り推奨")

    print(f"{'='*60}")
    print(f"  {'ランク':<4} {'コード':<6} {'銘柄名':<16} {'スコア':>6} {'比率':>6} {'当日':>6} {'加速度':>8}")
    print("  " + "─" * 62)

    for _, r in top_a.iterrows():
        rank = get_rank(r)
        surge_str = f"{float(r['today_rise']):>+5.1f}%"
        print(f"  {rank:<4} {r['code']:<6} {r['name']:<16} {r['score']:>6.2f} "
              f"{r['ratio']:>5.1f}倍 {surge_str} {r['accel']:>+8.0f}%")

    save_a = [{"scan_date": TODAY, "code": r["code"], "name": r["name"],
               "score": r["score"], "trend": r["trend"], "accel": r["accel"],
               "ratio": r["ratio"], "actual_top20": pd.NA,
               "market_condition": condition}
              for _, r in top_a.iterrows()]
    new_a = pd.DataFrame(save_a).astype({"actual_top20": "object"})
    if os.path.exists(SCAN_CSV):
        ex = pd.read_csv(SCAN_CSV, encoding="utf-8-sig").astype({"actual_top20": "object"})
        ex = ex[ex["scan_date"] != TODAY]
        if ex.empty:
            new_a = new_a.copy()
        elif new_a.empty:
            new_a = ex.copy()
        else:
            ex = ex.dropna(axis=1, how="all")
            new_a = new_a.dropna(axis=1, how="all")
            new_a = pd.concat([ex, new_a], ignore_index=True)
    new_a.to_csv(SCAN_CSV, index=False, encoding="utf-8-sig")

    # ── 戦略B表示 ──
    print(f"\n{'='*60}")
    print(f"【戦略B】逆張り候補（翌朝寄付き買い・翌日引け売り）")
    print(f"  条件: 当日-5%以下 かつ リバウンドスコア3点以上")
    print(f"  期待値: +2.46%/トレード（バックテスト実績）")

    if condition == "PANIC":
        print(f"  🚨 本日地合いPANIC: エントリー非推奨（翌日も続落リスク高）")
    elif condition == "WEAK":
        print(f"  ⚠️  本日地合い軟調: 損切りラインを-3%に引き締めて慎重に")

    print(f"{'='*60}")

    if not results_b:
        print("  本日は該当銘柄なし")
    else:
        df_b = pd.DataFrame(results_b).sort_values("today_rise")
        print(f"  {'コード':<6} {'銘柄名':<16} {'当日騰落':>8} {'終値':>8} {'RBスコア':>8} {'理由'}")
        print("  " + "─" * 72)
        for _, r in df_b.iterrows():
            icon = "💥" if r["today_rise"] <= -20 else ("🔻" if r["today_rise"] <= -15 else "↘️")
            print(f"  {r['code']:<6} {r['name']:<16} "
                  f"{icon}{r['today_rise']:>+6.1f}% "
                  f"{r['close']:>8.0f}円 "
                  f"{r['rebound_score']:>6}点  "
                  f"{r['rebound_reason']}")
        print(f"\n  ⚠️  購入前にチャートとニュースを必ず確認してください")
        print(f"  ⚠️  損切りライン: 寄付き価格から-5%を厳守してください")

        # ★ 変更: rebound_score / rebound_reason を保存列に追加
        save_b = [{"buy_date": TODAY, "code": r["code"], "name": r["name"],
                   "today_rise": r["today_rise"], "buy_price": r["close"],
                   "score": r["score"], "ratio": r["ratio"],
                   "rebound_score":  r.get("rebound_score", 0),
                   "rebound_reason": r.get("rebound_reason", ""),
                   "market_condition": condition,
                   "next_rise": None, "next_open": None, "next_close": None}
                  for _, r in df_b.iterrows()]
        new_b = pd.DataFrame(save_b)
        if os.path.exists(WATCHLIST_CSV):
            ex = pd.read_csv(WATCHLIST_CSV, encoding="utf-8-sig")
            ex = ex[ex["buy_date"] != TODAY]
            if ex.empty:
                new_b = new_b.copy()
            elif new_b.empty:
                new_b = ex.copy()
            else:
                ex = ex.dropna(axis=1, how="all")
                new_b = new_b.dropna(axis=1, how="all")
                new_b = pd.concat([ex, new_b], ignore_index=True)
        new_b.to_csv(WATCHLIST_CSV, index=False, encoding="utf-8-sig")
        print(f"\n  ✅ 監視リスト保存: {WATCHLIST_CSV}")

    scan_strategy_c(df_today, name_dict, exclude_codes, mc)

    # ── 分足データ収集 ──
    scan_a_codes = [r["code"] for r in save_a]
    scan_b_codes = [r["code"] for r in save_b] if results_b else []
    save_intraday(scan_a_codes, scan_b_codes)

    print(f"\n✅ 完了")    


if __name__ == "__main__":
    main()
