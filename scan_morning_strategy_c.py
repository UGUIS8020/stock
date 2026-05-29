# -*- coding: utf-8 -*-
"""
scan_morning_strategy_c.py - 戦略C（暴落逆行高銘柄）スキャン

scan_daily.py から分割。今後ほぼ変更不要な安定ロジック。
地合いPANIC日に逆行して上昇した銘柄を記録・検証する。
目的: 材料・業種パターンの蓄積（次回PANIC日の参考）
"""

import os
import pandas as pd
from datetime import datetime
from pathlib import Path

_BASE_DIR      = Path(__file__).parent
STRATEGY_C_CSV = str(_BASE_DIR / "out" / "strategy_c_log.csv")

MIN_VOLUME   = 50_000
MIN_TURNOVER = 50_000_000


def scan_strategy_c(df_today, name_dict, exclude_codes, mc):
    """PANIC日に逆行高した銘柄を抽出してCSVに保存する。"""
    if mc["condition"] != "PANIC":
        return

    today = datetime.now().strftime("%Y-%m-%d")

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
            "date":          today,
            "code":          code4,
            "name":          name,
            "today_rise":    round(rise, 2),
            "close":         r["Close"],
            "volume":        r["Volume"],
            "nikkei_change": mc["nikkei_change"],
            "ad_ratio":      mc["ad_ratio"],
        })

    print(f"\n  ※ チャート・ニュースで材料を確認し、業種傾向を記録しておきましょう")

    new_c = pd.DataFrame(save_rows)
    if os.path.exists(STRATEGY_C_CSV):
        ex = pd.read_csv(STRATEGY_C_CSV, encoding="utf-8-sig")
        ex = ex[ex["date"] != today]
        new_c = pd.concat([ex, new_c], ignore_index=True)
    new_c.to_csv(STRATEGY_C_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  ✅ 戦略Cログ保存: {STRATEGY_C_CSV}")


def verify_strategy_c(df_today, name_dict):
    """戦略C（PANIC日逆行高銘柄）の翌日成績を検証する。"""
    if not os.path.exists(STRATEGY_C_CSV):
        return
    sc = pd.read_csv(STRATEGY_C_CSV, encoding="utf-8-sig")

    for col in ["next_open", "next_close", "next_rise"]:
        if col not in sc.columns:
            sc[col] = None

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
        t    = t.iloc[0]
        rise = round((t["Close"] - t["Open"]) / t["Open"] * 100, 2)
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
