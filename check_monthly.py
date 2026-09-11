# -*- coding: utf-8 -*-
"""
check_monthly.py - 月次レビュー（過去3か月）

毎月の定例チェック用。check_weekly.py（先週の成績のみ）を拡張し、
①実運用成績（戦略別・月別トレンド）②相場環境（AD比率・日経225）
③システム健全性（エラー件数）を3か月スパンでまとめて出す。

パラメータ調整は一切しない（読むだけ）。「バックテストの想定と実運用がズレていないか」
「相場環境が変わっていないか」に早めに気づくための定点観測。

使い方:
    python check_monthly.py              # 直近3か月
    python check_monthly.py --months 6   # 期間を変える
"""
import sys
import argparse
import sqlite3
from datetime import datetime, timedelta, timezone

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import numpy as np
import db

JST = timezone(timedelta(hours=9))
WEAK_AD, STRONG_AD = 0.35, 0.60


def classify(ad, nk):
    if ad <= 0.20:
        return "PANIC" if (nk is not None and nk <= -2.0) else "WEAK"
    if ad >= STRONG_AD:
        return "STRONG" if (nk is not None and nk >= 0.5) else "NORMAL"
    if ad <= WEAK_AD:
        return "WEAK"
    return "NORMAL"


def section(title):
    print(f"\n{'='*78}\n  {title}\n{'='*78}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=3)
    args = ap.parse_args()

    today = datetime.now(JST).date()
    start = (today.replace(day=1) - timedelta(days=30 * args.months)).strftime("%Y-%m-01")
    print(f"月次レビュー: {start} 〜 {today}（過去{args.months}か月・読み取り専用、パラメータ変更なし）")

    conn = db.get_conn()

    # ── 1. 実運用成績 ──────────────────────────────────────
    section("1. 実運用成績（戦略別）")
    pos = pd.read_sql(
        "SELECT date, code, strategy, shares, buy_price, sell_price, pnl_pct, status, exit_reason "
        "FROM positions WHERE date >= ? AND status='closed'",
        conn, params=[start],
    )
    if pos.empty:
        print("  該当期間の決済済みトレードなし")
    else:
        pos["pnl_yen"] = pos["shares"] * pos["buy_price"] * pos["pnl_pct"] / 100
        g = pos.groupby("strategy").agg(
            n=("pnl_pct", "count"),
            win_rate=("pnl_pct", lambda s: (s > 0).mean() * 100),
            avg_pct=("pnl_pct", "mean"),
            total_yen=("pnl_yen", "sum"),
        ).round(2)
        print(f"  {'戦略':<6}{'件数':>6}{'勝率':>8}{'平均%':>9}{'損益合計(概算円)':>18}")
        for strat, r in g.sort_values("n", ascending=False).iterrows():
            print(f"  {strat:<6}{int(r['n']):>6}{r['win_rate']:>7.1f}%{r['avg_pct']:>+8.2f}%{r['total_yen']:>17,.0f}")
        print(f"  {'合計':<6}{len(pos):>6}{(pos['pnl_pct']>0).mean()*100:>7.1f}%"
              f"{pos['pnl_pct'].mean():>+8.2f}%{pos['pnl_yen'].sum():>17,.0f}")

        print("\n  月別トレンド（全戦略合計）")
        pos["ym"] = pos["date"].str[:7]
        gm = pos.groupby("ym").agg(n=("pnl_pct", "count"),
                                   win_rate=("pnl_pct", lambda s: (s > 0).mean() * 100),
                                   avg_pct=("pnl_pct", "mean"),
                                   total_yen=("pnl_yen", "sum"))
        for ym, r in gm.iterrows():
            print(f"    {ym}  n={int(r['n']):>3}  勝率{r['win_rate']:>5.1f}%  "
                  f"平均{r['avg_pct']:>+6.2f}%  損益{r['total_yen']:>+11,.0f}円")

    # 現在の保有
    openp = pd.read_sql("SELECT date, code, name, strategy, shares, buy_price FROM positions WHERE status='open'", conn)
    print(f"\n  現在の保有: {len(openp)}件" + (f"（{', '.join(openp['code'])}）" if len(openp) else ""))

    # ── 2. 相場環境 ──────────────────────────────────────
    section("2. 相場環境（AD比率・地合い・日経225）")
    df = pd.read_sql("SELECT Date, Close, prev_close FROM daily_prices WHERE prev_close>0 AND Close>0 AND Date>=?",
                     conn, params=[start])
    conn.close()
    df["chg"] = (df["Close"] - df["prev_close"]) / df["prev_close"] * 100
    ad = df.groupby("Date")["chg"].apply(lambda s: (s > 0).sum() / len(s)).reset_index()
    ad.columns = ["date", "ad"]

    try:
        import yfinance as yf
        nk = yf.download("^N225", start=start, end=str(today + timedelta(days=1)), interval="1d", progress=False)["Close"]
        if hasattr(nk, "columns"):
            nk = nk.iloc[:, 0]
        nk.index = [str(x)[:10] for x in nk.index]
        nk_chg = (nk.pct_change() * 100).dropna()
        nk_map = nk_chg.to_dict()
    except Exception as e:
        print(f"  ⚠️ 日経225取得失敗（yfinance）: {e}")
        nk_map = {}

    ad["nk"] = ad["date"].map(nk_map)
    ad["regime"] = ad.apply(lambda r: classify(r["ad"], r["nk"]), axis=1)
    ad["ym"] = ad["date"].str[:7]

    gm = ad.groupby("ym")
    print(f"  {'年月':<9}{'AD平均':>8}{'WEAK+PANIC率':>14}{'日経月次':>10}")
    for ym, sub in gm:
        wp = sub["regime"].isin(["WEAK", "PANIC"]).mean() * 100
        month_dates = [d for d in nk_map if d.startswith(ym)]
        month_ret = sum(nk_map[d] for d in sorted(month_dates)) if month_dates else None
        mr = f"{month_ret:+.1f}%" if month_ret is not None else "  n/a"
        print(f"  {ym:<9}{sub['ad'].mean():>7.3f}{wp:>13.1f}%{mr:>10}")

    if len(ad) >= 5:
        print(f"\n  直近5営業日   AD平均 {ad['ad'].tail(5).mean():.3f}   "
              f"WEAK+PANIC率 {ad['regime'].tail(5).isin(['WEAK','PANIC']).mean()*100:.0f}%")
    if len(ad) >= 20:
        print(f"  直近20営業日  AD平均 {ad['ad'].tail(20).mean():.3f}   "
              f"WEAK+PANIC率 {ad['regime'].tail(20).isin(['WEAK','PANIC']).mean()*100:.0f}%")

    # ── 3. システム健全性 ──────────────────────────────────
    section("3. システム健全性（エラーログ）")
    import glob
    import re
    log_files = sorted(glob.glob("logs/run_daily_*.log"))
    err_by_month = {}
    for f in log_files:
        m = re.search(r"run_daily_(\d{4})(\d{2})(\d{2})\.log", f)
        if not m:
            continue
        ym = f"{m.group(1)}-{m.group(2)}"
        if ym < start[:7]:
            continue
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
            n_err = text.count("Traceback") + len(re.findall(r"KeyError", text))
            err_by_month[ym] = err_by_month.get(ym, 0) + n_err
        except Exception:
            pass
    if err_by_month:
        for ym, n in sorted(err_by_month.items()):
            flag = "  ⚠️" if n > 0 else ""
            print(f"  {ym}: Traceback/KeyError {n}件{flag}")
    else:
        print("  ログ未取得（このマシンに logs/ が無い場合はEC2上で実行してください）")

    # ── 4. 注意フラグ（単純な閾値チェックのみ、判断はしない） ──
    section("4. 注意フラグ（自動判定・要人間確認）")
    flags = []
    if not pos.empty and len(gm := pos.groupby("ym")) >= 2:
        yms = sorted(gm.groups.keys())
        recent = pos[pos["ym"] == yms[-1]]
        prior = pos[pos["ym"].isin(yms[:-1])]
        if len(recent) >= 5 and len(prior) >= 5:
            if recent["pnl_pct"].mean() < prior["pnl_pct"].mean() - 0.5:
                flags.append(f"直近月の平均リターンが以前より 0.5pt 以上悪化"
                             f"（直近{recent['pnl_pct'].mean():+.2f}% vs 以前{prior['pnl_pct'].mean():+.2f}%）")
    if len(ad) >= 20:
        recent_wp = ad["regime"].tail(20).isin(["WEAK", "PANIC"]).mean() * 100
        base_wp = ad["regime"].isin(["WEAK", "PANIC"]).mean() * 100
        if recent_wp > base_wp + 10:
            flags.append(f"直近20営業日のWEAK+PANIC率が期間平均より+10pt以上高い"
                         f"（直近{recent_wp:.0f}% vs 期間平均{base_wp:.0f}%）")
    if any(n > 0 for n in err_by_month.values()):
        flags.append("期間中にエラー（Traceback/KeyError）が発生した月がある。上記②で内容確認")
    if flags:
        for f in flags:
            print(f"  ⚠️ {f}")
    else:
        print("  特筆すべき閾値超過なし")


if __name__ == "__main__":
    main()
