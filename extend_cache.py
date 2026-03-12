"""
extend_cache.py
既存の6ヶ月キャッシュを1年分に拡張する
既存ファイルの最古日より前のデータを取得して先頭に追加する
所要時間: 約1〜2時間（途中停止→再実行で続きから再開可能）
"""
import jquantsapi
from dotenv import load_dotenv
import os, time, sys
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

load_dotenv()
cli = jquantsapi.ClientV2(api_key=os.getenv("JQUANTS_API_KEY"))

CACHE_DIR    = "out/cache"
MONTHS_TOTAL = 12  # 合計何ヶ月分にするか

today     = datetime.now()
from_date = (today - relativedelta(months=MONTHS_TOTAL)).strftime("%Y%m%d")

def extend_one(code4):
    cache_path = f"{CACHE_DIR}/{code4}.csv"
    if not os.path.exists(cache_path):
        return "no_cache"

    hist = pd.read_csv(cache_path)
    hist["Date"] = pd.to_datetime(hist["Date"])
    oldest = hist["Date"].min()

    # すでに1年分あればスキップ
    target_start = pd.Timestamp(today - relativedelta(months=MONTHS_TOTAL))
    if oldest <= target_start + pd.Timedelta(days=10):
        return "skip"

    # 古いデータを取得（既存の最古日の前まで）
    to_yyyymmdd   = (oldest - pd.Timedelta(days=1)).strftime("%Y%m%d")
    from_yyyymmdd = from_date

    try:
        code5 = code4 + "0" if len(code4) == 4 and code4.isdigit() else code4
        df_new = cli.get_eq_bars_daily(
            code=code5,
            from_yyyymmdd=from_yyyymmdd,
            to_yyyymmdd=to_yyyymmdd
        )
        if df_new.empty:
            return "empty"

        df_new = df_new.rename(columns={"O":"Open","H":"High","L":"Low","C":"Close","Vo":"Volume"})
        df_new["Date"] = pd.to_datetime(df_new["Date"])

        # 結合して保存
        combined = pd.concat([df_new, hist], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Date"]).sort_values("Date")
        combined["prev_close"] = combined["Close"].shift(1)
        combined.to_csv(cache_path, index=False)
        return "updated"

    except Exception as e:
        return f"error:{e}"

def main():
    files = [f.replace(".csv","") for f in os.listdir(CACHE_DIR) if f.endswith(".csv")]
    print(f"=== キャッシュ拡張（6ヶ月 → {MONTHS_TOTAL}ヶ月）===")
    print(f"  対象: {len(files)}銘柄")
    print(f"  取得期間: {from_date} 〜 現在\n")

    updated = skipped = empty = errors = 0

    for i, code4 in enumerate(files):
        result = extend_one(code4)
        if result == "updated":
            updated += 1
        elif result == "skip":
            skipped += 1
        elif result == "empty":
            empty += 1
        else:
            errors += 1

        if (i+1) % 100 == 0:
            pct = (i+1) / len(files) * 100
            print(f"  [{i+1:>4}/{len(files)}] {pct:.1f}%  "
                  f"更新:{updated} スキップ:{skipped} 空:{empty} エラー:{errors}")
            sys.stdout.flush()

        time.sleep(0.5)  # init_cacheより短め（追加分のみ取得なので）

    print(f"\n✅ 完了！")
    print(f"  更新: {updated}銘柄")
    print(f"  スキップ（既に1年分）: {skipped}銘柄")
    print(f"  データなし: {empty}銘柄")
    print(f"  エラー: {errors}銘柄")
    print(f"\n次は python Backtest_scanner_tomorrow.py を実行して")
    print(f"1年分のバックテスト結果を確認してください")

if __name__ == "__main__":
    main()
