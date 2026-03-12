"""
init_cache.py
全銘柄の6ヶ月履歴を一括取得してキャッシュに保存する（初回のみ実行）
所要時間の目安: 約2〜3時間（4,000銘柄 × 1秒待機）
途中で止まっても再実行すれば続きから再開できます
"""
import jquantsapi
from dotenv import load_dotenv
import os, time, sys
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

load_dotenv()
cli = jquantsapi.ClientV2(api_key=os.getenv("JQUANTS_API_KEY"))

CACHE_DIR   = "out/cache"
MONTHS_BACK = 6
os.makedirs(CACHE_DIR, exist_ok=True)

today     = datetime.now()
from_date = (today - relativedelta(months=MONTHS_BACK)).strftime("%Y%m%d")
to_date   = today.strftime("%Y%m%d")

def get_all_codes() -> list[str]:
    """全銘柄コード（4桁）を取得"""
    print("全銘柄リストを取得中...")
    master = cli.get_eq_master()
    master["code4"] = master["Code"].astype(str).str[:4]
    # ETF・REIT除外（数字4桁 or 数字3桁+英字）
    codes = master[
        master["code4"].str.match(r"^\d{4}$") |
        master["code4"].str.match(r"^\d{3}[A-Z]$")
    ]["code4"].unique().tolist()
    print(f"対象銘柄数: {len(codes)}銘柄")
    return codes

def fetch_and_cache(code4: str) -> bool:
    """1銘柄の履歴を取得してキャッシュに保存"""
    cache_path = f"{CACHE_DIR}/{code4}.csv"
    
    # キャッシュが存在すればスキップ
    if os.path.exists(cache_path):
        return False  # スキップ

    try:
        code5 = code4 + "0" if len(code4) == 4 and code4.isdigit() else code4
        df = cli.get_eq_bars_daily(code=code5, from_yyyymmdd=from_date, to_yyyymmdd=to_date)
        if df.empty:
            return True
        df = df.rename(columns={"O":"Open","H":"High","L":"Low","C":"Close","Vo":"Volume"})
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        df["prev_close"] = df["Close"].shift(1)
        df.to_csv(cache_path, index=False)
        return True
    except Exception:
        return True

def main():
    codes = get_all_codes()
    
    # 既にキャッシュ済みの件数を確認
    cached = [c for c in codes if os.path.exists(f"{CACHE_DIR}/{c}.csv")]
    print(f"取得済み: {len(cached)}銘柄 / 残り: {len(codes)-len(cached)}銘柄\n")
    
    if len(cached) == len(codes):
        print("✅ 全銘柄のキャッシュが揃っています")
        return

    print(f"取得開始（残り{len(codes)-len(cached)}銘柄）")
    print(f"所要時間目安: 約{(len(codes)-len(cached))//60}分\n")
    
    done = 0
    skipped = 0
    errors  = 0

    for i, code4 in enumerate(codes):
        if os.path.exists(f"{CACHE_DIR}/{code4}.csv"):
            skipped += 1
            continue

        result = fetch_and_cache(code4)
        if result:
            done += 1
        else:
            errors += 1

        # 進捗表示（100件ごと）
        total_done = done + skipped
        if total_done % 100 == 0:
            pct = total_done / len(codes) * 100
            print(f"  [{total_done:>4}/{len(codes)}] {pct:.1f}% 完了 "
                  f"（取得:{done} スキップ:{skipped}）")
            sys.stdout.flush()

        time.sleep(1.0)

    print(f"\n✅ 完了！")
    print(f"  取得: {done}銘柄")
    print(f"  スキップ（既存）: {skipped}銘柄")
    print(f"  エラー: {errors}銘柄")
    print(f"\n次は python scan_daily.py を実行してください")

if __name__ == "__main__":
    main()