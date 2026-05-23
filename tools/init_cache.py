"""
init_cache.py
全銘柄の株価履歴を一括取得してDBに保存する（初回のみ実行）
所要時間の目安: 約2〜3時間（4,000銘柄 × 1.5秒待機）
途中で止まっても再実行すれば続きから再開できます

実行方法:
    python tools/init_cache.py             # デフォルト: 直近18ヶ月
    python tools/init_cache.py --months 12 # 直近12ヶ月
    python tools/init_cache.py --force     # 既存データも上書き再取得
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jquantsapi
import argparse
from dotenv import load_dotenv
import time
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import db

load_dotenv()
cli = jquantsapi.ClientV2(api_key=os.getenv("JQUANTS_API_KEY"))

SLEEP_NORMAL = 1.5   # 通常の待機（秒）
SLEEP_429    = 60.0  # 429レート制限時の待機（秒）
MAX_RETRY    = 3     # 429時の最大リトライ回数


def _call_with_retry(fn, *args, **kwargs):
    """429レート制限に対してリトライしながらAPI呼び出し"""
    for attempt in range(MAX_RETRY):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            if "429" in msg and attempt < MAX_RETRY - 1:
                wait = SLEEP_429 * (attempt + 1)
                print(f"  ⏳ 429レート制限。{wait:.0f}秒待機してリトライ（{attempt+1}/{MAX_RETRY}）...",
                      flush=True)
                time.sleep(wait)
            else:
                raise


def get_all_codes() -> list[str]:
    """全銘柄コード（4桁）を取得"""
    print("全銘柄リストを取得中...")
    master = _call_with_retry(cli.get_eq_master)
    master["code4"] = master["Code"].astype(str).str[:4]
    codes = master[
        master["code4"].str.match(r"^\d{4}$") |
        master["code4"].str.match(r"^\d{3}[A-Z]$")
    ]["code4"].unique().tolist()
    print(f"対象銘柄数: {len(codes)}銘柄")
    return codes


def fetch_and_save(code4: str, from_date: str, to_date: str, force: bool) -> str:
    """1銘柄の履歴を取得してDBに保存。戻り値: 'skip'|'done'|'error:<msg>'"""
    if not force and db.code_exists(code4):
        return "skip"

    try:
        code5 = code4 + "0" if len(code4) == 4 and code4.isdigit() else code4
        df = _call_with_retry(cli.get_eq_bars_daily,
                              code=code5, from_yyyymmdd=from_date, to_yyyymmdd=to_date)
        if df.empty:
            return "done"
        df = df.rename(columns={"O": "Open", "H": "High", "L": "Low", "C": "Close", "Vo": "Volume"})
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        df["prev_close"] = df["Close"].shift(1)
        db.bulk_insert_daily_prices(code4, df)
        return "done"
    except Exception as e:
        return f"error:{e}"


def main():
    parser = argparse.ArgumentParser(description="全銘柄株価履歴をDBに一括取得")
    parser.add_argument("--months", type=int, default=18, help="取得する期間（月数、デフォルト18）")
    parser.add_argument("--force",  action="store_true",  help="既存データも上書き再取得")
    args = parser.parse_args()

    db.init_db()

    today     = datetime.now()
    from_date = (today - relativedelta(months=args.months)).strftime("%Y%m%d")
    to_date   = today.strftime("%Y%m%d")

    print(f"=== 株価DB初期化 ===")
    print(f"  取得期間: {from_date} 〜 {to_date}（直近{args.months}ヶ月）")
    print(f"  上書きモード: {'有効' if args.force else '無効（既存銘柄はスキップ）'}")
    print()

    codes = get_all_codes()

    # 既存件数確認
    existing = sum(1 for c in codes if db.code_exists(c))
    remaining = len(codes) - existing if not args.force else len(codes)
    print(f"DB登録済み: {existing}銘柄 / 残り: {remaining}銘柄\n")

    if remaining == 0 and not args.force:
        print("✅ 全銘柄のデータが揃っています")
        print("   最新データを追加するには scan_daily.py を実行してください")
        return

    print(f"取得開始（残り{remaining}銘柄）")
    print(f"所要時間目安: 約{int(remaining * SLEEP_NORMAL // 60)}分\n")

    done = skipped = errors = 0

    for code4 in codes:
        result = fetch_and_save(code4, from_date, to_date, args.force)
        if result == "skip":
            skipped += 1
        elif result == "done":
            done += 1
        else:
            errors += 1
            if errors <= 5:  # 最初の5件だけエラー内容を表示
                print(f"  ⚠️  エラー({code4}): {result[6:]}", flush=True)

        total_done = done + skipped + errors
        if total_done % 100 == 0:
            pct = total_done / len(codes) * 100
            print(f"  [{total_done:>4}/{len(codes)}] {pct:.1f}% 完了 "
                  f"（取得:{done} スキップ:{skipped} エラー:{errors}）")
            sys.stdout.flush()

        if result != "skip":
            time.sleep(SLEEP_NORMAL)

    print(f"\n✅ 完了！")
    print(f"  取得: {done}銘柄")
    print(f"  スキップ（既存）: {skipped}銘柄")
    print(f"  エラー: {errors}銘柄")
    if errors > 0:
        print(f"  ※エラー銘柄は再実行すれば取得を再試行します")
    print(f"\n次は python scan_daily.py を実行してください")


if __name__ == "__main__":
    main()
