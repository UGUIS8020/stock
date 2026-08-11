# -*- coding: utf-8 -*-
"""
check_api_calls.py - 立花証券APIへの1日の呼び出し回数を確認する

tachibana_order.log_api_call() が logs/api_calls_YYYYMMDD.log に追記した
実際のログを集計する。同時実行される複数プロセス(market_watch.py・
daytime.py・closing_watch.py・position_monitor.py等)からの呼び出しを
すべて合算できる。

使い方:
    python check_api_calls.py            # 本日分
    python check_api_calls.py 20260812    # 指定日分
"""
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

JST = timezone(timedelta(hours=9))
THRESHOLD = 10_000  # 立花証券から警告があった目安件数


def main():
    date_str = sys.argv[1] if len(sys.argv) > 1 else datetime.now(JST).strftime("%Y%m%d")
    path = Path(__file__).parent / "logs" / f"api_calls_{date_str}.log"

    if not path.exists():
        print(f"  ログが見つかりません: {path}")
        print(f"  （その日の取引がまだ実行されていないか、api_calls機能導入前の日付です）")
        return

    counts = Counter()
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            source = parts[1] if len(parts) > 1 else "unknown"
            counts[source] += 1
            total += 1

    print(f"{'='*55}")
    print(f"  {date_str} の立花証券API呼び出し回数")
    print(f"{'='*55}")
    for source, n in counts.most_common():
        print(f"  {source:38}: {n:6,}回")
    print(f"  {'-'*51}")
    print(f"  {'合計':38}: {total:6,}回")
    print(f"{'='*55}")

    if total >= THRESHOLD:
        print(f"  ⚠️  {THRESHOLD:,}回を超えています。対策の見直しが必要です。")
    else:
        margin = THRESHOLD - total
        print(f"  ✅ {THRESHOLD:,}回未満です（余裕 {margin:,}回）")


if __name__ == "__main__":
    main()
