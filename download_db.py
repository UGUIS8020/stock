# -*- coding: utf-8 -*-
"""
download_db.py - S3から共有データをダウンロードする（Mac/Windows両対応）

【ダウンロード対象】
  stock-db/stock.db          → out/stock.db       （日次株価DB・毎日更新）
  stock-optimize/*.csv/txt   → out/              （最適化結果・月1回更新）

【使い方】
  python download_db.py            # DB + 最適化データ 両方ダウンロード
  python download_db.py --db-only  # DBのみ（朝スキャン前の高速更新用）
"""

import sys
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

S3_BUCKET      = os.getenv("S3_BUCKET", "shibuya8020")
S3_DB_KEY      = "stock-db/stock.db"
S3_OPT_PREFIX  = "stock-optimize/"

# 最適化ファイル（S3キー → ローカル保存先）
OPT_FILES = [
    "opt_b_trades.csv",
    "opt_b_stage1.csv",
    "opt_b_stage2.csv",
    "opt_b_report.txt",
    "evolve_b_best.csv",
    "evolve_b_history.csv",
    "evolve_b_report.txt",
]


def download_db(s3, out_dir):
    db_path = out_dir / "stock.db"
    print("S3からDBをダウンロード中...")
    s3.download_file(S3_BUCKET, S3_DB_KEY, str(db_path))
    size_mb = db_path.stat().st_size / 1024 / 1024
    print(f"  ✅ stock.db ({size_mb:.0f}MB)")


def download_optimize(s3, out_dir):
    print("S3から最適化データをダウンロード中...")
    downloaded = 0
    for fname in OPT_FILES:
        key = S3_OPT_PREFIX + fname
        local = out_dir / fname
        try:
            s3.download_file(S3_BUCKET, key, str(local))
            print(f"  ✅ {fname}")
            downloaded += 1
        except Exception:
            print(f"  ── {fname}（S3未存在 or 未生成）")
    if downloaded:
        print(f"  {downloaded}ファイルをダウンロードしました")
    else:
        print("  最適化ファイルはまだS3にありません（optimize_b.py / evolve_b.py を実行後にアップロードされます）")


def run(db_only=False):
    """他スクリプトから直接呼び出せるエントリポイント"""
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)

    try:
        import boto3
        s3 = boto3.client("s3")
    except ImportError:
        print("❌ boto3がインストールされていません: pip install boto3")
        sys.exit(1)

    try:
        download_db(s3, out_dir)
    except Exception as e:
        print(f"❌ DBダウンロード失敗: {e}")
        sys.exit(1)

    if not db_only:
        try:
            download_optimize(s3, out_dir)
        except Exception as e:
            print(f"⚠️  最適化データのダウンロード中にエラー: {e}")

    print("\n✅ ダウンロード完了")


def main():
    parser = argparse.ArgumentParser(description="S3からデータをダウンロード")
    parser.add_argument("--db-only", action="store_true", help="DBのみダウンロード（最適化データはスキップ）")
    args = parser.parse_args()
    run(db_only=args.db_only)


if __name__ == "__main__":
    main()
