# -*- coding: utf-8 -*-
"""
download_db.py - S3からstock.dbをダウンロードする（Mac/Windows両対応）
"""

import sys
import os
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def main():
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)
    db_path = out_dir / "stock.db"

    s3_bucket = "shibuya8020"
    s3_key = "stock-db/stock.db"

    print("S3からDBをダウンロード中...")
    try:
        import boto3
        s3 = boto3.client("s3")
        s3.download_file(s3_bucket, s3_key, str(db_path))
        print("✅ ダウンロード完了")
    except Exception as e:
        print(f"❌ ダウンロード失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
