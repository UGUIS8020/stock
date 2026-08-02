# -*- coding: utf-8 -*-
"""S3から stock.db をダウンロードし、ダッシュボードのdata/へ配置する。
売買botのout/ディレクトリとは独立させ、常にこのダッシュボード専用のコピーを持つ。
cronから5分おきに呼び出す想定。
"""
import os
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET", "shibuya8020")
S3_DB_KEY = "stock-db/stock.db"


def main():
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    db_path = data_dir / "stock.db"
    tmp_path = data_dir / "stock.db.tmp"

    s3 = boto3.client("s3")
    s3.download_file(S3_BUCKET, S3_DB_KEY, str(tmp_path))
    tmp_path.replace(db_path)  # アプリが読み取り中のファイルを直接上書きしないよう、原子的に差し替える

    size_mb = db_path.stat().st_size / 1024 / 1024
    print(f"OK: {db_path} updated ({size_mb:.1f}MB)")


if __name__ == "__main__":
    main()
