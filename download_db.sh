#!/bin/bash
echo "S3からDBをダウンロード中..."
mkdir -p out
aws s3 cp s3://shibuya8020/stock-db/stock.db out/stock.db
if [ $? -eq 0 ]; then
    echo "✅ ダウンロード完了"
else
    echo "❌ ダウンロード失敗"
fi
