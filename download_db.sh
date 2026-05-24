#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AWS_CMD="$SCRIPT_DIR/.venv/bin/aws"
[ -f "$AWS_CMD" ] || AWS_CMD="aws"
echo "S3からDBをダウンロード中..."
mkdir -p out
"$AWS_CMD" s3 cp s3://shibuya8020/stock-db/stock.db out/stock.db
if [ $? -eq 0 ]; then
    echo "✅ ダウンロード完了"
else
    echo "❌ ダウンロード失敗"
fi
