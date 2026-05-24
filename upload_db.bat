@echo off
echo DBをS3にアップロード中...
aws s3 cp out\stock.db s3://shibuya8020/stock-db/stock.db
if %errorlevel% == 0 (
    echo ✅ アップロード完了 s3://shibuya8020/stock-db/stock.db
) else (
    echo ❌ アップロード失敗
)
