@echo off
cd /d C:\Users\Owner\Desktop\website\stock
call venv\Scripts\activate

echo ===== %date% %time% 開始 =====

echo [1/2] 当日TOP20スコアリング...
python surge_analysis.py >> logs\surge_%date:~0,4%%date:~5,2%%date:~8,2%.log 2>&1

echo [2/2] 過去データ補完...
python fetch_historical.py >> logs\historical_%date:~0,4%%date:~5,2%%date:~8,2%.log 2>&1

echo ===== 完了 =====
