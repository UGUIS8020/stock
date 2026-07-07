@echo off
cd /d "C:\Users\Owner\Desktop\website\stock"
call venv\Scripts\activate.bat
python run_daily.py >> out\run_daily_auto.log 2>&1
