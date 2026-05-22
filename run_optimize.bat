@echo off
cd /d %~dp0

echo ============================================================
echo  Strategy B Monthly Optimization  %date% %time%
echo ============================================================
echo.

call venv\Scripts\activate.bat

echo [Step 1/2] Updating trade data... (10-20 min)
echo.
python optimize\optimize_b.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: optimize_b.py failed.
    pause
    exit /b 1
)

echo.
echo [Step 2/2] Genetic algorithm optimization... (2-3 min)
echo.
python optimize\evolve_b.py --pop 20000 --gen 100
if %errorlevel% neq 0 (
    echo.
    echo ERROR: evolve_b.py failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Done!
echo    out\evolve_b_best.csv
echo    out\evolve_b_report.txt
echo ============================================================
echo.
pause
