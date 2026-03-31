@echo off
chcp 65001 > nul
cd /d %~dp0

echo ============================================================
echo  戦略B 月次最適化  %date% %time%
echo ============================================================
echo.

REM 仮想環境を有効化
call venv\Scripts\activate.bat

echo [Step 1/2] トレードデータを最新化中...
echo  （10〜20分かかることがあります）
echo.
python optimize_b.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ optimize_b.py でエラーが発生しました。終了します。
    pause
    exit /b 1
)

echo.
echo [Step 2/2] 遺伝的アルゴリズムで最適化中...
echo  （約2〜3分かかります）
echo.
python evolve_b.py --pop 20000 --gen 100
if %errorlevel% neq 0 (
    echo.
    echo ❌ evolve_b.py でエラーが発生しました。
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  完了！結果ファイル:
echo    out\evolve_b_best.csv   （推奨戦略一覧）
echo    out\evolve_b_report.txt （サマリーレポート）
echo ============================================================
echo.
pause
