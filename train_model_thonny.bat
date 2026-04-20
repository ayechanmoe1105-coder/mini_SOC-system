@echo off
cd /d "%~dp0"
set "THONNY_PY=%LOCALAPPDATA%\Programs\Thonny\python.exe"
if not exist "%THONNY_PY%" (
    echo Thonny Python was not found at:
    echo   %THONNY_PY%
    echo Open model_trainer.py in Thonny and press F5 instead.
    pause
    exit /b 1
)
echo.
echo  Training with Thonny Python (same as F5 in Thonny) — NumPy 1.x
echo  Folder: %CD%
echo  Uses:   data\CICIDS2017_WebAttacks.csv
echo  Writes: models\trained_model.pkl
echo  After training, start the dashboard with:  run_dashboard_thonny.bat
echo  (Do not use run_dashboard.bat — that is py-3 / NumPy 2 and mismatches this pickle.)
echo.
"%THONNY_PY%" model_trainer.py
echo.
pause
