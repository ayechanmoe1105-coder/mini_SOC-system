@echo off
cd /d "%~dp0"
set "THONNY_PY=%LOCALAPPDATA%\Programs\Thonny\python.exe"
if not exist "%THONNY_PY%" (
    echo Thonny Python was not found at:
    echo   %THONNY_PY%
    echo Install Thonny, or start the SOC with run_dashboard.bat using py -3 instead.
    pause
    exit /b 1
)
echo.
echo  SOC Dashboard — Thonny Python (NumPy 1.x — same stack as train_model_thonny.bat)
echo  Folder: %CD%
echo  If the ML model was trained with train_model.bat / py -3, it will NOT load here.
echo  Fix: run train_model_thonny.bat first (or model_trainer.py F5 in Thonny), then this file.
echo  URL:  http://127.0.0.1:15500
echo.
"%THONNY_PY%" working_app.py
pause
