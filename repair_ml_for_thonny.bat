@echo off
cd /d "%~dp0"
set "THONNY_PY=%LOCALAPPDATA%\Programs\Thonny\python.exe"
if not exist "%THONNY_PY%" (
    echo Thonny Python not found at:
    echo   %THONNY_PY%
    echo Install Thonny from https://thonny.org or open model_trainer.py in Thonny and press F5.
    pause
    exit /b 1
)

echo.
echo  This fixes: No module named numpy._core  (NumPy 2 pickle on NumPy 1 / Thonny)
echo  Folder: %CD%
echo.

if exist "models\trained_model.pkl" (
    echo Backing up current model to models\trained_model.pkl.backup_before_thonny_rebuild ...
    copy /Y "models\trained_model.pkl" "models\trained_model.pkl.backup_before_thonny_rebuild" >nul
)

echo Checking Thonny NumPy (must be 1.x for typical Thonny) ...
"%THONNY_PY%" -c "import numpy as n; v=n.__version__.split('.')[0]; assert v=='1', 'Expected NumPy 1.x in Thonny, got '+n.__version__; print('  NumPy OK:', n.__version__)"
if errorlevel 1 (
    echo.
    echo If this failed, Thonny has NumPy 2 — use train_model.bat + run_dashboard.bat instead.
    pause
    exit /b 1
)

echo.
echo Training (needs data\CICIDS2017_WebAttacks.csv) ...
"%THONNY_PY%" model_trainer.py
if errorlevel 1 (
    echo.
    echo Training failed. Restore backup if needed:
    echo   copy /Y models\trained_model.pkl.backup_before_thonny_rebuild models\trained_model.pkl
    pause
    exit /b 1
)

echo.
echo Done. Start the SOC from Thonny %%Run working_app.py OR run_dashboard_thonny.bat
echo.
pause
