@echo off
cd /d "%~dp0"
echo.
echo  Training ML model from this folder: %CD%
echo.
echo  IMPORTANT — Thonny vs Command Prompt:
echo   * Thonny uses NumPy 1.x.  "py -3" on newer Python often uses NumPy 2.x.
echo   * A pickle saved with NumPy 2 CANNOT load in Thonny (numpy._core error).
echo.
echo   Thonny SOC: do NOT use this file. Open model_trainer.py in Thonny, press F5.
echo   OR use train_model_thonny.bat (Thonny's python.exe).
echo.
echo   Batch / py -3 SOC: use this file, then always start the dashboard with
echo   run_dashboard.bat (same Python) — not Thonny.
echo.
echo  Requires: data\CICIDS2017_WebAttacks.csv
echo  Writes:   models\trained_model.pkl
echo.
py -3 model_trainer.py
echo.
pause
