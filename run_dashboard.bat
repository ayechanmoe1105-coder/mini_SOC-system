@echo off
cd /d "%~dp0"
echo.
echo  Starting SOC from this folder: %CD%
echo  Uses py -3 (same stack as train_model.bat).
echo  Thonny users: use run_dashboard_thonny.bat instead — NumPy 1 / NumPy 2 pickles differ.
echo  Dashboard: http://127.0.0.1:15500
echo.
py -3 working_app.py
pause
