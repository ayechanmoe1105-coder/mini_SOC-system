@echo off
REM Install mlxtend with the same Python launcher (adjust if you use Thonny only).
cd /d "%~dp0"
echo Installing mlxtend for Apriori (full algorithm)...
py -3 -m pip install mlxtend
if errorlevel 1 python -m pip install mlxtend
if errorlevel 1 (
  echo.
  echo If both failed: open Thonny -^> Tools -^> Manage packages -^> search mlxtend -^> Install.
  pause
  exit /b 1
)
echo.
echo Done. Restart working_app.py and click Run Apriori again. You should see Algorithm: apriori
pause
