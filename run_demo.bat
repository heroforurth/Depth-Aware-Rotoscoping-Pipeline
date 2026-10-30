@echo off
REM Double-click me after setup.bat. Runs the pipeline on a sample video and opens the result.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please double-click setup.bat first.
  if not "%NOPAUSE%"=="1" pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\run_demo.py
set CODE=%errorlevel%
if not "%NOPAUSE%"=="1" pause
exit /b %CODE%
