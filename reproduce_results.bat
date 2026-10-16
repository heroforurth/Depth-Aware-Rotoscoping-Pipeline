@echo off
REM Re-runs the evaluation and compares your numbers with reference_results\.
REM   reproduce_results.bat          quick: Bonn depth + DAVIS segmentation (~3.3 GB, ~1 hour)
REM   reproduce_results.bat full     + Sintel + YouTubeMatte (~13 GB, ~3 hours)
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please double-click setup.bat first.
  if not "%NOPAUSE%"=="1" pause
  exit /b 1
)
set LEVEL=%1
if "%LEVEL%"=="" set LEVEL=quick
".venv\Scripts\python.exe" scripts\reproduce.py %LEVEL%
if not "%NOPAUSE%"=="1" pause
