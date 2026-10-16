@echo off
REM Double-click me first. Builds .venv, installs everything, downloads the AI models.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_env.ps1" %*
if errorlevel 1 (
  echo.
  echo SETUP FAILED - read the message above, fix it, then double-click setup.bat again.
  if not "%NOPAUSE%"=="1" pause
  exit /b 1
)
if not "%NOPAUSE%"=="1" pause
