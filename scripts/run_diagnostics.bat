@echo off
setlocal
cd /d "%~dp0.."

if "%~1"=="" (
  echo Usage: scripts\run_diagnostics.bat dashboard [COM5] [duration_seconds]
  echo        scripts\run_diagnostics.bat cli       [COM5] [duration_seconds]
  exit /b 2
)

set "MODE=%~1"
set "PORT=%~2"
if "%PORT%"=="" set "PORT=COM5"
set "DURATION=%~3"
if "%DURATION%"=="" set "DURATION=60"

if exist "venv\Scripts\python.exe" (
  set "PYTHON=venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

echo Running %MODE% diagnostics on %PORT% for %DURATION% seconds...
"%PYTHON%" scripts\diagnose.py "%MODE%" --port "%PORT%" --duration "%DURATION%"
echo.
echo The diagnostic run is complete. Attach the newest folder under diagnostics\
pause
