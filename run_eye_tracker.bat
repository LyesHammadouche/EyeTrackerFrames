@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo   Drawing Analysis Eye Tracker Launcher
echo ==========================================

REM Check if python is available
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python is not found in PATH.
    echo Please install Python 3.10+ and ensure it is added to your PATH.
    pause
    exit /b 1
)

echo Starting Application...
python "drawing_analysis/app/main.py"

if %errorlevel% neq 0 (
    echo.
    echo Application crashed with error code %errorlevel%.
    pause
)
endlocal
