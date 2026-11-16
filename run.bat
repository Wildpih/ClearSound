@echo off
chcp 65001 >nul 2>&1
title ClearSound
cd /d "%~dp0"

if not exist "%~dp0venv\Scripts\activate.bat" (
    echo.
    echo ========================================================
    echo  [ERROR] Virtual environment not found!
    echo  Please run install.bat first to set up ClearSound.
    echo ========================================================
    echo.
    pause
    exit /b 1
)

call "%~dp0venv\Scripts\activate.bat"

echo.
echo ========================================================
echo  Starting ClearSound GUI...
echo ========================================================
echo.

"%~dp0venv\Scripts\python.exe" -m clearsound.ui.app %*
if errorlevel 1 (
    echo.
    echo ========================================================
    echo  ClearSound exited with an error. See details above.
    echo ========================================================
    echo.
    pause
)
