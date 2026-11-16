@echo off
chcp 65001 >nul 2>&1
title ClearSound
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo.
    echo ========================================================
    echo  [ERROR] Virtual environment (venv) not found!
    echo  Please run install.bat first to set up ClearSound.
    echo ========================================================
    echo.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

REM Launch GUI (with or without drag-and-dropped file)
if "%~1"=="" (
    start "" python -m clearsound.ui.app
) else (
    start "" python -m clearsound.ui.app "%~1"
)
exit /b 0
