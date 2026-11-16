@echo off
chcp 65001 >nul 2>&1
title ClearSound - Setup
color 0B

cd /d "%~dp0"

echo.
echo ========================================================
echo               ClearSound - Setup
echo ========================================================
echo.

REM --- 1. Checking Python ---
echo [1/3] Checking Python installation...
set "PY_CMD="

python --version >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=python"
)

if not defined PY_CMD (
    py -3 --version >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=py -3"
    )
)

if not defined PY_CMD (
    if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
        set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    ) else if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    ) else if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
        set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    )
)

if not defined PY_CMD (
    echo [ERROR] Python not found!
    echo Please install Python 3.9+ from https://www.python.org/
    echo Make sure to check 'Add Python to PATH' during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('%PY_CMD% --version 2^>^&1') do echo OK: %%v

REM --- 2. Create Virtual Environment ---
echo.
echo [2/3] Setting up virtual environment (venv)...
if not exist "venv\Scripts\activate.bat" (
    %PY_CMD% -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo OK: venv created successfully.
) else (
    echo OK: venv already exists.
)

REM --- 3. Install Dependencies ---
echo.
echo [3/3] Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install packages from requirements.txt!
    pause
    exit /b 1
)
python -m pip install -e .

echo.
echo ========================================================
echo               Installation Completed!
echo               Run run.bat to launch ClearSound
echo ========================================================
echo.
pause
