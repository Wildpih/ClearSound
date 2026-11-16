@echo off
chcp 65001 >nul 2>&1
title ClearSound CLI
color 0B
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo.
    echo [ERROR] Virtual environment not found! Run install.bat first.
    echo.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

if not "%~1"=="" (
    echo.
    echo Processing: %~nx1
    echo.
    python -m clearsound.main process "%~1"
    echo.
    pause
    exit /b 0
)

:menu
cls
echo.
echo ========================================================
echo  ClearSound CLI - Console Menu
echo ========================================================
echo.
echo   [1] Launch Graphical Interface (GUI)
echo   [2] Process file in console (Full pipeline)
echo   [3] Remove vocals only
echo   [4] Sound library summary
echo   [5] Generate basic sounds
echo   [6] Open output folder
echo   [0] Exit
echo.
set /p choice="Select option (0-6): "

if "%choice%"=="1" (
    start "" python -m clearsound.ui.app
    exit /b 0
)
if "%choice%"=="2" goto process
if "%choice%"=="3" goto remove_only
if "%choice%"=="4" goto library
if "%choice%"=="5" goto generate
if "%choice%"=="6" goto open_output
if "%choice%"=="0" exit /b 0
goto menu

:process
cls
echo.
set /p inputfile="Enter file path: "
set inputfile=%inputfile:"=%
if not exist "%inputfile%" (
    echo File not found!
    pause
    goto menu
)
python -m clearsound.main process "%inputfile%"
pause
goto menu

:remove_only
cls
echo.
set /p inputfile="Enter file path: "
set inputfile=%inputfile:"=%
if not exist "%inputfile%" (
    echo File not found!
    pause
    goto menu
)
python -m clearsound.main remove-vocals "%inputfile%"
pause
goto menu

:library
cls
echo.
python -m clearsound.main library
pause
goto menu

:generate
cls
echo.
python -m clearsound.main generate
pause
goto menu

:open_output
explorer "%~dp0output"
goto menu
