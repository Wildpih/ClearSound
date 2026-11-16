@echo off
chcp 65001 >nul 2>&1
title ClearSound Console CLI
color 0B

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo.
    echo  [ОШИБКА] Виртуальное окружение не найдено!
    echo  Сначала запустите install.bat
    echo.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

:: Если файл перетащен на .bat
if not "%~1"=="" (
    echo.
    echo  Обрабатываем: %~nx1
    echo.
    python -m clearsound.main process "%~1"
    echo.
    pause
    exit /b 0
)

:menu
cls
echo.
echo  ══════════════════════════════════════════════
echo   ClearSound CLI — Меню командной строки
echo  ══════════════════════════════════════════════
echo.
echo   [1] Запустить графический интерфейс (UI)
echo   [2] Обработать файл через консоль
echo   [3] Только удалить голос
echo   [4] Показать библиотеку звуков
echo   [5] Сгенерировать базовые звуки
echo   [6] Открыть папку output
echo   [0] Выход
echo.
set /p choice="  Выберите действие: "

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
set /p inputfile="  Введите путь к файлу: "
set inputfile=%inputfile:"=%
if not exist "%inputfile%" (
    echo  Файл не найден!
    pause
    goto menu
)
python -m clearsound.main process "%inputfile%"
pause
goto menu

:remove_only
cls
echo.
set /p inputfile="  Введите путь к файлу: "
set inputfile=%inputfile:"=%
if not exist "%inputfile%" (
    echo  Файл не найден!
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
