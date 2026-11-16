@echo off
chcp 65001 >nul 2>&1
title ClearSound
color 0B

cd /d "%~dp0"

:: ── Проверка установки ───────────────────────────────────────
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo  [ОШИБКА] Виртуальное окружение не найдено!
    echo  Сначала запустите install.bat
    echo.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

:: ── Если файл перетащен на .bat ──────────────────────────────
if not "%~1"=="" (
    echo.
    echo  Обрабатываем: %~nx1
    echo.
    python -m clearsound.main process "%~1"
    echo.
    pause
    exit /b 0
)

:: ── Главное меню ─────────────────────────────────────────────
:menu
cls
echo.
echo   ██████╗██╗     ███████╗ █████╗ ██████╗
echo  ██╔════╝██║     ██╔════╝██╔══██╗██╔══██╗
echo  ██║     ██║     █████╗  ███████║██████╔╝
echo  ██║     ██║     ██╔══╝  ██╔══██║██╔══██╗
echo  ╚██████╗███████╗███████╗██║  ██║██║  ██║
echo   ╚═════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝
echo  ███████╗ ██████╗ ██╗   ██╗███╗   ██╗██████╗
echo  ██╔════╝██╔═══██╗██║   ██║████╗  ██║██╔══██╗
echo  ███████╗██║   ██║██║   ██║██╔██╗ ██║██║  ██║
echo  ╚════██║██║   ██║██║   ██║██║╚██╗██║██║  ██║
echo  ███████║╚██████╔╝╚██████╔╝██║ ╚████║██████╔╝
echo  ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═════╝
echo.
echo  ══════════════════════════════════════════════
echo   Удаление голоса и восстановление звука
echo  ══════════════════════════════════════════════
echo.
echo   [1] Обработать файл (полный pipeline)
echo   [2] Только удалить голос
echo   [3] Обработать файл (без предпрослушивания)
echo   [4] Показать библиотеку звуков
echo   [5] Сгенерировать базовые звуки
echo   [6] Добавить звук в библиотеку
echo   [7] Открыть папку output
echo   [0] Выход
echo.
set /p choice="  Выберите действие: "

if "%choice%"=="1" goto process
if "%choice%"=="2" goto remove_only
if "%choice%"=="3" goto process_auto
if "%choice%"=="4" goto library
if "%choice%"=="5" goto generate
if "%choice%"=="6" goto add_sound
if "%choice%"=="7" goto open_output
if "%choice%"=="0" goto quit
goto menu

:: ── 1. Полная обработка ──────────────────────────────────────
:process
cls
echo.
echo  ══════════════════════════════════════════════
echo   Полная обработка (с предпрослушиванием)
echo  ══════════════════════════════════════════════
echo.
echo  Введите путь к файлу (или перетащите файл сюда):
echo.
set /p inputfile="  Файл: "
:: Убираем кавычки если есть
set inputfile=%inputfile:"=%

if not exist "%inputfile%" (
    echo.
    echo  [ОШИБКА] Файл не найден: %inputfile%
    pause
    goto menu
)

echo.
set /p outputfile="  Сохранить как (Enter = автоматически): "

if "%outputfile%"=="" (
    python -m clearsound.main process "%inputfile%"
) else (
    set outputfile=%outputfile:"=%
    python -m clearsound.main process "%inputfile%" -o "%outputfile%"
)
echo.
pause
goto menu

:: ── 2. Только удаление голоса ────────────────────────────────
:remove_only
cls
echo.
echo  ══════════════════════════════════════════════
echo   Только удаление голоса (без восстановления)
echo  ══════════════════════════════════════════════
echo.
echo  Введите путь к файлу:
echo.
set /p inputfile="  Файл: "
set inputfile=%inputfile:"=%

if not exist "%inputfile%" (
    echo.
    echo  [ОШИБКА] Файл не найден: %inputfile%
    pause
    goto menu
)

echo.
set /p outputfile="  Сохранить как (Enter = автоматически): "

if "%outputfile%"=="" (
    python -m clearsound.main remove-vocals "%inputfile%"
) else (
    set outputfile=%outputfile:"=%
    python -m clearsound.main remove-vocals "%inputfile%" -o "%outputfile%"
)
echo.
pause
goto menu

:: ── 3. Обработка без интерактива ─────────────────────────────
:process_auto
cls
echo.
echo  ══════════════════════════════════════════════
echo   Обработка без предпрослушивания
echo  ══════════════════════════════════════════════
echo.
echo  Введите путь к файлу:
echo.
set /p inputfile="  Файл: "
set inputfile=%inputfile:"=%

if not exist "%inputfile%" (
    echo.
    echo  [ОШИБКА] Файл не найден: %inputfile%
    pause
    goto menu
)

echo.
set /p outputfile="  Сохранить как (Enter = автоматически): "

if "%outputfile%"=="" (
    python -m clearsound.main process "%inputfile%" --no-interactive
) else (
    set outputfile=%outputfile:"=%
    python -m clearsound.main process "%inputfile%" -o "%outputfile%" --no-interactive
)
echo.
pause
goto menu

:: ── 4. Библиотека ────────────────────────────────────────────
:library
cls
echo.
python -m clearsound.main library
echo.
pause
goto menu

:: ── 5. Генерация звуков ──────────────────────────────────────
:generate
cls
echo.
python -m clearsound.main generate
echo.
pause
goto menu

:: ── 6. Добавить звук ─────────────────────────────────────────
:add_sound
cls
echo.
echo  ══════════════════════════════════════════════
echo   Добавить звук в библиотеку
echo  ══════════════════════════════════════════════
echo.
echo  Введите путь к аудио-файлу:
set /p soundfile="  Файл: "
set soundfile=%soundfile:"=%

if not exist "%soundfile%" (
    echo.
    echo  [ОШИБКА] Файл не найден: %soundfile%
    pause
    goto menu
)

echo.
echo  Выберите категорию:
echo   [1] ambient  — фоновые шумы
echo   [2] music    — музыкальные элементы
echo   [3] nature   — природа
echo   [4] urban    — город
echo   [5] foley    — звуковые эффекты
echo.
set /p cat_choice="  Категория (1-5): "

if "%cat_choice%"=="1" set category=ambient
if "%cat_choice%"=="2" set category=music
if "%cat_choice%"=="3" set category=nature
if "%cat_choice%"=="4" set category=urban
if "%cat_choice%"=="5" set category=foley
if not defined category set category=foley

python -m clearsound.main add-sound "%soundfile%" -c %category%
echo.
pause
goto menu

:: ── 7. Открыть output ────────────────────────────────────────
:open_output
explorer "%~dp0output"
goto menu

:: ── Выход ────────────────────────────────────────────────────
:quit
echo.
echo  До встречи!
echo.
exit /b 0
