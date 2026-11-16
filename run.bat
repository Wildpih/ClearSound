@echo off
chcp 65001 >nul 2>&1
title ClearSound UI
cd /d "%~dp0"

:: ── Проверка установки ───────────────────────────────────────
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo  [ОШИБКА] Виртуальное окружение не найдено!
    echo  Сначала запустите install.bat для установки всех компонентов.
    echo.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

:: ── Запуск графического интерфейса ClearSound ────────────────
if "%~1"=="" (
    start "" python -m clearsound.ui.app
) else (
    start "" python -m clearsound.ui.app "%~1"
)
exit /b 0
