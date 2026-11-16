@echo off
chcp 65001 >nul 2>&1
title ClearSound — Установка
color 0B

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║         ClearSound — Установка               ║
echo  ╚══════════════════════════════════════════════╝
echo.

cd /d "%~dp0"

:: ── Проверка Python ──────────────────────────────────────────
echo  [1/3] Проверяем Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ОШИБКА] Python не найден!
    echo  Установите Python 3.9+ с https://www.python.org/
    echo  Убедитесь, что он добавлен в PATH.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do (
    echo  OK: Python %%v
)

:: ── Создание виртуального окружения ──────────────────────────
echo.
echo  [2/3] Создаём виртуальное окружение...
if not exist "venv" (
    python -m venv venv
    echo  OK: venv создано
) else (
    echo  OK: venv уже существует
)

:: ── Установка зависимостей ───────────────────────────────────
echo.
echo  [3/3] Устанавливаем зависимости...
call venv\Scripts\activate.bat
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
pip install -e . >nul 2>&1

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║        Установка завершена!                  ║
echo  ║        Запустите run.bat для работы           ║
echo  ╚══════════════════════════════════════════════╝
echo.
pause
