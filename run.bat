@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo .venv ne naiden. Snachala zapustite setup.bat
    pause
    exit /b 1
)
.venv\Scripts\python.exe main.py
if errorlevel 1 (
    pause
)
