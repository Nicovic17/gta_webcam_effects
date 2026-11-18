@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Ambiente Python mancante. Esegui prima SETUP_GTA.ps1.
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -B "%~dp0gta.py" %*
if errorlevel 1 pause
