@echo off
setlocal
chcp 65001
cd /d "%~dp0"

if not exist "venv" (
    echo [ERROR] venv not found. Please run setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate
if "%VDC_SERVER_NAME%"=="" set "VDC_SERVER_NAME=0.0.0.0"
if "%VDC_SERVER_PORT%"=="" set "VDC_SERVER_PORT=7860"
echo [INFO] Starting VoiceDesignCloner...
echo [INFO] Browser will open automatically.
echo [INFO] Default URL: http://127.0.0.1:%VDC_SERVER_PORT%
echo [INFO] Bind address: %VDC_SERVER_NAME%:%VDC_SERVER_PORT%
python app.py
pause
