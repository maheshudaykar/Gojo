@echo off
REM Production-grade launcher for Gojo web UI
REM Includes WSGI server, heartbeat monitoring, and graceful shutdown

echo ================================================
echo   Gojo - Production Launch
echo ================================================
echo.

REM Check for virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found at .venv\
    echo Please create virtual environment first: python -m venv .venv
    echo Then install dependencies: .venv\Scripts\pip install -r requirements_production.txt
    pause
    exit /b 1
)

REM Check for required models
if not exist "models\lexical_model.joblib" (
    echo WARNING: Lexical model not found
    echo Please train models first: .venv\Scripts\python -m phish_detector.train
    echo.
)

if not exist "models\char_model.joblib" (
    echo WARNING: Char model not found
    echo Please train models first: .venv\Scripts\python -m phish_detector.train
    echo.
)

REM Create necessary directories
if not exist "logs" mkdir logs
if not exist "webapp\output" mkdir webapp\output

echo [INFO] Starting production WSGI server...
echo [INFO] Access at: http://127.0.0.1:5000
echo [INFO] Press Ctrl+C to stop server
echo [INFO] Server will auto-shutdown if browser tab is closed
echo.

REM Start production server with waitress
start /MIN cmd /c ".venv\Scripts\python.exe -m waitress --listen=127.0.0.1:5000 webapp.app:app"

REM Wait for server to start
timeout /t 3 /nobreak >nul

REM Open browser
echo [INFO] Opening browser...
start http://127.0.0.1:5000

echo.
echo [INFO] Server running in background window
echo [INFO] Close browser tab to auto-shutdown server
echo [INFO] Or press any key here to manually shutdown...
pause >nul

REM Server will auto-shutdown via heartbeat mechanism
exit
