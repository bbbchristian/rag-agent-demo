@echo off
echo ====================================
echo   AI Agent Demo Launcher
echo ====================================
echo.

cd /d "%~dp0"

echo [*] Starting Streamlit app...
echo.
echo Open http://localhost:8501 in your browser
echo.
streamlit run app.py --server.port 8501

if errorlevel 1 (
    echo.
    echo Install dependencies first: pip install -r requirements.txt
    pause
)
