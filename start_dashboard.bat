@echo off
echo ================================================================================
echo   AttnRetrofit Dashboard Launcher
echo ================================================================================
echo.
echo Starting FastAPI server...
echo.
echo Open your browser to:
echo   - http://localhost:8000        (Dashboard)
echo   - http://localhost:8000/docs   (API Documentation)
echo.
echo Press Ctrl+C to stop the server.
echo ================================================================================
echo.

call conda activate attnretrofit
python api_server.py
