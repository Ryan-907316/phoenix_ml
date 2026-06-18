@echo off
cd /d "%~dp0"
echo Loading phoenix_ml, please wait...
python app.py
if errorlevel 1 (
    echo.
    echo Something went wrong. See the error above.
    pause
)
