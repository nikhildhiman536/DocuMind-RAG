@echo off
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found.
    echo Please create it and install requirements first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python -m streamlit run app.py

pause