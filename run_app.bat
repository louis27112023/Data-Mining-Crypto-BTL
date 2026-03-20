@echo off
echo =======================================================
echo     KHOI DONG HE THONG WEB APP - DATA MINING
echo =======================================================
echo.

cd /d "%~dp0"

:: 1. Kiem tra va kich hoat moi truong ao (.venv)
echo Dang nap moi truong Python (.venv)...
if exist "..\.venv\Scripts\activate.bat" (
    call "..\.venv\Scripts\activate.bat"
) else (
    echo Khong tim thay moi truong o thuc muc ..\.venv\
    pause
    exit /b
)

:: 2. Cai dat luon Streamlit neu May tinh User chua co trong Venv (de phong)
echo Dang kiem tra thu vien (chac chung 5 giay thoi)...
pip install streamlit pandas matplotlib seaborn scikit-learn >nul 2>&1

:: 3. Chay Web App
echo.
echo =======================================================
echo 🚀 HEO THONG GIAO DIEN CHUAN BI MO TREN TRINH DUYET!
echo =======================================================
python -m streamlit run app.py

pause
