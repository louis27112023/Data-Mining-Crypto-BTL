@echo off
echo ===================================================
echo   KHOI TAO MOI TRUONG VA CHAY DATA MINING PIPELINE
echo ===================================================
echo.
echo [1/3] Dang tao moi truong ao (venv)...
c:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe -m venv venv

echo [2/3] Dang cai dat thu vien (pip install)...
call venv\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo [3/3] Dang chay Pipeline hoc may...
python scripts\run_pipeline.py

echo.
echo ===================================================
echo HOAN TAT! Tat ca ket qua duoc luu trong thu muc outputs/tables/
echo ===================================================
pause
