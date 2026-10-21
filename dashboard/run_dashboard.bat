@echo off
echo Installing required Python packages...
pip install fastapi uvicorn numpy websockets
echo.
echo Starting the CSI Translation Bridge and Dashboard Server...
cd backend
python app.py
pause
