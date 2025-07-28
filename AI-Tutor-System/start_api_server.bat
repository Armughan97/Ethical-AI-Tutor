@echo off
REM Activate virtual environment
cd /d "%~dp0AI-Tutor-System"
call myvenv\Scripts\activate

REM Start Conversation API (port 8500)
start "Conversation API" cmd /k python -m uvicorn api:app --host 127.0.0.1 --port 8500 --reload

REM Return to original directory
cd /d "%~dp0"
echo All servers and simulator started!
pause