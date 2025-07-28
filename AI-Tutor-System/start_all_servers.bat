@echo off
REM Activate virtual environment
cd /d "%~dp0AI-Tutor-System"
call myvenv\Scripts\activate

REM Start Intent API (port 9000)
start "Intent API" cmd /k python -m uvicorn intent_api:app --host 127.0.0.1 --port 9000 --reload

REM Start LLM API (port 8000)
start "LLM API" cmd /k python -m uvicorn llm_api:app --host 127.0.0.1 --port 8000 --reload

@REM REM Start Conversation API (port 8500)
@REM start "Conversation API" cmd /k python -m uvicorn api:app --host 127.0.0.1 --port 8500 --reload

@REM REM Start Simulator
@REM start "Simulator" cmd /k python simulator.py

REM Return to original directory
cd /d "%~dp0"
echo All servers and simulator started!
pause