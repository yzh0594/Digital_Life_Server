@echo off
set SCRIPT_NAME=SocketServer.py
set API_URL=http://localhost:11434/api/chat
set PROXY=http://127.0.0.1:7890
set STREAM=y
set CHARACTER=paimon
set MODEL=qwen2.5
set PROMPT=AICustomerService


.\venv\Scripts\python.exe %SCRIPT_NAME% --apiUrl %API_URL% --stream %STREAM% --character %CHARACTER% --model %MODEL% --prompt %PROMPT%
