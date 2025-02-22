@echo off
set SCRIPT_NAME=SocketServer.py
set API_URL=http://localhost:3000/api/chat/completions
set PROXY=http://127.0.0.1:7890
set STREAM=y
set CHARACTER=paimon
set MODEL=deepseek-chat
set MODELB=qwen2.5:latest
set MODELa=deepseek-r1:1.5b
set PROMPT=AICustomerService


.\venv\Scripts\python.exe %SCRIPT_NAME% --apiUrl %API_URL% --stream %STREAM% --character %CHARACTER% --model %MODEL% --prompt %PROMPT%
