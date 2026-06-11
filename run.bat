@echo off
REM run.bat — Load .env variables and start the server.
REM Usage: run.bat (double-click or type run.bat in Command Prompt)

REM Read each line from .env and set it as an environment variable.
REM Lines starting with # are treated as comments and skipped.
for /f "usebackq tokens=1,* delims==" %%A in (`findstr /v "^#" .env`) do (
    set "%%A=%%B"
    echo Set %%A
)

echo.
echo Starting server...
python -m uvicorn app.main:app --reload
