@echo off
title AGV Simulator
cd /d "%~dp0"
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo Python 3 is not installed on this PC.
  echo Install it from https://www.python.org/downloads/
  echo IMPORTANT: tick "Add python.exe to PATH" in the installer, then double-click me again.
  pause
  exit /b 1
)
if not exist .venv\Scripts\python.exe (
  echo First start: setting up - this takes a minute, it needs internet once...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo Could not set up Python here. Try moving this folder to Documents and start again.
    pause
    exit /b 1
  )
  .venv\Scripts\python -m pip install --upgrade pip -q
)
.venv\Scripts\python -m pip install -q -r requirements.txt
if errorlevel 1 (
  echo Install failed - check the internet connection and start again.
  pause
  exit /b 1
)
echo.
echo AGV Simulator is starting - your browser opens by itself. Keep this window open while you work.
.venv\Scripts\python app.py
pause
