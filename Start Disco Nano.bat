@echo off
title Disco Diffusion Nano
setlocal
rem Settings left by other tools that would point Python at the wrong libraries, or make
rem its downloads fail: SSLKEYLOGFILE crashes the standalone Python that setup can install.
set "PYTHONPATH="
set "PYTHONHOME="
set "SSLKEYLOGFILE="
set "HOME_DIR=%~dp0data"
if exist "%~dp0home.txt" set /p HOME_DIR=<"%~dp0home.txt"
if defined DISCO_NANO_HOME set "HOME_DIR=%DISCO_NANO_HOME%"
set "PY=%HOME_DIR%\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Could not find the environment at %HOME_DIR%. Run setup.bat first.
  pause
  exit /b 1
)
cd /d "%~dp0app"
echo Starting Disco Diffusion Nano. The app opens in your browser when it is ready.
"%PY%" -B server.py
if errorlevel 1 pause
