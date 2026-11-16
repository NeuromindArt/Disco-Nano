@echo off
title Disco Diffusion Nano setup
setlocal
rem Settings left by other tools that would point Python at the wrong libraries, or make
rem its downloads fail: SSLKEYLOGFILE crashes the standalone Python that setup can install.
set "PYTHONPATH="
set "PYTHONHOME="
set "SSLKEYLOGFILE="
set "HOME_DIR=%~dp0data"
if exist "%~dp0home.txt" set /p HOME_DIR=<"%~dp0home.txt"
if defined DISCO_NANO_HOME set "HOME_DIR=%DISCO_NANO_HOME%"
set "VENV_PY=%HOME_DIR%\.venv\Scripts\python.exe"
set "UV_VERSION=0.11.6"

echo "%HOME_DIR%" | find /i "OneDrive" >nul
if errorlevel 1 goto home_ok
echo.
echo The data folder would be inside OneDrive: %HOME_DIR%
echo It holds the environment and models, over 10GB, which OneDrive would try to upload.
echo Move the Disco Diffusion Nano folder somewhere like C:\Disco Diffusion Nano first,
echo or put another folder's path in a file named home.txt beside setup.bat.
echo.
choice /m "Continue anyway"
if errorlevel 2 exit /b 1
:home_ok

if not exist "%HOME_DIR%" mkdir "%HOME_DIR%" || goto failed
if exist "%VENV_PY%" goto install

py -3.12 -c "pass" >nul 2>nul
if errorlevel 1 goto own_python
echo Making the environment with your Python 3.12.
py -3.12 -m venv "%HOME_DIR%\.venv" || goto failed
goto install

:own_python
rem No Python 3.12 on this PC, so uv fetches one into the data folder. Nothing is installed system-wide.
echo Downloading Python 3.12 into %HOME_DIR%
set "UV=%HOME_DIR%\tools\uv.exe"
set "UV_PYTHON_INSTALL_DIR=%HOME_DIR%\python"
set "UV_CACHE_DIR=%HOME_DIR%\cache\uv"
set "UV_PYTHON_PREFERENCE=only-managed"
set "UV_LINK_MODE=copy"
set "UV_NO_CONFIG=1"
rem Use the Windows certificate store, so antivirus HTTPS scanning and company proxies work.
set "UV_SYSTEM_CERTS=1"
if exist "%UV%" goto have_uv
if not exist "%HOME_DIR%\tools" mkdir "%HOME_DIR%\tools"
curl.exe -L --fail -s -S -o "%HOME_DIR%\tools\uv.zip" "https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/uv-x86_64-pc-windows-msvc.zip" || goto failed
tar.exe -xf "%HOME_DIR%\tools\uv.zip" -C "%HOME_DIR%\tools" || goto failed
del "%HOME_DIR%\tools\uv.zip"
:have_uv
"%UV%" venv --quiet --seed --clear --python 3.12 "%HOME_DIR%\.venv" || goto failed

:install
rem Ignore pip.ini files left by other tools, such as nvidia-pyindex, which add package servers or turn off the cache.
set "PIP_CONFIG_FILE=nul"
echo.
echo Installing the libraries Nano uses. The first time is about 6GB and takes a while.
"%VENV_PY%" -m pip install --disable-pip-version-check -r "%~dp0app\requirements.txt" || goto failed
"%VENV_PY%" -B "%~dp0app\fetch_code.py" || goto failed
echo.
echo Looking for models other apps already have, then downloading the main model, about 2GB, if needed.
"%VENV_PY%" -B "%~dp0app\download_models.py" || echo The download failed. It will try again on your first render.
echo.
rem A .bat file can't have an icon of its own, so a shortcut beside it carries the Disco icon.
echo Making a Disco Nano shortcut with the app's icon.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s = (New-Object -ComObject WScript.Shell).CreateShortcut('%~dp0Disco Nano.lnk'); $s.TargetPath = '%~dp0Start Disco Nano.bat'; $s.WorkingDirectory = '%~dp0'; $s.IconLocation = '%~dp0app\static\icon.ico'; $s.Description = 'Disco Diffusion Nano'; $s.Save()" >nul 2>&1
echo.
echo Setup finished. Double-click "Disco Nano" to open the app.
echo Optional models download the first time you use them.
pause
exit /b 0

:failed
echo.
echo Setup failed. See the messages above. Running setup.bat again picks up where it stopped.
pause
exit /b 1
