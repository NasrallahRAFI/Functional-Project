@echo off
setlocal
rem The legacy Functional Project entry point opens the current V4 monitor.
rem This file is hard-linked into that directory; explicit V3 launches stay V3.
if exist "%~dp0..\project.layout.json" goto :launch_v4
cd /d "%~dp0"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  where py >nul 2>nul
  if errorlevel 1 (
    echo Python 3 is required. Install Python 3.12 and try again.
    pause
    exit /b 1
  )
  py -3.12 -m venv .venv
  if errorlevel 1 goto :failed
)
"%PYTHON_EXE%" scripts\check_live_environment.py >nul 2>nul
if errorlevel 1 (
  echo Preparing the local detector Python runtime. This may take a few minutes.
  "%PYTHON_EXE%" -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-live.txt
  if errorlevel 1 goto :failed
)
"%PYTHON_EXE%" scripts\check_live_environment.py
if errorlevel 1 goto :failed
"%PYTHON_EXE%" scripts\live_detector.py --check
if errorlevel 1 goto :failed
"%PYTHON_EXE%" scripts\launch_detector.py
set "LAUNCH_EXIT_CODE=%ERRORLEVEL%"
if not "%LAUNCH_EXIT_CODE%"=="0" (
  echo The local detector did not start cleanly. Read the error above, then check the local runtime and model assets.
  pause
)
exit /b %LAUNCH_EXIT_CODE%
:failed
echo Could not prepare or start the local detector. Read the error above and check Python 3.12, dependencies, and model files.
pause
exit /b 1

:launch_v4
if not exist "%~dp0..\workspaces\study-v4\start_detector.bat" (
  echo The V4 launcher is missing from workspaces\study-v4. Restore that workspace and retry.
  exit /b 1
)
echo Starting CNN V4.0 at http://127.0.0.1:8766/live/index.html
call "%~dp0..\workspaces\study-v4\start_detector.bat"
exit /b %ERRORLEVEL%
