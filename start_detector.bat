@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=%~dp0..\..\local\environments\study-v3\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo The registered Python 3.12 runtime is missing.
  exit /b 1
)
"%PYTHON_EXE%" scripts\check_live_environment.py
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" scripts\live_detector.py --check
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" scripts\launch_detector.py
exit /b %ERRORLEVEL%
