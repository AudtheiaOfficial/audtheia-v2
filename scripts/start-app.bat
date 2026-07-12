@echo off
rem Audtheia desktop app-window launcher for Windows.
rem
rem Double-click this to start Audtheia in its own desktop window instead of a
rem browser tab. It finds a suitable Python and hands control to the launcher with
rem the window option; any extra options are forwarded, for example:
rem
rem   start-app.bat               open Audtheia in a desktop window
rem   start-app.bat --no-browser  do not fall back to the browser
rem
setlocal
set "SCRIPT_DIR=%~dp0"

where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3 "%SCRIPT_DIR%bootstrap_start.py" --window %*
  exit /b %ERRORLEVEL%
)

where python >nul 2>nul
if %ERRORLEVEL%==0 (
  python "%SCRIPT_DIR%bootstrap_start.py" --window %*
  exit /b %ERRORLEVEL%
)

echo Python 3 was not found. Run the desktop setup first.
exit /b 1
