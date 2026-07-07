@echo off
rem Audtheia desktop launcher for Windows.
rem
rem This file can be double-clicked to start Audtheia. It finds a suitable Python
rem and hands control to the launcher, which starts the application and opens the
rem interface. Options are forwarded, for example:
rem
rem   start.bat               start Audtheia and offer to open the browser
rem   start.bat --tray        run with a system-tray icon when available
rem   start.bat --no-browser  start without opening or offering the browser
rem
setlocal
set "SCRIPT_DIR=%~dp0"

where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3 "%SCRIPT_DIR%bootstrap_start.py" %*
  exit /b %ERRORLEVEL%
)

where python >nul 2>nul
if %ERRORLEVEL%==0 (
  python "%SCRIPT_DIR%bootstrap_start.py" %*
  exit /b %ERRORLEVEL%
)

echo Python 3 was not found. Run the desktop setup first.
exit /b 1
