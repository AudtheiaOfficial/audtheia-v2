@echo off
rem Audtheia field-station provisioning for Windows.
rem
rem This is a thin wrapper. It finds a suitable Python and hands control to the
rem provisioning orchestrator, which connects to a Pi over SSH and stands the
rem station up. Options are forwarded, for example:
rem
rem   connect-pi.bat --station-id <id> --host <pi-address> --user <pi-user>
rem   connect-pi.bat --station-id <id> --dry-run   preview without contacting a Pi
rem
setlocal
set "SCRIPT_DIR=%~dp0"

where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3 "%SCRIPT_DIR%bootstrap_setup_pi.py" %*
  exit /b %ERRORLEVEL%
)

where python >nul 2>nul
if %ERRORLEVEL%==0 (
  python "%SCRIPT_DIR%bootstrap_setup_pi.py" %*
  exit /b %ERRORLEVEL%
)

echo Python 3 was not found. Install Python 3.11 or newer and try again.
exit /b 1
