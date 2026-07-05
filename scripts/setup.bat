@echo off
rem Audtheia desktop setup for Windows.
rem
rem This is a thin wrapper. It finds a suitable Python and hands control to the
rem cross-platform bootstrap, which does all of the real work. Any options you
rem pass are forwarded, for example:
rem
rem   setup.bat               install everything and fetch the essential models
rem   setup.bat --full        also fetch the field-station models to stage
rem   setup.bat --skip-models set up the environment and database only
rem
setlocal
set "SCRIPT_DIR=%~dp0"

where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3 "%SCRIPT_DIR%bootstrap_setup.py" %*
  exit /b %ERRORLEVEL%
)

where python >nul 2>nul
if %ERRORLEVEL%==0 (
  python "%SCRIPT_DIR%bootstrap_setup.py" %*
  exit /b %ERRORLEVEL%
)

echo Python 3 was not found. Install Python 3.11 or newer and run setup again.
exit /b 1
