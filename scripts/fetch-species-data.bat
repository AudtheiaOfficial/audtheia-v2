@echo off
rem Audtheia per-species reference fetch for Windows.
rem
rem This is a thin wrapper. It prefers the environment that setup created (which
rem has the network library installed) and hands control to the fetcher. Options
rem are forwarded, for example:
rem
rem   fetch-species-data.bat                          fetch every station's target species
rem   fetch-species-data.bat --species "Panthera leo" fetch one species by name
rem   fetch-species-data.bat --refresh                update species already on file
rem
setlocal
set "SCRIPT_DIR=%~dp0"
set "VENV_PY=%SCRIPT_DIR%..\.venv\Scripts\python.exe"

if exist "%VENV_PY%" (
  "%VENV_PY%" "%SCRIPT_DIR%bootstrap_fetch_species.py" %*
  exit /b %ERRORLEVEL%
)

where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3 "%SCRIPT_DIR%bootstrap_fetch_species.py" %*
  exit /b %ERRORLEVEL%
)

where python >nul 2>nul
if %ERRORLEVEL%==0 (
  python "%SCRIPT_DIR%bootstrap_fetch_species.py" %*
  exit /b %ERRORLEVEL%
)

echo Python 3 was not found. Run the desktop setup first.
exit /b 1
