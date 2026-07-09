@echo off
REM Run an Audtheia station on this desktop (Windows), with no field hardware.
REM Any arguments pass straight through to the orchestrator.
python "%~dp0bootstrap_run_desktop.py" %*
