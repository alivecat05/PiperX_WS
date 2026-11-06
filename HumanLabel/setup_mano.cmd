@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_mano.ps1"
exit /b %errorlevel%
