@echo off
REM Use the same CPU-only client; DP defaults to port 8001.
if not defined SERVER_URL set "SERVER_URL=ws://127.0.0.1:8001"
call "%~dp0run_client.cmd" %*
