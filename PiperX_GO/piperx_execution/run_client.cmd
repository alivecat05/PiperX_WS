@echo off
REM Run in the PC's existing PiperX_Teleop Python environment.
REM Set SERVER_URL or edit the default below to the reachable inference server.
if not defined SERVER_URL set "SERVER_URL=ws://127.0.0.1:8000"
python "%~dp0client.py" ^
  --server "%SERVER_URL%" ^
  --config "%~dp0config.pc.json" ^
  --rtc ^
  --rtc-lookahead-steps 3 ^
  --blend-steps 5 ^
  %*
