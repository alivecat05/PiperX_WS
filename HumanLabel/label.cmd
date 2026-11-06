@echo off
setlocal
set "PYTHONUTF8=1"
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.cmd first.
    popd
    exit /b 1
)
if exist "..\PiperX_Teleop\.conda\envs\teleop\Library\bin" set "PATH=%~dp0..\PiperX_Teleop\.conda\envs\teleop\Library\bin;%PATH%"
".venv\Scripts\python.exe" label.py %*
set "label_exit=%errorlevel%"
popd
exit /b %label_exit%
