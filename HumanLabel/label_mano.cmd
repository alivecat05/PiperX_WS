@echo off
setlocal
set "PYTHONUTF8=1"
pushd "%~dp0"
if not exist ".mano-venv\Scripts\python.exe" (
    echo Run setup_mano.cmd first.
    popd
    exit /b 1
)
if exist "..\PiperX_Teleop\.conda\envs\teleop\Library\bin" set "PATH=%~dp0..\PiperX_Teleop\.conda\envs\teleop\Library\bin;%PATH%"
".mano-venv\Scripts\python.exe" label.py --backend mano --output result/mano %*
set "mano_exit=%errorlevel%"
popd
exit /b %mano_exit%
