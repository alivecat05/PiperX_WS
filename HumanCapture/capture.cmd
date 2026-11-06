@echo off
setlocal
set "PYTHONUTF8=1"
pushd "%~dp0"
if exist ".venv\Scripts\python.exe" (
    if exist "..\PiperX_Teleop\.conda\envs\teleop\Library\bin" set "PATH=%~dp0..\PiperX_Teleop\.conda\envs\teleop\Library\bin;%PATH%"
    ".venv\Scripts\python.exe" -m human_capture.app %*
) else if exist "..\PiperX_Teleop\.conda\envs\teleop\python.exe" (
    call "..\PiperX_Teleop\.conda\Scripts\activate.bat" "%~dp0..\PiperX_Teleop\.conda\envs\teleop"
    call :run_existing %*
) else (
    echo Python environment missing. Run setup.cmd first.
    popd
    exit /b 1
)
set "capture_exit=%errorlevel%"
popd
exit /b %capture_exit%

:run_existing
if errorlevel 1 exit /b 1
"..\PiperX_Teleop\.conda\envs\teleop\python.exe" -m human_capture.app %*
exit /b %errorlevel%
