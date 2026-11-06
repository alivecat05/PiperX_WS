$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        $sharedPython = Join-Path $PSScriptRoot '..\PiperX_Teleop\.conda\envs\teleop\python.exe'
        if (Test-Path -LiteralPath $sharedPython) {
            $sharedEnv = Split-Path -Parent $sharedPython
            $env:PATH = "$sharedEnv\Library\bin;$sharedEnv\Scripts;$sharedEnv;" + $env:PATH
            & $sharedPython -m venv .venv
        } else {
            & py -3.12 -m venv .venv
        }
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 environment creation failed.' }
    }
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        & uv pip install --python .venv\Scripts\python.exe -r requirements.txt
    } else {
        & .\.venv\Scripts\python.exe -m ensurepip
        if ($LASTEXITCODE -ne 0) { throw 'pip setup failed.' }
        & .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    }
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    Write-Host 'Ready. Run label.cmd; results are saved under HumanLabel/result.'
    exit 0
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
