$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
function Run-Checked($Program, $Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Program (exit $LASTEXITCODE)" }
}
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        $sharedPython = Join-Path $PSScriptRoot '..\PiperX_Teleop\.conda\envs\teleop\python.exe'
        if (Test-Path -LiteralPath $sharedPython) {
            # Keep Conda DLLs available when creating a separate virtual environment.
            $sharedEnv = Split-Path -Parent $sharedPython
            $env:PATH = "$sharedEnv\Library\bin;$sharedEnv\Scripts;$sharedEnv;" + $env:PATH
            Run-Checked $sharedPython @('-m', 'venv', '.venv')
        } elseif (Get-Command py -ErrorAction SilentlyContinue) {
            Run-Checked 'py' @('-3.12', '-m', 'venv', '.venv')
        } else {
            throw 'Install Python 3.12 (with Tk), or set up the sibling PiperX_Teleop environment first.'
        }
    }
    Run-Checked '.\.venv\Scripts\python.exe' @('-m', 'pip', 'install', '-r', 'requirements.txt')
    if (-not (Test-Path -LiteralPath 'config.json')) {
        Copy-Item -LiteralPath 'config.example.json' -Destination 'config.json'
    }
    Write-Host 'Ready. Edit config.json camera serials, then run capture.cmd.'
    exit 0
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
