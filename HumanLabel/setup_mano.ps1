$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
function Run-Checked($Program, $Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Program (exit $LASTEXITCODE)" }
}
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' }
    if (-not (Test-Path -LiteralPath '.mano-venv\Scripts\python.exe')) {
        $sharedPython = Join-Path $PSScriptRoot '..\PiperX_Teleop\.conda\envs\teleop\python.exe'
        if (Test-Path -LiteralPath $sharedPython) {
            Run-Checked 'uv' @('venv', '--python', $sharedPython, '.mano-venv')
        } else {
            Run-Checked 'uv' @('venv', '--python', '3.12', '.mano-venv')
        }
    }
    Run-Checked 'uv' @('pip', 'install', '--link-mode', 'copy', '--python', '.mano-venv\Scripts\python.exe',
        'torch==2.5.1+cpu', 'torchvision==0.20.1+cpu', '--index-url', 'https://download.pytorch.org/whl/cpu')
    Run-Checked 'uv' @('pip', 'install', '--link-mode', 'copy', '--python', '.mano-venv\Scripts\python.exe', '-r', 'requirements-mano.txt')
    Run-Checked '.\.mano-venv\Scripts\python.exe' @('mano_assets.py')
    Write-Host 'Ready. Run label_mano.cmd; MANO results use result/mano.'
    exit 0
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
