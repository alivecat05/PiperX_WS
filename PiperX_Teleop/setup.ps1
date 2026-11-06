param([switch]$Export)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Set-Location -LiteralPath $PSScriptRoot

function Run-Checked($Program, $Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Program (exit $LASTEXITCODE)" }
}

function Download-File($Url, $Destination) {
    # Never expose an interrupted download as a finished installer or archive.
    $partial = "$Destination.part"
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            if ($curl) {
                Run-Checked $curl.Source @('--fail', '--location', '--silent', '--show-error',
                    '--connect-timeout', '30', '--max-time', '900', '--output', $partial, $Url)
            } else {
                Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $partial -TimeoutSec 900
            }
            Move-Item -LiteralPath $partial -Destination $Destination -Force
            return
        } catch {
            if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
            if ($attempt -eq 3) { throw }
            Write-Warning "Download failed (attempt $attempt/3): $($_.Exception.Message). Retrying..."
            Start-Sleep -Seconds 3
        }
    }
}

function Get-MiniforgeInstaller($Installer) {
    # Pin the release and published SHA256 so mirror updates cannot mix versions.
    # https://github.com/conda-forge/miniforge/releases/tag/26.9.4-1
    $version = '26.9.4-1'
    $expected = '8cb96eed35effdd5b2d0860c5e85eed2224e72e71dd45a3fd6b1afe3d70585bb'
    $fileName = "Miniforge3-$version-Windows-x86_64.exe"
    if (Test-Path -LiteralPath $Installer) {
        if ((Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash -eq $expected) {
            Write-Host 'Using verified cached Miniforge installer.'
            return
        }
        Write-Warning 'Cached installer is incomplete or a different version; downloading again.'
    }
    $urls = @(
        "https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/$fileName",
        "https://github.com/conda-forge/miniforge/releases/download/$version/$fileName"
    )
    foreach ($url in $urls) {
        try {
            Write-Host "Downloading Miniforge: $url"
            Download-File $url $Installer
            if ((Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash -ne $expected) {
                throw 'Installer SHA256 checksum mismatch'
            }
            return
        } catch {
            Write-Warning "Cannot use ${url}: $($_.Exception.Message)"
        }
    }
    throw 'Unable to download a verified Miniforge installer. Check network/proxy access and rerun setup.cmd.'
}

function Install-Miniforge($Installer, $Destination) {
    # NSIS requires /D last and unquoted; wait for this GUI installer to finish.
    $arguments = "/S /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /NoShortcuts=1 /D=$Destination"
    $process = Start-Process -FilePath $Installer -ArgumentList $arguments -WindowStyle Hidden -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Miniforge installation failed (exit $($process.ExitCode)), destination: $Destination"
    }
}

try {
    New-Item -ItemType Directory -Force '.setup' | Out-Null
    $condaExe = Join-Path $PSScriptRoot '.conda\Scripts\conda.exe'
    if (-not (Test-Path $condaExe)) {
        Write-Host 'Installing local Conda (Miniforge)...'
        $installer = Join-Path $PSScriptRoot '.setup\Miniforge3-Windows-x86_64.exe'
        Get-MiniforgeInstaller $installer
        Install-Miniforge $installer (Join-Path $PSScriptRoot '.conda')
        if (-not (Test-Path $condaExe)) { throw 'Conda installation did not finish successfully' }
    }

    $envName = if ($Export) { 'export' } else { 'teleop' }
    $envFile = if ($Export) { 'environment-export.yml' } else { 'environment.yml' }
    $envPath = Join-Path $PSScriptRoot ".conda\envs\$envName"
    $operation = if (Test-Path "$envPath\python.exe") { 'update' } else { 'create' }
    $envArgs = @('env', $operation, '--prefix', $envPath, '--file', $envFile)
    if ($operation -eq 'create') { $envArgs += '--yes' }
    Run-Checked $condaExe $envArgs
    if (-not $Export) {
        Run-Checked $condaExe @('run', '--prefix', $envPath, '--no-capture-output', 'python', '-m', 'pip', 'install', './vendor/python-can-agx-cando-master', './vendor/pyAgxArm-master')
        if (-not (Test-Path '.setup\platform-tools\adb.exe')) {
            Download-File 'https://dl.google.com/android/repository/platform-tools-latest-windows.zip' '.setup\platform-tools.zip'
            Expand-Archive -LiteralPath '.setup\platform-tools.zip' -DestinationPath '.setup' -Force
        }
    }
    if (-not (Test-Path 'config.json')) { Copy-Item 'config.example.json' 'config.json' }
    Write-Host "Ready: $envName"
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
