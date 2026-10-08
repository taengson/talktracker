[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Requirements = Join-Path $ProjectRoot 'requirements.txt'
$VenvRoot = Join-Path $ProjectRoot '.venv'
$Python = Join-Path $VenvRoot 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $Python)) {
    $PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
    if (-not $PythonLauncher) {
        throw "Python 3.11을 찾지 못했습니다. Python 3.11을 설치한 뒤 이 스크립트를 다시 실행해 주세요."
    }

    Write-Host "Creating this project's Python environment..."
    & $PythonLauncher.Source -3.11 -m venv $VenvRoot
    if ($LASTEXITCODE -ne 0) {
        throw '프로젝트 Python 환경을 만들지 못했습니다.'
    }
}

& $Python -m pip --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Repairing pip in this project environment...'
    & $Python -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) {
        throw '프로젝트 Python 환경에 pip을 준비하지 못했습니다.'
    }
}

Write-Host 'Installing TalkTracker packages...'
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw 'pip 업데이트에 실패했습니다.'
}

& $Python -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) {
    throw 'TalkTracker 패키지 설치에 실패했습니다.'
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning 'FFmpeg를 찾지 못했습니다. 음성 파일 처리를 위해 FFmpeg를 설치하고 PATH에 추가해 주세요.'
}

Write-Host ''
Write-Host 'Done. The project environment is ready.' -ForegroundColor Green
Write-Host 'Next: run scripts\windows\Download-Models.ps1 to download the local models.'
