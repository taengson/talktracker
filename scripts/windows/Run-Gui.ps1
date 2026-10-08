[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$SourceRoot = Join-Path $ProjectRoot 'src'

if (-not (Test-Path -LiteralPath $Python)) {
    throw "프로젝트 Python 환경을 찾지 못했습니다: $Python`n먼저 Install-Dependencies.ps1을 실행해 주세요."
}

if (-not (Test-Path -LiteralPath $SourceRoot)) {
    throw "앱 소스 폴더를 찾지 못했습니다: $SourceRoot"
}

$env:PYTHONPATH = if ($env:PYTHONPATH) { "$SourceRoot;$env:PYTHONPATH" } else { $SourceRoot }
& $Python -m talktracker
exit $LASTEXITCODE
