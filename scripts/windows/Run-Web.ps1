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
$Arguments = @('-m', 'talktracker.server')
# Do not declare PowerShell parameters here.  The server's --host, --port,
# and other long options are forwarded unchanged, like the macOS/Linux runner.
$ServerArguments = @()
for ($index = 0; $index -lt $args.Count; $index += 1) {
    $argument = [string]$args[$index]
    if ($argument -in @('-BindHost', '-Host', '-Port')) {
        if (($index + 1) -ge $args.Count) {
            throw "$argument 옵션에는 값이 필요합니다."
        }
        $option = if ($argument -eq '-Port') { '--port' } else { '--host' }
        $index += 1
        $ServerArguments += @($option, [string]$args[$index])
    } else {
        $ServerArguments += $argument
    }
}
$Arguments += $ServerArguments

& $Python @Arguments
exit $LASTEXITCODE
