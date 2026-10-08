[CmdletBinding()]
param(
    [ValidateSet('small', 'turbo', 'large-v3')]
    [string]$WhisperProfile = 'turbo'
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Converter = Join-Path $ProjectRoot '.venv\Scripts\ct2-transformers-converter.exe'
$WhisperModels = @{
    'small' = @{ SourceFolder = 'whisper-small'; DestinationFolder = 'whisper-small-ct2'; DisplayName = 'Whisper Small'; Quantization = 'int8' }
    'turbo' = @{ SourceFolder = 'whisper-large-v3-turbo'; DestinationFolder = 'whisper-large-v3-turbo-ct2'; DisplayName = 'Whisper Large-v3 Turbo'; Quantization = 'int8' }
    'large-v3' = @{ SourceFolder = 'whisper-large-v3'; DestinationFolder = 'whisper-large-v3-ct2'; DisplayName = 'Whisper Large-v3'; Quantization = 'float16' }
}

$SelectedWhisper = $WhisperModels[$WhisperProfile]
$Source = Join-Path $ProjectRoot "models\$($SelectedWhisper.SourceFolder)"
$Destination = Join-Path $ProjectRoot "models\$($SelectedWhisper.DestinationFolder)"

if (-not (Test-Path -LiteralPath (Join-Path $Source 'model.safetensors'))) {
    throw "$($SelectedWhisper.DisplayName) 원본 모델을 찾지 못했습니다: $Source`n먼저 Download-Models.ps1 -WhisperProfile $WhisperProfile 을 실행해 주세요."
}

if (-not (Test-Path -LiteralPath $Converter)) {
    throw "CPU 최적화 변환 도구를 찾지 못했습니다: $Converter`n먼저 Install-Dependencies.ps1을 실행해 주세요."
}

if (Test-Path -LiteralPath (Join-Path $Destination 'model.bin')) {
    Write-Host "Already ready: $Destination" -ForegroundColor Green
    return
}

if (Test-Path -LiteralPath $Destination) {
    $existing = Get-ChildItem -LiteralPath $Destination -Force
    if ($existing) {
        throw "불완전한 모델 변환 폴더가 있습니다: $Destination`n폴더 내용을 확인한 뒤 다시 실행해 주세요."
    }
    Remove-Item -LiteralPath $Destination -Force
}

Write-Host "Preparing $($SelectedWhisper.DisplayName) for local inference..."
Write-Host "Source: $Source"
Write-Host "Destination: $Destination"

& $Converter `
    --model $Source `
    --output_dir $Destination `
    --copy_files tokenizer.json preprocessor_config.json `
    --quantization $SelectedWhisper.Quantization

if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $Destination 'model.bin'))) {
    throw 'Whisper 모델 준비에 실패했습니다.'
}

Write-Host ''
Write-Host "Done. TalkTracker will use the prepared model from: $Destination" -ForegroundColor Green
