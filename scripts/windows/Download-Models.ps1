[CmdletBinding()]
param(
    [ValidateSet('small', 'turbo', 'large-v3')]
    [string]$WhisperProfile = 'turbo',
    [switch]$SkipDiarization,
    [switch]$PrepareForLocalUse
)

$ErrorActionPreference = 'Stop'

$WhisperModels = @{
    'small' = @{ Repository = 'openai/whisper-small'; FolderName = 'whisper-small'; DisplayName = 'Whisper Small (빠른 전사)' }
    'turbo' = @{ Repository = 'openai/whisper-large-v3-turbo'; FolderName = 'whisper-large-v3-turbo'; DisplayName = 'Whisper Large-v3 Turbo (권장)' }
    'large-v3' = @{ Repository = 'openai/whisper-large-v3'; FolderName = 'whisper-large-v3'; DisplayName = 'Whisper Large-v3 (GPU 필요)' }
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\\..')).Path
$ModelsRoot = Join-Path $ProjectRoot 'models'
$HuggingFace = Join-Path $ProjectRoot '.venv\\Scripts\\hf.exe'
$PrepareScript = Join-Path $PSScriptRoot 'Prepare-FasterWhisper.ps1'

if (-not (Test-Path -LiteralPath $HuggingFace)) {
    throw "Hugging Face 도구를 찾지 못했습니다: $HuggingFace`n먼저 이 프로젝트의 Python 환경과 의존성을 준비해 주세요."
}

New-Item -ItemType Directory -Force -Path $ModelsRoot | Out-Null

function Download-Model {
    param(
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][string]$FolderName
    )

    $Destination = Join-Path $ModelsRoot $FolderName
    Write-Host "Downloading $Repository"
    Write-Host "Destination: $Destination"
    & $HuggingFace download $Repository `
        --local-dir $Destination `
        --exclude '*.gguf' `
        --exclude '*.nemo' `
        --exclude '*.bin' `
        --exclude '*.onnx' `
        --exclude '*.onnx_data' `
        --exclude '*.mp4' `
        --exclude '*.gif' `
        --exclude '*.png'
    if ($LASTEXITCODE -ne 0) {
        throw "다운로드에 실패했습니다: $Repository"
    }
}

$SelectedWhisper = $WhisperModels[$WhisperProfile]
Write-Host "Selected transcription model: $($SelectedWhisper.DisplayName)" -ForegroundColor Cyan
Download-Model -Repository $SelectedWhisper.Repository -FolderName $SelectedWhisper.FolderName

if (-not $SkipDiarization) {
    Download-Model -Repository 'nvidia/Nemotron-3-Diarization' -FolderName 'nemotron-3-diarization'
}

if ($PrepareForLocalUse) {
    if (-not (Test-Path -LiteralPath $PrepareScript)) {
        throw "모델 준비 스크립트를 찾지 못했습니다: $PrepareScript"
    }
    Write-Host ''
    Write-Host 'Preparing the downloaded transcription model for TalkTracker...' -ForegroundColor Cyan
    & $PrepareScript -WhisperProfile $WhisperProfile
}

Write-Host ""
Write-Host "Done. TalkTracker will use models from: $ModelsRoot" -ForegroundColor Green
