[CmdletBinding()]
param(
    [ValidateSet('small', 'turbo', 'large-v3')]
    [string]$WhisperProfile = 'turbo',
    [ValidateSet('auto', 'cpu', 'cuda', 'both')]
    [string]$Runtime = 'auto'
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Converter = Join-Path $ProjectRoot '.venv\Scripts\ct2-transformers-converter.exe'
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$WhisperModels = @{
    'small' = @{ SourceFolder = 'whisper-small'; CpuDestinationFolder = 'whisper-small-ct2'; CudaDestinationFolder = 'whisper-small-ct2-cuda-fp16'; DisplayName = 'Whisper Small' }
    'turbo' = @{ SourceFolder = 'whisper-large-v3-turbo'; CpuDestinationFolder = 'whisper-large-v3-turbo-ct2'; CudaDestinationFolder = 'whisper-large-v3-turbo-ct2-cuda-fp16'; DisplayName = 'Whisper Large-v3 Turbo' }
    # This was already the GPU FP16 output location before automatic selection was added.
    'large-v3' = @{ SourceFolder = 'whisper-large-v3'; CpuDestinationFolder = $null; CudaDestinationFolder = 'whisper-large-v3-ct2'; DisplayName = 'Whisper Large-v3' }
}

$SelectedWhisper = $WhisperModels[$WhisperProfile]
$Source = Join-Path $ProjectRoot "models\$($SelectedWhisper.SourceFolder)"

if (-not (Test-Path -LiteralPath (Join-Path $Source 'model.safetensors'))) {
    throw "$($SelectedWhisper.DisplayName) 원본 모델을 찾지 못했습니다: $Source`n먼저 Download-Models.ps1 -WhisperProfile $WhisperProfile 을 실행해 주세요."
}

if (-not (Test-Path -LiteralPath $Converter)) {
    throw "Whisper 변환 도구를 찾지 못했습니다: $Converter`n먼저 Install-Dependencies.ps1을 실행해 주세요."
}

function Test-CudaWhisperRuntime {
    if (-not (Test-Path -LiteralPath $Python)) {
        return $false
    }

    $Probe = @'
import sys
try:
    import ctranslate2
    has_cuda = ctranslate2.get_cuda_device_count() > 0
    supports_fp16 = "float16" in ctranslate2.get_supported_compute_types("cuda")
    sys.exit(0 if has_cuda and supports_fp16 else 1)
except Exception:
    sys.exit(1)
'@
    & $Python -c $Probe
    return $LASTEXITCODE -eq 0
}

function Prepare-WhisperRuntime {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('cpu', 'cuda')][string]$Target
    )

    if ($Target -eq 'cpu') {
        $DestinationFolder = $SelectedWhisper.CpuDestinationFolder
        $Quantization = 'int8'
        $RuntimeName = 'CPU INT8'
    } else {
        $DestinationFolder = $SelectedWhisper.CudaDestinationFolder
        $Quantization = 'float16'
        $RuntimeName = 'CUDA GPU FP16'
    }

    if (-not $DestinationFolder) {
        throw "$($SelectedWhisper.DisplayName)은 CPU 실행을 지원하지 않습니다. 호환 CUDA GPU가 필요합니다."
    }

    $Destination = Join-Path $ProjectRoot "models\$DestinationFolder"
    if (Test-Path -LiteralPath (Join-Path $Destination 'model.bin')) {
        Write-Host "Already ready ($RuntimeName): $Destination" -ForegroundColor Green
        return
    }

    if (Test-Path -LiteralPath $Destination) {
        $existing = Get-ChildItem -LiteralPath $Destination -Force
        if ($existing) {
            throw "불완전한 모델 변환 폴더가 있습니다: $Destination`n폴더 내용을 확인한 뒤 다시 실행해 주세요."
        }
        Remove-Item -LiteralPath $Destination -Force
    }

    Write-Host "Preparing $($SelectedWhisper.DisplayName) for $RuntimeName inference..."
    Write-Host "Source: $Source"
    Write-Host "Destination: $Destination"

    & $Converter `
        --model $Source `
        --output_dir $Destination `
        --copy_files tokenizer.json preprocessor_config.json `
        --quantization $Quantization

    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $Destination 'model.bin'))) {
        throw "$RuntimeName Whisper 모델 준비에 실패했습니다."
    }
}

$CudaAvailable = Test-CudaWhisperRuntime
switch ($Runtime) {
    'cpu' {
        $Targets = @('cpu')
    }
    'cuda' {
        if (-not $CudaAvailable) {
            throw 'CUDA GPU FP16 실행을 사용할 수 없습니다. NVIDIA 드라이버, CUDA 12.x, cuDNN 8을 확인해 주세요.'
        }
        $Targets = @('cuda')
    }
    'both' {
        if (-not $CudaAvailable) {
            throw 'CUDA GPU FP16 실행을 사용할 수 없습니다. NVIDIA 드라이버, CUDA 12.x, cuDNN 8을 확인해 주세요.'
        }
        $Targets = if ($SelectedWhisper.CpuDestinationFolder) { @('cpu', 'cuda') } else { @('cuda') }
    }
    default {
        if ($CudaAvailable) {
            $Targets = if ($SelectedWhisper.CpuDestinationFolder) { @('cpu', 'cuda') } else { @('cuda') }
        } elseif ($SelectedWhisper.CpuDestinationFolder) {
            $Targets = @('cpu')
        } else {
            throw 'Whisper Large-v3는 호환 CUDA GPU가 필요합니다. NVIDIA 드라이버, CUDA 12.x, cuDNN 8을 확인해 주세요.'
        }
    }
}

if ($CudaAvailable) {
    if ($SelectedWhisper.CpuDestinationFolder) {
        Write-Host 'CUDA GPU를 감지했습니다. CPU 대체 모델과 CUDA GPU 모델을 함께 준비합니다.' -ForegroundColor Cyan
    } else {
        Write-Host 'CUDA GPU를 감지했습니다. CUDA GPU 모델을 준비합니다.' -ForegroundColor Cyan
    }
} else {
    Write-Host 'CUDA GPU 실행 환경을 찾지 못했습니다. CPU INT8 모델을 준비합니다.' -ForegroundColor Cyan
}

foreach ($Target in $Targets) {
    Prepare-WhisperRuntime -Target $Target
}

Write-Host ''
Write-Host 'Done. TalkTracker will automatically select the best prepared Whisper runtime.' -ForegroundColor Green
