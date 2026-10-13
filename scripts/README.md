# 운영체제별 초기 설정 스크립트

TalkTracker는 프로젝트 내부의 `.venv`, `.tools`, `models/` 폴더를 사용합니다. 모델은 Hugging Face 공용 캐시에 의존하지 않으며, 내려받은 모델 파일은 Git에 포함되지 않습니다.

Windows용 PowerShell 스크립트는 `scripts/windows`, macOS용 Bash 스크립트는 `scripts/macos`, Linux용 Bash 스크립트는 `scripts/linux`에 있습니다.

## Windows

먼저 Python 3.11을 설치한 뒤, 프로젝트 최상위 폴더의 PowerShell에서 아래 순서대로 실행합니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\Install-Dependencies.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\windows\Download-Models.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\windows\Prepare-FasterWhisper.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\windows\Run-Gui.ps1
```

CUDA GPU 실행이 준비된 Windows 환경에서는 모델 준비 스크립트가 CPU `int8` 대체 모델과 CUDA GPU `float16` 모델을 함께 만듭니다. 앱은 GPU를 우선 사용하고, 실행 중 GPU를 쓰지 못하면 Small·Turbo를 CPU 모델로 자동 전환합니다. CTranslate2 GPU 실행에는 NVIDIA 드라이버, CUDA 12.x 런타임, cuDNN 9가 필요합니다.

## macOS

프로젝트 최상위 폴더의 터미널에서 아래 순서대로 실행합니다. `Install-Dependencies.sh`는 설치된 Python 3.11+Tkinter를 재사용합니다. 없으면 프로젝트 내부 `.tools`에 사전 빌드 Python 3.11을 내려받으므로 Homebrew가 Python·Tkinter를 소스에서 컴파일하지 않습니다. 같은 위치에 Nemotron 화자 분리용 공식 네이티브 실행기도 설치합니다. Homebrew는 FFmpeg 설치에만 사용하며, 사전 빌드 bottle을 우선 요청합니다. Homebrew가 없다면 [먼저 설치](https://brew.sh)해 주세요.

```bash
bash scripts/macos/Install-Dependencies.sh
bash scripts/macos/Download-Models.sh
bash scripts/macos/Run-Gui.sh
```

macOS에서는 빠른 전사(Small)와 권장(Turbo) 프로필을 CPU로 사용할 수 있습니다. `Download-Models.sh`는 CTranslate2 형식으로 준비된 모델을 직접 받아 Intel Mac에서 로컬 INT8 변환 중 발생할 수 있는 충돌을 피하고, Nemotron의 공식 GGUF 화자 분리 모델도 함께 받습니다. 기존 원본 Whisper 파일만 이미 받아 둔 경우에는 `Prepare-FasterWhisper.sh`를 실행하면 같은 준비된 모델을 내려받습니다. 최고 정확도(Large-v3)는 CUDA GPU가 필요해 macOS에서는 실행할 수 없습니다.

화자 분리는 Python PyTorch 경로가 아니라 공식 NeMo-Speech.cpp 실행기를 씁니다. Intel Mac에서는 CPU, Apple Silicon에서는 Metal을 자동 선택하며 실행기는 프로젝트 내부 `.tools/nemo-speech`에만 설치됩니다.

## Linux

프로젝트 최상위 폴더의 터미널에서 아래 순서대로 실행합니다. 설치 스크립트는 apt, dnf, pacman을 감지해 필요한 시스템 패키지를 설치하려고 합니다. 배포판에 Python 3.11 또는 FFmpeg 패키지가 없으면 배포판의 별도 저장소를 설정한 뒤 다시 실행하세요.

```bash
bash scripts/linux/Install-Dependencies.sh
bash scripts/linux/Download-Models.sh
bash scripts/linux/Run-Gui.sh
```

Linux는 CTranslate2 형식으로 준비된 Whisper 모델을 직접 받아 로컬 PyTorch 변환을 하지 않습니다. 앱은 CUDA GPU가 준비되면 GPU `float16`으로 실행하고, 실행 중 GPU를 쓰지 못하면 Small·Turbo를 CPU `int8`으로 자동 전환합니다. NVIDIA GPU를 감지하면 설치 스크립트가 CUDA 12용 cuBLAS와 cuDNN 9 런타임을 프로젝트 `.venv`에 설치하고 실행 시 필요한 라이브러리 경로도 설정합니다. NVIDIA GPU가 보이는데도 PyPI가 CPU 전용 CTranslate2를 설치한 Linux ARM64 환경에서는 설치 스크립트가 CTranslate2 CUDA 바인딩을 프로젝트 내부 `.tools`에 한 번 빌드합니다. 이 단계에는 CUDA Toolkit(`nvcc`)과 cuDNN 개발 파일(`cudnn.h`, `libcudnn.so`)이 있어야 하며, 처음만 몇 분 걸릴 수 있습니다.

화자 분리는 Nemotron GGUF와 공식 NeMo-Speech.cpp 실행기를 사용합니다. 설치 스크립트는 실행기를 프로젝트 내부 `.tools/nemo-speech`에 설치하며, NVIDIA 드라이버가 보이면 CUDA를, 그렇지 않으면 CPU를 선택합니다.

기존에 설치를 마친 뒤 업데이트했다면 `bash scripts/linux/Install-Dependencies.sh`를 한 번 더 실행해 네이티브 화자 분리 실행기를 준비하세요.

실제 Linux PC에서 설치·CPU 전사·CUDA 전사를 차례대로 확인하려면 [Linux 검증 가이드](../docs/linux-test-guide.md)를 사용하세요. 배포판에서 Python 3.11 또는 FFmpeg를 바로 제공하지 않는 경우의 확인 사항도 포함합니다.

### Linux 웹 서버 실행

Linux에서는 GUI 대신 브라우저 요청을 받는 서버 모드를 실행할 수 있습니다. `Install-Dependencies.sh`를 실행하면 웹 서버에 필요한 Python 패키지도 함께 설치됩니다.

```bash
bash scripts/linux/Run-Web.sh
```

기본 주소는 `127.0.0.1:8080`이므로 SSH 터널을 통해 다른 PC의 브라우저에서 안전하게 접속할 수 있습니다.

```bash
ssh -L 8080:127.0.0.1:8080 user@remote-host
```

신뢰할 수 있는 내부 네트워크에서 직접 접속하게 하려면 `--host 0.0.0.0`을 지정합니다. 이 경우 토큰은 필요하지 않습니다. 외부 인터넷에 직접 공개하지 마세요.

```bash
bash scripts/linux/Run-Web.sh --host 0.0.0.0
```

내부망 사용자도 제한해야 한다면 `TALKTRACKER_WEB_TOKEN`을 설정합니다. 이 경우에만 웹 화면에 접근 토큰 입력란이 표시됩니다.

```bash
TALKTRACKER_WEB_TOKEN='길고_임의적인_비밀값' \
  bash scripts/linux/Run-Web.sh --host 0.0.0.0
```

서버는 업로드 파일을 `outputs/jobs/` 아래에 작업별로 보관하고, GPU·메모리 충돌을 피하기 위해 분석을 한 번에 하나씩 처리합니다. 업로드 크기·보관 기간·포트는 `TALKTRACKER_WEB_MAX_UPLOAD_MB`, `TALKTRACKER_WEB_RETENTION_DAYS`, `TALKTRACKER_WEB_PORT` 환경 변수로 조정할 수 있습니다.

## 전사 품질 선택

모델 다운로드와 준비 스크립트에는 같은 프로필을 지정합니다. 지정하지 않으면 권장 Turbo를 사용합니다.

| 프로필 | Windows PowerShell 예시 | macOS Bash 예시 | Linux Bash 예시 |
| --- | --- | --- | --- |
| 빠른 전사 | `powershell -ExecutionPolicy Bypass -File .\scripts\windows\Download-Models.ps1 -WhisperProfile small` | `bash scripts/macos/Download-Models.sh --whisper-profile small` | `bash scripts/linux/Download-Models.sh --whisper-profile small` |
| 권장 | `powershell -ExecutionPolicy Bypass -File .\scripts\windows\Download-Models.ps1 -WhisperProfile turbo` | `bash scripts/macos/Download-Models.sh --whisper-profile turbo` | `bash scripts/linux/Download-Models.sh --whisper-profile turbo` |
| 최고 정확도 (CUDA GPU 필요) | `powershell -ExecutionPolicy Bypass -File .\scripts\windows\Download-Models.ps1 -WhisperProfile large-v3` | 다운로드는 가능하지만 macOS에서는 실행할 수 없음 | `bash scripts/linux/Download-Models.sh --whisper-profile large-v3` |

macOS와 Linux는 다운로드한 모델이 곧바로 실행 가능한 CTranslate2 형식입니다. 기존 원본 Whisper 파일만 받아 둔 경우에는 각 운영체제의 `Prepare-FasterWhisper` 스크립트에 `--whisper-profile`을 지정해 준비된 모델만 받을 수 있습니다. Windows에서는 `-PrepareForLocalUse`를 다운로드 명령 끝에 붙여 변환까지 이어서 실행할 수 있습니다.

macOS·Linux의 실행용 Whisper 폴더는 모델마다 필요한 파일 구성이 다릅니다. Turbo·Large-v3는 `config.json`, `preprocessor_config.json`, `model.bin`, `tokenizer.json`, `vocabulary.json`을 사용하고, Small의 공식 CTranslate2 모델은 `config.json`, `model.bin`, `tokenizer.json`, `vocabulary.txt`를 사용합니다. `Download-Models.sh`와 `Prepare-FasterWhisper.sh`는 선택한 모델에 맞는 파일만 검사합니다. 일부 파일만 빠진 경우에는 이미 받은 큰 `model.bin`을 다시 받지 않고 누락 파일만 강제로 내려받은 뒤 다시 검사합니다.

예를 들어 Linux Large-v3 폴더를 복구하면서 화자 분리 모델 다운로드를 건너뛰려면 다음을 실행합니다.

```bash
bash scripts/linux/Download-Models.sh --whisper-profile large-v3 --skip-diarization
```

완료 뒤 GUI의 **상태 새로고침**을 누릅니다. 앱도 선택한 모델의 필수 파일을 확인하므로 `model.bin`만 있는 불완전한 폴더를 준비 완료로 표시하지 않습니다.

Windows에서 모델 준비 대상을 직접 지정하려면 `Prepare-FasterWhisper.ps1`에 `-Runtime cpu`, `-Runtime cuda`, 또는 `-Runtime both`를 붙일 수 있습니다. Linux는 하나의 준비된 CTranslate2 모델을 CPU `int8`과 CUDA GPU `float16`에서 모두 사용하며, 앱이 실행 시 실제 CTranslate2 CUDA 가능 여부를 확인해 선택합니다.

Linux Large-v3가 전사 중 실패하면 CUDA 설치를 반복하기 전에 `outputs/diagnostics/whisper-large-v3-*.log`를 확인하세요. 이 로그에는 실제 실행 환경, 모델 구성 파일 상태, CTranslate2·faster-whisper 버전, 원본 예외가 기록됩니다.
