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

CUDA GPU 실행이 준비된 Windows 환경에서는 모델 준비 스크립트가 CPU `int8` 대체 모델과 CUDA GPU `float16` 모델을 함께 만듭니다. 앱은 GPU를 우선 사용하고, 실행 중 GPU를 쓰지 못하면 Small·Turbo를 CPU 모델로 자동 전환합니다. CTranslate2 GPU 실행에는 NVIDIA 드라이버, CUDA 12.x, cuDNN 8이 필요합니다.

## macOS

프로젝트 최상위 폴더의 터미널에서 아래 순서대로 실행합니다. `Install-Dependencies.sh`는 설치된 Python 3.11+Tkinter를 재사용합니다. 없으면 프로젝트 내부 `.tools`에 사전 빌드 Python 3.11을 내려받으므로 Homebrew가 Python·Tkinter를 소스에서 컴파일하지 않습니다. Homebrew는 FFmpeg 설치에만 사용하며, 사전 빌드 bottle을 우선 요청합니다. Homebrew가 없다면 [먼저 설치](https://brew.sh)해 주세요.

```bash
bash scripts/macos/Install-Dependencies.sh
bash scripts/macos/Download-Models.sh
bash scripts/macos/Prepare-FasterWhisper.sh
bash scripts/macos/Run-Gui.sh
```

macOS에서는 빠른 전사(Small)와 권장(Turbo) 프로필을 CPU로 사용할 수 있습니다. 최고 정확도(Large-v3)는 CUDA GPU가 필요해 macOS에서는 실행할 수 없습니다.

## Linux

프로젝트 최상위 폴더의 터미널에서 아래 순서대로 실행합니다. 설치 스크립트는 apt, dnf, pacman을 감지해 필요한 시스템 패키지를 설치하려고 합니다. 배포판에 Python 3.11 또는 FFmpeg 패키지가 없으면 배포판의 별도 저장소를 설정한 뒤 다시 실행하세요.

```bash
bash scripts/linux/Install-Dependencies.sh
bash scripts/linux/Download-Models.sh
bash scripts/linux/Prepare-FasterWhisper.sh
bash scripts/linux/Run-Gui.sh
```

CUDA GPU 실행이 준비된 Linux 환경에서는 모델 준비 스크립트가 CPU `int8` 대체 모델과 CUDA GPU `float16` 모델을 함께 만듭니다. 앱은 GPU를 우선 사용하고, 실행 중 GPU를 쓰지 못하면 Small·Turbo를 CPU 모델로 자동 전환합니다. NVIDIA 드라이버, CUDA 12.x, cuDNN 8이 필요합니다.

실제 Linux PC에서 설치·CPU 전사·CUDA 전사를 차례대로 확인하려면 [Linux 검증 가이드](../docs/linux-test-guide.md)를 사용하세요. 배포판에서 Python 3.11 또는 FFmpeg를 바로 제공하지 않는 경우의 확인 사항도 포함합니다.

## 전사 품질 선택

모델 다운로드와 준비 스크립트에는 같은 프로필을 지정합니다. 지정하지 않으면 권장 Turbo를 사용합니다.

| 프로필 | Windows PowerShell 예시 | macOS Bash 예시 | Linux Bash 예시 |
| --- | --- | --- | --- |
| 빠른 전사 | `powershell -ExecutionPolicy Bypass -File .\scripts\windows\Download-Models.ps1 -WhisperProfile small` | `bash scripts/macos/Download-Models.sh --whisper-profile small` | `bash scripts/linux/Download-Models.sh --whisper-profile small` |
| 권장 | `powershell -ExecutionPolicy Bypass -File .\scripts\windows\Download-Models.ps1 -WhisperProfile turbo` | `bash scripts/macos/Download-Models.sh --whisper-profile turbo` | `bash scripts/linux/Download-Models.sh --whisper-profile turbo` |
| 최고 정확도 (CUDA GPU 필요) | `powershell -ExecutionPolicy Bypass -File .\scripts\windows\Download-Models.ps1 -WhisperProfile large-v3` | 다운로드는 가능하지만 macOS에서는 실행할 수 없음 | `bash scripts/linux/Download-Models.sh --whisper-profile large-v3` |

파일 다운로드 후 실행 형식으로 바로 준비하려면 Windows에서는 `-PrepareForLocalUse`, macOS에서는 `--prepare-for-local-use`를 다운로드 명령 끝에 붙일 수 있습니다.

Windows에서 모델 준비 대상을 직접 지정하려면 `Prepare-FasterWhisper.ps1`에 `-Runtime cpu`, `-Runtime cuda`, 또는 `-Runtime both`를 붙일 수 있습니다. Linux에서는 `Prepare-FasterWhisper.sh`에 `--runtime cpu`, `--runtime cuda`, 또는 `--runtime both`를 붙일 수 있습니다. 두 운영체제 모두 기본값 `auto`가 실제 CTranslate2 CUDA 실행 가능 여부를 확인해 적절한 모델을 준비합니다.
