# TalkTracker

로컬 음성 파일에서 **누가 언제 말했는지**와 **무슨 말을 했는지**를 한 화면에서 정리하는 Windows·macOS·Linux용 데스크톱 프로그램입니다. 음성 파일과 분석 결과는 PC 밖으로 전송하지 않습니다.

> **운영체제 지원 범위:** Windows, macOS, Linux를 지원합니다. 설치·모델 다운로드·실행 스크립트는 각각 `scripts/windows`, `scripts/macos`, `scripts/linux`에 있습니다. macOS에서는 CPU 전사 프로필(빠른 전사·권장)을 사용합니다. Windows와 Linux는 CTranslate2 CUDA 실행 환경이 준비되어 있으면 GPU 전사를 자동 선택합니다. 최고 정확도 Large-v3는 CUDA GPU가 필요합니다.

> **화자 분리:** macOS와 Linux는 Nemotron의 공식 GGUF 모델과 프로젝트 내부의 네이티브 실행기를 사용합니다. Intel Mac은 CPU로, Apple Silicon은 Metal로, Linux는 사용 가능한 CPU 또는 CUDA로 화자 분리를 실행하므로 PyTorch·Transformers의 호환성에 영향을 받지 않습니다.

## 현재 가능한 기능

- M4A, MP3, WAV, FLAC, AAC, OGG 음성 파일 선택
- Nemotron 기반 화자 구간 분리
- 전사 목적에 따라 Whisper Small 또는 Large-v3 Turbo 선택
- 호환 CUDA GPU를 감지하면 Small·Turbo도 GPU `float16`으로 자동 전사하고, 사용할 수 없으면 CPU `int8`으로 자동 전환
- 호환 CUDA GPU가 있을 때 Whisper Large-v3 최고 정확도 모드 선택
- 화자별 타임라인을 전사 중에도 먼저 표시
- 경과 시간과 진행 상태 표시
- 분석 중단 시, 실행 중인 네이티브 화자 분리 작업을 즉시 종료하고 그 시점까지 완성된 화자 구간과 전사 결과 유지
- 읽기 쉬운 Markdown 형식으로 결과 저장

## 현재 작업 상태

현재 버전에서는 화면 시제품만이 아니라, 실제 로컬 분석 흐름까지 연결되어 있습니다.

- 파일 선택부터 분석 시작·중단·결과 저장까지 데스크톱 화면에서 실행
- Nemotron으로 화자별 구간을 먼저 찾고, 분석 도중에도 화자 타임라인을 표시
- Whisper Small(빠른 확인), Large-v3 Turbo(권장), Large-v3(GPU 필요)를 선택해 전사
- Windows와 Linux에서 CUDA GPU가 준비되면 GPU `float16`을 우선 사용하고, GPU 로드 실패 시 Small·Turbo는 CPU `int8`으로 자동 전환
- 전사 중 경과 시간과 상태 문구를 표시해 긴 파일의 진행 여부를 확인
- 분석을 중단하면 이미 완료된 화자 분리와 전사 결과만으로 화면 및 저장 결과 구성
- 모델 가중치는 프로젝트의 `models/`에 보관하며, Git에는 포함하지 않음

## Windows에서 실행하기

처음 한 번만 필요한 준비입니다.

이 프로젝트가 제공하는 실행 스크립트는 모두 `scripts/windows`에 있습니다. PowerShell에서 프로젝트 최상위 폴더를 연 뒤 아래 순서대로 실행합니다.

1. PowerShell에서 `scripts\\windows\\Install-Dependencies.ps1`을 실행합니다.
2. `scripts\\windows\\Download-Models.ps1`을 실행해 기본 전사 모델을 받습니다.
3. `scripts\\windows\\Prepare-FasterWhisper.ps1`을 실행해 로컬 실행용 모델을 준비합니다.
4. 이후에는 `scripts\\windows\\Run-Gui.ps1`을 실행하면 됩니다.

모델은 프로젝트의 `models` 폴더에 저장됩니다. 화면에서 전사 품질을 고르면 그 선택에 맞는 모델만 준비 메뉴에서 내려받아 사용할 수 있습니다.

> **Windows·Linux CUDA GPU 사용:** 모델 준비 스크립트는 CTranslate2가 CUDA GPU를 사용할 수 있으면 CPU `int8` 대체 모델과 GPU `float16` 모델을 함께 준비합니다. NVIDIA 드라이버가 필요하며, Linux 설치 스크립트는 GPU를 감지하면 필요한 CUDA 12 cuBLAS·cuDNN 9 런타임을 프로젝트 `.venv`에 자동으로 설치하고 실행 환경을 설정합니다. Linux ARM64에서 PyPI가 CPU 전용 CTranslate2를 제공하면 설치 스크립트가 프로젝트 내부에서 CUDA용 바인딩을 한 번 빌드합니다.

## macOS에서 실행하기

처음 한 번만 필요한 준비입니다. Python 3.11, Tkinter, FFmpeg가 필요합니다. 설치 스크립트는 이미 설치된 Python 3.11+Tkinter를 우선 재사용하고, 없으면 프로젝트의 `.tools` 폴더에 사전 빌드 Python 3.11을 내려받아 만듭니다. 또한 화자 분리에 쓸 공식 네이티브 실행기를 같은 `.tools` 폴더에 설치합니다. Python과 Tkinter 때문에 Homebrew가 소스 컴파일을 시작하지 않습니다. FFmpeg 설치에만 Homebrew가 필요하므로, 없다면 [먼저 설치](https://brew.sh)해 주세요.

터미널에서 프로젝트 최상위 폴더를 연 뒤 아래 순서대로 실행합니다.

1. `bash scripts/macos/Install-Dependencies.sh`
2. `bash scripts/macos/Download-Models.sh` — macOS에서 바로 실행할 수 있는 Whisper 형식과 Nemotron GGUF 화자 분리 모델을 받습니다.
3. 이후에는 `bash scripts/macos/Run-Gui.sh`

모델은 프로젝트의 `models` 폴더에만 저장됩니다. macOS는 Intel Mac에서 발생할 수 있는 로컬 INT8 변환 오류를 피하기 위해, CTranslate2 형식으로 이미 준비된 Whisper 모델을 직접 받습니다. 전사는 CPU `int8` 설정을 사용합니다. 화자 분리는 Nemotron GGUF를 공식 네이티브 실행기로 처리하며, Intel Mac에서도 실제 화자 구간을 분리합니다.

> **Intel Mac 사용 전 안내:** Intel Mac에서는 Whisper 전사와 Nemotron 화자 분리가 모두 CPU로 실행됩니다. 특히 긴 녹음은 화자 분리 단계부터 시간이 많이 걸려 실사용 효율이 크게 떨어질 수 있습니다. **빠른 전사(Small)**를 선택하면 Whisper 전사 시간은 줄지만 화자 분리 시간은 줄지 않습니다. 긴 녹음의 실제 분석은 CUDA GPU를 사용할 수 있는 Linux 또는 Windows PC를 권장합니다.

### 내 Mac 확인 방법

터미널에서 아래 명령을 실행합니다.

```bash
system_profiler SPHardwareDataType | grep -E 'Chip|Processor Name|Total Number of Cores|Memory'
```

- `Processor Name: Intel ...`이 보이면 Intel Mac입니다. 위의 CPU 전용 안내가 적용됩니다.
- `Chip: Apple M1`, `Chip: Apple M2`처럼 표시되면 Apple Silicon입니다. Nemotron 화자 분리기는 Metal을 자동 사용하지만, Whisper 전사는 현재 CPU 설정으로 동작합니다.

`uname -m`도 보조 확인에 쓸 수 있습니다. 일반적으로 Intel Mac은 `x86_64`, Apple Silicon은 `arm64`를 표시하지만, Apple Silicon에서 Rosetta 터미널을 열면 `x86_64`가 표시될 수 있으므로 `system_profiler` 결과를 우선하세요.

## Linux에서 실행하기

Ubuntu·Debian·Fedora·Arch 계열의 패키지 관리자를 감지해 Python 3.11, Tkinter, FFmpeg 설치를 안내하거나 시도합니다. 배포판에 따라 Python 3.11이나 FFmpeg 패키지가 별도 저장소에 있을 수 있으며, 이 경우 스크립트의 안내에 따라 해당 저장소를 먼저 설정해 주세요.

터미널에서 프로젝트 최상위 폴더를 연 뒤 아래 순서대로 실행합니다.

1. `bash scripts/linux/Install-Dependencies.sh`
2. `bash scripts/linux/Download-Models.sh` — 바로 실행 가능한 Whisper 및 Nemotron 모델을 받습니다.
3. 이후에는 `bash scripts/linux/Run-Gui.sh`

Linux는 로컬 PyTorch 변환 없이 CTranslate2 형식으로 준비된 Whisper 모델을 직접 받습니다. NVIDIA GPU가 감지되면 설치 스크립트가 CUDA 12 cuBLAS·cuDNN 9 런타임을 프로젝트 환경에 설치해 GPU `float16` 실행을 준비합니다. Small·Turbo는 GPU 로드·전사 실패 시 CPU `int8`으로 자동 전환되지만, Large-v3는 GPU 전용입니다. 화자 분리는 Nemotron GGUF와 프로젝트 내부의 네이티브 실행기를 사용합니다.

> **한국어 전사 기본값:** TalkTracker는 한국어 녹음을 기본 대상으로 하므로 Whisper에 한국어를 명시합니다. 이는 일부 Linux CUDA 환경에서 Large-v3의 자동 언어 감지 중 발생하는 오류를 피합니다. 다른 한 언어의 녹음만 처리하려면 GUI를 실행하기 전에 `TALKTRACKER_LANGUAGE=en`처럼 [Whisper 언어 코드](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py)를 지정할 수 있습니다. 자동 감지가 꼭 필요하면 `TALKTRACKER_LANGUAGE=`로 빈 값으로 설정할 수 있지만, Large-v3 CUDA에서는 권장하지 않습니다.

> **Large-v3 CUDA 호환 모드:** 일부 Linux CUDA 환경에서 Large-v3의 단어 단위 타임스탬프 정렬이 CTranslate2 오류로 끝날 수 있어, Large-v3는 전사 구간 단위 시간으로 화자를 연결합니다. 대화 내용과 화자 구분은 그대로 만들며, Small·Turbo는 단어 단위 시간을 계속 사용합니다.

실제 Linux PC에서 설치와 CPU·CUDA 동작을 검증할 때는 [Linux 검증 가이드](docs/linux-test-guide.md)를 따르세요. 배포판별 Python 3.11·FFmpeg 저장소 차이와 기대 결과를 함께 정리했습니다.

## 처리 순서

1. 음성 파일을 분석용 형식으로 정리합니다.
2. Nemotron이 화자별 발화 구간을 찾습니다.
3. Whisper가 음성을 글로 옮깁니다.
4. 시간 정보를 기준으로 전사문과 화자를 연결합니다.

현재는 화자 분리 후 전사를 이어서 수행하는 순차 처리 방식입니다. CPU 환경에서는 긴 파일의 전사에 시간이 걸릴 수 있으나, 타임라인·경과 시간·중단 버튼으로 현재 상태를 확인할 수 있습니다.

## 상세 문서

- [개발 계획서](docs/development-plan.md)
- [현재 구현 상태와 검증 메모](docs/implementation-status.md)
- [Linux 검증 가이드](docs/linux-test-guide.md)
- [스크립트 안내](scripts/README.md)
