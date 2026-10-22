# TalkTracker

로컬 음성 파일에서 **누가 언제 말했는지**와 **무슨 말을 했는지**를 정리하는 Windows·macOS·Linux용 프로그램입니다. 데스크톱 화면과 다른 PC의 브라우저 요청을 처리하는 웹 서버 모드를 세 운영체제에서 제공합니다. 음성 파일과 분석 결과는 TalkTracker가 실행 중인 PC 안에서 처리합니다.

> **운영체제 지원 범위:** Windows, macOS, Linux를 지원합니다. 설치·모델 다운로드·실행 스크립트는 각각 `scripts/windows`, `scripts/macos`, `scripts/linux`에 있습니다. macOS에서는 CPU 전사 프로필(빠른 전사·권장)을 사용합니다. Windows와 Linux는 CTranslate2 CUDA 실행 환경이 준비되어 있으면 GPU 전사를 자동 선택합니다. 최고 정확도 Large-v3는 CUDA GPU가 필요합니다.

> **화자 분리:** macOS와 Linux는 Nemotron의 공식 GGUF 모델과 프로젝트 내부의 네이티브 실행기를 사용합니다. Intel Mac은 CPU로, Apple Silicon은 Metal로, Linux는 사용 가능한 CPU 또는 CUDA로 화자 분리를 실행하므로 PyTorch·Transformers의 호환성에 영향을 받지 않습니다.

## 현재 가능한 기능

- M4A, MP3, WAV, FLAC, AAC, OGG 음성 파일 선택
- Nemotron 기반 화자 구간 분리
- 전사 목적에 따라 Whisper Small 또는 Large-v3 Turbo 선택
- 호환 CUDA GPU를 감지하면 Small·Turbo도 GPU `float16`으로 자동 전사하고, 사용할 수 없으면 CPU `int8`으로 자동 전환
- 호환 CUDA GPU가 있을 때 Whisper Large-v3 최고 정확도 모드 선택
- 화자별 타임라인을 전사 중에도 먼저 표시
- 데스크톱과 웹에서 분석의 경과 시간·진행 상태 표시, Whisper 전사 중에는 처리된 녹음 시각 표시
- 분석 중단 시, 실행 중인 네이티브 화자 분리 작업을 즉시 종료하고 그 시점까지 완성된 화자 구간과 전사 결과 유지
- 웹 서버 모드에서 브라우저 업로드, 단일 작업 대기열, 실제 분석 시간·진행 상태·화자 타임라인, 중단, Markdown 결과 다운로드
- 읽기 쉬운 Markdown 형식으로 결과 저장
- OpenAI 호환 로컬 LLM 서버 연결: 모델 목록 조회, 모델 선택·직접 입력, 짧은 생성 요청으로 연결 확인
- 로컬 AI로 다듬은 대화·요약 만들기, 진행·경과 시간 표시, 결과별 별도 Markdown 저장

## 현재 작업 상태

현재 버전에서는 화면 시제품만이 아니라, 실제 로컬 분석 흐름까지 연결되어 있습니다.

- 파일 선택부터 분석 시작·중단·결과 저장까지 데스크톱 화면에서 실행
- Nemotron으로 화자별 구간을 먼저 찾고, 분석 도중에도 화자 타임라인을 표시
- Whisper Small(빠른 확인), Large-v3 Turbo(권장), Large-v3(GPU 필요)를 선택해 전사
- Windows와 Linux에서 CUDA GPU가 준비되면 GPU `float16`을 우선 사용하고, GPU 로드 실패 시 Small·Turbo는 CPU `int8`으로 자동 전환
- 데스크톱과 웹 모두 전사 중 경과 시간과 상태 문구, 처리된 녹음 시각을 표시해 긴 파일의 진행 여부를 확인. 웹의 경과 시간은 업로드·대기열이 아닌 실제 화자 분리·전사 실행 시간만 계산
- 분석을 중단하면 이미 완료된 화자 분리와 전사 결과만으로 화면 및 저장 결과 구성
- 모델 가중치는 프로젝트의 `models/`에 보관하며, Git에는 포함하지 않음
- 상단의 **로컬 AI 연결**에서 같은 PC 또는 신뢰하는 사설망의 OpenAI 호환 LLM 서버를 연결할 수 있음
- 최근 성공한 서버 주소 목록과 마지막 모델 ID를 저장하고, 목록의 주소는 입력칸에서 다시 선택 가능. API 키는 현재 앱 실행 중 메모리에만 유지

## 로컬 AI 연결과 AI 정리 — 1~2단계 완료

로컬 LLM 연결 기반과 전사문의 AI 정리 기능을 구현했습니다.

1. 사용자가 OpenAI 호환 로컬 LLM 서버를 별도로 실행합니다.
2. 데스크톱 앱 상단의 **로컬 AI 연결 필요**를 눌러 서버 주소를 입력합니다.
3. **모델 불러오기**로 `/v1/models` 목록을 읽거나, 모델 ID를 직접 입력합니다.
4. **연결 확인**은 선택한 모델에 아주 짧은 생성 요청을 보내 실제 사용 가능 여부를 확인합니다.

서버 주소는 같은 PC의 `localhost`뿐 아니라 사용자가 관리하는 사설망 LLM 서버의 IP 주소 또는 호스트 이름도 입력할 수 있으며, 주소 끝의 `/v1`은 자동으로 맞춥니다. 주소에는 API 키·계정 정보·쿼리를 넣을 수 없습니다. 연결 확인에 성공한 주소만 최근 연결 목록에 남아 다음에 입력칸에서 선택할 수 있습니다. API 키가 필요하면 연결 확인 시에만 입력할 수 있고 디스크·결과 문서·오류 메시지에는 저장하지 않습니다. 향후 AI 정리 기능을 실행하면 전사문이 선택한 서버로 전달되므로, 신뢰하는 서버만 선택하세요. 로컬 LLM 모델 가중치는 TalkTracker의 `models/` 폴더나 모델 다운로드 스크립트로 관리하지 않으며, 사용자가 선택한 LLM 서버가 관리합니다.

웹 서버 모드에서도 OpenAI 호환 LLM 서버 주소를 입력할 수 있는 연결 패널을 제공합니다. 기본인 `127.0.0.1` 또는 `localhost`에서는 바로 사용할 수 있습니다. `--host 0.0.0.0`으로 내부망에 웹 서버를 열 때는 패널을 기본적으로 숨기며, 신뢰하는 사내 폐쇄망에서만 `--allow-local-ai-over-network` 옵션으로 명시적으로 열 수 있습니다. 이 옵션을 사용하면 브라우저에서 입력한 API 키와 AI 요청이 네트워크를 통해 서버로 전달되므로, 인터넷에 노출된 서버에서는 사용하지 마세요.

분석이 끝나면 원본 전사는 자동으로 보존되어 바로 볼 수 있고, 저장할 때도 항상 별도 Markdown 문서로 만들어집니다. 웹의 **결과 만들기** 영역에서는 원본 전사를 펼쳐 보고 대화의 목적·배경 안내를 입력할 수 있습니다. **다듬은 대화**와 **요약**은 하나 또는 둘 다 선택해 현재 연결한 로컬 LLM으로 생성합니다. AI 결과는 원본을 덮어쓰지 않으며, 다듬은 대화를 만든 뒤 요약을 추가해도 앞선 다듬은 대화가 유지됩니다. 결과 화면에서는 원본 전사·다듬은 대화·요약을 전환해 보고 각각 별도 문서로 내려받을 수 있습니다. 이미 만든 문서는 로컬 AI 연결이 해제되었거나 비활성화된 뒤에도 내려받을 수 있습니다.

 AI 처리 중에는 현재 단계와 경과 시간을 표시하고 중단할 수 있습니다. 전사문은 작업 중에만 임시 Markdown 파일로 만들고, Agent Chat 최신 첨부 문서 정책과 동일하게 처리합니다. 파일 크기는 5MiB까지 허용하며 240KiB 이하면 전체를 한 번에 전달합니다. 더 길면 16KiB 문서 단위로 나눈 뒤, 사용자 요청 관련 단위와 앞·뒤 문맥을 골라 최대 240KiB만 한 번에 전달합니다. 관련성을 찾지 못하면 앞 70%·뒤 30% 발췌본을 전달합니다. 모델에는 `[첨부 파일]`과 `[사용자 요청]` 형식의 일반 대화만 보내며, 내부 구간 ID·JSON을 요구하지 않습니다. 모델이 만든 자유 형식 Markdown은 **다듬은 대화** 또는 **요약**의 별도 문서로 보존하고, 입력 선택 정보도 결과 화면에서 확인할 수 있습니다. **결과 만들기**에서 매 요청마다 추론 강도를 `기본(서버 설정)`, `medium`, `low`, `none` 중 선택합니다. 기본은 필드를 보내지 않아 서버의 모델 설정을 그대로 사용하고, 나머지는 선택값만 한 번 명시합니다. 자동 추론 강도 변경·응답 길이 재시도·모델 호출 분할은 하지 않으며, 선택값을 지원하지 않는 서버에는 기본값으로 몰래 되돌아가지 않고 안내를 표시합니다. 임시 전사 파일은 성공·실패·중단 후 모두 정리하며 원본 전사는 어느 단계에서도 바꾸지 않습니다. 대화의 목적·배경과 화자 안내는 현재 요청의 정확도를 높이는 용도로만 로컬 LLM 서버에 전달되며, 배경·역할 안내는 결과 문서에 자동으로 저장하지 않습니다.

### 전사문 파일 첨부 방식

일반 길이의 전사문(예: 18KiB)은 나누지 않고 전체를 하나의 임시 Markdown 첨부 문서로 전달합니다. 원본 전사는 그대로 두고, 작업 전용 임시 폴더에 파일을 만든 뒤 아래 한 개의 사용자 메시지를 구성합니다.

```text
[첨부 파일: <원본이름>-전사문.md]
<전사문 전체 또는 선택된 문서 단위>
[첨부 파일 끝]

[사용자 요청]
<다듬기 또는 요약 지시>
```

임시 파일 생성·문서 선택·로컬 서버 전송 준비는 작은 전사문에서 거의 즉시 끝납니다. 화면에 **다듬은 대화 문서로 만들고 있습니다** 또는 **요약 문서로 만들고 있습니다**가 표시된 뒤의 대기 시간은 보통 로컬 LLM이 전사문을 읽고 결과 Markdown을 생성하는 시간입니다. 작업을 중단하거나 성공·실패하면 임시 파일과 임시 폴더는 모두 삭제됩니다.

## 다음 구현 예정

다음 단계에서는 분석이 끝난 대화 기록을 더 쉽게 정리할 수 있도록 아래 기능을 추가합니다. 이 항목들은 현재 구현 전입니다.

- **OpenRouter 무료 모델 연결:** 로컬 LLM을 쓰기 어려운 경우 사용자가 선택적으로 OpenRouter의 무료 모델을 연결할 수 있게 합니다. 이 경로는 사용자가 API 키와 모델을 직접 선택할 때만 사용하며, 분석에 필요한 전사 텍스트만 전송한다는 점을 화면에서 분명히 알립니다.
- **화자 이름 지정과 저장:** `화자 1`, `화자 2`처럼 분리된 화자에 실제 이름 또는 역할을 지정하고, 화면·대화 기록·Markdown 저장 결과에 일관되게 반영합니다. 이름 지정은 분석 결과와 함께 다시 열어볼 수 있도록 저장합니다.

## Windows에서 실행하기

처음 한 번만 필요한 준비입니다.

이 프로젝트가 제공하는 실행 스크립트는 모두 `scripts/windows`에 있습니다. PowerShell에서 프로젝트 최상위 폴더를 연 뒤 아래 순서대로 실행합니다.

1. PowerShell에서 `scripts\\windows\\Install-Dependencies.ps1`을 실행합니다.
2. `scripts\\windows\\Download-Models.ps1`을 실행해 기본 전사 모델을 받습니다.
3. `scripts\\windows\\Prepare-FasterWhisper.ps1`을 실행해 로컬 실행용 모델을 준비합니다.
4. 이후에는 `scripts\\windows\\Run-Gui.ps1`을 실행하면 됩니다.

모델은 프로젝트의 `models` 폴더에 저장됩니다. 화면에서 전사 품질을 고르면 그 선택에 맞는 모델만 준비 메뉴에서 내려받아 사용할 수 있습니다.

브라우저로 사용할 때는 `powershell -ExecutionPolicy Bypass -File .\scripts\windows\Run-Web.ps1`을 실행합니다. [웹 서버로 실행하기](#웹-서버로-실행하기)의 접속·포트 안내를 따르세요.

> **Windows·Linux CUDA GPU 사용:** Windows는 CTranslate2가 CUDA GPU를 사용할 수 있으면 Small·Turbo용 CPU `int8` 대체 모델과 GPU `float16` 모델을 함께 준비합니다. Linux는 준비된 CTranslate2 모델 하나를 GPU `float16`과 CPU `int8`에 함께 사용합니다. NVIDIA 드라이버가 필요하며, Linux 설치 스크립트는 GPU를 감지하면 필요한 CUDA 12 cuBLAS·cuDNN 9 런타임을 프로젝트 `.venv`에 자동으로 설치하고 실행 환경을 설정합니다. Linux ARM64에서 PyPI가 CPU 전용 CTranslate2를 제공하면 설치 스크립트가 프로젝트 내부에서 CUDA용 바인딩을 한 번 빌드합니다.

## macOS에서 실행하기

처음 한 번만 필요한 준비입니다. Python 3.11, Tkinter, FFmpeg가 필요합니다. 설치 스크립트는 이미 설치된 Python 3.11+Tkinter를 우선 재사용하고, 없으면 프로젝트의 `.tools` 폴더에 사전 빌드 Python 3.11을 내려받아 만듭니다. 또한 화자 분리에 쓸 공식 네이티브 실행기를 같은 `.tools` 폴더에 설치합니다. Python과 Tkinter 때문에 Homebrew가 소스 컴파일을 시작하지 않습니다. FFmpeg 설치에만 Homebrew가 필요하므로, 없다면 [먼저 설치](https://brew.sh)해 주세요.

터미널에서 프로젝트 최상위 폴더를 연 뒤 아래 순서대로 실행합니다.

1. `bash scripts/macos/Install-Dependencies.sh`
2. `bash scripts/macos/Download-Models.sh` — macOS에서 바로 실행할 수 있는 Whisper 형식과 Nemotron GGUF 화자 분리 모델을 받습니다.
3. 이후에는 `bash scripts/macos/Run-Gui.sh`

모델은 프로젝트의 `models` 폴더에만 저장됩니다. macOS는 Intel Mac에서 발생할 수 있는 로컬 INT8 변환 오류를 피하기 위해, CTranslate2 형식으로 이미 준비된 Whisper 모델을 직접 받습니다. 전사는 CPU `int8` 설정을 사용합니다. 화자 분리는 Nemotron GGUF를 공식 네이티브 실행기로 처리하며, Intel Mac에서도 실제 화자 구간을 분리합니다.

브라우저로 사용할 때는 `bash scripts/macos/Run-Web.sh`을 실행합니다. [웹 서버로 실행하기](#웹-서버로-실행하기)의 접속·포트 안내를 따르세요.

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

Whisper 실행용 모델은 `model.bin` 하나만으로 준비가 끝난 것이 아닙니다. Turbo·Large-v3는 `config.json`, `preprocessor_config.json`, `tokenizer.json`, `vocabulary.json`을 함께 사용하며, Small의 공식 CTranslate2 모델은 `preprocessor_config.json` 대신 `vocabulary.txt`를 사용합니다. 앱과 macOS·Linux 스크립트는 선택한 모델에 맞는 파일 구성을 검사하고, 일부만 빠진 경우에는 모델 전체를 다시 받지 않고 누락된 파일만 복구합니다. 업데이트 뒤 모델 관리 화면에서 다운로드 필요 상태가 보이면 **모델 다운로드 시작**을 누르거나 아래 명령을 실행하세요.

```bash
bash scripts/linux/Download-Models.sh --whisper-profile large-v3 --skip-diarization
```

> **한국어 전사 기본값:** TalkTracker는 한국어 녹음을 기본 대상으로 하므로 Whisper에 한국어를 명시합니다. 이는 일부 Linux CUDA 환경에서 Large-v3의 자동 언어 감지 중 발생하는 오류를 피합니다. 다른 한 언어의 녹음만 처리하려면 GUI를 실행하기 전에 `TALKTRACKER_LANGUAGE=en`처럼 [Whisper 언어 코드](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py)를 지정할 수 있습니다. 자동 감지가 꼭 필요하면 `TALKTRACKER_LANGUAGE=`로 빈 값으로 설정할 수 있지만, Large-v3 CUDA에서는 권장하지 않습니다.

> **Large-v3 CUDA 호환 모드:** 일부 Linux CUDA 환경에서 Large-v3의 단어 단위 타임스탬프 정렬이 CTranslate2 오류로 끝날 수 있어, Large-v3는 전사 구간 단위 시간으로 화자를 연결합니다. 대화 내용과 화자 구분은 그대로 만들며, Small·Turbo는 단어 단위 시간을 계속 사용합니다. 전사 중에는 `처리한 시각 / 전체 길이`가 표시되므로 실제 진행 여부를 확인할 수 있습니다.

> **Large-v3 오류 확인:** 누락된 모델 파일은 분석 전에 바로 안내됩니다. CUDA GPU가 이미 시작된 뒤 발생한 전사 오류는 설치 오류로 단정하지 않으며, 앱은 실행 환경, 모델 구성, CTranslate2·faster-whisper 버전, 원본 예외를 `outputs/diagnostics/whisper-large-v3-*.log`에 기록합니다.

### SSH로 컴퓨터 사이에 파일 주고받기

두 컴퓨터가 SSH로 연결되어 있다면 Samba를 따로 설정하지 않아도 `scp`로 녹음 파일과 결과를 전송할 수 있습니다. 아래 예시의 `user@remote-host`는 **SSH 원격 컴퓨터의 사용자와 호스트 이름**으로 바꾸세요.

처음 한 번, SSH 원격 컴퓨터에 녹음 파일을 둘 폴더를 만듭니다. 이 명령은 현재 컴퓨터의 터미널에서 실행합니다.

```bash
ssh user@remote-host 'mkdir -p ~/talktracker-recordings'
```

#### 보내기: 현재 컴퓨터 → SSH 원격 컴퓨터

현재 컴퓨터에 있는 녹음 파일을 SSH 원격 컴퓨터로 보냅니다. 이 명령은 **보내는 쪽인 현재 컴퓨터**에서 실행합니다.

```bash
scp "/경로/녹음파일.m4a" user@remote-host:~/talktracker-recordings/
```

#### 가져오기: SSH 원격 컴퓨터 → 현재 컴퓨터

SSH 원격 컴퓨터에 있는 녹음 파일 또는 결과 파일을 현재 컴퓨터의 `Downloads` 폴더로 가져옵니다. 이 명령은 **받는 쪽인 현재 컴퓨터**에서 실행합니다.

```bash
scp user@remote-host:~/talktracker-recordings/파일명.m4a ~/Downloads/
```

대용량 파일을 여러 번 옮기거나 중단 후 이어받아야 하면 같은 경로에 `rsync -avP`를 사용할 수 있습니다.

실제 Linux PC에서 설치와 CPU·CUDA 동작을 검증할 때는 [Linux 검증 가이드](docs/linux-test-guide.md)를 따르세요. 배포판별 Python 3.11·FFmpeg 저장소 차이와 기대 결과를 함께 정리했습니다.

### 웹 서버로 실행하기

모델을 준비한 컴퓨터에서 웹 서버를 시작하면, 다른 PC는 브라우저만으로 녹음 파일을 올리고 결과를 받을 수 있습니다. `Run-Web`은 요청을 다른 서버로 전달하는 프로그램이 아니라, **실행한 PC에서 직접 분석하는 서버**입니다. 따라서 처리 성능과 모델이 있는 PC에서 실행하고, 다른 PC는 그 서버 주소를 브라우저로 열어 사용하세요. 기본 주소는 서버 PC 내부에서만 접속할 수 있는 `127.0.0.1:8080`입니다.

| 서버 운영체제 | 실행 명령 |
| --- | --- |
| Windows | `powershell -ExecutionPolicy Bypass -File .\scripts\windows\Run-Web.ps1` |
| macOS | `bash scripts/macos/Run-Web.sh` |
| Linux | `bash scripts/linux/Run-Web.sh` |

포트 번호를 8081로 바꾸는 예시는 다음과 같습니다.

```bash
# macOS
bash scripts/macos/Run-Web.sh --port 8081
# Linux
bash scripts/linux/Run-Web.sh --port 8081
```

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File .\scripts\windows\Run-Web.ps1 --port 8081
```

다른 PC에서는 서버에 SSH 터널을 연 뒤 브라우저에서 `http://localhost:8080`으로 접속하세요.

```bash
ssh -L 8080:127.0.0.1:8080 user@remote-host
```

서버 포트를 바꿨다면 SSH 터널 명령의 마지막 포트도 맞춥니다. 예를 들어 서버가 8081 포트라면 `ssh -L 8081:127.0.0.1:8081 user@remote-host`를 실행한 뒤 `http://localhost:8081`로 접속합니다.

신뢰할 수 있는 내부 네트워크에서만 사용할 경우에는 다음처럼 직접 열 수 있습니다. 이 경우 접근 토큰 입력은 필요 없습니다. 인터넷에 직접 공개하는 용도로는 사용하지 마세요.

```bash
# macOS
bash scripts/macos/Run-Web.sh --host 0.0.0.0
# Linux
bash scripts/linux/Run-Web.sh --host 0.0.0.0
```

```powershell
# Windows: 내부망에서 10000 포트로 직접 접속
powershell -ExecutionPolicy Bypass -File .\scripts\windows\Run-Web.ps1 --host 0.0.0.0 --port 10000
```

내부망의 다른 사용자를 제한해야 한다면 선택적으로 `TALKTRACKER_WEB_TOKEN`을 설정할 수 있습니다. 이때만 웹 화면에 **접근 토큰** 입력란이 표시됩니다.

```bash
# macOS
TALKTRACKER_WEB_TOKEN='길고_임의적인_비밀값' \
  bash scripts/macos/Run-Web.sh --host 0.0.0.0
# Linux
TALKTRACKER_WEB_TOKEN='길고_임의적인_비밀값' \
  bash scripts/linux/Run-Web.sh --host 0.0.0.0
```

```powershell
$env:TALKTRACKER_WEB_TOKEN = '길고_임의적인_비밀값'
powershell -ExecutionPolicy Bypass -File .\scripts\windows\Run-Web.ps1 --host 0.0.0.0
```

### 사내 폐쇄망에서 웹 AI 정리 사용하기

내부망에 직접 연 웹 화면에서도 로컬 LLM 연결, 다듬은 대화, 요약 기능을 사용하려면 서버를 시작할 때 `--allow-local-ai-over-network`를 추가합니다. 이 옵션은 네트워크로 API 키와 전사문을 보내도 되는 **신뢰하는 폐쇄망**임을 운영자가 명시하는 용도입니다. 접근 토큰은 필요할 때 별도로 함께 설정할 수 있습니다.

```bash
# Linux 예시
bash scripts/linux/Run-Web.sh --host 0.0.0.0 --port 12200 --allow-local-ai-over-network

# macOS 예시
bash scripts/macos/Run-Web.sh --host 0.0.0.0 --allow-local-ai-over-network
```

```powershell
# Windows 예시
powershell -ExecutionPolicy Bypass -File .\scripts\windows\Run-Web.ps1 --host 0.0.0.0 --allow-local-ai-over-network
```

웹 서버는 한 번에 하나의 분석 작업만 실행해 GPU·메모리 충돌을 피합니다. 업로드와 결과는 `outputs/jobs/`에 작업별로 보관하며, 서버를 시작할 때 기본 7일보다 오래된 작업 폴더를 정리합니다. `TALKTRACKER_WEB_MAX_UPLOAD_MB`(기본 2048MB), `TALKTRACKER_WEB_RETENTION_DAYS`(기본 7일), `TALKTRACKER_WEB_PORT`(기본 8080) 환경 변수로 조정할 수 있습니다.

웹 서버의 **로컬 AI 연결** 패널은 TalkTracker 웹 서버를 기본 주소인 `127.0.0.1` 또는 `localhost`로 실행하면 표시됩니다. 내부망에 공개한 `--host 0.0.0.0` 서버에서는 `--allow-local-ai-over-network`를 함께 준 경우에만 표시됩니다. 이때 웹 화면에는 **로컬 연결 허용 모드**와 API 키·AI 요청이 네트워크를 통해 전달된다는 안내가 표시됩니다.

`reasoning_effort`는 모든 OpenAI 호환 로컬 서버의 공통 기능이 아닙니다. TalkTracker는 **결과 만들기**에서 사용자가 고른 `medium`, `low`, `none`만 요청에 넣습니다. `기본(서버 설정)`은 이 필드를 보내지 않습니다. 선택한 필드를 서버가 거절하면 기본 설정으로 자동 재시도하지 않고, 지원하는 값 또는 기본을 고르도록 안내합니다. 어느 경우에도 응답 길이에 따라 추론 강도를 바꾸거나 낮은 강도로 다시 시도하지 않습니다.

웹 화면에서도 데스크톱과 같은 흐름으로 분석 뒤 원본 전사를 자동 보존하고, **다듬은 대화**와 **요약**을 선택해 로컬 LLM으로 만들 수 있습니다. 실행 중인 단계·경과 시간·중단 버튼을 표시하며, 만든 결과는 원본·다듬은 대화·요약별로 전환하고 각각 Markdown 문서로 받을 수 있습니다.

웹 화면에서는 이 PC에서 실제로 실행 가능한 전사 모델만 선택할 수 있습니다. 준비되지 않은 모델은 **준비 필요**로 비활성화되며, 직접 요청을 보내더라도 서버가 파일을 작업 대기열에 넣기 전에 거절합니다. 하나라도 실행 가능한 전사 프로필이 있으면 일반 업로드 화면을 표시합니다. 반대로 전사·화자 분리에 필요한 모델이 하나도 준비되지 않았다면, 업로드 화면 대신 **분석 요청 수신 불가** 상태 페이지를 표시하며 파일을 받지 않습니다. 원하는 품질이 비활성화되어 있으면 서버가 표시하는 현재 운영체제용 준비 명령을 실행한 뒤 브라우저를 새로고침하세요. Windows는 모델 다운로드 뒤 로컬 실행용 변환도 필요하므로 안내 명령에 `-PrepareForLocalUse`가 함께 포함됩니다.

화자 분리가 끝나면 웹 화면의 현재 작업 영역에 화자별 발화 구간을 녹음 전체 시간축에 맞춘 색상 레인으로 표시합니다. 각 구간에 마우스를 올리면 화자와 시작·종료 시각을 확인할 수 있습니다.

## 처리 순서

1. 음성 파일을 분석용 형식으로 정리합니다.
2. Nemotron이 화자별 발화 구간을 찾습니다.
3. Whisper가 음성을 글로 옮깁니다.
4. 시간 정보를 기준으로 전사문과 화자를 연결합니다. Small·Turbo는 단어 시간 정보를, Linux Large-v3는 전사 구간 시간을 사용합니다.

현재는 화자 분리 후 전사를 이어서 수행하는 순차 처리 방식입니다. CPU 환경에서는 긴 파일의 전사에 시간이 걸릴 수 있으나, 타임라인·경과 시간·전사 처리 시각·중단 버튼으로 현재 상태를 확인할 수 있습니다.

## 상세 문서

- [개발 계획서](docs/development-plan.md)
- [웹 서버 모드 구현 계획](docs/web-server-plan.md)
- [현재 구현 상태와 검증 메모](docs/implementation-status.md)
- [Linux 검증 가이드](docs/linux-test-guide.md)
- [로컬 LLM 전사 다듬기·요약 계획](docs/local-llm-refinement-plan.md)
- [로컬 모델 저장소 안내](models/README.md)
- [스크립트 안내](scripts/README.md)
