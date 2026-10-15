# Linux 검증 가이드

이 문서는 TalkTracker의 Linux 설치·모델 준비·GUI 실행을 실제 배포판에서 확인할 때 사용합니다. 검증 대상은 그래픽 데스크톱 환경의 Python 3.11입니다.

## 시작 전 확인

프로젝트 최상위 폴더에서 아래 명령으로 배포판과 환경을 기록합니다.

```bash
uname -a
cat /etc/os-release
```

설치 스크립트는 `apt-get`, `dnf`, `pacman`을 순서대로 감지해 Python 3.11, Tkinter, FFmpeg 설치를 시도합니다. 시스템 패키지를 설치할 때 관리자 암호를 요청할 수 있습니다.

- Ubuntu·Debian에서 기본 저장소에 Python 3.11이 없으면 Python 3.11을 제공하는 저장소를 먼저 설정해야 합니다.
- Fedora 계열에서는 FFmpeg가 별도 저장소에 있을 수 있습니다.
- Arch 계열은 현재 기본 Python 버전이 3.11이 아닐 수 있으므로, Python 3.11을 제공하는 방법을 먼저 마련해야 합니다.
- GUI는 그래픽 세션에서 실행해야 합니다. SSH만 연결된 서버에서는 모델 다운로드·변환은 할 수 있지만 Tkinter 창은 열 수 없습니다.

## CPU 기본 검증

먼저 빠른 전사 프로필로 전체 흐름을 확인합니다. 프로젝트 최상위 폴더에서 차례대로 실행합니다.

```bash
bash scripts/linux/Install-Dependencies.sh
bash scripts/linux/Download-Models.sh --whisper-profile small
bash scripts/linux/Run-Gui.sh
```

기대 결과는 다음과 같습니다.

- `.venv`가 생성되고 필요한 Python 패키지가 설치됩니다.
- `models/whisper-small-ct2/`에 `config.json`, `preprocessor_config.json`, `model.bin`, `tokenizer.json`, `vocabulary.json`이 모두 내려받아지고, `models/nemotron-3-diarization/Nemotron-3-Diarization.q8_0.gguf`도 준비됩니다.
- 프로젝트 내부 `.tools/nemo-speech`에 화자 분리 실행기가 준비됩니다.
- GUI에서 **빠른 전사**를 선택하면 모델 상태가 준비됨으로 표시됩니다.
- 짧은 음성 파일을 선택해 분석과 Markdown 저장까지 완료할 수 있습니다. Whisper 전사 단계에서는 `처리한 시각 / 전체 길이`가 갱신됩니다.

## 로컬 AI 연결 검증 — 1단계

전사 다듬기·요약 기능은 아직 구현 전이지만, 같은 Linux PC에서 실행 중인 OpenAI 호환 LLM 서버에 연결하고 모델을 확인할 수 있습니다. LLM 서버는 TalkTracker가 설치하거나 시작하지 않으므로 먼저 원하는 서버와 모델을 준비해 실행하세요.

1. `bash scripts/linux/Run-Gui.sh`로 TalkTracker를 엽니다.
2. 상단의 **로컬 AI 연결 필요**를 누릅니다.
3. 서버 주소에 `http://127.0.0.1:포트` 또는 `http://localhost:포트`를 입력합니다. `/v1`은 넣어도 되고 생략해도 됩니다.
4. 서버가 모델 목록을 제공하면 **모델 불러오기**를 누르고, 그렇지 않으면 모델 ID를 직접 입력합니다.
5. **연결 확인**을 눌러 선택한 모델이 짧은 생성 요청을 처리하는지 확인합니다.

성공하면 상단 상태가 `로컬 AI 연결됨 · 모델명`으로 바뀌고 연결 창에도 성공 문구가 표시됩니다. 일부 추론 모델은 연결 확인에 눈에 보이는 문장을 돌려주지 않을 수 있지만, 응답 구조가 정상이라면 연결은 성공으로 처리합니다.

보안을 위해 `192.168.x.x` 같은 내부망 주소나 인터넷 주소는 현재 허용하지 않습니다. LLM 서버는 TalkTracker와 같은 PC에서 실행해야 하며, API 키가 필요하다면 연결 확인 시에만 입력합니다. API 키는 파일이나 설정에 저장되지 않습니다.

## NVIDIA CUDA GPU 검증

NVIDIA GPU가 있다면 드라이버 상태를 먼저 확인합니다.

```bash
nvidia-smi
bash scripts/linux/Install-Dependencies.sh
bash scripts/linux/Download-Models.sh --whisper-profile small
```

CUDA 실행 환경이 준비된 경우 앱은 받은 CTranslate2 Whisper 모델을 GPU `float16`으로 실행합니다. GPU를 시작하거나 전사하는 중 실패하면 Small·Turbo는 같은 모델을 CPU `int8`으로 다시 실행합니다. 최근 CTranslate2/faster-whisper 조합은 CUDA 12용 cuBLAS와 cuDNN 9가 필요합니다. 설치 스크립트는 NVIDIA GPU를 감지하면 이 런타임을 프로젝트 `.venv`에 자동으로 설치하고, GUI 실행 스크립트가 필요한 라이브러리 경로를 자동으로 설정합니다.

Linux ARM64 NVIDIA 시스템은 PyPI에서 CPU 전용 CTranslate2가 설치될 수 있습니다. 이때 업데이트된 `Install-Dependencies.sh`는 프로젝트 내부 `.tools/ctranslate2-cuda`에 CUDA용 CTranslate2를 한 번 빌드합니다. CUDA Toolkit의 `nvcc`와 cuDNN 개발 파일(`cudnn.h`, `libcudnn.so`)이 없으면 설치를 멈추고 필요한 항목을 정확히 안내합니다. 빌드가 완료되면 다음 명령으로 상태를 확인합니다.

```bash
source scripts/linux/Configure-Cuda-Environment.sh
.venv/bin/python - <<'PY'
import ctranslate2 as ct
print("CUDA devices:", ct.get_cuda_device_count())
print("CUDA compute types:", sorted(ct.get_supported_compute_types("cuda")))
PY
```

앱은 GPU를 우선 사용합니다. CUDA가 감지되지 않거나 준비되지 않은 경우에는 같은 준비된 모델을 CPU로 실행하며, 이는 정상 동작입니다. Large-v3는 CPU 대체가 없으므로 CUDA 환경을 먼저 준비해야 합니다. 다만 GPU가 정상 시작된 뒤 전사 중에 실패한 경우에는 설치 문제로 단정하지 말고 아래의 진단 로그를 먼저 확인합니다. 화자 분리는 `.tools/nemo-speech`의 공식 네이티브 실행기를 사용하며, NVIDIA 드라이버가 보이면 CUDA를, 그렇지 않으면 CPU를 선택합니다.

## 모델 파일 완전성 및 Large-v3 확인

CTranslate2 Whisper는 `model.bin`만으로 실행할 수 없습니다. `config.json`, `preprocessor_config.json`, `tokenizer.json`, `vocabulary.json`도 같은 모델 폴더에 있어야 합니다. 앱의 모델 관리 화면과 최신 모델 스크립트는 이 다섯 파일을 모두 확인합니다.

업데이트 전 내려받은 Large-v3 폴더에 일부 파일이 빠져 있다면, 아래 명령은 이미 있는 큰 가중치 파일은 유지하고 누락된 파일만 다시 받습니다.

```bash
bash scripts/linux/Download-Models.sh --whisper-profile large-v3 --skip-diarization
```

Large-v3 GPU 전사를 확인할 때는 GUI에서 **최고 정확도 · Whisper Large-v3**를 선택하고, 별도 터미널에서 `nvidia-smi`를 실행해 Python 프로세스와 GPU 메모리 사용량을 확인합니다. Large-v3는 일부 Linux CUDA 조합의 단어 단위 정렬 오류를 피하기 위해 전사 구간 시간으로 화자를 연결합니다. 대화 내용과 화자 분리 결과는 유지됩니다.

실행 중 전사 오류가 나면 앱이 아래 위치에 진단 로그를 남깁니다.

```text
outputs/diagnostics/whisper-large-v3-YYYYMMDD-HHMMSS.log
```

이 로그에는 선택 프로필, 실제 실행 환경, 모델 구성 파일 값의 형식, CTranslate2·faster-whisper 버전, 원본 예외와 Python traceback이 들어 있습니다. CUDA 설치 스크립트를 반복 실행하기 전에 이 로그로 모델 파일 누락과 전사 실행 오류를 구분합니다.

## 테스트 결과에 포함할 내용

문제가 생겼거나 검증 결과를 공유할 때에는 민감한 정보(사용자명, 홈 경로, 접근 토큰)를 지운 뒤 아래 내용을 포함해 주세요.

- `cat /etc/os-release`와 `uname -a` 출력
- CPU만 사용했는지, NVIDIA CUDA GPU를 사용했는지
- 실패한 단계의 전체 출력
- 생성된 `models/*-ct2*/` 폴더와 다섯 필수 파일의 존재 여부
- 짧은 음성 파일의 분석·Markdown 저장 성공 여부
- 실패 시 `outputs/diagnostics/whisper-*.log` 내용

