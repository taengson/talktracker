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

## 로컬 AI 연결과 결과 만들기 검증

TalkTracker는 같은 Linux PC 또는 신뢰하는 사설망에서 실행 중인 OpenAI 호환 LLM 서버로 전사문을 보내 다듬은 대화·요약을 만들 수 있습니다. LLM 서버와 모델은 TalkTracker가 설치하거나 시작하지 않으므로 먼저 원하는 서버와 모델을 준비해 실행하세요.

1. `bash scripts/linux/Run-Gui.sh`로 TalkTracker를 엽니다.
2. 상단의 **로컬 AI 연결 필요**를 누릅니다.
3. 서버 주소에 `http://127.0.0.1:포트` 또는 `http://localhost:포트`를 입력합니다. `/v1`은 넣어도 되고 생략해도 됩니다.
4. 서버가 모델 목록을 제공하면 **모델 불러오기**를 누르고, 그렇지 않으면 모델 ID를 직접 입력합니다.
5. **연결 확인**을 눌러 선택한 모델이 짧은 생성 요청을 처리하는지 확인합니다.
6. 짧은 음성 파일을 분석합니다. 원본 전사는 자동으로 보존되어 결과 화면에서 바로 확인할 수 있습니다.
7. **다듬은 대화** 또는 **요약**을 선택합니다. 둘을 함께 선택할 수도 있습니다.
   - 둘 다 선택하면 다듬은 대화를 먼저 만든 뒤 해당 결과를 요약합니다. 요약만 선택했을 때 이전에 만들어 둔 다듬은 대화가 있으면 그 내용을 요약 입력으로 사용합니다.
8. **추론 강도**에서 `기본(서버 설정)`, `medium`, `low`, `none` 중 이번 요청에 쓸 값을 고릅니다. 필요하면 대화의 목적·배경, 화자별 표시 이름·역할을 입력한 뒤 **선택한 결과 만들기**를 누릅니다.
9. 현재 단계와 경과 시간이 보이는지, 완료 후 원본·다듬은 대화·요약을 각각 전환하고 별도 Markdown 파일로 저장할 수 있는지 확인합니다.

성공하면 상단 상태가 `로컬 AI 연결됨 · 모델명`으로 바뀌고 연결 창에도 성공 문구가 표시됩니다. 일부 추론 모델은 연결 확인에 눈에 보이는 문장을 돌려주지 않을 수 있지만, 응답 구조가 정상이라면 연결은 성공으로 처리합니다.

같은 PC의 LLM 서버뿐 아니라 사용자가 신뢰하는 사설망 LLM 서버의 IP 주소 또는 호스트 이름도 입력할 수 있습니다. AI 정리 기능을 실행하면 전사문이 선택한 서버로 전달되므로, 신뢰하는 서버만 선택하세요. API 키가 필요하다면 연결 확인과 현재 실행 중인 AI 요청에만 쓰며, 파일이나 설정에 저장되지 않습니다. 전사문은 작업 중에만 임시 Markdown 파일로 만들어 Agent Chat 최신 첨부 정책과 같이 전달합니다. 5MiB까지는 허용하며, 모델 본문은 240KiB 이하면 전체를 보내고 더 길면 16KiB 문서 단위 중 요청 관련 부분과 앞·뒤 문맥만 골라 한 번에 보냅니다. 관련성을 찾지 못하면 앞 70%·뒤 30% 발췌본을 보냅니다. 모델에는 자유 형식 Markdown을 요청합니다. `기본(서버 설정)`은 추론 강도 필드를 보내지 않고, `medium`·`low`·`none`은 선택한 값만 보냅니다. 자동 응답 길이 재시도·발화 묶음 재분할·추론 강도 변경은 사용하지 않으며, 서버가 선택값을 지원하지 않으면 기본값으로 되돌아가지 않고 안내합니다. 임시 파일은 성공·실패·중단 후 모두 정리됩니다.

웹 서버를 `127.0.0.1` 또는 SSH 터널로 열었다면 브라우저 화면에서도 위와 같은 연결·결과 만들기·문서 저장 흐름을 확인할 수 있습니다. 사내 폐쇄망에서 `--host 0.0.0.0`으로 직접 공개한 화면에서도 사용하려면 `--allow-local-ai-over-network`를 함께 지정합니다. 이 옵션은 브라우저의 API 키와 AI 요청이 네트워크를 통해 전달되는 것을 운영자가 허용한다는 뜻이므로, 신뢰하는 내부망에서만 사용하세요.

```bash
bash scripts/linux/Run-Web.sh --host 0.0.0.0 --port 12200 --allow-local-ai-over-network
```

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

