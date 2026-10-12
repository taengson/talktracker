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
- `models/whisper-small-ct2/model.bin`과 `models/nemotron-3-diarization/Nemotron-3-Diarization.q8_0.gguf`가 내려받아집니다.
- 프로젝트 내부 `.tools/nemo-speech`에 화자 분리 실행기가 준비됩니다.
- GUI에서 **빠른 전사**를 선택하면 모델 상태가 준비됨으로 표시됩니다.
- 짧은 음성 파일을 선택해 분석과 Markdown 저장까지 완료할 수 있습니다.

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

```text
models/whisper-small-ct2/model.bin
```

앱은 GPU를 우선 사용합니다. CUDA가 감지되지 않거나 준비되지 않은 경우에는 같은 준비된 모델을 CPU로 실행하며, 이는 정상 동작입니다. Large-v3는 CPU 대체가 없으므로 설치 스크립트가 CUDA 런타임 설치에 실패했다는 메시지가 나오면 먼저 그 스크립트를 다시 실행해야 합니다. 화자 분리는 `.tools/nemo-speech`의 공식 네이티브 실행기를 사용하며, NVIDIA 드라이버가 보이면 CUDA를, 그렇지 않으면 CPU를 선택합니다.

GPU 상태를 좀 더 확인하려면 설치 후 아래 명령을 실행합니다.

## 테스트 결과에 포함할 내용

문제가 생겼거나 검증 결과를 공유할 때에는 민감한 정보(사용자명, 홈 경로, 접근 토큰)를 지운 뒤 아래 내용을 포함해 주세요.

- `cat /etc/os-release`와 `uname -a` 출력
- CPU만 사용했는지, NVIDIA CUDA GPU를 사용했는지
- 실패한 단계의 전체 출력
- 생성된 `models/*-ct2*/model.bin` 경로
- 짧은 음성 파일의 분석·Markdown 저장 성공 여부

