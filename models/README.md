# 로컬 모델 저장소

TalkTracker는 실행에 필요한 모델을 이 폴더에 보관합니다. Hugging Face 공용 캐시가 아닌 프로젝트 폴더를 사용하므로, 모델을 한 번 준비한 뒤에는 인터넷 연결 없이 분석할 수 있습니다.

모델 파일은 크기가 크므로 Git에 포함하지 않습니다. 이 폴더에서는 이 안내와 `.gitkeep`만 Git으로 관리합니다.

## 모델 준비 방법

프로젝트 최상위 폴더에서 현재 운영체제에 맞는 다운로드 스크립트를 실행하세요. 기본값은 권장 모델인 Whisper Large-v3 Turbo입니다.

| 운영체제 | 모델 다운로드 |
| --- | --- |
| Windows | `powershell -ExecutionPolicy Bypass -File .\scripts\windows\Download-Models.ps1 -PrepareForLocalUse` |
| macOS | `bash scripts/macos/Download-Models.sh` |
| Linux | `bash scripts/linux/Download-Models.sh` |

특정 전사 품질을 준비하려면 `small`, `turbo`, `large-v3` 중 하나를 지정합니다. 실행 예시와 모델 준비 옵션은 [스크립트 안내](../scripts/README.md)를 참고하세요.

```bash
# Linux에서 최고 정확도 Large-v3 준비
bash scripts/linux/Download-Models.sh --whisper-profile large-v3
```

Windows에서는 원본 Whisper 모델을 받은 뒤 CTranslate2 실행 모델로 변환합니다. macOS와 Linux는 이미 변환된 CTranslate2 모델을 직접 받으므로 로컬 변환을 하지 않습니다.

## 전사 모델 프로필과 실행 조건

| 프로필 | 모델 폴더 예시 | macOS | Windows·Linux |
| --- | --- | --- | --- |
| 빠른 전사 | `whisper-small-ct2` | CPU `int8` | CUDA GPU `float16` 우선, 불가하면 CPU `int8` |
| 권장 | `whisper-large-v3-turbo-ct2` | CPU `int8` | CUDA GPU `float16` 우선, 불가하면 CPU `int8` |
| 최고 정확도 | `whisper-large-v3-ct2` | 실행 불가 | 호환 CUDA GPU `float16` 필요 |

Windows는 변환 과정에서 CPU·CUDA 전용 폴더를 각각 만들 수 있습니다. 예를 들어 Small·Turbo는 `-ct2` CPU 모델과 `-ct2-cuda-fp16` GPU 모델을 함께 준비합니다. Linux는 하나의 준비된 CTranslate2 모델을 CPU와 CUDA GPU 모두에서 사용합니다.

현재 선택한 프로필과 Nemotron만 내려받아도 됩니다. 위 표의 모든 Whisper 모델을 한 번에 보관할 필요는 없습니다.

## 로컬 LLM 모델은 이 폴더에 넣지 않습니다

TalkTracker의 **로컬 AI 연결**은 OpenAI 호환 API를 제공하는 별도 로컬 LLM 서버에 접속합니다. 해당 LLM의 가중치와 실행 환경은 그 서버가 관리하므로, TalkTracker의 `models/` 폴더에 복사하거나 운영체제별 `Download-Models` 스크립트로 받을 필요가 없습니다.

TalkTracker에는 Whisper 전사 모델과 Nemotron 화자 분리 모델만 이 폴더에 둡니다. LLM 서버 주소와 선택 모델 ID는 사용자 설정에 저장할 수 있지만 API 키는 저장하지 않습니다.

## 웹 서버에서의 모델 확인

웹 서버 모드도 데스크톱 앱과 똑같이 Whisper 전사 모델과 Nemotron 화자 분리 모델의 실행 가능 여부를 확인합니다. 실행 가능한 전사 프로필이 하나도 없으면 업로드 화면 대신 **분석 요청 수신 불가** 상태 페이지를 보여 주고 파일을 받지 않습니다. 이 상태는 모델을 의도적으로 지운 PC에서 정상이며, 해당 PC를 분석 서버로 쓰지 않고 모델과 처리 성능이 있는 다른 PC의 서버 주소로 접속하면 됩니다.

## CTranslate2 Whisper 폴더의 필수 파일

`model.bin`만 있다고 모델 준비가 끝난 것이 아닙니다. 공식 CTranslate2 저장소의 파일 구성은 모델별로 다릅니다.

| 프로필 | 필수 파일 |
| --- | --- |
| Small | `config.json`, `model.bin`, `tokenizer.json`, `vocabulary.txt` |
| Turbo·Large-v3 | `config.json`, `preprocessor_config.json`, `model.bin`, `tokenizer.json`, `vocabulary.json` |

TalkTracker와 최신 macOS·Linux 다운로드 스크립트는 선택한 모델에 맞는 파일을 검사합니다. 하나라도 없으면 모델 관리 화면에서 **모델 다운로드 시작**을 누르거나 현재 운영체제용 다운로드 스크립트를 다시 실행하세요.

예를 들어 Linux의 Large-v3 폴더에 구성 파일만 빠진 경우, 아래 명령은 이미 받은 큰 `model.bin`을 유지하고 누락된 파일만 다시 받습니다.

```bash
bash scripts/linux/Download-Models.sh --whisper-profile large-v3 --skip-diarization
```

macOS에서도 같은 방식으로 현재 선택한 모델의 누락 파일만 복구할 수 있습니다.

```bash
bash scripts/macos/Download-Models.sh --whisper-profile turbo --skip-diarization
```

Windows의 원본 모델과 변환 결과가 불완전하면 `Download-Models.ps1`과 `Prepare-FasterWhisper.ps1`을 다시 실행해 준비합니다. 폴더를 임의로 복사하거나 파일을 섞기보다 운영체제별 스크립트를 사용하는 편이 안전합니다.

## 화자 분리 모델

모든 운영체제에서 `models/nemotron-3-diarization/`을 사용하지만 실행 방식은 다릅니다.

| 운영체제 | 주요 모델 파일 | 실행 방식 |
| --- | --- | --- |
| macOS·Linux | `Nemotron-3-Diarization.q8_0.gguf` | 프로젝트 내부 `.tools/nemo-speech` 네이티브 실행기 |
| Windows | `model.safetensors` 등 원본 저장소 파일 | Python·Transformers 실행 경로 |

macOS와 Linux에서 화자 분리 실행기가 없다면 모델 다운로드가 아니라 해당 운영체제의 `Install-Dependencies` 스크립트를 다시 실행해야 합니다.

## 문제 확인

- 앱의 **로컬 모델 관리**에서 각 모델의 준비 상태를 먼저 확인합니다.
- Linux Large-v3 전사 단계에서 오류가 나면 `outputs/diagnostics/whisper-large-v3-*.log`를 확인합니다. 이 로그에는 실제 CUDA 실행 환경, 모델 구성, 라이브러리 버전, 원본 오류가 기록됩니다.
- 모델을 다른 PC로 옮겨야 할 때도 대상 PC에서 운영체제별 스크립트를 실행하는 방식을 우선합니다. 모델은 용량이 크고 운영체제별 준비 방식이 다릅니다.

모델 다운로드·CUDA 준비·전사 품질 선택의 상세 내용은 [README](../README.md)와 [스크립트 안내](../scripts/README.md)에 있습니다.
