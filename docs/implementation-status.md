# TalkTracker 구현 상태 및 검증 메모

이 문서는 개발 계획서와 별개로, 현재 프로그램에 실제 반영된 기능과 확인 중인 항목을 기록합니다.

## 현재 구현 완료

### 데스크톱 화면

- Python/Tkinter 기반 Windows·macOS·Linux 데스크톱 화면
- 음성 파일 선택, 분석 시작, 중단, 선택 취소 버튼
- 로컬 모델 준비 상태 표시 및 모델 준비 메뉴
- 분석 단계와 경과 시간 표시, Whisper 전사 중 처리된 녹음 시각 표시
- 화자 분리가 끝나면 전사가 진행 중이어도 화자 타임라인 먼저 표시
- 분석 중단 시 이미 완성된 결과만 화면에 남김
- Markdown 결과 저장

### 로컬 AI 연결과 짧은 AI 정리 — 1~2단계

- 상단의 **로컬 AI 연결 필요** 상태 버튼에서 같은 PC 또는 신뢰하는 사설망의 OpenAI 호환 LLM 서버를 연결
- `/v1/models`로 모델 목록을 읽고, 목록을 제공하지 않는 서버는 모델 ID를 직접 입력
- 선택 모델에 짧은 생성 요청을 보내 실제 응답 경로까지 확인한 뒤 `로컬 AI 연결됨 · 모델명` 상태로 표시
- 같은 PC의 루프백 주소와 사용자가 지정한 사설망 LLM 서버 주소를 허용하고, 주소 끝의 `/v1` 유무는 자동 처리. URL에는 인증정보·쿼리·조각 식별자를 넣을 수 없음
- 최근 성공한 서버 주소 목록과 마지막 모델 ID를 사용자 설정에 저장하며, 목록의 주소는 최근 연결 목록에서 다시 선택 가능. API 키는 메모리에만 유지하고 오류 문구·저장 결과에는 포함하지 않음
- API 키 입력은 **표시/숨기기**와 글자 수 표시를 제공하며, 현재 입력 위치는 화면에서 확인 가능
- 모델 목록·연결 확인 오류는 인증, API 경로, 서버 과부하, 응답 형식별로 다음 조치를 안내
- 웹 화면도 같은 연결 API를 제공하되, 웹 서버가 `127.0.0.1`에 바인딩된 경우에만 연결 패널을 노출
- Linux GUI에서 OpenAI 호환 서버의 모델 목록 조회와 짧은 연결 확인을 수동 검증
- 분석 완료 뒤 원본 전사는 자동 보존·별도 저장. AI 결과로는 다듬은 대화와 요약을 하나 또는 둘 다 선택해 생성
- 대화의 목적·배경 안내와 화자별 이름·역할 안내를 현재 요청에만 전달. 원본 화자·시간·순서는 바꾸지 않고, 입력한 표시 이름은 다듬은 대화에만 반영
- AI 응답은 구간 ID 기반 JSON으로 검증. 잘못되거나 누락된 구간은 원문을 유지하고, AI 결과를 원본 전사에 덮어쓰지 않음
- AI가 응답 길이 한도에 닿으면 더 큰 길이로 한 번 재시도. 성공한 길이 설정은 같은 서버 주소·모델 조합에만 저장하며 API 키는 저장하지 않음
- GUI와 `127.0.0.1` 웹 화면 모두에서 AI 정리 단계·경과 시간·중단·결과 전환을 제공. 음성 분석과 AI 정리는 한 번에 하나만 실행
- 원본·다듬은 대화·요약은 화면에서 각각 전환해 보고 별도 Markdown 파일로 저장. 나중에 요약을 추가해도 기존 다듬은 대화는 유지
- 짧은 전사의 전체 전송과 선택 요약은 구현했으며, 긴 전사의 문맥 한도 재분할·단계적 재결합은 다음 단계 범위

### 음성 분석 흐름

1. FFmpeg로 입력 파일을 16 kHz 모노 WAV로 정리합니다.
2. `nvidia/Nemotron-3-Diarization`으로 화자별 발화 구간을 분석합니다.
3. 사용자가 고른 Whisper 모델을 CTranslate2 형식으로 실행합니다. Windows는 원본 모델을 로컬에서 변환하고, macOS·Linux는 준비된 CTranslate2 모델을 직접 받습니다. Windows·Linux에서 CUDA GPU가 준비되면 Small·Turbo는 GPU `float16`을 우선 사용하고, 그렇지 않으면 CPU `int8`을 사용합니다. GPU 로드가 실패하면 Small·Turbo는 CPU로 자동 전환합니다. Large-v3는 CUDA GPU `float16` 전용이며 Linux에서는 구간 단위 시간으로 화자를 연결합니다.
4. 전사 구간과 화자 구간의 시간대를 겹쳐 화자별 대화 기록으로 만듭니다.

## 모델과 저장 위치

모든 모델은 프로젝트 내부 `models` 폴더를 사용합니다.

| 용도 | 폴더 | 비고 |
| --- | --- | --- |
| 화자 분리 | `models/nemotron-3-diarization` | macOS·Linux는 GGUF, Windows는 Python 모델 파일 |
| macOS·Linux Whisper | `models/whisper-*-ct2` | 내려받은 CTranslate2 실행 모델 |
| Windows Whisper 원본 | `models/whisper-*` | Hugging Face 형식 원본 모델 |
| Windows CPU·CUDA 실행 모델 | `models/whisper-*-ct2*` | 원본을 CTranslate2로 변환한 모델 |

macOS·Linux의 CTranslate2 Whisper 폴더는 `config.json`, `preprocessor_config.json`, `model.bin`, `tokenizer.json`, `vocabulary.json`이 모두 있어야 합니다. 앱과 최신 모델 스크립트는 전체 파일을 검사하며, 일부 파일만 빠졌다면 누락분만 복구합니다.

## 전사 품질 선택

| 화면 선택 | 모델 | 실행 조건 |
| --- | --- | --- |
| 빠른 전사 | `openai/whisper-small` | CUDA GPU 우선 · CPU 대체 가능 |
| 권장 | `openai/whisper-large-v3-turbo` | CUDA GPU 우선 · CPU 대체 가능 · 기본값 |
| 최고 정확도 | `openai/whisper-large-v3` | 호환 CUDA GPU 필요 |

선택한 전사 모델만 다운로드합니다. Windows에서는 CUDA GPU가 준비되면 GPU 실행본과 CPU 대체 실행본을 함께 변환합니다. Linux는 하나의 준비된 CTranslate2 모델을 GPU `float16` 또는 CPU `int8`로 실행합니다. 결과 Markdown에는 실제 사용한 전사 모델과 실행 환경을 기록합니다.

## 전사 오류 진단

- Large-v3가 GPU를 시작한 뒤 전사 중 실패하면, 앱은 이를 단순 CUDA 설치 실패로 표시하지 않습니다.
- 오류 시 실행 환경, 모델 구성, CTranslate2·faster-whisper 버전, 원본 예외를 `outputs/diagnostics/whisper-large-v3-*.log`에 기록합니다.
- `model.bin`만 있는 불완전한 모델 폴더는 분석 시작 전에 감지합니다. 모델 관리의 **모델 다운로드 시작** 또는 운영체제별 `Download-Models` 스크립트로 빠진 파일만 복구할 수 있습니다.

## Windows·macOS·Linux 준비 및 실행

| 작업 | 스크립트 |
| --- | --- |
| 가상 환경 및 패키지 설치 | Windows: `scripts/windows/Install-Dependencies.ps1` · macOS: `scripts/macos/Install-Dependencies.sh` · Linux: `scripts/linux/Install-Dependencies.sh` |
| 원본 모델 다운로드 | Windows: `scripts/windows/Download-Models.ps1` · macOS: `scripts/macos/Download-Models.sh` · Linux: `scripts/linux/Download-Models.sh` |
| Whisper 모델 준비 | Windows: `scripts/windows/Prepare-FasterWhisper.ps1` · macOS: `scripts/macos/Prepare-FasterWhisper.sh` · Linux: `scripts/linux/Prepare-FasterWhisper.sh` |
| 프로그램 실행 | Windows: `scripts/windows/Run-Gui.ps1` · macOS: `scripts/macos/Run-Gui.sh` · Linux: `scripts/linux/Run-Gui.sh` |

FFmpeg는 음성 형식 변환에 필요합니다. 설치되어 있지 않으면 준비 스크립트가 안내를 표시합니다. Windows·Linux에서 CUDA GPU 전사를 사용하려면 NVIDIA 드라이버, CUDA 12.x 런타임, cuDNN 9가 필요합니다. 조건을 충족하지 않거나 GPU 모델 로드에 실패하면 Small·Turbo는 CPU `int8`으로 실행합니다.

macOS에서는 Small·Turbo 프로필을 CPU `int8`로 실행합니다. Windows·Linux는 CTranslate2 CUDA 실행 환경을 감지하면 GPU를 우선 사용합니다. Large-v3는 CUDA GPU가 필요하므로 macOS에서는 선택할 수 없습니다.

macOS 설치 스크립트는 기존 Python 3.11+Tkinter를 재사용하고, 없는 경우 프로젝트 내부에 사전 빌드 Python 3.11을 내려받아 Homebrew의 긴 Python·Tkinter 소스 컴파일을 피합니다. Homebrew는 FFmpeg의 사전 빌드 bottle 설치에만 사용합니다.

## 긴 녹음 파일 검증 현황

- 약 1시간 49분 길이의 M4A 녹음으로 실제 분석 흐름을 확인 중입니다.
- 화자 분리는 완료되면 전사 전에 타임라인에 반영됩니다.
- 전사는 CPU 환경에서 가장 오래 걸리는 단계입니다. 현재 UI는 전사 중 처리된 녹음 시각과 전체 길이를 함께 표시합니다.
- 현재 CPU 전사 속도와 전체 소요 시간은 이 실제 녹음의 완료 시점에 기록할 예정입니다.

## 중단 동작

- 사용자가 **중단**을 누르면 화면은 즉시 중단 요청 상태로 바뀝니다.
- 현재 처리 단위가 끝나는 안전한 지점에서 분석을 멈춥니다.
- 완료된 화자 분리 구간과 이미 생성된 전사문만 결과로 표시하고 저장할 수 있습니다.
- Nemotron의 한 번의 계산 도중에는 즉시 멈추지 않을 수 있으며, 다음 안전 지점에서 반영됩니다.

## 다음 검증 항목

- [Linux 검증 가이드](linux-test-guide.md)에 따른 Ubuntu·Debian·Fedora·Arch 계열의 설치 및 GUI 실행 확인
- Linux CPU 전사와 NVIDIA CUDA GPU 전사의 실제 처리 시간·CPU/VRAM 사용량 기록
- Linux Large-v3의 전체 녹음 처리와 `outputs/diagnostics` 오류 기록 경로 확인
- 긴 M4A 파일 전체 처리 시간 측정
- 화자 수와 화자 구간의 품질 확인
- Whisper 전사문과 화자 매칭의 자연스러움 확인
- 중단 후 Markdown 저장 결과 확인
- 다양한 입력 형식(MP3, WAV, AAC) 점검
