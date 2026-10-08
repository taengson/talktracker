# TalkTracker 구현 상태 및 검증 메모

이 문서는 개발 계획서와 별개로, 현재 프로그램에 실제 반영된 기능과 확인 중인 항목을 기록합니다.

## 현재 구현 완료

### 데스크톱 화면

- Python/Tkinter 기반 Windows 데스크톱 화면
- 음성 파일 선택, 분석 시작, 중단, 선택 취소 버튼
- 로컬 모델 준비 상태 표시 및 모델 준비 메뉴
- 분석 단계와 경과 시간 표시
- 화자 분리가 끝나면 전사가 진행 중이어도 화자 타임라인 먼저 표시
- 분석 중단 시 이미 완성된 결과만 화면에 남김
- Markdown 결과 저장

### 음성 분석 흐름

1. FFmpeg로 입력 파일을 16 kHz 모노 WAV로 정리합니다.
2. `nvidia/Nemotron-3-Diarization`으로 화자별 발화 구간을 분석합니다.
3. 사용자가 고른 Whisper 모델을 CTranslate2 형식으로 변환한 뒤 전사합니다. Small과 Turbo는 CPU `int8`, Large-v3는 GPU `float16`을 사용합니다.
4. 전사 구간과 화자 구간의 시간대를 겹쳐 화자별 대화 기록으로 만듭니다.

## 모델과 저장 위치

모든 모델은 프로젝트 내부 `models` 폴더를 사용합니다.

| 용도 | 폴더 | 비고 |
| --- | --- | --- |
| 화자 분리 | `models/nemotron-3-diarization` | Nemotron 원본 모델 |
| 원본 전사 모델 | `models/whisper-large-v3-turbo` | Hugging Face 형식 |
| CPU 전사용 모델 | `models/whisper-large-v3-turbo-ct2` | faster-whisper/CTranslate2 `int8` 형식 |

## 전사 품질 선택

| 화면 선택 | 모델 | 실행 조건 |
| --- | --- | --- |
| 빠른 전사 | `openai/whisper-small` | CPU 사용 가능 |
| 권장 | `openai/whisper-large-v3-turbo` | CPU 사용 가능 · 기본값 |
| 최고 정확도 | `openai/whisper-large-v3` | 호환 CUDA GPU 필요 |

선택한 모델만 다운로드·변환합니다. 결과 Markdown에는 사용한 전사 모델도 함께 기록합니다.

## Windows 준비 및 실행

| 작업 | 스크립트 |
| --- | --- |
| 가상 환경 및 패키지 설치 | `scripts/windows/Install-Dependencies.ps1` |
| 원본 모델 다운로드 | `scripts/windows/Download-Models.ps1` |
| CPU 전사용 Whisper 모델 준비 | `scripts/windows/Prepare-FasterWhisper.ps1` |
| 프로그램 실행 | `scripts/windows/Run-Gui.ps1` |

FFmpeg는 음성 형식 변환에 필요합니다. 설치되어 있지 않으면 준비 스크립트가 안내를 표시합니다.

## 긴 녹음 파일 검증 현황

- 약 1시간 49분 길이의 M4A 녹음으로 실제 분석 흐름을 확인 중입니다.
- 화자 분리는 완료되면 전사 전에 타임라인에 반영됩니다.
- 전사는 CPU 환경에서 가장 오래 걸리는 단계입니다.
- 현재 CPU 전사 속도와 전체 소요 시간은 이 실제 녹음의 완료 시점에 기록할 예정입니다.

## 중단 동작

- 사용자가 **중단**을 누르면 화면은 즉시 중단 요청 상태로 바뀝니다.
- 현재 처리 단위가 끝나는 안전한 지점에서 분석을 멈춥니다.
- 완료된 화자 분리 구간과 이미 생성된 전사문만 결과로 표시하고 저장할 수 있습니다.
- Nemotron의 한 번의 계산 도중에는 즉시 멈추지 않을 수 있으며, 다음 안전 지점에서 반영됩니다.

## 다음 검증 항목

- 긴 M4A 파일 전체 처리 시간 측정
- 화자 수와 화자 구간의 품질 확인
- Whisper 전사문과 화자 매칭의 자연스러움 확인
- 중단 후 Markdown 저장 결과 확인
- 다양한 입력 형식(MP3, WAV, AAC) 점검
