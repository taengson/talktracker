# 로컬 모델 동작 검증 기록

**검증일:** 2026-10-08  
**상태:** 통과

## 검증 환경

| 항목 | 값 |
| --- | --- |
| 운영체제 | Windows |
| CPU | Intel Core i7-9700 (8 논리 프로세서) |
| 시스템 메모리 | 24GB |
| GPU | GeForce GT 710, 2GB VRAM |
| 실행 방식 | CPU 전용 |
| Python | 3.11 격리 환경 (`.venv`) |
| 오디오 변환 | FFmpeg 8.1.1 |

GT 710의 2GB VRAM은 Whisper Turbo와 Nemotron을 GPU로 동시에 실행하기에 부족하므로, 이 검증은 CPU에서 진행했다.

## 로컬 모델 경로

모든 가중치는 프로젝트 폴더에 저장했으며, 실행 시 Hugging Face 공용 캐시나 원격 모델 ID를 사용하지 않는다.

| 모델 | 프로젝트 내 경로 | 확인된 가중치 크기 |
| --- | --- | --- |
| Whisper large-v3-turbo | `models/whisper-large-v3-turbo/` | 약 1.51GB |
| Nemotron-3-Diarization | `models/nemotron-3-diarization/` | 약 0.37GB |

## 실행한 테스트

1. 공개 다화자 MP3 샘플을 받았다.
2. FFmpeg로 20초 길이의 AAC M4A 파일로 변환했다.
3. 스모크 테스트가 입력 M4A를 16kHz 모노 WAV로 정규화했다.
4. 로컬 Whisper Turbo 폴더에서 전사와 단어별 시간 정보를 생성했다.
5. 로컬 Nemotron 폴더에서 화자 구간을 생성했다.

### 결과

- M4A 입력 파일 정규화: 성공
- Whisper 전사: 성공
- 단어별 시간 정보: 58개 생성
- Nemotron 화자 분리: 성공
- 생성된 화자 구간: 4개

처음 20초에서 확인된 화자 구간은 다음과 같다.

| 화자 | 구간 |
| --- | --- |
| `speaker_0` | 0.35s – 9.31s |
| `speaker_1` | 9.14s – 13.82s |
| `speaker_2` | 13.54s – 18.32s |
| `speaker_3` | 18.38s – 20.00s |

검증 보고서 원본은 `tmp/smoke-test/report.json`에 있으며 Git 추적 대상이 아니다.

## 결론과 다음 검증

TalkTracker의 핵심 조합인 Whisper Turbo와 Nemotron은 이 PC에서 CPU만으로도 로컬 M4A 파일을 처리할 수 있음이 확인됐다. 사용자 화면에서는 이러한 모델·CPU 설정을 노출하지 않고, 앱이 기본값으로 처리한다. 결과는 우선 Markdown과 읽기 좋은 HTML 문서로 제공한다. 실제 사용 전에 한국어 iPhone 음성 메모와 긴 회의 파일로 다음을 추가 측정해야 한다.

- 30분 이상 파일의 처리 시간과 메모리 사용량
- 한국어 전사 정확도
- 실제 회의에서의 화자 전환·동시 발화 품질
- CPU 실행 시 허용 가능한 대기 시간
