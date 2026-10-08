# Windows 초기 설정 스크립트

현재는 Windows만 지원합니다. 이 폴더에는 앱을 실행하기 전에 한 번 준비하는 최소한의 스크립트만 둡니다. 모델 파일은 항상 프로젝트의 `models/` 폴더에 저장되며, Hugging Face 공용 캐시에 의존하지 않습니다.

먼저 PC에 Python 3.11을 설치한 뒤, 프로젝트 최상위 폴더에서 아래 순서대로 실행합니다.

## 1. 앱 패키지 준비

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\Install-Dependencies.ps1
```

프로젝트 내부의 `.venv` 환경을 만들고 `requirements.txt`에 적힌 Python 패키지만 설치합니다. 음성 파일 변환에는 FFmpeg도 필요하며, 설치 스크립트는 FFmpeg가 없을 때 안내를 표시합니다.

## 2. 모델 다운로드

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\Download-Models.ps1
```

기본 실행은 권장 모델인 Whisper Large-v3 Turbo와 Nemotron을 `models/`에 저장합니다. 프로그램 화면에서 시작하면 다운로드 뒤 실행용 변환까지 하나의 작업으로 이어집니다. 스크립트에서 직접 지정할 수도 있습니다.

```powershell
# 빠른 전사
powershell -ExecutionPolicy Bypass -File .\scripts\windows\Download-Models.ps1 -WhisperProfile small

# 권장 (기본값)
powershell -ExecutionPolicy Bypass -File .\scripts\windows\Download-Models.ps1 -WhisperProfile turbo

# 최고 정확도: 호환 CUDA GPU 필요
powershell -ExecutionPolicy Bypass -File .\scripts\windows\Download-Models.ps1 -WhisperProfile large-v3
```

## 3. Whisper CPU 최적화 준비

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\Prepare-FasterWhisper.ps1
```

선택한 Whisper 모델을 로컬 실행 형식으로 변환합니다. 기본 Turbo와 Small은 CPU `int8` 형식으로, Large-v3는 GPU `float16` 형식으로 준비됩니다. 첫 실행에만 필요합니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\Prepare-FasterWhisper.ps1 -WhisperProfile small
```

## 4. 프로그램 실행

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\Run-Gui.ps1
```

파일 선택, 분석 시작, 중단, 결과 저장 버튼을 사용할 수 있습니다. 분석 시작을 누르면 Nemotron 화자 분리와 선택한 Whisper 전사를 순차로 실행한 뒤 실제 결과를 표시합니다.
