# TalkTracker

로컬 음성 파일에서 **누가 언제 말했는지**와 **무슨 말을 했는지**를 한 화면에서 정리하는 Windows용 데스크톱 프로그램입니다. 음성 파일과 분석 결과는 PC 밖으로 전송하지 않습니다.

> **운영체제 지원 범위:** 현재는 **Windows만 지원**합니다. 설치·모델 다운로드·실행을 위한 PowerShell 스크립트는 `scripts/windows`에만 제공하며, macOS와 Linux용 스크립트는 아직 제공하지 않습니다.

## 현재 가능한 기능

- M4A, MP3, WAV, FLAC, AAC, OGG 음성 파일 선택
- Nemotron 기반 화자 구간 분리
- 전사 목적에 따라 Whisper Small 또는 Large-v3 Turbo 선택
- 호환 GPU가 있을 때 Whisper Large-v3 최고 정확도 모드 선택
- 화자별 타임라인을 전사 중에도 먼저 표시
- 경과 시간과 진행 상태 표시
- 분석 중단 시, 그 시점까지 완성된 화자 구간과 전사 결과 유지
- 읽기 쉬운 Markdown 형식으로 결과 저장

## 현재 작업 상태

현재 버전에서는 화면 시제품만이 아니라, 실제 로컬 분석 흐름까지 연결되어 있습니다.

- 파일 선택부터 분석 시작·중단·결과 저장까지 데스크톱 화면에서 실행
- Nemotron으로 화자별 구간을 먼저 찾고, 분석 도중에도 화자 타임라인을 표시
- Whisper Small(빠른 확인), Large-v3 Turbo(권장), Large-v3(GPU 필요)를 선택해 전사
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

## 처리 순서

1. 음성 파일을 분석용 형식으로 정리합니다.
2. Nemotron이 화자별 발화 구간을 찾습니다.
3. Whisper가 음성을 글로 옮깁니다.
4. 시간 정보를 기준으로 전사문과 화자를 연결합니다.

현재는 화자 분리 후 전사를 이어서 수행하는 순차 처리 방식입니다. CPU 환경에서는 긴 파일의 전사에 시간이 걸릴 수 있으나, 타임라인·경과 시간·중단 버튼으로 현재 상태를 확인할 수 있습니다.

## 상세 문서

- [개발 계획서](docs/development-plan.md)
- [현재 구현 상태와 검증 메모](docs/implementation-status.md)
- [스크립트 안내](scripts/README.md)
