#!/usr/bin/env bash

# Download TalkTracker's selected Whisper model and the Nemotron diarization model.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
models_root="$project_root/models"
hugging_face="$project_root/.venv/bin/hf"
whisper_profile="turbo"
skip_diarization=false
prepare_for_local_use=false

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup script is for macOS. Run the matching script for your operating system." >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage: bash scripts/macos/Download-Models.sh [options]

Options:
  --whisper-profile PROFILE  small, turbo (default), or large-v3
  --skip-diarization         Do not download Nemotron.
  --prepare-for-local-use    Kept for compatibility; the macOS download is already ready to run.
  -h, --help                 Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --whisper-profile)
      [[ $# -ge 2 ]] || { echo "--whisper-profile needs a value." >&2; exit 2; }
      whisper_profile="$2"
      shift 2
      ;;
    --skip-diarization)
      skip_diarization=true
      shift
      ;;
    --prepare-for-local-use)
      prepare_for_local_use=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$whisper_profile" in
  small)
    whisper_repository="Systran/faster-whisper-small"
    whisper_folder="whisper-small-ct2"
    whisper_display_name="Whisper Small (fast transcription)"
    ;;
  turbo)
    # This is the CTranslate2 model used by faster-whisper for the Turbo alias.
    whisper_repository="mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    whisper_folder="whisper-large-v3-turbo-ct2"
    whisper_display_name="Whisper Large-v3 Turbo (recommended)"
    ;;
  large-v3)
    whisper_repository="Systran/faster-whisper-large-v3"
    whisper_folder="whisper-large-v3-ct2"
    whisper_display_name="Whisper Large-v3 (CUDA GPU required to run)"
    ;;
  *)
    echo "Unknown Whisper profile: $whisper_profile" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ ! -x "$hugging_face" ]]; then
  echo "Hugging Face CLI was not found: $hugging_face" >&2
  echo "Run scripts/macos/Install-Dependencies.sh first." >&2
  exit 1
fi

mkdir -p "$models_root"

download_prepared_whisper() {
  local destination="$models_root/$whisper_folder"
  local required_files=(config.json preprocessor_config.json model.bin tokenizer.json vocabulary.json)
  local missing_files=()
  local file
  for file in "${required_files[@]}"; do
    [[ -f "$destination/$file" ]] || missing_files+=("$file")
  done
  if [[ ${#missing_files[@]} -eq 0 ]]; then
    echo "Already ready: $destination"
    return
  fi

  echo "Completing pre-converted $whisper_repository"
  echo "Destination: $destination"
  echo "Missing files: ${missing_files[*]}"
  "$hugging_face" download "$whisper_repository" \
    --local-dir "$destination" \
    --force-download \
    "${missing_files[@]}"

  for file in "${required_files[@]}"; do
    [[ -f "$destination/$file" ]] || {
      echo "Whisper model download did not provide the required file: $file" >&2
      exit 1
    }
  done
}

download_diarization_model() {
  local destination="$models_root/nemotron-3-diarization"
  echo "Downloading the ready-to-run Nemotron speaker-diarization model"
  echo "Destination: $destination"
  "$hugging_face" download "nvidia/Nemotron-3-Diarization" \
    --local-dir "$destination" \
    --include Nemotron-3-Diarization.q8_0.gguf
}

echo "Selected transcription model: $whisper_display_name"
echo "macOS downloads the ready-to-run CTranslate2 model directly; no local conversion is needed."
download_prepared_whisper

if [[ "$skip_diarization" != true ]]; then
  download_diarization_model
fi

if [[ "$prepare_for_local_use" == true ]]; then
  echo
  echo "The downloaded Whisper model is already prepared for local use on macOS."
fi

echo
echo "Done. TalkTracker will use models from: $models_root"
