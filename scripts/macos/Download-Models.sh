#!/usr/bin/env bash

# Download TalkTracker's selected Whisper model and the Nemotron diarization model.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
models_root="$project_root/models"
hugging_face="$project_root/.venv/bin/hf"
prepare_script="$script_dir/Prepare-FasterWhisper.sh"
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
  --prepare-for-local-use    Convert Whisper for TalkTracker after downloading it.
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
    whisper_repository="openai/whisper-small"
    whisper_folder="whisper-small"
    whisper_display_name="Whisper Small (fast transcription)"
    ;;
  turbo)
    whisper_repository="openai/whisper-large-v3-turbo"
    whisper_folder="whisper-large-v3-turbo"
    whisper_display_name="Whisper Large-v3 Turbo (recommended)"
    ;;
  large-v3)
    whisper_repository="openai/whisper-large-v3"
    whisper_folder="whisper-large-v3"
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

download_model() {
  local repository="$1"
  local folder_name="$2"
  local destination="$models_root/$folder_name"

  echo "Downloading $repository"
  echo "Destination: $destination"
  "$hugging_face" download "$repository" \
    --local-dir "$destination" \
    --exclude '*.gguf' \
    --exclude '*.nemo' \
    --exclude '*.bin' \
    --exclude '*.onnx' \
    --exclude '*.onnx_data' \
    --exclude '*.mp4' \
    --exclude '*.gif' \
    --exclude '*.png'
}

echo "Selected transcription model: $whisper_display_name"
download_model "$whisper_repository" "$whisper_folder"

if [[ "$skip_diarization" != true ]]; then
  download_model "nvidia/Nemotron-3-Diarization" "nemotron-3-diarization"
fi

if [[ "$prepare_for_local_use" == true ]]; then
  echo
  echo "Preparing the downloaded transcription model for TalkTracker..."
  bash "$prepare_script" --whisper-profile "$whisper_profile"
fi

echo
echo "Done. TalkTracker will use models from: $models_root"
