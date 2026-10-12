#!/usr/bin/env bash

# Convert a downloaded Whisper model to the local CTranslate2 format used by TalkTracker.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
converter="$project_root/.venv/bin/ct2-transformers-converter"
whisper_profile="turbo"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup script is for macOS. Run the matching script for your operating system." >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage: bash scripts/macos/Prepare-FasterWhisper.sh [--whisper-profile PROFILE]

PROFILE can be small, turbo (default), or large-v3.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --whisper-profile)
      [[ $# -ge 2 ]] || { echo "--whisper-profile needs a value." >&2; exit 2; }
      whisper_profile="$2"
      shift 2
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
    source_folder="whisper-small"
    destination_folder="whisper-small-ct2"
    display_name="Whisper Small"
    quantization="int8"
    ;;
  turbo)
    source_folder="whisper-large-v3-turbo"
    destination_folder="whisper-large-v3-turbo-ct2"
    display_name="Whisper Large-v3 Turbo"
    quantization="int8"
    ;;
  large-v3)
    source_folder="whisper-large-v3"
    destination_folder="whisper-large-v3-ct2"
    display_name="Whisper Large-v3"
    quantization="float16"
    ;;
  *)
    echo "Unknown Whisper profile: $whisper_profile" >&2
    usage >&2
    exit 2
    ;;
esac

source="$project_root/models/$source_folder"
destination="$project_root/models/$destination_folder"

if [[ ! -f "$source/model.safetensors" ]]; then
  echo "$display_name source model was not found: $source" >&2
  echo "Run Download-Models.sh --whisper-profile $whisper_profile first." >&2
  exit 1
fi

if [[ ! -x "$converter" ]]; then
  echo "CTranslate2 converter was not found: $converter" >&2
  echo "Run scripts/macos/Install-Dependencies.sh first." >&2
  exit 1
fi

if [[ -f "$destination/model.bin" ]]; then
  echo "Already ready: $destination"
  exit 0
fi

if [[ -e "$destination" ]]; then
  if [[ -n "$(find "$destination" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "An incomplete model conversion folder exists: $destination" >&2
    echo "Inspect it before running this script again." >&2
    exit 1
  fi
  rmdir "$destination"
fi

echo "Preparing $display_name for local inference..."
echo "Source: $source"
echo "Destination: $destination"

"$converter" \
  --model "$source" \
  --output_dir "$destination" \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization "$quantization"

if [[ ! -f "$destination/model.bin" ]]; then
  echo "Whisper model preparation did not complete successfully." >&2
  exit 1
fi

echo
echo "Done. TalkTracker will use the prepared model from: $destination"
