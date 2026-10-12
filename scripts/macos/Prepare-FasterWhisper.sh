#!/usr/bin/env bash

# Download a CTranslate2-format Whisper model ready for TalkTracker on macOS.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
hugging_face="$project_root/.venv/bin/hf"
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
    destination_folder="whisper-small-ct2"
    display_name="Whisper Small"
    repository="Systran/faster-whisper-small"
    ;;
  turbo)
    destination_folder="whisper-large-v3-turbo-ct2"
    display_name="Whisper Large-v3 Turbo"
    # This is the repository resolved by faster-whisper for the Turbo alias.
    repository="mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    ;;
  large-v3)
    destination_folder="whisper-large-v3-ct2"
    display_name="Whisper Large-v3"
    repository="Systran/faster-whisper-large-v3"
    ;;
  *)
    echo "Unknown Whisper profile: $whisper_profile" >&2
    usage >&2
    exit 2
    ;;
esac

destination="$project_root/models/$destination_folder"

if [[ ! -x "$hugging_face" ]]; then
  echo "Hugging Face CLI was not found: $hugging_face" >&2
  echo "Run scripts/macos/Install-Dependencies.sh first." >&2
  exit 1
fi

if [[ -f "$destination/model.bin" ]]; then
  echo "Already ready: $destination"
  exit 0
fi

echo "Preparing $display_name for local inference..."
echo "Downloading a pre-converted CTranslate2 model; no local INT8 conversion is performed on macOS."
echo "Repository: $repository"
echo "Destination: $destination"

"$hugging_face" download "$repository" \
  --local-dir "$destination" \
  --include config.json preprocessor_config.json model.bin tokenizer.json 'vocabulary.*'

if [[ ! -f "$destination/model.bin" ]]; then
  echo "Whisper model preparation did not complete successfully." >&2
  exit 1
fi

echo
echo "Done. TalkTracker will use the prepared model from: $destination"
