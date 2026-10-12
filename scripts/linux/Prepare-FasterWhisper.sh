#!/usr/bin/env bash

# Convert Whisper models for the best available local CTranslate2 runtime on Linux.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
converter="$project_root/.venv/bin/ct2-transformers-converter"
python="$project_root/.venv/bin/python"
whisper_profile="turbo"
runtime="auto"

usage() {
  cat <<'EOF'
Usage: bash scripts/linux/Prepare-FasterWhisper.sh [options]

Options:
  --whisper-profile PROFILE  small, turbo (default), or large-v3
  --runtime TARGET           auto (default), cpu, cuda, or both
  -h, --help                 Show this help.
EOF
}

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This setup script is for Linux. Run the matching script for your operating system." >&2
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --whisper-profile)
      [[ $# -ge 2 ]] || { echo "--whisper-profile needs a value." >&2; exit 2; }
      whisper_profile="$2"
      shift 2
      ;;
    --runtime)
      [[ $# -ge 2 ]] || { echo "--runtime needs a value." >&2; exit 2; }
      runtime="$2"
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
    source_folder="whisper-small"; cpu_folder="whisper-small-ct2"; cuda_folder="whisper-small-ct2-cuda-fp16"; display_name="Whisper Small"
    ;;
  turbo)
    source_folder="whisper-large-v3-turbo"; cpu_folder="whisper-large-v3-turbo-ct2"; cuda_folder="whisper-large-v3-turbo-ct2-cuda-fp16"; display_name="Whisper Large-v3 Turbo"
    ;;
  large-v3)
    source_folder="whisper-large-v3"; cpu_folder=""; cuda_folder="whisper-large-v3-ct2"; display_name="Whisper Large-v3"
    ;;
  *)
    echo "Unknown Whisper profile: $whisper_profile" >&2
    usage >&2
    exit 2
    ;;
esac

case "$runtime" in auto|cpu|cuda|both) ;; *) echo "Unknown runtime target: $runtime" >&2; usage >&2; exit 2 ;; esac

source_model="$project_root/models/$source_folder"
if [[ ! -f "$source_model/model.safetensors" ]]; then
  echo "$display_name source model was not found: $source_model" >&2
  echo "Run Download-Models.sh --whisper-profile $whisper_profile first." >&2
  exit 1
fi
if [[ ! -x "$converter" || ! -x "$python" ]]; then
  echo "The project conversion tools were not found. Run scripts/linux/Install-Dependencies.sh first." >&2
  exit 1
fi

if "$python" -c 'import ctranslate2, sys; sys.exit(0 if ctranslate2.get_cuda_device_count() > 0 and "float16" in ctranslate2.get_supported_compute_types("cuda") else 1)'; then
  cuda_available=true
else
  cuda_available=false
fi

if [[ "$runtime" == "cpu" ]]; then
  targets=(cpu)
elif [[ "$runtime" == "cuda" ]]; then
  [[ "$cuda_available" == true ]] || { echo "CUDA FP16 is unavailable. Check the NVIDIA driver, CUDA 12.x, and cuDNN 8." >&2; exit 1; }
  targets=(cuda)
elif [[ "$runtime" == "both" ]]; then
  [[ "$cuda_available" == true ]] || { echo "CUDA FP16 is unavailable. Check the NVIDIA driver, CUDA 12.x, and cuDNN 8." >&2; exit 1; }
  [[ -n "$cpu_folder" ]] && targets=(cpu cuda) || targets=(cuda)
elif [[ "$cuda_available" == true ]]; then
  [[ -n "$cpu_folder" ]] && targets=(cpu cuda) || targets=(cuda)
elif [[ -n "$cpu_folder" ]]; then
  targets=(cpu)
else
  echo "Whisper Large-v3 requires a compatible CUDA GPU." >&2
  exit 1
fi

prepare_runtime() {
  local target="$1" destination_folder quantization runtime_name destination
  if [[ "$target" == "cpu" ]]; then
    [[ -n "$cpu_folder" ]] || { echo "$display_name requires CUDA GPU execution." >&2; exit 1; }
    destination_folder="$cpu_folder"; quantization="int8"; runtime_name="CPU INT8"
  else
    destination_folder="$cuda_folder"; quantization="float16"; runtime_name="CUDA GPU FP16"
  fi
  destination="$project_root/models/$destination_folder"

  if [[ -f "$destination/model.bin" ]]; then
    echo "Already ready ($runtime_name): $destination"
    return
  fi
  if [[ -e "$destination" ]]; then
    if [[ -n "$(find "$destination" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "An incomplete model conversion folder exists: $destination" >&2
      exit 1
    fi
    rmdir "$destination"
  fi

  echo "Preparing $display_name for $runtime_name inference..."
  "$converter" --model "$source_model" --output_dir "$destination" \
    --copy_files tokenizer.json preprocessor_config.json --quantization "$quantization"
  [[ -f "$destination/model.bin" ]] || { echo "$runtime_name model preparation did not complete successfully." >&2; exit 1; }
}

if [[ "$cuda_available" == true && -n "$cpu_folder" ]]; then
  echo "CUDA GPU detected. Preparing GPU FP16 and CPU INT8 fallback models."
elif [[ "$cuda_available" == true ]]; then
  echo "CUDA GPU detected. Preparing the GPU FP16 model."
else
  echo "CUDA GPU execution is unavailable. Preparing the CPU INT8 model."
fi

for target in "${targets[@]}"; do
  prepare_runtime "$target"
done

echo
echo "Done. TalkTracker will automatically select the best prepared Whisper runtime."
