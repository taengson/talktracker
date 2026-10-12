#!/usr/bin/env bash

# Create TalkTracker's project-local Python environment on macOS.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
requirements="$project_root/requirements.txt"
venv_root="$project_root/.venv"
python="$venv_root/bin/python"
tools_root="$project_root/.tools"
uv_root="$tools_root/uv"
uv="$uv_root/uv"
nemo_speech_root="$tools_root/nemo-speech"
nemo_speech="$nemo_speech_root/bin/nemo-speech"

find_python311_with_tk() {
  local candidate version
  for candidate in python3.11 /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11; do
    if [[ -x "$candidate" ]] || command -v "$candidate" >/dev/null 2>&1; then
      if [[ "$candidate" == */* ]]; then
        [[ -x "$candidate" ]] || continue
      else
        candidate="$(command -v "$candidate")"
      fi
      version="$($candidate -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      if [[ "$version" == "3.11" ]] && "$candidate" -c 'import tkinter' >/dev/null 2>&1; then
        printf '%s\n' "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

install_project_uv() {
  if [[ -x "$uv" ]]; then
    return
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to download the project-local Python runtime." >&2
    exit 1
  fi

  local installer
  installer="$(mktemp -t talktracker-uv.XXXXXX)"
  echo "Downloading the small project-local uv launcher..."
  if ! curl --fail --location --proto '=https' --tlsv1.2 --silent --show-error \
    https://astral.sh/uv/install.sh --output "$installer"; then
    rm -f "$installer"
    echo "Could not download uv. Check the network connection and run this script again." >&2
    exit 1
  fi

  mkdir -p "$uv_root"
  if ! UV_UNMANAGED_INSTALL="$uv_root" sh "$installer"; then
    rm -f "$installer"
    echo "Could not install the project-local uv launcher." >&2
    exit 1
  fi
  rm -f "$installer"
  [[ -x "$uv" ]] || { echo "uv was not installed where expected: $uv" >&2; exit 1; }
}

create_managed_venv() {
  install_project_uv
  echo "Downloading a prebuilt Python 3.11 runtime and creating this project's environment..."
  UV_CACHE_DIR="$tools_root/uv-cache" \
    UV_PYTHON_INSTALL_DIR="$tools_root/python" \
    "$uv" venv --managed-python --python 3.11 "$venv_root"
}

verify_python_dependencies() {
  "$python" - <<'PY'
import importlib
import sys

modules = [
    "numpy",
    "huggingface_hub",
    "ctranslate2",
    "faster_whisper",
]
if sys.platform != "darwin":
    modules.extend((
    "torch",
    "transformers",
    "librosa",
    ))

for module_name in modules:
    try:
        importlib.import_module(module_name)
    except Exception as error:
        print(f"Could not import {module_name}: {error}", file=sys.stderr)
        raise SystemExit(1)

print("TalkTracker Python packages are ready.")
PY
}

install_ffmpeg_with_homebrew() {
  if command -v ffmpeg >/dev/null 2>&1; then
    return
  fi
  if ! command -v brew >/dev/null 2>&1; then
    cat >&2 <<'EOF'
FFmpeg is required but was not found.
Install Homebrew from https://brew.sh, then run this script again.
EOF
    exit 1
  fi

  echo "Installing FFmpeg from a prebuilt Homebrew package..."
  HOMEBREW_NO_AUTO_UPDATE=1 brew install --force-bottle ffmpeg
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "FFmpeg was not found after Homebrew finished. Check Homebrew's output, then run this script again." >&2
    exit 1
  fi
}

install_nemo_speech() {
  if [[ -x "$nemo_speech" ]]; then
    return
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to download the macOS speaker-diarization runtime." >&2
    exit 1
  fi

  local installer backend
  backend="cpu"
  if [[ "$(uname -m)" == "arm64" ]]; then
    backend="metal"
  fi
  installer="$(mktemp -t talktracker-nemo-speech.XXXXXX)"
  echo "Downloading the prebuilt macOS speaker-diarization runtime..."
  if ! curl --fail --location --proto '=https' --tlsv1.2 --silent --show-error \
    https://raw.githubusercontent.com/NVIDIA/NeMo-Speech.cpp/main/scripts/install.sh --output "$installer"; then
    rm -f "$installer"
    echo "Could not download the speaker-diarization runtime installer. Check the network connection and run this script again." >&2
    exit 1
  fi

  # Prefer the verified native archive and fail clearly instead of unexpectedly
  # starting a lengthy C++ source build on the user's Mac.
  if ! sh "$installer" --prefix "$nemo_speech_root" --backend "$backend" --binary-only --no-modify-path; then
    rm -f "$installer"
    echo "Could not install the prebuilt macOS speaker-diarization runtime." >&2
    echo "No source build was started. Check the message above, then run this script again." >&2
    exit 1
  fi
  rm -f "$installer"
  [[ -x "$nemo_speech" ]] || { echo "The speaker-diarization runtime was not installed where expected: $nemo_speech" >&2; exit 1; }
  "$nemo_speech" --version
}

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup script is for macOS. Run the matching script for your operating system." >&2
  exit 1
fi

case "$(uname -m)" in
  x86_64) echo "Detected an Intel Mac. Installing the native CPU speech runtime..." ;;
  arm64) echo "Detected an Apple Silicon Mac. Installing the native Metal speech runtime..." ;;
  *) echo "Detected an unsupported macOS CPU architecture: $(uname -m)" >&2; exit 1 ;;
esac

if [[ ! -x "$python" ]]; then
  system_python="$(find_python311_with_tk || true)"
  if [[ -n "$system_python" ]]; then
    echo "Creating this project's Python environment from the installed Python 3.11..."
    "$system_python" -m venv "$venv_root"
  else
    create_managed_venv
  fi
fi

if ! "$python" -c 'import tkinter' >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Tkinter is unavailable in this existing project environment.
Remove the project .venv folder and run this script again to create it with the prebuilt managed Python runtime.
EOF
  exit 1
fi

install_ffmpeg_with_homebrew
install_nemo_speech

echo "Installing TalkTracker packages..."
if [[ -x "$uv" ]]; then
  UV_CACHE_DIR="$tools_root/uv-cache" "$uv" pip install --python "$python" --upgrade pip
  UV_CACHE_DIR="$tools_root/uv-cache" "$uv" pip install --python "$python" -r "$requirements"
else
  "$python" -m pip --version >/dev/null 2>&1 || "$python" -m ensurepip --upgrade
  "$python" -m pip install --upgrade pip
  "$python" -m pip install -r "$requirements"
fi

if ! dependency_check_output="$(verify_python_dependencies 2>&1)"; then
  echo "TalkTracker package verification failed:" >&2
  echo "$dependency_check_output" >&2
  echo "Fix the reported package error, then run this script again." >&2
  exit 1
fi
echo "$dependency_check_output"

echo
echo "Done. The project environment is ready."
echo "Next: run scripts/macos/Download-Models.sh to download the local models."
