#!/usr/bin/env bash

# Create TalkTracker's project-local Python environment on Linux.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
requirements="$project_root/requirements.txt"
venv_root="$project_root/.venv"
python="$venv_root/bin/python"

run_as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "Administrator access is required to install $1. Install it manually, then run this script again." >&2
    exit 1
  fi
}

install_linux_package() {
  local dependency="$1"
  if command -v apt-get >/dev/null 2>&1; then
    case "$dependency" in
      python) run_as_root apt-get update; run_as_root apt-get install -y python3.11 python3.11-venv ;;
      tkinter)
        if apt-cache show python3.11-tk >/dev/null 2>&1; then
          run_as_root apt-get install -y python3.11-tk
        else
          run_as_root apt-get install -y python3-tk
        fi
        ;;
      ffmpeg) run_as_root apt-get install -y ffmpeg ;;
    esac
  elif command -v dnf >/dev/null 2>&1; then
    case "$dependency" in
      python) run_as_root dnf install -y python3.11 ;;
      tkinter) run_as_root dnf install -y python3-tkinter ;;
      ffmpeg) run_as_root dnf install -y ffmpeg ;;
    esac
  elif command -v pacman >/dev/null 2>&1; then
    case "$dependency" in
      python) run_as_root pacman -Sy --needed python ;;
      tkinter) run_as_root pacman -Sy --needed tk ;;
      ffmpeg) run_as_root pacman -Sy --needed ffmpeg ;;
    esac
  else
    cat >&2 <<EOF
Could not find apt-get, dnf, or pacman to install $dependency automatically.
Install Python 3.11 (with venv and Tkinter) and FFmpeg using your distribution's package manager, then run this script again.
EOF
    exit 1
  fi
}

find_python311() {
  local candidate version
  for candidate in python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      version="$($candidate -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      if [[ "$version" == "3.11" ]]; then
        command -v "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This setup script is for Linux. Run the matching script for your operating system." >&2
  exit 1
fi

if [[ ! -x "$python" ]]; then
  system_python="$(find_python311 || true)"
  if [[ -z "$system_python" ]]; then
    echo "Python 3.11 was not found. Attempting to install it..."
    install_linux_package python
    system_python="$(find_python311 || true)"
  fi
  if [[ -z "$system_python" ]]; then
    echo "Python 3.11 is still unavailable. Install Python 3.11 and its venv package for your distribution, then run this script again." >&2
    exit 1
  fi

  if ! "$system_python" -c 'import tkinter' >/dev/null 2>&1; then
    install_linux_package tkinter
  fi
  if ! "$system_python" -c 'import tkinter' >/dev/null 2>&1; then
    echo "Tkinter is unavailable in Python 3.11 after installation." >&2
    exit 1
  fi

  echo "Creating this project's Python environment..."
  "$system_python" -m venv "$venv_root"
fi

"$python" -m pip --version >/dev/null 2>&1 || "$python" -m ensurepip --upgrade

if ! "$python" -c 'import tkinter' >/dev/null 2>&1; then
  echo "Tkinter is unavailable in this project environment. Recreate .venv after installing the distribution Tkinter package." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  install_linux_package ffmpeg
fi

echo "Installing TalkTracker packages..."
"$python" -m pip install --upgrade pip
"$python" -m pip install -r "$requirements"

if "$python" -c 'import ctranslate2, sys; sys.exit(0 if ctranslate2.get_cuda_device_count() > 0 and "float16" in ctranslate2.get_supported_compute_types("cuda") else 1)'; then
  echo "CUDA GPU Whisper execution is ready. Model preparation will create GPU and CPU fallback models."
elif command -v nvidia-smi >/dev/null 2>&1; then
  echo "Warning: An NVIDIA GPU was found, but CTranslate2 CUDA execution is not ready. Install CUDA 12.x and cuDNN 8 to enable GPU transcription; TalkTracker will use CPU INT8 meanwhile." >&2
fi

echo
echo "Done. The project environment is ready."
echo "Next: run scripts/linux/Download-Models.sh to download the local models."
