#!/usr/bin/env bash

# Create TalkTracker's project-local Python environment on Linux.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
requirements="$project_root/requirements.txt"
venv_root="$project_root/.venv"
python="$venv_root/bin/python"

probe_ctranslate2_cuda() {
  "$python" - <<'PY'
import sys

import ctranslate2 as ct

try:
    compute_types = ct.get_supported_compute_types("cuda")
except (OSError, RuntimeError, ValueError) as error:
    print(error, file=sys.stderr)
    if "not compiled with CUDA support" in str(error):
        raise SystemExit(2)
    raise SystemExit(3)

device_count = ct.get_cuda_device_count()
if device_count < 1:
    print("CTranslate2 can use CUDA but no GPU is visible to this process.", file=sys.stderr)
    raise SystemExit(4)
if "float16" not in compute_types:
    print(f"The detected GPU does not support Whisper FP16: {sorted(compute_types)}", file=sys.stderr)
    raise SystemExit(5)
print(f"CUDA GPU Whisper execution is ready ({device_count} GPU(s)).")
PY
}

has_nvidia_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

verify_python_dependencies() {
  "$python" - <<'PY'
import importlib
import sys

for module_name in (
    "torch",
    "transformers",
    "huggingface_hub",
    "ctranslate2",
    "faster_whisper",
    "librosa",
):
    try:
        importlib.import_module(module_name)
    except Exception as error:
        print(f"Could not import {module_name}: {error}", file=sys.stderr)
        raise SystemExit(1)

print("TalkTracker Python packages are ready.")
PY
}

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

source "$script_dir/Configure-Cuda-Environment.sh"
if ! dependency_check_output="$(verify_python_dependencies 2>&1)"; then
  echo "TalkTracker package verification failed:" >&2
  echo "$dependency_check_output" >&2
  echo "Fix the reported package error, then run this script again." >&2
  exit 1
fi
echo "$dependency_check_output"

cuda_probe_output=""
if cuda_probe_output="$(probe_ctranslate2_cuda 2>&1)"; then
  echo "$cuda_probe_output"
  echo "Model preparation will create GPU and CPU fallback models."
else
  cuda_probe_status=$?
  if has_nvidia_gpu && [[ "$cuda_probe_status" -eq 2 ]]; then
    cat <<'EOF'
An NVIDIA GPU is visible, but PyPI installed a CPU-only CTranslate2 package.
TalkTracker will now build a project-local CUDA-enabled CTranslate2 binding.
This is expected on some Linux ARM64 NVIDIA systems and takes a few minutes once.
EOF
    bash "$script_dir/Build-CTranslate2-CUDA.sh"
    source "$script_dir/Configure-Cuda-Environment.sh"
    cuda_probe_output="$(probe_ctranslate2_cuda 2>&1)" || {
      echo "CTranslate2 CUDA verification failed after the build:" >&2
      echo "$cuda_probe_output" >&2
      exit 1
    }
    echo "$cuda_probe_output"
    echo "Model preparation will create GPU and CPU fallback models."
  elif has_nvidia_gpu; then
    cat >&2 <<EOF
Warning: An NVIDIA GPU was found, but CTranslate2 CUDA execution is not ready.
$cuda_probe_output
Install CUDA 12.x runtime libraries and cuDNN 9, or ensure the GPU is passed through to this Linux environment. TalkTracker can use CPU INT8 meanwhile.
EOF
  fi
fi

echo
echo "Done. The project environment is ready."
echo "Next: run scripts/linux/Download-Models.sh to download the local models."
