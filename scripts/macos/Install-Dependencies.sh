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

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup script is for macOS. Run the matching script for your operating system." >&2
  exit 1
fi

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

echo "Installing TalkTracker packages..."
if [[ -x "$uv" ]]; then
  UV_CACHE_DIR="$tools_root/uv-cache" "$uv" pip install --python "$python" --upgrade pip
  UV_CACHE_DIR="$tools_root/uv-cache" "$uv" pip install --python "$python" -r "$requirements"
else
  "$python" -m pip --version >/dev/null 2>&1 || "$python" -m ensurepip --upgrade
  "$python" -m pip install --upgrade pip
  "$python" -m pip install -r "$requirements"
fi

echo
echo "Done. The project environment is ready."
echo "Next: run scripts/macos/Download-Models.sh to download the local models."
