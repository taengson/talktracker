#!/usr/bin/env bash

# Create TalkTracker's project-local Python environment on macOS.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
requirements="$project_root/requirements.txt"
venv_root="$project_root/.venv"
python="$venv_root/bin/python"

install_with_homebrew() {
  local package="$1"
  if ! command -v brew >/dev/null 2>&1; then
    cat >&2 <<EOF
$package is required but Homebrew is not installed.
Install the requirement yourself, or install Homebrew from https://brew.sh and run this script again.
EOF
    exit 1
  fi

  if ! brew list --versions "$package" >/dev/null 2>&1; then
    echo "Installing $package with Homebrew..."
    brew install "$package"
  fi
}

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup script is for macOS. Run the matching script for your operating system." >&2
  exit 1
fi

if [[ ! -x "$python" ]]; then
  if ! command -v python3.11 >/dev/null 2>&1; then
    install_with_homebrew "python@3.11"
  fi

  system_python="$(command -v python3.11 || true)"
  if [[ -z "$system_python" ]] && command -v brew >/dev/null 2>&1; then
    homebrew_python="$(brew --prefix python@3.11)/bin/python3.11"
    if [[ -x "$homebrew_python" ]]; then
      system_python="$homebrew_python"
    fi
  fi
  if [[ -z "$system_python" ]]; then
    echo "Python 3.11 could not be found after installation." >&2
    exit 1
  fi

  if ! "$system_python" -c 'import tkinter' >/dev/null 2>&1; then
    install_with_homebrew "python-tk@3.11"
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
  install_with_homebrew "python-tk@3.11"
  if ! "$python" -c 'import tkinter' >/dev/null 2>&1; then
    echo "Tkinter is still unavailable in this Python environment. Recreate .venv, then run this script again." >&2
    exit 1
  fi
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  install_with_homebrew "ffmpeg"
fi

echo "Installing TalkTracker packages..."
"$python" -m pip install --upgrade pip
"$python" -m pip install -r "$requirements"

echo
echo "Done. The project environment is ready."
echo "Next: run scripts/macos/Download-Models.sh to download the local models."
