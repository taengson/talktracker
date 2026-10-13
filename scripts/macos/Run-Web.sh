#!/usr/bin/env bash

# Launch TalkTracker's browser-accessible server on macOS.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
python="$project_root/.venv/bin/python"
source_root="$project_root/src"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This launch script is for macOS. Run the matching script for your operating system." >&2
  exit 1
fi
if [[ ! -x "$python" ]]; then
  echo "The project Python environment was not found: $python" >&2
  echo "Run scripts/macos/Install-Dependencies.sh first." >&2
  exit 1
fi
if [[ ! -d "$source_root" ]]; then
  echo "The app source folder was not found: $source_root" >&2
  exit 1
fi

export PYTHONPATH="$source_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" -m talktracker.server "$@"
