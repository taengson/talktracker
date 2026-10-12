#!/usr/bin/env bash

# Load the project-local CTranslate2 CUDA library, when TalkTracker built one.
# This file is intended to be sourced by the Linux launcher and model tools.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file from a TalkTracker Linux script; do not run it directly." >&2
  exit 1
fi

linux_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
linux_project_root="$(cd "$linux_script_dir/../.." && pwd)"
local_ct2_library="$linux_project_root/.tools/ctranslate2-cuda/lib"
project_python="$linux_project_root/.venv/bin/python"
library_paths=""

if [[ -d "$local_ct2_library" ]]; then
  library_paths="$local_ct2_library"
fi

# faster-whisper/CTranslate2 needs cuBLAS and cuDNN at the moment it loads a
# Whisper model.  The Linux installer keeps compatible CUDA 12 runtime wheels
# in the project virtual environment, so include their directories only for
# TalkTracker.  A system-wide CUDA installation remains supported too.
if [[ -x "$project_python" ]]; then
  nvidia_library_paths="$("$project_python" - <<'PY' 2>/dev/null || true
from pathlib import Path

paths = []
for module_name in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
    try:
        module = __import__(module_name, fromlist=["__file__"])
    except ImportError:
        continue
    path = Path(module.__file__).resolve().parent
    if path.is_dir():
        paths.append(str(path))
print(":".join(paths))
PY
)"
  if [[ -n "$nvidia_library_paths" ]]; then
    library_paths="${library_paths:+$library_paths:}$nvidia_library_paths"
  fi
fi

if [[ -n "$library_paths" ]]; then
  export LD_LIBRARY_PATH="$library_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
