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

if [[ -d "$local_ct2_library" ]]; then
  export LD_LIBRARY_PATH="$local_ct2_library${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
