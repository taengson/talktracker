#!/usr/bin/env bash

# Build a CUDA-enabled CTranslate2 wheel when PyPI supplied a CPU-only wheel.
# This is normally needed only on Linux ARM64 NVIDIA systems, such as DGX Spark.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
python="$project_root/.venv/bin/python"
tools_root="$project_root/.tools"
source_root="$tools_root/ctranslate2-4.8.2"
build_root="$tools_root/ctranslate2-4.8.2-build"
install_root="$tools_root/ctranslate2-cuda"
ctranslate2_version="4.8.2"

run_as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "Administrator access is required to install $1. Install the build packages manually, then run this script again." >&2
    exit 1
  fi
}

install_build_packages() {
  if command -v apt-get >/dev/null 2>&1; then
    run_as_root apt-get update
    run_as_root apt-get install -y build-essential cmake git pkg-config libopenblas-dev
  elif command -v dnf >/dev/null 2>&1; then
    run_as_root dnf install -y gcc-c++ make cmake git pkgconf-pkg-config openblas-devel
  elif command -v pacman >/dev/null 2>&1; then
    run_as_root pacman -Sy --needed base-devel cmake git pkgconf openblas
  else
    cat >&2 <<'EOF'
Could not find apt-get, dnf, or pacman to install the CTranslate2 CUDA build tools.
Install a C++17 compiler, CMake 3.15+, Git, pkg-config, and OpenBLAS development files, then run this script again.
EOF
    exit 1
  fi
}

find_cuda_root() {
  local nvcc
  nvcc="$(command -v nvcc || true)"
  if [[ -z "$nvcc" ]]; then
    return 1
  fi
  cd "$(dirname "$nvcc")/.." && pwd
}

find_cudnn_include() {
  local candidate
  for candidate in \
    "$cuda_root/include" \
    /usr/include \
    /usr/local/include \
    /usr/include/aarch64-linux-gnu \
    /usr/include/x86_64-linux-gnu; do
    if [[ -f "$candidate/cudnn.h" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

find_cudnn_library() {
  local candidate
  for candidate in \
    "$cuda_root/lib64/libcudnn.so" \
    "$cuda_root/lib/libcudnn.so" \
    /usr/lib/aarch64-linux-gnu/libcudnn.so \
    /usr/lib/x86_64-linux-gnu/libcudnn.so \
    /usr/lib64/libcudnn.so \
    /usr/local/lib/libcudnn.so; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This CUDA build script is for Linux." >&2
  exit 1
fi
if [[ ! -x "$python" ]]; then
  echo "The project Python environment was not found: $python" >&2
  echo "Run scripts/linux/Install-Dependencies.sh first." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "No usable NVIDIA GPU was detected. A CUDA CTranslate2 build is not needed." >&2
  exit 1
fi

cuda_root="$(find_cuda_root || true)"
if [[ -z "$cuda_root" ]]; then
  cat >&2 <<'EOF'
The NVIDIA driver is visible, but the CUDA Toolkit compiler (nvcc) is not installed or not on PATH.
Install the CUDA Toolkit and make nvcc available, then run this script again.
EOF
  exit 1
fi

cudnn_include="$(find_cudnn_include || true)"
cudnn_library="$(find_cudnn_library || true)"
if [[ -z "$cudnn_include" || -z "$cudnn_library" ]]; then
  cat >&2 <<EOF
cuDNN development files were not found for the CUDA Toolkit at $cuda_root.
Install the matching cuDNN development package (header cudnn.h and library libcudnn.so), then run this script again.
EOF
  exit 1
fi

echo "Installing CTranslate2 CUDA build tools (one-time setup)..."
install_build_packages

mkdir -p "$tools_root"
if [[ ! -d "$source_root/.git" ]]; then
  echo "Downloading CTranslate2 $ctranslate2_version source..."
  git clone --depth 1 --branch "v$ctranslate2_version" --recurse-submodules \
    https://github.com/OpenNMT/CTranslate2.git "$source_root"
fi

echo "Building CUDA-enabled CTranslate2. This can take a few minutes..."
cmake -S "$source_root" -B "$build_root" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$install_root" \
  -DCUDA_TOOLKIT_ROOT_DIR="$cuda_root" \
  -DCUDNN_INCLUDE_DIR="$cudnn_include" \
  -DCUDNN_LIBRARIES="$cudnn_library" \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN=ON \
  -DCUDA_DYNAMIC_LOADING=ON \
  -DWITH_MKL=OFF \
  -DWITH_OPENBLAS=ON \
  -DOPENMP_RUNTIME=COMP \
  -DBUILD_CLI=OFF
cmake --build "$build_root" --parallel
cmake --install "$build_root"

echo "Installing the matching Python binding in the project environment..."
"$python" -m pip install --upgrade pybind11 setuptools wheel
CTRANSLATE2_ROOT="$install_root" "$python" -m pip install --no-build-isolation --force-reinstall "$source_root/python"

# The local library is deliberately scoped to TalkTracker; only its scripts add
# it to LD_LIBRARY_PATH.
source "$script_dir/Configure-Cuda-Environment.sh"
"$python" - <<'PY'
import ctranslate2 as ct

compute_types = sorted(ct.get_supported_compute_types("cuda"))
device_count = ct.get_cuda_device_count()
if device_count < 1 or "float16" not in compute_types:
    raise SystemExit(
        f"CTranslate2 CUDA verification failed (devices={device_count}, types={compute_types})."
    )
print(f"CTranslate2 CUDA is ready: {device_count} GPU(s), compute types: {', '.join(compute_types)}")
PY
