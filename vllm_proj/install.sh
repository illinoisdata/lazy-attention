#!/usr/bin/env bash
#
# Build vLLM from source.
#
# You almost certainly do NOT need this. LazyAttention and BlockAttention add no
# C++/CUDA of their own -- they are Python monkey patches plus Triton kernels
# that are JIT-compiled at runtime -- so the prebuilt wheel that
# `scripts/install.sh` installs already has every compiled kernel they need.
#
# Build from source only when you are modifying vLLM's own C++/CUDA.
#
# Environment overrides:
#   VLLM_COMMIT       vLLM commit/tag to build       (default: v0.9.2)
#   CUDA_HOME         CUDA toolkit root              (default: auto-detected)
#   MAX_JOBS          parallel compile jobs          (default: nproc, capped at 8)
#   NVCC_THREADS      threads per nvcc invocation    (default: 4)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

VLLM_COMMIT="${VLLM_COMMIT:-v0.9.2}"
NVCC_THREADS="${NVCC_THREADS:-4}"
if [[ -z "${MAX_JOBS:-}" ]]; then
    MAX_JOBS="$(nproc)"
    (( MAX_JOBS > 8 )) && MAX_JOBS=8
fi

die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# ------------------------------------------------------- prerequisites ----
command -v git   >/dev/null || die "git not found"
command -v cmake >/dev/null || die "cmake not found (pip install cmake ninja)"
command -v ninja >/dev/null || die "ninja not found (pip install cmake ninja)"
python -c "import torch" 2>/dev/null \
    || die "torch must be installed before building vLLM"

# Locate a CUDA toolkit instead of hardcoding a cluster-specific path.
if [[ -z "${CUDA_HOME:-}" ]]; then
    if command -v nvcc >/dev/null; then
        CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
    elif [[ -d /usr/local/cuda ]]; then
        CUDA_HOME=/usr/local/cuda
    else
        die "no CUDA toolkit found; set CUDA_HOME (on a module system: module load cuda/...)"
    fi
fi
[[ -x "${CUDA_HOME}/bin/nvcc" ]] || die "no nvcc under CUDA_HOME=${CUDA_HOME}"
export CUDA_HOME

say "Build configuration"
echo "  vLLM commit : ${VLLM_COMMIT}"
echo "  CUDA_HOME   : ${CUDA_HOME}"
echo "  nvcc        : $("${CUDA_HOME}/bin/nvcc" --version | tail -1)"
echo "  torch       : $(python -c 'import torch; print(torch.__version__)')"
echo "  jobs        : ${MAX_JOBS} (nvcc threads: ${NVCC_THREADS})"

# ccache makes rebuilds far cheaper, but is optional -- don't try to install it.
CCACHE_ARGS=()
if command -v ccache >/dev/null; then
    CCACHE_ARGS=(
        -DCMAKE_C_COMPILER_LAUNCHER=ccache
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
    )
else
    echo "  ccache      : not found (rebuilds will be slower)"
fi

# ------------------------------------------------------------- source ----
say "Fetching vLLM ${VLLM_COMMIT}"
if [[ -d vllm/.git ]]; then
    git -C vllm fetch --tags origin
else
    rm -rf vllm
    git clone https://github.com/vllm-project/vllm.git vllm
fi
git -C vllm checkout --force "${VLLM_COMMIT}"

# Our only change to upstream: a NO_C=1 escape hatch so the editable install
# below skips re-running the extension build we do by hand here.
cp setup.py vllm/setup.py

cd vllm
pip install -r requirements/build.txt

# -------------------------------------------------------------- build ----
say "Configuring"
rm -rf build
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX=. \
    -DVLLM_TARGET_DEVICE=cuda \
    -DVLLM_PYTHON_EXECUTABLE="$(which python)" \
    -DVLLM_PYTHON_PATH="$(python -c 'import sys; print(":".join(sys.path))')" \
    -DFETCHCONTENT_BASE_DIR="$(pwd)/.deps" \
    -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
    -DNVCC_THREADS="${NVCC_THREADS}" \
    -DCMAKE_JOB_POOL_COMPILE:STRING=compile \
    -DCMAKE_JOB_POOLS:STRING="compile=${MAX_JOBS}" \
    "${CCACHE_ARGS[@]}"

say "Building _C and _vllm_fa2_C"
# fa3 is omitted deliberately: it needs far more memory to compile.
cmake --build build --target _C _vllm_fa2_C -j "${MAX_JOBS}"

say "Installing extensions"
cmake --install build --component _C
cmake --install build --component _vllm_fa2_C

if [[ -d vllm_flash_attn ]]; then
    cp -r vllm_flash_attn ./vllm/
    rm -rf vllm_flash_attn
fi

say "Installing vLLM (editable, extensions already built)"
NO_C=1 pip install -e . --no-build-isolation

# ------------------------------------------------------------- verify ----
say "Verifying"
python -c "import vllm, vllm._C; print(f'vLLM {vllm.__version__} + compiled extensions OK')"
