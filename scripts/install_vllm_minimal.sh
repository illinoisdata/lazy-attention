#!/bin/bash
# Minimal vLLM 0.8.5.post1 rebuild for RTX 5070 Ti (sm_120)
# Skips: MoE, FlashAttn, FlashMLA, DeepGEMM, _C_stable_libtorch
# Keeps: _C (core ops with all sources) + cumem_allocator
#
# Prerequisites:
#   - CUDA 13.x toolkit (with nvToolsExt shim)
#   - PyTorch 2.7.0+cu128  (has sm_120 support)
#   - Triton >= 3.4.0       (has sm_120 support)
#   - cmake, ninja, setuptools_scm
set -euo pipefail

VLLM_VERSION="v0.8.5.post1"
BUILD_DIR="/tmp/vllm_rebuild_$$"

echo "=== Minimal vLLM rebuild for sm_120 (RTX 5070 Ti) ==="
echo "torch: $(python -c 'import torch; print(torch.__version__)')"
echo "triton: $(python -c "import triton; print(triton.__version__)")"
echo "cuda:   $(python -c 'import torch; print(torch.version.cuda)')"
echo "archs:  $(python -c 'import torch; print(torch.cuda.get_arch_list())')"

# 1) Fetch source
echo ""
echo "[1/6] Fetching vLLM ${VLLM_VERSION} source..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
git clone --depth 1 --branch "${VLLM_VERSION}" https://github.com/vllm-project/vllm.git src 2>&1 | tail -3
cd src

# 2) Patch pyproject.toml
echo "[2/6] Patching pyproject.toml..."
sed -i 's/license = "Apache-2.0"/license = {text = "Apache-2.0"}/' pyproject.toml
sed -i '/license-files/d' pyproject.toml
sed -i 's/torch == 2.6.0/torch >= 2.6.0/' pyproject.toml
sed -i 's/>=3.9,<3.13/>=3.9/' pyproject.toml

# 3) Patch setup.py: remove MoE, flash_attn, flashmla, deepgemm, _C_stable_libtorch
echo "[3/6] Patching setup.py..."
python3 << 'PYEOF'
import re
with open("setup.py") as f:
    src = f.read()
minimal_ext = """ext_modules = []

# --- MINIMAL BUILD: _C + cumem_allocator only ---
if _is_cuda() or _is_hip():
    ext_modules.append(CMakeExtension(name="vllm.cumem_allocator"))
    ext_modules.append(CMakeExtension(name="vllm.triton_kernels", optional=True))

if _build_custom_ops():
    ext_modules.append(CMakeExtension(name="vllm._C"))
    # Skipped: _C_stable_libtorch (quant kernels not needed for BF16 inference)"""
pat = r'(ext_modules = \[\].*?)(package_data = \{)'
m = re.search(pat, src, re.DOTALL)
assert m, "Could not find ext_modules block"
src = src[:m.start()] + minimal_ext + '\n' + m.group(2) + src[m.end():]
with open("setup.py", "w") as f:
    f.write(src)
print("OK")
PYEOF

# 4) Patch CMakeLists.txt: remove external projects, add nvToolsExt shim
echo "[4/6] Patching CMakeLists.txt..."
mkdir -p cmake
cat > cmake/fix_nvtoolsext.cmake << 'EOF'
if(NOT TARGET CUDA::nvToolsExt)
  find_library(NVTOOLSEXT_LIB nvToolsExt PATHS /usr/local/cuda/lib64 /usr/local/cuda-12.6/targets/x86_64-linux/lib /usr/lib/x86_64-linux-gnu NO_DEFAULT_PATH)
  if(NVTOOLSEXT_LIB)
    add_library(CUDA::nvToolsExt SHARED IMPORTED)
    set_target_properties(CUDA::nvToolsExt PROPERTIES IMPORTED_LOCATION "${NVTOOLSEXT_LIB}")
    message(STATUS "Created CUDA::nvToolsExt shim at ${NVTOOLSEXT_LIB}")
  endif()
endif()
EOF

python3 << 'PYEOF'
with open("CMakeLists.txt") as f:
    cml = f.read()
# Remove external projects
old = """# For CUDA we also build and ship some external projects.
if (VLLM_GPU_LANG STREQUAL "CUDA")
    include(cmake/external_projects/deepgemm.cmake)
    include(cmake/external_projects/flashmla.cmake)
    include(cmake/external_projects/qutlass.cmake)

    # vllm-flash-attn should be last as it overwrites some CMake functions
    include(cmake/external_projects/vllm_flash_attn.cmake)
endif ()"""
cml = cml.replace(old, "# MINIMAL: skipped external projects")
# Inject nvToolsExt shim
cml = cml.replace('find_package(Torch REQUIRED)',
    'find_package(Torch REQUIRED)\n\ninclude(${CMAKE_CURRENT_SOURCE_DIR}/cmake/fix_nvtoolsext.cmake)')
with open("CMakeLists.txt", "w") as f:
    f.write(cml)
# Fix cub::Sum/Min/Max for CUDA 13.x
import glob
for f in glob.glob("csrc/**/*.cu", recursive=True) + glob.glob("csrc/**/*.cuh", recursive=True):
    with open(f) as fh:
        content = fh.read()
    changed = False
    if "cub::Sum{}" in content:
        content = content.replace("cub::Sum{}", "cuda::std::plus<void>()")
        changed = True
    if "cub::Min{}" in content:
        content = content.replace("cub::Min{}", "cuda::minimum<void>()")
        changed = True
    if "cub::Max{}" in content:
        content = content.replace("cub::Max{}", "cuda::maximum<void>()")
        changed = True
    if changed:
        with open(f, "w") as fh:
            fh.write(content)
        print(f"  Fixed cub functors: {f}")
print("OK")
PYEOF

# 5) Build
echo "[5/6] Building (MAX_JOBS=2, TORCH_CUDA_ARCH_LIST=12.0)..."
export MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST="12.0"
export NVCC_THREADS=2
export CMAKE_BUILD_TYPE=Release

mkdir -p build && cd build
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_TARGET_DEVICE=cuda \
  -DVLLM_PYTHON_EXECUTABLE=$(which python) \
  -DNVCC_THREADS=2 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build . -j=2 --target=cumem_allocator --target=_C

# 6) Install
echo "[6/6] Installing..."
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
pip install vllm==0.8.5.post1 --no-deps --force-reinstall 2>&1 | tail -3
cp _C.abi3.so "$SITE_PACKAGES/vllm/_C.abi3.so"
cp cumem_allocator.abi3.so "$SITE_PACKAGES/vllm/cumem_allocator.abi3.so"
echo "Installed rebuilt .so files"

python -c "import vllm; print(f'vLLM {vllm.__version__} OK')" 2>&1
echo "=== Done ==="
