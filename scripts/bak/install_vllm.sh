# ----------------------------------------------------------------
# For delta platform
# ----------------------------------------------------------------
module load cuda/12.4.0
# Make sure the exact matched version is installed
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# ----------------------------------------------------------------

pushd vllm_proj  # leave the root
rm -rf build
mkdir build
pushd build

# Get config
cmake .. \
    -G Ninja \
    -DCMAKE_INSTALL_PREFIX=.. \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DVLLM_TARGET_DEVICE=cuda \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -DCMAKE_HIP_COMPILER_LAUNCHER=ccache \
    -DVLLM_PYTHON_EXECUTABLE=$(which python) \
    -DVLLM_PYTHON_PATH=$(python -c "import sys; print(':'.join(sys.path))") \
    -DFETCHCONTENT_BASE_DIR=$(pwd)/../.deps \
    -DNVCC_THREADS=16 \
    -DCMAKE_JOB_POOL_COMPILE:STRING=compile \
    -DCMAKE_JOB_POOLS:STRING=compile=16

# Build
cmake --build . --target _C _vllm_fa2_C  # fa3 easy to OOM

# Install
cmake --install . --component _C
cmake --install . --component _vllm_fa2_C

popd
cp -r vllm_flash_attn ./vllm/
rm -rf vllm_flash_attn

pip install setuptools==77.0.3
NO_C=1 pip install -e . --no-build-isolation
popd  # back the root