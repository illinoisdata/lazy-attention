#!/bin/bash
set -eo pipefail

# Check if vllm env exists, if not create one
ENV_NAME="vllm"
if conda env list | grep " ${ENV_NAME} " >/dev/null 2>&1; then
    echo "Conda environment '${ENV_NAME}' already exists."
else
    echo "Conda environment '${ENV_NAME}' not found. Creating it now..."
    conda create -n ${ENV_NAME} python=3.10 -y

    if [ $? -eq 0 ]; then
        echo " Conda environment '${ENV_NAME}' created successfully."
    else
        echo "Error: Failed to create Conda environment '${ENV_NAME}'. Exiting."
        exit 1
    fi
fi
conda activate vllm

# ----------------------------------------------------------------
# Checkout to 0.8.5.post1
# ----------------------------------------------------------------
rm -rf vllm
git clone https://github.com/vllm-project/vllm.git
pushd vllm
git checkout 3015d5634e74d59704e2b39bab0dbe2e6f86a38a

# ----------------------------------------------------------------
# Check ccache, install it if not found
# ----------------------------------------------------------------
if ! command -v ccache &> /dev/null; then
    echo "ccache could not be found, installing it via conda..."
    conda install -c conda-forge ccache -y
fi

# ----------------------------------------------------------------
# Apply changes
# ----------------------------------------------------------------
cp ../setup.py .  # to override the setup.py in vllm

# ----------------------------------------------------------------
# For delta platform
# ----------------------------------------------------------------
module load gcc/11.4.0
module load cuda/12.4.0
module load gcc/11.4.0

# Make sure the exact matched version is installed
pip install -r requirements/build.txt
# ----------------------------------------------------------------

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
    -DCMAKE_JOB_POOL_COMPILE:STRING=compile \
    -DCMAKE_JOB_POOLS:STRING=compile=16 \
    -DCUDA_TOOLKIT_ROOT_DIR=/sw/user/cudatoolkits/installs/cuda-12.4.0

# Build
cmake --build . --target _C _vllm_fa2_C -- -j 4 # fa3 easy to OOM

# Install
cmake --install . --component _C
cmake --install . --component _vllm_fa2_C

popd
cp -r vllm_flash_attn ./vllm/
rm -rf vllm_flash_attn

pip install setuptools==77.0.3
NO_C=1 pip install -e . --no-build-isolation

popd

# Check
python -c "import vllm"
