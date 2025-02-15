# Dev Guide with cuda and C code

This guide is about how to quickly compile your change to Cuda and C code.

## step 1: Create your branch from hanxi/install
Hanxi disabled some irrelevant extensions like flash_attn, making the compilation quicker.

## step 2: Install build tools into your virtual environment
``
conda install ccache
``

``
pip install -r requirements-build.txt
``

## step 3: Create a dir for build result and enter that dir
``
mkdir build && cd build
``

## step 4: Configure C make
Depending on your local virtual env, different people have different command for this step. Hanxi's command is:

``
cmake .. -G Ninja -DCMAKE_INSTALL_PREFIX=.. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVLLM_TARGET_DEVICE=cuda -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DCMAKE_HIP_COMPILER_LAUNCHER=ccache -DVLLM_PYTHON_EXECUTABLE=/u/hfang4/miniconda3/envs/vllm-env1/bin/python -DVLLM_PYTHON_PATH=:/u/hfang4/miniconda3/envs/vllm-env1/lib/python39.zip:/u/hfang4/miniconda3/envs/vllm-env1/lib/python3.9:/u/hfang4/miniconda3/envs/vllm-env1/lib/python3.9/lib-dynload:/u/hfang4/miniconda3/envs/vllm-env1/lib/python3.9/site-packages -DFETCHCONTENT_BASE_DIR=/u/hfang4/research/DynamicRAG/.deps -DNVCC_THREADS=1 -DCMAKE_JOB_POOL_COMPILE:STRING=compile -DCMAKE_JOB_POOLS:STRING=compile=1
``


One way to check your own command is to run:
``
pip install --verbose -e .
``
You don't need to wait until it's done, which may take about 40 mins. Instead, You should be able to see the right command to use from the terminal very soon by **searching the terminal output with keyword "Print configure command"**
But note that you'll have to specify ``-DCMAKE_INSTALL_PREFIX=..`` manually because setup.py didn't do that.

If you're using vscode's cmake extension, you need to set "-DCMAKE_MAKE_PROGRAM=/u/hfang4/miniconda3/envs/vllm-env1/bin/ninja" additionally; And add "C_Cpp.default.compileCommands": "${workspaceFolder}/build/compile_commands.json" to settings.json

## step 5: cmake build only the component we need!
``
cmake --build . --target _C
``

## step 6: install (i.e.,move) the built .so files to the right position!
``
cmake --install . --component _C
``

## step 7: install python package without re-installing any C extension
``
NO_C=1 pip install -e . --no-build-isolation
``

Note that you only need to run step 5 - 6 in the future each time you change your code. Step 1 to 4 only needs to be run once in your virtual environment's lifetime! 


## pytest dynamic attention
cuda unit test:
``
pytest -v -s tests/kernels/test_attention.py::test_dynamic_paged_attention
``
integration test:
``
pytest tests/test_decode.py
``

