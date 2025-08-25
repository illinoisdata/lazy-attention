module load cuda/12.4.0
module load gcc/11.4.0
# /sw/spack/deltas11-2023-03/apps/linux-rhel8-x86_64/gcc-8.5.0/gcc-11.4.0-yycklku/
# /sw/spack/deltas11-2023-03/apps/linux-rhel8-x86_64/gcc-8.5.0/gcc-13.2.0-blv4b5f/lib64/libstdc++.so.6
# conda install -c conda-forge libstdcxx-ng
# cd ~/.conda/envs/vllm/lib
# rm -f libstdc++.so.6
# ln -s libstdc++.so.6.0.34 libstdc++.so.6
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sw/spack/deltas11-2023-03/apps/linux-rhel8-x86_64/gcc-8.5.0/gcc-13.2.0-blv4b5f/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sw/spack/deltas11-2023-03/apps/linux-rhel8-x86_64/gcc-8.5.0/gcc-11.4.0-yycklku/lib64:$LD_LIBRARY_PATH
conda install pandas datasets numpy -y

if [ ! -f ShareGPT_V3_unfiltered_cleaned_split.json ]; then
    wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
fi

# VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --swap-space 16 \
    --disable-log-requests &

# Wait for the server to start
sleep 100

python vllm/benchmarks/benchmark_serving.py \
    --model meta-llama/Llama-3.1-8B-Instruct  \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json

# ----------------------------------------------------------------
# A40
# ----------------------------------------------------------------
# Triton
    # ============ Serving Benchmark Result ============
    # Successful requests:                     1000      
    # Benchmark duration (s):                  105.98    
    # Total input tokens:                      215196    
    # Total generated tokens:                  197090    
    # Request throughput (req/s):              9.44      
    # Output token throughput (tok/s):         1859.76   
    # Total Token throughput (tok/s):          3890.36   
    # ---------------Time to First Token----------------
    # Mean TTFT (ms):                          29614.19  
    # Median TTFT (ms):                        27353.40  
    # P99 TTFT (ms):                           69900.03  
    # -----Time per Output Token (excl. 1st token)------
    # Mean TPOT (ms):                          125.54    
    # Median TPOT (ms):                        114.87    
    # P99 TPOT (ms):                           304.13    
    # ---------------Inter-token Latency----------------
    # Mean ITL (ms):                           102.79    
    # Median ITL (ms):                         77.98     
    # P99 ITL (ms):                            306.77    
    # ==================================================

# ---------------------------------------------------------------
# A100
# ---------------------------------------------------------------
# Default
============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  57.83
Total input tokens:                      215196
Total generated tokens:                  197328
Request throughput (req/s):              17.29
Output token throughput (tok/s):         3411.94
Total Token throughput (tok/s):          7132.82
---------------Time to First Token----------------
Mean TTFT (ms):                          16452.24
Median TTFT (ms):                        14621.18
P99 TTFT (ms):                           40214.49
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          69.51
Median TPOT (ms):                        67.72
P99 TPOT (ms):                           145.25
---------------Inter-token Latency----------------
Mean ITL (ms):                           58.03
Median ITL (ms):                         45.67
P99 ITL (ms):                            148.74
==================================================
# Flash
    # Failed to run
