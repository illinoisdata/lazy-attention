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

# Flash
    # Failed to run
