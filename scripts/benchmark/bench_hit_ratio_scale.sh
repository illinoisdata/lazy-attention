#!/bin/bash
# RQ2 / Table 1: VRAM document-KV cache hit ratio vs KV cache budget.
#
# Varies --gpu-memory-utilization (the KV budget knob) under a SKEWED (Zipfian,
# alpha=2.1 by default, matching the paper) RAG workload over a fixed document
# pool. The lazy doc-KV hit ratio is emitted by LazyKVCacheManager as
# "[LAZY_DOC_KV_HIT] hits=.. queries=.. ratio=.." in the job log; the last such
# line per budget is the run-aggregate hit ratio. Parse with
# scripts/benchmark/extract_hit_ratio.sh.
#
# Env knobs (defaults in []):
#   HIT_SUT [lazyrag]   GPU_UTIL_LIST [0.30,0.40,0.50,0.60,0.70,0.80,0.90]
#   HIT_REQUESTS [200]  HIT_MAX_CONCURRENCY [16]  HIT_DOC_LEN [2048]
#   HIT_NUM_DOCS [100] (pool)  HIT_DOCS_PER_PROMPT [5]  HIT_ZIPF [2.1]
source scripts/benchmark/bench_lib.sh
benchmark_setup_pythonpath
benchmark_install_interrupt_handler

HIT_SUT="${HIT_SUT:-lazyrag}"
GPU_UTIL_LIST="${GPU_UTIL_LIST:-0.30,0.40,0.50,0.60,0.70,0.80,0.90}"
HIT_REQUESTS="${HIT_REQUESTS:-200}"
HIT_MAX_CONCURRENCY="${HIT_MAX_CONCURRENCY:-16}"
HIT_DOC_LEN="${HIT_DOC_LEN:-2048}"
HIT_NUM_DOCS="${HIT_NUM_DOCS:-100}"
HIT_DOCS_PER_PROMPT="${HIT_DOCS_PER_PROMPT:-5}"
HIT_ZIPF="${HIT_ZIPF:-2.1}"

prepare_sut "${HIT_SUT}"
make_sut_args "${HIT_SUT}" sut_args
read -r -a sut_cli_args <<< "${sut_args}"

benchmark_split_csv "${GPU_UTIL_LIST}" GPU_UTILS
echo "Hit-ratio KV-budget sweep: sut=${HIT_SUT} utils=${GPU_UTIL_LIST}"
echo "  reqs=${HIT_REQUESTS} concurrency=${HIT_MAX_CONCURRENCY} doc_len=${HIT_DOC_LEN}"
echo "  pool=${HIT_NUM_DOCS} docs_per_prompt=${HIT_DOCS_PER_PROMPT} zipf=${HIT_ZIPF}"
for util in "${GPU_UTILS[@]}"; do
    echo "@@@@ HIT_RATIO_SWEEP gpu_util=${util} sut=${HIT_SUT} @@@@"
    python benchmarks/benchmark_rag_serving.py \
        --exp "hit_ratio_u${util}_${HIT_SUT}" \
        --dataset-name random \
        --sample-requests "${HIT_REQUESTS}" \
        --random-num-prompts "${HIT_REQUESTS}" \
        --random-input-len 16 \
        --random-output-len 8 \
        --random-document-len "${HIT_DOC_LEN}" \
        --random-num-documents "${HIT_NUM_DOCS}" \
        --random-num-documents-per-prompt "${HIT_DOCS_PER_PROMPT}" \
        --document-sampling-method zipf --document-zipf-param "${HIT_ZIPF}" \
        --max-concurrency "${HIT_MAX_CONCURRENCY}" \
        "${sut_cli_args[@]}" \
        --gpu-memory-utilization "${util}"
done
echo "HIT_RATIO_SWEEP_DONE sut=${HIT_SUT}"
