#!/bin/bash
# RQ1 / Figure 4: mean TTFT vs request rate under different document-popularity
# traffic. Documents' KV are precomputed/cached; each request queries
# RATE_DOCS_PER_PROMPT docs sampled either UNIFORMLY (low reuse, Fig 4a) or with
# a Zipfian skew (high reuse, Fig 4b). We sweep --request-rate (open-loop Poisson
# arrivals) and record mean TTFT per rate, holding the memory budget fixed so all
# methods compete under identical paging/budget conditions.
#
# Env knobs (defaults in []):
#   RATE_SUT [lazyrag]        method/SUT (lazyrag|llmrag|baseline|blockattnrag|...)
#   TRAFFIC  [skewed]         uniform | skewed
#   RATE_LIST [0.5,1,2,3,4]   request rates (req/s) to sweep
#   RATE_REQUESTS [120]       requests per rate point
#   RATE_DOC_LEN [2048]  RATE_NUM_DOCS [100] (pool)  RATE_DOCS_PER_PROMPT [5]
#   RATE_INPUT_LEN [32]  RATE_OUTPUT_LEN [16]  RATE_ZIPF [2.1]
#   RATE_GPU_UTIL [0.9]       common KV budget for all methods
#   RATE_EXTRA_ARGS []        e.g. "--enforce-eager" for local sm_120
source scripts/benchmark/bench_lib.sh
benchmark_setup_pythonpath
benchmark_install_interrupt_handler

RATE_SUT="${RATE_SUT:-lazyrag}"
# Result-file label, defaults to the SUT. Override so variants of the same SUT
# don't collide (e.g. MEPIC runs as RATE_SUT=lazyrag + LAZY_ATTENTION_VARIANT=mepic,
# so pass RATE_LABEL=mepic to name results ttft_rate_*_mepic.json).
RATE_LABEL="${RATE_LABEL:-${RATE_SUT}}"
TRAFFIC="${TRAFFIC:-skewed}"
RATE_LIST="${RATE_LIST:-0.5,1,2,3,4}"
RATE_REQUESTS="${RATE_REQUESTS:-120}"
RATE_DOC_LEN="${RATE_DOC_LEN:-2048}"
RATE_NUM_DOCS="${RATE_NUM_DOCS:-100}"
RATE_DOCS_PER_PROMPT="${RATE_DOCS_PER_PROMPT:-5}"
RATE_INPUT_LEN="${RATE_INPUT_LEN:-32}"
RATE_OUTPUT_LEN="${RATE_OUTPUT_LEN:-16}"
RATE_ZIPF="${RATE_ZIPF:-2.1}"
RATE_GPU_UTIL="${RATE_GPU_UTIL:-0.9}"

case "${TRAFFIC}" in
    uniform) doc_sampling=(--document-sampling-method uniform) ;;
    skewed)  doc_sampling=(--document-sampling-method zipf --document-zipf-param "${RATE_ZIPF}") ;;
    *) echo "ERROR: TRAFFIC must be uniform|skewed, got '${TRAFFIC}'"; exit 1 ;;
esac

prepare_sut "${RATE_SUT}"
make_sut_args "${RATE_SUT}" sut_args
read -r -a sut_cli_args <<< "${sut_args}"
read -r -a rate_extra_args <<< "${RATE_EXTRA_ARGS:-}"

benchmark_split_csv "${RATE_LIST}" RATES
echo "TTFT-vs-rate sweep: sut=${RATE_SUT} traffic=${TRAFFIC} rates=${RATE_LIST}"
echo "  reqs/point=${RATE_REQUESTS} doc_len=${RATE_DOC_LEN} pool=${RATE_NUM_DOCS}"
echo "  docs/prompt=${RATE_DOCS_PER_PROMPT} in=${RATE_INPUT_LEN} out=${RATE_OUTPUT_LEN} util=${RATE_GPU_UTIL}"
for rate in "${RATES[@]}"; do
    echo "@@@@ TTFT_RATE_SWEEP rate=${rate} traffic=${TRAFFIC} sut=${RATE_SUT} label=${RATE_LABEL} @@@@"
    python benchmarks/benchmark_rag_serving.py \
        --exp "ttft_rate_r${rate}_${TRAFFIC}_${RATE_LABEL}" \
        --dataset-name random \
        --sample-requests "${RATE_REQUESTS}" \
        --random-num-prompts "${RATE_REQUESTS}" \
        --random-input-len "${RATE_INPUT_LEN}" \
        --random-output-len "${RATE_OUTPUT_LEN}" \
        --random-document-len "${RATE_DOC_LEN}" \
        --random-num-documents "${RATE_NUM_DOCS}" \
        --random-num-documents-per-prompt "${RATE_DOCS_PER_PROMPT}" \
        "${doc_sampling[@]}" \
        --request-rate "${rate}" \
        "${sut_cli_args[@]}" \
        --gpu-memory-utilization "${RATE_GPU_UTIL}" \
        "${rate_extra_args[@]}"
done
echo "TTFT_RATE_SWEEP_DONE sut=${RATE_SUT} label=${RATE_LABEL} traffic=${TRAFFIC}"
