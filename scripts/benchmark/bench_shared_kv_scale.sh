#!/bin/bash
source scripts/benchmark/bench_lib.sh

benchmark_setup_pythonpath
benchmark_install_interrupt_handler

SHARED_KV_SUT="${SHARED_KV_SUT:-lazyrag}"
SHARED_KV_EXP_PREFIX="${SHARED_KV_EXP_PREFIX:-shared_kv_scale}"
SHARED_KV_CONCURRENCY_LIST="${SHARED_KV_CONCURRENCY_LIST:-1,8,32,128,256,512,1000}"
SHARED_KV_INPUT_LEN="${SHARED_KV_INPUT_LEN:-1}"
SHARED_KV_OUTPUT_LEN="${SHARED_KV_OUTPUT_LEN:-64}"
SHARED_KV_DOCUMENT_LEN="${SHARED_KV_DOCUMENT_LEN:-16384}"
SHARED_KV_NUM_DOCUMENTS="${SHARED_KV_NUM_DOCUMENTS:-1}"
SHARED_KV_DOCS_PER_PROMPT="${SHARED_KV_DOCS_PER_PROMPT:-1}"
SHARED_KV_DISABLE_CASCADE_ATTN="${SHARED_KV_DISABLE_CASCADE_ATTN:-1}"
SHARED_KV_MAX_NUM_SEQS_MULTIPLIER="${SHARED_KV_MAX_NUM_SEQS_MULTIPLIER:-2}"
SHARED_KV_MAX_NUM_SEQS_MIN="${SHARED_KV_MAX_NUM_SEQS_MIN:-64}"
SHARED_KV_MAX_NUM_BATCHED_TOKENS="${SHARED_KV_MAX_NUM_BATCHED_TOKENS:-8192}"

benchmark_split_csv "${SHARED_KV_CONCURRENCY_LIST}" SHARED_KV_CONCURRENCIES

echo "Running shared-KV scaling sweep with LAZY_ATTENTION_VARIANT=${LAZY_ATTENTION_VARIANT:-lazy}"
echo "  SUT=${SHARED_KV_SUT}"
echo "  CONCURRENCY_LIST=${SHARED_KV_CONCURRENCY_LIST}"
echo "  INPUT_LEN=${SHARED_KV_INPUT_LEN}"
echo "  OUTPUT_LEN=${SHARED_KV_OUTPUT_LEN}"
echo "  DOCUMENT_LEN=${SHARED_KV_DOCUMENT_LEN}"
echo "  NUM_DOCUMENTS=${SHARED_KV_NUM_DOCUMENTS}"
echo "  DOCS_PER_PROMPT=${SHARED_KV_DOCS_PER_PROMPT}"
echo "  DISABLE_CASCADE_ATTN=${SHARED_KV_DISABLE_CASCADE_ATTN}"
echo "  MAX_NUM_SEQS_MULTIPLIER=${SHARED_KV_MAX_NUM_SEQS_MULTIPLIER}"
echo "  MAX_NUM_SEQS_MIN=${SHARED_KV_MAX_NUM_SEQS_MIN}"
echo "  MAX_NUM_BATCHED_TOKENS=${SHARED_KV_MAX_NUM_BATCHED_TOKENS}"
echo "  Note: set NUM_DOCUMENTS=1 and DOCS_PER_PROMPT=1 to force all logical requests to share one document."

for concurrency in "${SHARED_KV_CONCURRENCIES[@]}"; do
    max_num_seqs=$(( concurrency * SHARED_KV_MAX_NUM_SEQS_MULTIPLIER ))
    if [ "${max_num_seqs}" -lt "${SHARED_KV_MAX_NUM_SEQS_MIN}" ]; then
        max_num_seqs="${SHARED_KV_MAX_NUM_SEQS_MIN}"
    fi

    echo
    echo "=== shared_kv concurrency=${concurrency} ==="
    BENCH_EXTRA_ARGS=(
        --dataset-name random
        --random-num-prompts "${concurrency}"
        --max-concurrency "${concurrency}"
        --max-num-seqs "${max_num_seqs}"
        --max-num-batched-tokens "${SHARED_KV_MAX_NUM_BATCHED_TOKENS}"
        --random-input-len "${SHARED_KV_INPUT_LEN}"
        --random-output-len "${SHARED_KV_OUTPUT_LEN}"
        --random-document-len "${SHARED_KV_DOCUMENT_LEN}"
        --random-num-documents "${SHARED_KV_NUM_DOCUMENTS}"
        --random-num-documents-per-prompt "${SHARED_KV_DOCS_PER_PROMPT}"
        --disable-tqdm
        --metadata "shared_kv=1" "logical_requests=${concurrency}" "shared_docs=${SHARED_KV_NUM_DOCUMENTS}"
    )
    if [ "${SHARED_KV_DISABLE_CASCADE_ATTN}" = "1" ]; then
        BENCH_EXTRA_ARGS+=(--disable-cascade-attn)
    fi
    benchmark_run_case "${SHARED_KV_EXP_PREFIX}_c${concurrency}" "${SHARED_KV_SUT}" randtiny
done
