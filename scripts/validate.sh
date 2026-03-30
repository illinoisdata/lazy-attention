#!/bin/bash
source scripts/benchmark/bench_lib.sh

benchmark_setup_pythonpath
benchmark_install_interrupt_handler

if ! benchmark_resolve_matrix_args "$@"; then
    echo "Invalid number of arguments (expected <= 2), $# provided"
    echo 'Example: bash scripts/benchmark/validate.sh'
    echo 'Example: bash scripts/benchmark/validate.sh parrot'
    echo 'Example: bash scripts/benchmark/validate.sh parrot,llmrag randtiny,sqa,narrativeqa'
    exit 1
fi

echo "Running validate on [ ${BENCH_SELECTED_SUTS[*]} ] x [ ${BENCH_SELECTED_DATA_KEYS[*]} ]"

BENCH_SELECTED_SUT="${BENCH_SELECTED_SUTS[0]}"
BENCH_SELECTED_DATANAME="${BENCH_SELECTED_DATA_KEYS[0]}"
BENCH_EXTRA_ARGS=()

benchmark_run_case \
    exp4 \
    "${BENCH_SELECTED_SUT}" \
    "${BENCH_SELECTED_DATANAME}" \
    --ablation-reqs 1 \
    --doc-per-request 5 \
    --document-len 16 \
    --max-concurrency 1

benchmark_run_case \
    exp4 \
    "${BENCH_SELECTED_SUT}" \
    "${BENCH_SELECTED_DATANAME}" \
    --ablation-reqs 1 \
    --doc-per-request 5 \
    --document-len 30 \
    --max-concurrency 1
