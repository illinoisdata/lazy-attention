#!/bin/bash
source scripts/benchmark/bench_lib.sh

benchmark_setup_pythonpath
benchmark_install_interrupt_handler

if ! benchmark_resolve_matrix_args "$@"; then
    echo "Invalid number of arguments (expected <= 2), $# provided"
    echo 'Example: bash scripts/benchmark/bench_exp4.sh'
    echo 'Example: bash scripts/benchmark/bench_exp4.sh parrot'
    echo 'Example: bash scripts/benchmark/bench_exp4.sh parrot,llmrag randtiny,sqa,narrativeqa'
    exit 1
fi

echo "Running exp4 on [ ${BENCH_SELECTED_SUTS[*]} ] x [ ${BENCH_SELECTED_DATA_KEYS[*]} ]"

for sut in "${BENCH_SELECTED_SUTS[@]}"; do
    for datakey in "${BENCH_SELECTED_DATA_KEYS[@]}"; do
        BENCH_EXTRA_ARGS=()
        benchmark_run_case exp4 "${sut}" "${datakey}" \
            --ablation-reqs 1 \
            --doc-per-request 5 \
            --document-len 10000 \
            --max-concurrency 1
    done
done
