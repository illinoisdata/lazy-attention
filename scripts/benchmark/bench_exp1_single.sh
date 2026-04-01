#!/bin/bash
source scripts/benchmark/bench_lib.sh

benchmark_setup_pythonpath
benchmark_install_interrupt_handler

if ! benchmark_require_case_args "$@"; then
    echo "Require 2 argument (SUT, DATANAME), $# provided"
    echo 'Example: bash scripts/benchmark/bench_exp1_single.sh parrot randtiny'
    echo 'Example: bash scripts/benchmark/bench_exp1_single.sh llmrag sqa'
    exit 1
fi

echo "Running exp1 single with LAZY_ATTENTION_VARIANT=${LAZY_ATTENTION_VARIANT:-lazy}"

benchmark_run_case \
    exp1 \
    "${BENCH_SELECTED_SUT}" \
    "${BENCH_SELECTED_DATANAME}" \
    --sample-requests 100 \
    --max-concurrency 5
