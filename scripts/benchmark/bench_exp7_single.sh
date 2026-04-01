#!/bin/bash
source scripts/benchmark/bench_lib.sh

benchmark_setup_pythonpath
benchmark_install_interrupt_handler

if ! benchmark_require_case_args "$@"; then
    echo "Require 2 argument (SUT, DATANAME), $# provided"
    echo 'Example: bash scripts/benchmark/bench_exp7_single.sh lazyrag ablation'
    exit 1
fi

EXP7_REQS="${EXP7_REQS:-1}"
EXP7_DOCS_PER_REQUEST="${EXP7_DOCS_PER_REQUEST:-1}"
EXP7_DOCUMENT_LEN="${EXP7_DOCUMENT_LEN:-32768}"
EXP7_MAX_CONCURRENCY="${EXP7_MAX_CONCURRENCY:-1}"
EXP7_OUTPUT_LEN="${EXP7_OUTPUT_LEN:-512}"

benchmark_run_case \
    exp7 \
    "${BENCH_SELECTED_SUT}" \
    "${BENCH_SELECTED_DATANAME}" \
    --ablation-reqs "${EXP7_REQS}" \
    --doc-per-request "${EXP7_DOCS_PER_REQUEST}" \
    --document-len "${EXP7_DOCUMENT_LEN}" \
    --max-concurrency "${EXP7_MAX_CONCURRENCY}" \
    --ablation-output-len "${EXP7_OUTPUT_LEN}"
