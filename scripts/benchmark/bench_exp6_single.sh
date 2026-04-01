#!/bin/bash
source scripts/benchmark/bench_lib.sh

benchmark_setup_pythonpath
benchmark_install_interrupt_handler

if ! benchmark_require_case_args "$@"; then
    echo "Require 2 argument (SUT, DATANAME), $# provided"
    echo 'Example: bash scripts/benchmark/bench_exp6_single.sh lazyrag ablation'
    exit 1
fi

EXP6_REQS="${EXP6_REQS:-8}"
EXP6_DOCS_PER_REQUEST="${EXP6_DOCS_PER_REQUEST:-5}"
EXP6_DOCUMENT_LEN="${EXP6_DOCUMENT_LEN:-10000}"
EXP6_MAX_CONCURRENCY="${EXP6_MAX_CONCURRENCY:-8}"

benchmark_run_case \
    exp6 \
    "${BENCH_SELECTED_SUT}" \
    "${BENCH_SELECTED_DATANAME}" \
    --ablation-reqs "${EXP6_REQS}" \
    --doc-per-request "${EXP6_DOCS_PER_REQUEST}" \
    --document-len "${EXP6_DOCUMENT_LEN}" \
    --max-concurrency "${EXP6_MAX_CONCURRENCY}"
