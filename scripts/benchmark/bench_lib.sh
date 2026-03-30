#!/bin/bash

BENCH_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BENCH_LIB_DIR}/bench_consts.sh"

benchmark_setup_pythonpath() {
    export PYTHONPATH=.:./promptcache:./lazy_attn:./block_attn
}

benchmark_install_interrupt_handler() {
    int_handler() {
        echo "Interrupted."
        kill "$PPID"
        exit 1
    }
    trap 'int_handler' INT
}

benchmark_split_csv() {
    local raw_value="$1"
    local target_name="$2"
    local -a parsed_values=()
    IFS=',' read -r -a parsed_values <<< "${raw_value}"
    eval "${target_name}=(\"\${parsed_values[@]}\")"
}

benchmark_resolve_matrix_args() {
    if [ "$#" -eq 0 ]; then
        BENCH_SELECTED_SUTS=("${SUTS[@]}")
        BENCH_SELECTED_DATA_KEYS=("${DATA_KEYS[@]}")
        return 0
    fi

    if [ "$#" -eq 1 ]; then
        benchmark_split_csv "$1" BENCH_SELECTED_SUTS
        BENCH_SELECTED_DATA_KEYS=("${DATA_KEYS[@]}")
        return 0
    fi

    if [ "$#" -eq 2 ]; then
        benchmark_split_csv "$1" BENCH_SELECTED_SUTS
        benchmark_split_csv "$2" BENCH_SELECTED_DATA_KEYS
        return 0
    fi

    return 1
}

benchmark_require_case_args() {
    if [ "$#" -lt 2 ]; then
        return 1
    fi

    BENCH_SELECTED_SUT="$1"
    BENCH_SELECTED_DATANAME="$2"
    shift 2
    BENCH_EXTRA_ARGS=("$@")
    return 0
}

benchmark_print_case() {
    local sut="$1"
    local dataname="$2"
    local sut_args="$3"
    local datakey="$4"
    local dataargs="$5"
    shift 5
    local -a extra_args=("$@")

    echo "Using SUT=${sut}, DATANAME=${dataname}"
    echo "      sut_args=\"${sut_args}\""
    echo "      datakey=${datakey}, dataargs=\"${dataargs}\", EXTRA_ARGS=\"${extra_args[*]}\""
}

benchmark_run_case() {
    local exp_name="$1"
    local sut="$2"
    local dataname="$3"
    shift 3

    local -a profile_args=("$@")
    local sut_args=""
    local datakey=""
    local dataargs=""
    local -a sut_cli_args=()
    local -a data_cli_args=()
    local -a command=()

    make_sut_args "${sut}" sut_args
    make_data_args "${dataname}" datakey dataargs
    benchmark_print_case "${sut}" "${dataname}" "${sut_args}" "${datakey}" "${dataargs}" "${BENCH_EXTRA_ARGS[@]}"

    read -r -a sut_cli_args <<< "${sut_args}"
    read -r -a data_cli_args <<< "${dataargs}"

    prepare_sut "${sut}"

    command=(
        python benchmarks/benchmark_rag_serving.py
        --exp "${exp_name}_${sut}_${datakey}"
        "${profile_args[@]}"
        "${data_cli_args[@]}"
        "${sut_cli_args[@]}"
        "${BENCH_EXTRA_ARGS[@]}"
    )
    "${command[@]}"
}
