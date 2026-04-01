#!/bin/bash
source scripts/benchmark/bench_lib.sh

benchmark_setup_pythonpath
benchmark_install_interrupt_handler

SWEEP_OUTPUT_LEN="${SWEEP_OUTPUT_LEN:-128}"
SWEEP_SCENARIOS="${SWEEP_SCENARIOS:-light_single,light_async,medium_async,exp4_like,exp6_like,exp7_like,exp7_extreme}"
SWEEP_VARIANTS="${SWEEP_VARIANTS:-lazy,mepic0,mepic1}"
SWEEP_SUMMARY_PATH="${SWEEP_SUMMARY_PATH:-results/lazy_mepic_sweep_summary.tsv}"

mkdir -p "$(dirname "${SWEEP_SUMMARY_PATH}")"
: > "${SWEEP_SUMMARY_PATH}"
printf 'scenario\tvariant\treqs\tdocs_per_request\tdocument_len\tmax_concurrency\toutput_len\n' >> "${SWEEP_SUMMARY_PATH}"

run_variant() {
    local scenario="$1"
    local variant="$2"
    local reqs="$3"
    local docs_per_request="$4"
    local document_len="$5"
    local max_concurrency="$6"
    local output_len="$7"

    local exp_name="${scenario}_${variant}"

    case "${variant}" in
        lazy)
            export LAZY_ATTENTION_VARIANT="lazy"
            export LAZY_FORCE_SPLIT_DECODE="0"
            unset MEPIC_FORCE_FP32_ROTARY
            ;;
        lazy_split)
            export LAZY_ATTENTION_VARIANT="lazy"
            export LAZY_FORCE_SPLIT_DECODE="1"
            unset MEPIC_FORCE_FP32_ROTARY
            ;;
        mepic0)
            export LAZY_ATTENTION_VARIANT="mepic"
            export LAZY_FORCE_SPLIT_DECODE="0"
            export MEPIC_FORCE_FP32_ROTARY="0"
            ;;
        mepic1)
            export LAZY_ATTENTION_VARIANT="mepic"
            export LAZY_FORCE_SPLIT_DECODE="0"
            export MEPIC_FORCE_FP32_ROTARY="1"
            ;;
        *)
            echo "Unknown variant: ${variant}" >&2
            return 1
            ;;
    esac

    echo "==== Sweep scenario=${scenario} variant=${variant} reqs=${reqs} docs_per_request=${docs_per_request} document_len=${document_len} max_concurrency=${max_concurrency} output_len=${output_len} LAZY_ATTENTION_VARIANT=${LAZY_ATTENTION_VARIANT} LAZY_FORCE_SPLIT_DECODE=${LAZY_FORCE_SPLIT_DECODE:-0} MEPIC_FORCE_FP32_ROTARY=${MEPIC_FORCE_FP32_ROTARY:-unset} ===="

    BENCH_EXTRA_ARGS=()
    benchmark_run_case \
        "${exp_name}" \
        lazyrag \
        ablation \
        --ablation-reqs "${reqs}" \
        --doc-per-request "${docs_per_request}" \
        --document-len "${document_len}" \
        --max-concurrency "${max_concurrency}" \
        --ablation-output-len "${output_len}"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${scenario}" "${variant}" "${reqs}" "${docs_per_request}" "${document_len}" "${max_concurrency}" "${output_len}" \
        >> "${SWEEP_SUMMARY_PATH}"
}

benchmark_split_csv "${SWEEP_SCENARIOS}" SWEEP_SCENARIO_LIST
benchmark_split_csv "${SWEEP_VARIANTS}" SWEEP_VARIANT_LIST

echo "Running lazy/mepic sweep"
echo "Scenarios: ${SWEEP_SCENARIO_LIST[*]}"
echo "Variants: ${SWEEP_VARIANT_LIST[*]}"
echo "Summary path: ${SWEEP_SUMMARY_PATH}"

for scenario in "${SWEEP_SCENARIO_LIST[@]}"; do
    case "${scenario}" in
        light_single)
            reqs="1"
            docs_per_request="1"
            document_len="2048"
            max_concurrency="1"
            output_len="64"
            ;;
        light_async)
            reqs="8"
            docs_per_request="2"
            document_len="2048"
            max_concurrency="8"
            output_len="64"
            ;;
        medium_async)
            reqs="8"
            docs_per_request="2"
            document_len="4096"
            max_concurrency="8"
            output_len="128"
            ;;
        exp4_like)
            reqs="1"
            docs_per_request="5"
            document_len="10000"
            max_concurrency="1"
            output_len="128"
            ;;
        exp6_like)
            reqs="8"
            docs_per_request="5"
            document_len="10000"
            max_concurrency="8"
            output_len="128"
            ;;
        exp7_like)
            reqs="1"
            docs_per_request="1"
            document_len="32768"
            max_concurrency="1"
            output_len="512"
            ;;
        exp7_extreme)
            reqs="1"
            docs_per_request="1"
            document_len="65536"
            max_concurrency="1"
            output_len="1024"
            ;;
        *)
            echo "Unknown scenario: ${scenario}" >&2
            exit 1
            ;;
    esac

    for variant in "${SWEEP_VARIANT_LIST[@]}"; do
        run_variant "${scenario}" "${variant}" "${reqs}" "${docs_per_request}" "${document_len}" "${max_concurrency}" "${output_len}"
    done
done
