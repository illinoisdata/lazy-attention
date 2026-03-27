#!/bin/bash
export PYTHONPATH=.:./promptcache:./lazy_attn:./block_attn

int_handler() {
    echo "Interrupted."
    kill $PPID
    exit 1
}
trap 'int_handler' INT
source scripts/benchmark/bench_consts.sh

if [ "$#" -eq 0 ]
then
    suts=("${SUTS[@]}")
    data_keys=("${DATA_KEYS[@]}")
elif [ "$#" -eq 1 ]
then
    IFS=',' read -ra suts <<< "$1"
    data_keys=("${DATA_KEYS[@]}")
elif [ "$#" -eq 2 ]
then
    IFS=',' read -ra suts <<< "$1"
    IFS=',' read -ra data_keys <<< "$2"
else
    echo "Invalid number of arguments (expected <= 2), $# provided"
    echo 'Example: bash scripts/benchmark/validate.sh'
    echo 'Example: bash scripts/benchmark/validate.sh parrot'
    echo 'Example: bash scripts/benchmark/validate.sh parrot,llmrag randtiny,sqa,narrativeqa'
    exit 1
fi
echo "Running test on [ ${suts[*]} ] x [ ${data_keys[*]} ]"

SUT=${suts[0]}
DATANAME=${data_keys[0]}
# consume up to two positional args (safe when 0 or 1 args were passed)
if [ "$#" -ge 2 ]; then
    shift 2
else
    # shift by the remaining count (0 or 1) to clear positional parameters
    shift "$#"
fi
EXTRA_ARGS="$@"
make_sut_args ${SUT} sut_args
make_data_args ${DATANAME} datakey dataargs

echo "Using SUT=${SUT}, DATANAME=${DATANAME}"
echo "      sut_args=\"${sut_args}\""
echo "      datakey=${datakey}, dataargs=\"${dataargs}\", EXTRA_ARGS=\"${EXTRA_ARGS}\""
sleep 2

prepare_sut ${SUT}
python benchmarks/benchmark_rag_serving.py \
    --exp exp4_${SUT}_${DATANAME} \
    --ablation-reqs 1 \
    --doc-per-request 5 \
    --document-len 16 \
    --max-concurrency 1 \
    ${dataargs} \
    ${sut_args} \
    ${EXTRA_ARGS}

python benchmarks/benchmark_rag_serving.py \
    --exp exp4_${SUT}_${DATANAME} \
    --ablation-reqs 1 \
    --doc-per-request 5 \
    --document-len 30 \
    --max-concurrency 1 \
    ${dataargs} \
    ${sut_args} \
    ${EXTRA_ARGS}