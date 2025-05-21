#/!bin/bash
export PYTHONPATH=.:./promptcache
int_handler() {
    echo "Interrupted."
    kill $PPID
    exit 1
}
trap 'int_handler' INT
source scripts/bench_consts.sh

if [ "$#" -lt 2 ]
then
    echo "Require 2 argument (SUT, DATANAME), $# provided"
    echo 'Example: bash scripts/bench_exp1_single.sh parrot randtiny'
    echo 'Example: bash scripts/bench_exp1_single.sh llmrag sqa'
    exit 1
fi

SUT=$1
DATANAME=$2
shift 2
EXTRA_ARGS="$@"
make_sut_args ${SUT} sut_args
make_data_args ${DATANAME} datakey dataargs

echo "Using SUT=${SUT}, DATANAME=${DATANAME}"
echo "      sut_args=\"${sut_args}\""
echo "      datakey=${datakey}, dataargs=\"${dataargs}\""
sleep 2

prepare_sut ${SUT}
python benchmarks/benchmark_rag_serving.py \
    --exp exp1_${SUT}_${DATANAME} \
    ${dataargs} \
    ${sut_args} \
    ${EXTRA_ARGS}
