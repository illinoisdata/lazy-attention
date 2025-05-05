#/!bin/bash
export PYTHONPATH=.:./promptcache
int_handler() {
    echo "Interrupted."
    kill $PPID
    exit 1
}
trap 'int_handler' INT
source scripts/bench_consts.sh

if [ "$#" -eq 0 ]
then
    suts=("${SUTS[@]}")
    data_keys=("${DATA_KEYS[@]}")
elif [ "$#" -eq 1 ]
then
    IFS=',' read -ra suts <<< $1
    data_keys=("${DATA_KEYS[@]}")
elif [ "$#" -eq 2 ]
then
    IFS=',' read -ra suts <<< $1
    IFS=',' read -ra data_keys <<< $2
else
    echo "Invalid number of arguments (expected <= 2), $# provided"
    echo 'Example: bash scripts/bench_exp1.sh'
    echo 'Example: bash scripts/bench_exp1.sh parrot'
    echo 'Example: bash scripts/bench_exp1.sh parrot,llmrag randtiny,sqa,narrativeqa'
    exit 1
fi
echo "Running scripts/bench_exp1_single.sh on [ ${suts[*]} ] x [ ${data_keys[*]} ]"

for ((i = 0; i < ${#suts[@]}; i++)) do
    for ((j = 0; j < ${#data_keys[@]}; j++)) do
        bash scripts/bench_exp1_single.sh ${suts[$i]} ${data_keys[$j]}
    done
done
