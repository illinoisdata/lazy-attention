#/!bin/bash

int_handler() {
    echo "Interrupted."
    kill $PPID
    exit 1
}
trap 'int_handler' INT
source scripts/bench_consts.sh

if [ "$#" -eq 3 ]
then
    result_dir=$1
    IFS=',' read -ra suts <<< $2
    IFS=',' read -ra data_keys <<< $3
else
    echo "Invalid number of arguments (expected 3), $# provided"
    echo 'Example: bash scripts/bench_exp1_grade.sh results/ parrot,llmrag randtiny,sqa,narrativeqa'
    exit 1
fi
echo "Grading ${result_dir} on [ ${suts[*]} ] x [ ${data_keys[*]} ]"

# Grade one dataset at a time.
for ((j = 0; j < ${#data_keys[@]}; j++)) do
    # Build dataset arguments.
    dataname=${data_keys[$j]}
    make_data_args ${dataname} datakey dataargs

    # Build result paths.
    result_paths=""
    for ((i = 0; i < ${#suts[@]}; i++)) do
        sut=${suts[$i]}
        result_paths="${result_paths} ${result_dir}/exp1_${sut}_${dataname}.json"
    done

    # Grade.
    python benchmarks/grade_accuracy.py --result_paths ${result_paths} ${dataargs}
done
