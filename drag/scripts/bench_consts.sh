#/!bin/bash

##################
## Data mapping ##
##################

DATA_KEYS=(
    "randtiny"
    "rand"

    # ChatRAG-Bench
    "sqa"

    # Longbench
    "narrativeqa"
    "qasper"
    "multifieldqa_en"
    "multifieldqa_zh"
    "hotpotqa"
    "2wikimqa"
    "musique"
    "dureader"
    "gov_report"
    "qmsum"
    "multi_news"
    "vcsum"
    "trec"
    "triviaqa"
    "samsum"
    "lsht"
    "passage_count"
    "passage_retrieval_en"
    "passage_retrieval_zh"
    "lcc"
    "repobench-p"
)

GLOBAL_DATA_ARGS=""

function get_data_args() {
    local key=$1
    local RET_DATAARGS=$2
    if [[ $key == "randtiny" ]]
    then
        data_args="--dataset-name random --random-num-prompts 10 --random-input-len 128 --random-output-len 32 --random-document-len 16 --random-num-documents 4 --random-num-documents-per-prompt 1"
    elif [[ $key == "rand" ]]
    then
        data_args="--dataset-name random --sample-requests 2000 --random-num-prompts 100 --random-input-len 128 --random-output-len 32 --random-document-len 16 --random-num-documents 10 --random-num-documents-per-prompt 2"
    elif [[ $key == "sqa" ]]
    then
        data_args="--dataset-name chatragbench --eval_dataset sqa"
    elif [[ $key == "narrativeqa" ]]
    then
        data_args="--dataset-name longbench --longbench_dataset_name narrativeqa"
    elif [[ $key == "qasper" ]]
    then
        data_args="--dataset-name longbench --longbench_dataset_name qasper"
    elif [[ $key == "multifieldqa_en" ]]
    then
        data_args="--dataset-name longbench --longbench_dataset_name multifieldqa_en"
    elif [[ $key == "multifieldqa_zh" ]]
    then
        data_args="--dataset-name longbench --longbench_dataset_name multifieldqa_zh"
    elif [[ $key == "hotpotqa" ]]
    then
        data_args="--dataset-name longbench --longbench_dataset_name hotpotqa"
    elif [[ $key == "2wikimqa" ]]
    then
        data_args="--dataset-name longbench --longbench_dataset_name 2wikimqa"
    elif [[ $key == "musique" ]]
    then
        data_args="--dataset-name longbench --longbench_dataset_name musique"
    elif [[ $key == "dureader" ]]
    then
        data_args="--dataset-name longbench --longbench_dataset_name dureader"
    elif [[ $key == "gov_report" ]]
    then
        data_args="--dataset-name longbench --longbench_dataset_name gov_report"
    elif [[ $key == "qmsum" ]]
    then
        data_args="--dataset-name longbench --longbench_dataset_name qmsum"
    elif [[ $key == "multi_news" ]]
    then
        data_args="--dataset-name longbench --longbench_dataset_name multi_news"
    elif [[ $key == "vcsum" ]]
    then
        data_args="--dataset-name longbench --longbench_dataset_name vcsum"
    elif [[ $key == "trec" ]]
    then
        data_args="--dataset-name longbench --longbench_dataset_name trec"
    elif [[ $key == "triviaqa" ]]
    then
        data_args="--dataset-name longbench --longbench_dataset_name triviaqa"
    elif [[ $key == "samsum" ]]
    then
        data_args="--dataset-name longbench --longbench_dataset_name samsum"
    elif [[ $key == "lsht" ]]
    then
        data_args="--dataset-name longbench --longbench_dataset_name lsht"
    elif [[ $key == "passage_count" ]]
    then
        data_args="--dataset-name longbench --longbench_dataset_name passage_count"
    elif [[ $key == "passage_retrieval_en" ]]
    then
        data_args="--dataset-name longbench --longbench_dataset_name passage_retrieval_en"
    elif [[ $key == "passage_retrieval_zh" ]]
    then
        data_args="--dataset-name longbench --longbench_dataset_name passage_retrieval_zh"
    elif [[ $key == "lcc" ]]
    then
        data_args="--dataset-name longbench --longbench_dataset_name lcc"
    elif [[ $key == "repobench-p" ]]
    then
        data_args="--dataset-name longbench --longbench_dataset_name repobench-p"
    else
        echo "ERROR (get_data_args): Unknown dataset ${key}. standard dataset: [ ${DATA_KEYS[*]} ]"
        exit 1
    fi
    eval $RET_DATAARGS="'${data_args}'"
    return 0
}

function get_data_augment_args() {
    local key=$1
    local val=$2
    local RET_DATAARGS=$3
    if [[ $key == "example" ]]
    then
        data_args="--example_flag ${val}"
    else
        echo "ERROR (get_data_augment_args): Unknown key= ${key}, with value= ${value}"
        exit 1
    fi
    eval $RET_DATAARGS="'${data_args}'"
    return 0
}

# The mapping function.
function make_data_args() {
    local dataname=$1
    local RET_DATAKEY=$2
    local RET_DATAARGS=$3

    # Data key is everything before `[`.
    ret_datakey="${dataname%%\[*}"
    pairs="${dataname#*$ret_datakey}"

    # Get dataset arguments.
    get_data_args "${ret_datakey}" ret_dataargs

    # Parse optional flags in each pair of `[`` and `]`.
    while [[ "$pairs" =~ \[([^\]]*)\](.*) ]]; do
        pair="${BASH_REMATCH[1]}"
        pairs="${BASH_REMATCH[2]}"

        # Parse `key` or `key=value`.
        if [[ "$pair" == *"="* ]]; then
            # Split the pair into key and value
            IFS='=' read -r key value <<< "$pair"
        else
            # If no '=', treat the key as the entire pair and value as empty
            key="$pair"
            value=""
        fi

        # Map the short name to full argument format.
        get_data_augment_args "${key}" "${value}" new_nbargs
        ret_dataargs="${ret_dataargs} ${new_nbargs}"
    done

    # Append global data arguments.
    ret_dataargs="${ret_dataargs} ${GLOBAL_DATA_ARGS}"

    eval $RET_DATAKEY="'${ret_datakey}'"
    eval $RET_DATAARGS="'${ret_dataargs}'"
    return 0
}

##########
## SUTS ##
##########

SUTS=(
    "parrot"
    "llmrag"
    "trragr1"
    "trragr2"
    "trragm1"
    "trragm2"
    "trragm3"
    "pcrag"
    "drag"
)

# SUTS_MODEL="facebook/opt-125m"
SUTS_MODEL="meta-llama/Llama-3.1-8B-Instruct"

function make_sut_args() {
    local _SUT=$1
    local retVal=$2
    if [[ $_SUT == "parrot" ]]
    then
        sut_args="--rag_type=parrot --tokenizer ${SUTS_MODEL}"
    elif [[ $_SUT == "llmrag" ]]
    then
        sut_args="--rag_type=llmrag --tokenizer ${SUTS_MODEL} --model ${SUTS_MODEL}"
    elif [[ $_SUT == "trragr1" ]]
    then
        sut_args="--rag_type=trrag --tokenizer ${SUTS_MODEL} --trrag_lm_name ${SUTS_MODEL} --trrag_method r1"
    elif [[ $_SUT == "trragr2" ]]
    then
        sut_args="--rag_type=trrag --tokenizer ${SUTS_MODEL} --trrag_lm_name ${SUTS_MODEL} --trrag_method r2"
    elif [[ $_SUT == "trragm1" ]]
    then
        sut_args="--rag_type=trrag --tokenizer ${SUTS_MODEL} --trrag_lm_name ${SUTS_MODEL} --trrag_method m1"
    elif [[ $_SUT == "trragm2" ]]
    then
        sut_args="--rag_type=trrag --tokenizer ${SUTS_MODEL} --trrag_lm_name ${SUTS_MODEL} --trrag_method m2"
    elif [[ $_SUT == "trragm3" ]]
    then
        sut_args="--rag_type=trrag --tokenizer ${SUTS_MODEL} --trrag_lm_name ${SUTS_MODEL} --trrag_method m3"
    elif [[ $_SUT == "pcrag" ]]
    then
        sut_args="--rag_type=pcrag --tokenizer ${SUTS_MODEL} --pc_lm_name ${SUTS_MODEL}"
    elif [[ $_SUT == "drag" ]]
    then
        sut_args="--rag_type=drag --tokenizer ${SUTS_MODEL} --model ${SUTS_MODEL} --gpu-memory-utilization 0.9 --enforce-eager --enable-prefix-caching"
    else
        echo "ERROR (get_sut_args): Invalid SUT $_SUT, standard SUTS: [ ${SUTS[*]} ]"
        exit 1
    fi
    eval $retVal="'${sut_args}'"
    return 0
}

function prepare_sut() {
    local _SUT=$1
    if [[ $_SUT == "example" ]]
    then
        echo "Preparing example"
    else
        echo "Prepare SUT $_SUT with no-op."
    fi
    return 0
}
