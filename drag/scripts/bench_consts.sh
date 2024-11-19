#/!bin/bash

##################
## Data mapping ##
##################

DATA_KEYS=(
    "randtiny"
    "rand"
    "sqa"
    "narrativeqa"
)

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
    "pcrag"
    "drag"
)

function make_sut_args() {
    local _SUT=$1
    local retVal=$2
    if [[ $_SUT == "parrot" ]]
    then
        sut_args="--rag_type=parrot --tokenizer meta-llama/Llama-2-7b-chat-hf"
    elif [[ $_SUT == "llmrag" ]]
    then
        sut_args="--rag_type=llmrag --tokenizer meta-llama/Llama-2-7b-chat-hf --model meta-llama/Llama-2-7b-chat-hf"
    elif [[ $_SUT == "pcrag" ]]
    then
        sut_args="--rag_type=pcrag --tokenizer meta-llama/Llama-2-7b-chat-hf --pc_lm_name meta-llama/Llama-2-7b-chat-hf"
    elif [[ $_SUT == "drag" ]]
    then
        sut_args="--rag_type=drag --tokenizer meta-llama/Llama-2-7b-chat-hf --model meta-llama/Llama-2-7b-chat-hf --gpu-memory-utilization 0.9 --enforce-eager --enable-prefix-caching"
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
    elif [[ $_SUT == "parrot" || $_SUT == "llmrag" || $_SUT == "pcrag" || $_SUT == "drag" ]]
    then
        : # Do nothing
    else
        echo "ERROR (prepare_sut): Invalid SUT $_SUT, standard SUTS: [ ${SUTS[*]} ]"
        exit 1
    fi
    return 0
}
