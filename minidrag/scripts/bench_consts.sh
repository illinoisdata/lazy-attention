#/!bin/bash

##################
## Data mapping ##
##################

DATA_KEYS=(
    "randtiny"
    "rand"

    # ChatRAG-Bench
    "doc2dial"
    "convfinqa"
    "quac"
    "qrecc"
    "doqa_cooking"
    "doqa_travel"
    "doqa_movies"
    "coqa"
    "sqa"
    "topiocqa"
    "inscit"
    "hybridial"

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

# GLOBAL_DATA_ARGS=""
# --max-concurrency 16 
GLOBAL_DATA_ARGS="--request-sampling-method zipf --request-zipf-param 1.01 --sample-documents 10 --document-sampling-method zipf --document-zipf-param 2"

# For e1 2wikimqa
GLOBAL_DATA_ARGS+=" --sample-requests 200 --max-concurrency 1"


function get_data_args() {
    local key=$1
    local RET_DATAARGS=$2
    if [[ $key == "randtiny" ]]
    then
        data_args="--dataset-name random --random-num-prompts 10 --random-input-len 128 --random-output-len 32 --random-document-len 16 --random-num-documents 4 --random-num-documents-per-prompt 1"
    elif [[ $key == "rand" ]]
    then
        data_args="--dataset-name random --sample-requests 128 --random-num-prompts 64 --random-input-len 16 --random-output-len 16 --random-document-len 128 --random-num-documents 8 --random-num-documents-per-prompt 4"
    elif [[ $key == "doc2dial" ]]
    then
        data_args="--dataset-name chatragbench --eval_dataset doc2dial"
    elif [[ $key == "convfinqa" ]]
    then
        data_args="--dataset-name chatragbench --eval_dataset convfinqa"
    elif [[ $key == "quac" ]]
    then
        data_args="--dataset-name chatragbench --eval_dataset quac"
    elif [[ $key == "qrecc" ]]
    then
        data_args="--dataset-name chatragbench --eval_dataset qrecc"
    elif [[ $key == "doqa_cooking" ]]
    then
        data_args="--dataset-name chatragbench --eval_dataset doqa_cooking"
    elif [[ $key == "doqa_travel" ]]
    then
        data_args="--dataset-name chatragbench --eval_dataset doqa_travel"
    elif [[ $key == "doqa_movies" ]]
    then
        data_args="--dataset-name chatragbench --eval_dataset doqa_movies"
    elif [[ $key == "coqa" ]]
    then
        data_args="--dataset-name chatragbench --eval_dataset coqa"
    elif [[ $key == "sqa" ]]
    then
        data_args="--dataset-name chatragbench --eval_dataset sqa"
    elif [[ $key == "topiocqa" ]]
    then
        data_args="--dataset-name chatragbench --eval_dataset topiocqa"
    elif [[ $key == "inscit" ]]
    then
        data_args="--dataset-name chatragbench --eval_dataset inscit"
    elif [[ $key == "hybridial" ]]
    then
        data_args="--dataset-name chatragbench --eval_dataset hybridial"
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
    elif [[ $key == "2wikimqa_block" ]]
    then
        data_args="--dataset-name 2wikimqa_block"
    elif [[ $key == "hotpotqa_block" ]]
    then
        data_args="--dataset-name hotpotqa_block"
    elif [[ $key == "2wikimqa_cacheblend" ]]
    then
        data_args="--dataset-name 2wikimqa_cacheblend"
    elif [[ $key == "samsum_cacheblend" ]]
    then
        data_args="--dataset-name samsum_cacheblend"
    elif [[ $key == "musique_cacheblend" ]]
    then
        data_args="--dataset-name musique_cacheblend"
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
    elif [[ $key == "rdl" ]]
    then
        data_args="--random-document-len ${val}"
    elif [[ $key == "n" ]]
    then
        data_args="--sample-requests ${val}"
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
    "cachep"
    "cachepno"
    "cachepseq"
    "cachepptree"
    "llmrag"
    "trragr1"
    "trragr2"
    "trragm1"
    "trragm2"
    "trragm2v2"
    "trragm3"
    "pcrag"
    "cacheblend"
    "drag"
    "recllmrag"
    "reullmrag"
    "blockattnrag"
)

function get_sut_args() {
    local _SUT=$1
    local retVal=$2

    if [ -z "$SUTS_MODEL" ]
    then
        # SUTS_MODEL="facebook/opt-125m"
        # SUTS_MODEL="meta-llama/Llama-3.1-8B-Instruct"
        # SUTS_MODEL="meta-llama/Llama-3.1-8B-Instruct"
        # SUTS_MODEL="meta-llama/Llama-3.1-70B-Instruct"
        SUTS_MODEL="ldsjmdy/Tulu3-Block-FT"
        # SUTS_MODEL="ldsjmdy/Tulu3-RAG"
        # SUTS_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
    fi
    echo "Using SUTS_MODEL=${SUTS_MODEL}"

    if [[ $_SUT == "parrot" ]]
    then
        sut_args="--rag_type=parrot --tokenizer ${SUTS_MODEL}"
    elif [[ $_SUT == "cachep" ]]
    then
        sut_args="--rag_type=cachep --tokenizer ${SUTS_MODEL} --cachep_tokenizer ${SUTS_MODEL} --cachep_type lru"
    elif [[ $_SUT == "cachepno" ]]
    then
        sut_args="--rag_type=cachep --tokenizer ${SUTS_MODEL} --cachep_tokenizer ${SUTS_MODEL} --cachep_type no"
    elif [[ $_SUT == "cachepseq" ]]
    then
        sut_args="--rag_type=cachep --tokenizer ${SUTS_MODEL} --cachep_tokenizer ${SUTS_MODEL} --cachep_type seq"
    elif [[ $_SUT == "cachepptree" ]]
    then
        sut_args="--rag_type=cachep --tokenizer ${SUTS_MODEL} --cachep_tokenizer ${SUTS_MODEL} --cachep_type ptree"
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
    elif [[ $_SUT == "trragm2v2" ]]
    then
        sut_args="--rag_type=trrag --tokenizer ${SUTS_MODEL} --trrag_lm_name ${SUTS_MODEL} --trrag_method m2v2"
    elif [[ $_SUT == "trragm3" ]]
    then
        sut_args="--rag_type=trrag --tokenizer ${SUTS_MODEL} --trrag_lm_name ${SUTS_MODEL} --trrag_method m3"
    elif [[ $_SUT == "pcrag" ]]
    then
        sut_args="--rag_type=pcrag --tokenizer ${SUTS_MODEL} --pc_lm_name ${SUTS_MODEL}"
    elif [[ $_SUT == "cacheblend" ]]
    then
        sut_args="--rag_type=cacheblend --tokenizer ${SUTS_MODEL} --model ${SUTS_MODEL}"
    elif [[ $_SUT == "drag" ]]
    then
        sut_args="--rag_type=drag --tokenizer ${SUTS_MODEL} --model ${SUTS_MODEL} --gpu-memory-utilization 0.9 --enforce-eager --enable-prefix-caching"
    elif [[ $_SUT == "recllmrag" ]]
    then
        sut_args="--rag_type=recllmrag --tokenizer ${SUTS_MODEL} --model ${SUTS_MODEL}"
    elif [[ $_SUT == "reullmrag" ]]
    then
        sut_args="--rag_type=reullmrag --tokenizer ${SUTS_MODEL} --model ${SUTS_MODEL}"
    elif [[ $_SUT == "blockattnrag" ]]
    then
        sut_args="--rag_type=blockattnrag --tokenizer ${SUTS_MODEL} --model ${SUTS_MODEL}"
    else
        echo "ERROR (get_sut_args): Invalid SUT $_SUT, standard SUTS: [ ${SUTS[*]} ]"
        exit 1
    fi
    eval $retVal="'${sut_args}'"
    return 0
}

function get_sut_augment_args() {
    local key=$1
    local val=$2
    local RET_SUTARGS=$3
    if [[ $key == "example" ]]
    then
        sut_args="--example_flag ${val}"
    elif [[ $key == "id" ]]
    then
        sut_args=""  # No-op, used for repetitions
    elif [[ $key == "cpc" ]]
    then
        sut_args="--cachep_capacity ${val}"
    else
        echo "ERROR (get_sut_augment_args): Unknown key= ${key}, with value= ${value}"
        exit 1
    fi
    eval $RET_SUTARGS="'${sut_args}'"
    return 0
}

# The mapping function.
function make_sut_args() {
    local sutname=$1
    local RET_SUTARGS=$2

    # SUT key is everything before `[`.
    ret_sutkey="${sutname%%\[*}"
    pairs="${sutname#*$ret_sutkey}"

    # Get sutset arguments.
    get_sut_args "${ret_sutkey}" ret_sutargs

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
        get_sut_augment_args "${key}" "${value}" new_nbargs
        ret_sutargs="${ret_sutargs} ${new_nbargs}"
    done

    # Append global sut arguments.
    ret_sutargs="${ret_sutargs} ${GLOBAL_SUT_ARGS}"

    eval $RET_SUTARGS="'${ret_sutargs}'"
    return 0
}

function prepare_sut() {
    local _SUT=$1
    if [[ $_SUT == "example" ]]
    then
        echo "Preparing example"
    else
        echo "Prepare SUT $_SUT."
            if [[ $_SUT == "drag" ]]
            then
                echo "Enabling lazy attention"
                VLLM_USE_LAZY_ATTENTION="1"
                export VLLM_USE_LAZY_ATTENTION
                # echo "VLLM_USE_LAZY_ATTENTION=${VLLM_USE_LAZY_ATTENTION}"
            else
                VLLM_USE_LAZY_ATTENTION="0"
                export VLLM_USE_LAZY_ATTENTION
                # echo "VLLM_USE_LAZY_ATTENTION=${VLLM_USE_LAZY_ATTENTION}"
            fi
    fi

    return 0
}
