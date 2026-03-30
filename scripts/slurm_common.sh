#!/bin/bash

slurm_print_header() {
    echo "Job ID: ${SLURM_JOB_ID}"
    echo "Node: ${SLURM_NODELIST}"
    echo "Start time: $(date)"
    echo ""
}

slurm_activate_runtime() {
    local conda_env="${1:-vllm}"
    local cuda_module="${2:-cuda/12.4.0}"

    source ~/.bashrc
    conda activate "${conda_env}"
    module load "${cuda_module}"
}

slurm_enter_repo() {
    local repo_root="${1:-/projects/bdjx/${USER}/gh/lazy-attention}"
    cd "${repo_root}" || exit 1
}

slurm_run_logged() {
    local log_file="${1:-log_${USER}.txt}"
    shift

    echo "@@@@@@@@@@@@" >> "${log_file}"
    ("$@") 2>&1 | tee -a "${log_file}"
}
