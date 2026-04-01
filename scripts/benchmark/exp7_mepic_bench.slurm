#!/bin/bash
#SBATCH --job-name="exp7_mepic_bench"
#SBATCH --account=bdjx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm/exp7_mepic_bench_%j.out
#SBATCH --error=slurm/exp7_mepic_bench_%j.err

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm_common.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/scripts/slurm_common.sh"
else
    source "/projects/bdjx/${USER}/gh/lazy-attention/scripts/slurm_common.sh"
fi

slurm_print_header
slurm_activate_runtime
slurm_enter_repo

export VLLM_LOGGING_LEVEL=INFO
export LAZY_ATTENTION_VARIANT="${LAZY_ATTENTION_VARIANT:-mepic}"
export MEPIC_FORCE_FP32_ROTARY="${MEPIC_FORCE_FP32_ROTARY:-1}"
echo "Using LAZY_ATTENTION_VARIANT=${LAZY_ATTENTION_VARIANT} for exp7_mepic_bench"
echo "MEPIC_FORCE_FP32_ROTARY=${MEPIC_FORCE_FP32_ROTARY}"
echo "EXP7_REQS=${EXP7_REQS:-1} EXP7_DOCS_PER_REQUEST=${EXP7_DOCS_PER_REQUEST:-1} EXP7_DOCUMENT_LEN=${EXP7_DOCUMENT_LEN:-32768} EXP7_MAX_CONCURRENCY=${EXP7_MAX_CONCURRENCY:-1} EXP7_OUTPUT_LEN=${EXP7_OUTPUT_LEN:-512}"

slurm_run_logged "log_${USER}.txt" \
    bash scripts/benchmark/bench_exp7_single.sh lazyrag ablation
