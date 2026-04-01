#!/bin/bash
#SBATCH --job-name="exp6_mepic_bench"
#SBATCH --account=bdjx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm/exp6_mepic_bench_%j.out
#SBATCH --error=slurm/exp6_mepic_bench_%j.err

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
export MEPIC_FORCE_FP32_ROTARY="${MEPIC_FORCE_FP32_ROTARY:-0}"
echo "Using LAZY_ATTENTION_VARIANT=${LAZY_ATTENTION_VARIANT} for exp6_mepic_bench"
echo "MEPIC_FORCE_FP32_ROTARY=${MEPIC_FORCE_FP32_ROTARY}"
echo "EXP6_REQS=${EXP6_REQS:-8} EXP6_DOCS_PER_REQUEST=${EXP6_DOCS_PER_REQUEST:-5} EXP6_DOCUMENT_LEN=${EXP6_DOCUMENT_LEN:-10000} EXP6_MAX_CONCURRENCY=${EXP6_MAX_CONCURRENCY:-8}"

slurm_run_logged "log_${USER}.txt" \
    bash scripts/benchmark/bench_exp6_single.sh lazyrag ablation
