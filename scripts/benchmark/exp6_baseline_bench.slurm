#!/bin/bash
#SBATCH --job-name="exp6_baseline_bench"
#SBATCH --account=bdjx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm/exp6_baseline_bench_%j.out
#SBATCH --error=slurm/exp6_baseline_bench_%j.err

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm_common.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/scripts/slurm_common.sh"
else
    source "/projects/bdjx/${USER}/gh/lazy-attention/scripts/slurm_common.sh"
fi

slurm_print_header
slurm_activate_runtime
slurm_enter_repo

export VLLM_LOGGING_LEVEL=INFO
export BASELINE_PREPARE_PREFIX_CACHE="${BASELINE_PREPARE_PREFIX_CACHE:-1}"
echo "Using baseline for exp6_baseline_bench"
echo "BASELINE_PREPARE_PREFIX_CACHE=${BASELINE_PREPARE_PREFIX_CACHE}"
echo "EXP6_REQS=${EXP6_REQS:-8} EXP6_DOCS_PER_REQUEST=${EXP6_DOCS_PER_REQUEST:-5} EXP6_DOCUMENT_LEN=${EXP6_DOCUMENT_LEN:-10000} EXP6_MAX_CONCURRENCY=${EXP6_MAX_CONCURRENCY:-8}"

slurm_run_logged "log_${USER}.txt" \
    bash scripts/benchmark/bench_exp6_single.sh baseline ablation
