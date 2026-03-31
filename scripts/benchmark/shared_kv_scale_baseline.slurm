#!/bin/bash
#SBATCH --job-name="shared_kv_scale_baseline"
#SBATCH --account=bdjx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm/shared_kv_scale_baseline_%j.out
#SBATCH --error=slurm/shared_kv_scale_baseline_%j.err

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm_common.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/scripts/slurm_common.sh"
else
    source "/projects/bdjx/${USER}/gh/lazy-attention/scripts/slurm_common.sh"
fi

slurm_print_header
slurm_activate_runtime
slurm_enter_repo

export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export LAZY_ATTENTION_VARIANT="${LAZY_ATTENTION_VARIANT:-lazy}"
echo "Using baseline SUT on lazy-patched stack for shared_kv_scale_baseline"
echo "LAZY_ATTENTION_VARIANT=${LAZY_ATTENTION_VARIANT}"

slurm_run_logged "log_${USER}.txt" \
    env SHARED_KV_SUT=baseline bash scripts/benchmark/bench_shared_kv_scale.sh
