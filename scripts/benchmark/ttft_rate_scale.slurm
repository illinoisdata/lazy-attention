#!/bin/bash
#SBATCH --job-name="ttft_rate_scale"
#SBATCH --account=bdjx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm/ttft_rate_scale_%j.out
#SBATCH --error=slurm/ttft_rate_scale_%j.err

# RQ1 / Figure 4: mean TTFT vs request rate under uniform vs skewed traffic.
# Pick the system and traffic via --export, e.g.:
#   sbatch --export=ALL,RATE_SUT=lazyrag,TRAFFIC=skewed scripts/benchmark/ttft_rate_scale.slurm
# Results land in results/ttft_rate_r<rate>_<traffic>_<sut>.json; mean_ttft_ms is the metric.

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm_common.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/scripts/slurm_common.sh"
else
    source "/projects/bdjx/${USER}/gh/lazy-attention/scripts/slurm_common.sh"
fi
slurm_print_header
slurm_activate_runtime
slurm_enter_repo
export VLLM_LOGGING_LEVEL=INFO
export LAZY_ATTENTION_VARIANT="${LAZY_ATTENTION_VARIANT:-lazy}"
echo "TTFT-rate sweep: RATE_SUT=${RATE_SUT:-lazyrag} TRAFFIC=${TRAFFIC:-skewed} LAZY_ATTENTION_VARIANT=${LAZY_ATTENTION_VARIANT}"
slurm_run_logged "log_${USER}.txt" bash scripts/benchmark/bench_ttft_rate_scale.sh
