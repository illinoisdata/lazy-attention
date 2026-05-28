#!/bin/bash
#SBATCH --job-name="hit_ratio_scale"
#SBATCH --account=bdjx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm/hit_ratio_scale_%j.out
#SBATCH --error=slurm/hit_ratio_scale_%j.err

# RQ2 / Table 1: VRAM doc-KV cache hit ratio vs KV budget (gpu-memory-utilization
# sweep) under skewed Zipfian RAG traffic. Set HIT_SUT to pick the system:
#   sbatch --export=ALL,HIT_SUT=lazyrag   scripts/benchmark/hit_ratio_scale.slurm
# Read results with scripts/benchmark/extract_hit_ratio.sh on the .out log.

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm_common.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/scripts/slurm_common.sh"
else
    source "/projects/bdjx/${USER}/gh/lazy-attention/scripts/slurm_common.sh"
fi
slurm_print_header
slurm_activate_runtime
slurm_enter_repo
export VLLM_LOGGING_LEVEL=INFO
echo "Hit-ratio KV-budget sweep: HIT_SUT=${HIT_SUT:-lazyrag} LAZY_ATTENTION_VARIANT=${LAZY_ATTENTION_VARIANT:-lazy}"
slurm_run_logged "log_${USER}.txt" bash scripts/benchmark/bench_hit_ratio_scale.sh
