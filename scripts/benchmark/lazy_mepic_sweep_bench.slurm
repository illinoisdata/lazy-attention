#!/bin/bash
#SBATCH --job-name="lazy_mepic_sweep_bench"
#SBATCH --account=bdjx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm/lazy_mepic_sweep_bench_%j.out
#SBATCH --error=slurm/lazy_mepic_sweep_bench_%j.err

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm_common.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/scripts/slurm_common.sh"
else
    source "/projects/bdjx/${USER}/gh/lazy-attention/scripts/slurm_common.sh"
fi

slurm_print_header
slurm_activate_runtime
slurm_enter_repo

export VLLM_LOGGING_LEVEL=INFO

echo "SWEEP_SCENARIOS=${SWEEP_SCENARIOS:-light_single,light_async,medium_async,exp4_like,exp6_like,exp7_like,exp7_extreme}"
echo "SWEEP_VARIANTS=${SWEEP_VARIANTS:-lazy,mepic0,mepic1}"
echo "SWEEP_SUMMARY_PATH=${SWEEP_SUMMARY_PATH:-results/lazy_mepic_sweep_summary.tsv}"

slurm_run_logged "log_${USER}.txt" \
    bash scripts/benchmark/bench_lazy_mepic_sweep.sh
