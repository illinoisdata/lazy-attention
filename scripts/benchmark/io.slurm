#!/bin/bash
#SBATCH --constraint=perf
#SBATCH --job-name=profiler_io
#SBATCH --account=bdjx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=2:50:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm/profiler_io.out
#SBATCH --error=slurm/profiler_io.err

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/slurm_common.sh"

slurm_print_header
slurm_activate_runtime
slurm_enter_repo

export VLLM_LOGGING_LEVEL=INFO

export NO_LAZY=0  # whether disable lazy attention for ablation study

slurm_run_logged "log_${USER}.txt" ncu -o lazyio -f \
    --target-processes all \
    --section MemoryWorkloadAnalysis \
    bash scripts/benchmark/bench_exp4.sh lazyrag ablation

export NO_LAZY=1  # whether disable lazy attention for ablation study

slurm_run_logged "log_${USER}.txt" ncu -o wolazyio -f \
    --target-processes all \
    --section MemoryWorkloadAnalysis \
    bash scripts/benchmark/bench_exp4.sh lazyrag ablation
