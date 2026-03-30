#!/bin/bash
#SBATCH --job-name="exp1_lazy_bench"
#SBATCH --account=bdjx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm/exp1_lazy_bench_%j.out
#SBATCH --error=slurm/exp1_lazy_bench_%j.err


# Description of exp 1:
# Two groups of data
# 1. various batch size (batch_size=1, 2, 4, 8) [just use the original documents from the dataset]
# 2. various document length (each document length can be set to 1k 2k 4k 8k)


# Assumptions:
# (1) Logged in into huggingface (`huggingface-cli login`)
# (2) Cloned repo at /projects/bdjx/${USER}/gh/lazy-attention
# (3) Downloaded dataset via scripts/download_dataset.sh
# (4) Installed vllm in editable mode (pip install -e .) in the conda env vllm

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/slurm_common.sh"

slurm_print_header
slurm_activate_runtime
slurm_enter_repo

export VLLM_LOGGING_LEVEL=INFO
slurm_run_logged "log_${USER}.txt" bash scripts/benchmark/bench_exp1.sh baseline,lazyrag 2wikimqa

# baseline blk_2wiki,blk_tqa
