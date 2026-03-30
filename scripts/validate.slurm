#!/bin/bash
#SBATCH --job-name="validate"
#SBATCH --account=bdjx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=00:20:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm/validate_%j.out
#SBATCH --error=slurm/validate_%j.err


# Description of [validate] exp:
# We use this scripts to expose the internal token ids 
# and attention output to check the sanity of customized
# deferred positional encoding

# Assumptions:
# (1) Logged in into huggingface (`huggingface-cli login`)
# (2) Cloned repo at /projects/bdjx/${USER}/gh/lazy-attention
# (3) Downloaded dataset via scripts/download_dataset.sh
# (4) Installed vllm in editable mode (pip install -e .) in the conda env vllm

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/slurm_common.sh"

slurm_print_header
slurm_activate_runtime
slurm_enter_repo

export VLLM_LOGGING_LEVEL=INFO

# export NO_LAZY=1  # whether disable lazy attention for ablation study
export DEBUG_LAZY=1
slurm_run_logged "log_${USER}.txt" bash scripts/validate.sh lazyrag ablation
