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

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load necessary modules
source ~/.bashrc  # for conda path 
conda activate vllm
module load cuda/12.4.0

cd /projects/bdjx/${USER}/gh/lazy-attention
export VLLM_LOGGING_LEVEL=INFO

# export NO_LAZY=1  # whether disable lazy attention for ablation study
export DEBUG_LAZY=1
export output=log_${USER}.txt;
echo "@@@@@@@@@@@@" >> $output; (bash scripts/validate.sh lazyrag ablation) 2>& 1 | tee -a $output; unset output