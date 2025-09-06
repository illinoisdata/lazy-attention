#!/bin/bash
#SBATCH --job-name=lazy_test_generate
#SBATCH --account=bdjx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=0:30:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm/lazy_test_generate_%j.out
#SBATCH --error=slurm/lazy_test_generate_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load necessary modules
source ~/.bashrc  # for conda path 
conda activate vllm
module load cuda/12.4.0

# python -c "import vllm; print('load vllm successfully')"

cd /projects/bdjx/hxia3/gh/lazy-attention/lazyattn

export VLLM_LOGGING_LEVEL=DEBUG

PYTHONPATH=. pytest ./tests/test_base.py -v -s

