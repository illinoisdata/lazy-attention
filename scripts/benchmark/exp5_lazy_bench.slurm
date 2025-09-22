#!/bin/bash
#SBATCH --job-name="exp5_lazy_bench"
#SBATCH --account=bdjx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm/exp5_lazy_bench_%j.out
#SBATCH --error=slurm/exp5_lazy_bench_%j.err


# Description of exp 5:
# Test the generation quality of lazy attention with various datasets
# Datasets: 2wikimqa, naturalquestions, hotpotqa, triviaqa
# Model: Tulu3-blk-ft
# Metric: Exact match (EM)

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

export output=log_${USER}.txt;
echo "@@@@@@@@@@@@" >> $output; (bash scripts/benchmark/bench_exp5.sh lazyrag blk_2wiki,blk_tqa,blk_nq,blk_hqa) 2>& 1 | tee -a $output; unset output