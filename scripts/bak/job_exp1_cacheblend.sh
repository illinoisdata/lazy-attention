#!/bin/bash
#SBATCH --job-name="drag_job_exp1_cacheblend"
#SBATCH --output="drag_job_exp1.%j.%N.out"
#SBATCH --partition=gpuA40x4
##SBATCH --mail-type=END,FAIL 
##SBATCH --mail-user=your_email@example.com
#SBATCH --mem=62G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bdjx-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --no-requeue
#SBATCH -t 04:00:00
#SBATCH -e slurm/slurm-%j.err
#SBATCH -o slurm/slurm-%j.out

# Assumptions:
# (1) Logged in into huggingface (`huggingface-cli login`)
# (2) Cloned repo at /projects/bdjx/${USER}/DynamicRAG
# (3) Downloaded dataset via scripts/download_dataset.sh
# (4) Installed vllm (via `pip install -e install`) and drag (via `make install``)

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module load anaconda3_gpu
conda deactivate
conda deactivate
cd /projects/bdjx/${USER}/vllm/lazy
source .venv-cacheblend/bin/activate
module list  # job documentation and metadata
echo "Job is starting on `hostname`"

# TODO: Fix and add drag into SUT list.
export output=log_${USER}.txt; echo "@@@@@@@@@@@@" >> $output; (bash scripts/bench_exp1.sh cacheblend sqa,narrativeqa) 2>& 1 | tee -a $output; unset output
