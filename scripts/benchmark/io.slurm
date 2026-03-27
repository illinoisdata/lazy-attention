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

export NO_LAZY=0  # whether disable lazy attention for ablation study

export output=log_${USER}.txt;
echo "@@@@@@@@@@@@" >> $output; 
(ncu -o lazyio -f \
    --target-processes all \
    --section MemoryWorkloadAnalysis \
    bash scripts/benchmark/bench_exp4.sh lazyrag ablation) 2>& 1 | tee -a $output; unset output

export NO_LAZY=1  # whether disable lazy attention for ablation study

export output=log_${USER}.txt;
echo "@@@@@@@@@@@@" >> $output; 
(ncu -o wolazyio -f \
    --target-processes all \
    --section MemoryWorkloadAnalysis \
    bash scripts/benchmark/bench_exp4.sh lazyrag ablation) 2>& 1 | tee -a $output; unset output