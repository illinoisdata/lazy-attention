#!/bin/bash
#SBATCH --constraint=perf
#SBATCH --job-name=profiler_test_nsys
#SBATCH --account=bdjx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=0:10:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm/profiler_test_ncu.out
#SBATCH --error=slurm/profiler_test_ncu.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load necessary modules
module reset
source ~/.bashrc  # for conda path 
conda activate vllm
module load cuda/12.4.0

python -c "import vllm; print('load vllm successfully')"

cd /projects/bdjx/hxia3/gh/lazy-attention

ncu -o vllm-generate -f \
    --target-processes all \
    --section MemoryWorkloadAnalysis \
    python vllm_proj/vllm/examples/offline_inference/basic/generate.py

# /opt/nvidia/hpc_sdk/Linux_aarch64/24.3/profilers/Nsight_Systems/bin/nsys
# /opt/nvidia/hpc_sdk/Linux_aarch64/24.3/profilers/Nsight_Compute/ncu
