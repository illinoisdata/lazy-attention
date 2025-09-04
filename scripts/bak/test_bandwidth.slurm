#!/bin/bash
#SBATCH --job-name=gh200_kv_cache_test
#SBATCH --account=bdjx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=0:30:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=72
#SBATCH --output=bandwidth_test_%j.out
#SBATCH --error=bandwidth_test_%j.err

## Each Grace-Hopper GH200 superchip has a Grace ARM 72-core CPU with 120 GB of memory and a NVIDIA H100 GPU with 96GB of memory.
## srun example is srun --account bdjx-dtai-gh --time==0:30:00 -p ghx4 --gpus-per-node=1 --mem=110G --cpus-per-task=72 --pty bash

echo "Starting GH200 KV Cache vs Recomputation Analysis"
echo "================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load necessary modules
source ~/.bashrc  # for conda path 
conda activate vllm
module load cuda/12.4.0

python -c "import vllm; print('load vllm successfully')"


# Print system information
echo "System Information:"
echo "==================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
echo ""
lscpu | grep "Model name\|CPU(s)\|Architecture"
echo ""
free -h
echo ""

# Change to the script directory
cd /projects/bdjx/hxia3/gh/lazy-attention/scripts
# numactl --cpunodebind=0 --membind=0 python test_bw_torch_v2.py
python test_bw_torch_v2.py
