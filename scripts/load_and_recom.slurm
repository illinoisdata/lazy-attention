#!/bin/bash
#SBATCH --job-name=gh200_kv_cache_test
#SBATCH --account=bdjx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=0:30:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=72
#SBATCH --output=gh200_kv_cache_test_%j.out
#SBATCH --error=gh200_kv_cache_test_%j.err

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

# Run quick test with lightweight model first
echo "Running quick test with lightweight model..."
echo "============================================"
python3 load_and_recompute.py \
    --model facebook/opt-125m \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.7 \
    --intersection-only True \
    --token-lengths 128 256 512 1024 2048 4096 8192 \
    --max-tokens 1 \
    --num-runs 50 \
    --output-file gh200_quick_test_${SLURM_JOB_ID}.json

echo ""
echo "Quick test completed. Starting comprehensive analysis..."
echo "======================================================="

# Run comprehensive test with larger model
python3 load_and_recompute.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --intersection-only True \
    --token-lengths 128 256 512 1024 2048 4096 8192 \
    --max-tokens 1 \
    --num-runs 50 \
    --output-file gh200_comprehensive_${SLURM_JOB_ID}.json

echo ""
echo "Analysis completed at: $(date)"
echo "Results saved to:"
echo "  - gh200_quick_test_${SLURM_JOB_ID}.json"
echo "  - gh200_comprehensive_${SLURM_JOB_ID}.json"
echo ""
echo "Job completed successfully!"

# Optional: Copy results to a permanent location
# cp gh200_*_${SLURM_JOB_ID}.json /projects/bdjx/hxia3/gh/lazy-attention/results/