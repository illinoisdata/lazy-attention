#!/bin/bash
#SBATCH --job-name=gh200_kv_quick
#SBATCH --account=bdjx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=0:10:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=36
#SBATCH --output=gh200_kv_quick_%j.out
#SBATCH --error=gh200_kv_quick_%j.err

## Quick test script for GH200 KV Cache vs Recomputation Analysis
## Uses smaller resources for faster testing

echo "Starting GH200 KV Cache Quick Test"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load necessary modules
source ~/.bashrc  # for conda path 
conda activate vllm
module load cuda/12.4.0

python -c "import vllm; print('load vllm successfully')"
# Print basic system info
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
echo ""

# Change to the script directory
cd /projects/bdjx/hxia3/gh/lazy-attention/scripts

echo "Running KV cache intersection analysis (quick test)..."
echo "====================================================="

# Run only the lightweight model test with 5 runs for stability
python3 load_and_recompute.py \
    --model facebook/opt-125m \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.6 \
    --intersection-only \
    --token-lengths 64 128 256 512 \
    --max-tokens 20 \
    --num-runs 5 \
    --output-file gh200_quick_${SLURM_JOB_ID}.json

echo ""
echo "Quick test completed at: $(date)"
echo "Results saved to: gh200_quick_${SLURM_JOB_ID}.json"

# Display summary results
echo ""
echo "=== SUMMARY ==="
if [ -f "gh200_quick_${SLURM_JOB_ID}.json" ]; then
    echo "Analysis completed successfully!"
    # Extract key findings using python
    python3 -c "
import json
try:
    with open('gh200_quick_${SLURM_JOB_ID}.json', 'r') as f:
        data = json.load(f)
    if 'intersection_analysis' in data:
        analysis = data['intersection_analysis']['analysis']
        print('Key Findings:')
        for token_len, trends in analysis.get('performance_trends', {}).items():
            decision = trends.get('decision', 'unknown')
            print(f'  {token_len} tokens: {decision}')
        
        if analysis.get('intersection_points'):
            print('Intersection Points Found:')
            for key, point in analysis['intersection_points'].items():
                print(f'  {point[\"token_length\"]} tokens: {point[\"transition\"]}')
        else:
            print('No intersection points found in tested range')
except Exception as e:
    print(f'Could not parse results: {e}')
"
else
    echo "Results file not found!"
fi

echo ""
echo "Job completed!"
