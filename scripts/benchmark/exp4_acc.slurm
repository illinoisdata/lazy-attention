#!/bin/bash
#SBATCH --job-name="exp4_acc"
#SBATCH --account=bdjx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=00:45:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm/exp4_acc_%j.out
#SBATCH --error=slurm/exp4_acc_%j.err

# RQ4 accuracy on the 8B (authoritative). Pick SUT + dataset via --export, e.g.:
#   sbatch --export=ALL,EXP4_SUT=lazyrag,EXP4_DATA=2wikimqa exp4_acc.slurm
#   sbatch --export=ALL,EXP4_SUT=llmrag,EXP4_DATA=2wikimqa,VLLM_WORKER_MULTIPROC_METHOD=spawn exp4_acc.slurm
# Stock-vLLM SUTs (llmrag/recllmrag) need spawn to dodge the CUDA-fork crash.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm_common.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/scripts/slurm_common.sh"
else
    source "/projects/bdjx/${USER}/gh/lazy-attention/scripts/slurm_common.sh"
fi
slurm_print_header
slurm_activate_runtime
slurm_enter_repo
export VLLM_LOGGING_LEVEL=INFO
export LAZY_ATTENTION_VARIANT="${LAZY_ATTENTION_VARIANT:-lazy}"
echo "exp4_acc: SUT=${EXP4_SUT:-lazyrag} DATA=${EXP4_DATA:-2wikimqa} LAZY_VARIANT=${LAZY_ATTENTION_VARIANT} MULTIPROC=${VLLM_WORKER_MULTIPROC_METHOD:-default}"
slurm_run_logged "log_${USER}.txt" \
    bash scripts/benchmark/bench_exp4_single.sh "${EXP4_SUT:-lazyrag}" "${EXP4_DATA:-2wikimqa}"
