#!/bin/bash
#SBATCH --job-name="race_gif"
#SBATCH --account=bdjx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=72
#SBATCH --output=slurm/race_gif_%j.out
#SBATCH --error=slurm/race_gif_%j.err

# Render the Lazy-vs-Prefix-Caching TTFT GIF on the 8B (authoritative) model,
# whose answers are coherent (the local 1B is too weak for real QA). A GH200 has
# ~96 GB, so both 8B engines (lazy + stock) fit at once; demo_race still measures
# each in isolation so they never contend.
#   sbatch scripts/demo/record_race_gif.slurm
#   sbatch --export=ALL,DEMO_RECORD_INDEX=3 scripts/demo/record_race_gif.slurm
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm_common.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/scripts/slurm_common.sh"
else
    source "/projects/bdjx/${USER}/gh/lazy-attention/scripts/slurm_common.sh"
fi
slurm_print_header
slurm_activate_runtime
slurm_enter_repo
export LAZY_ATTENTION_VARIANT="${LAZY_ATTENTION_VARIANT:-lazy}"

export DEMO_MODEL="${DEMO_MODEL:-ldsjmdy/Tulu3-Block-FT}"   # 8B Block-FT (coherent)
export DEMO_GPU_MEM_UTIL="${DEMO_GPU_MEM_UTIL:-0.42}"        # 2 x 8B on one GH200 (~96GB)
export DEMO_OUT="${DEMO_OUT:-analysis/lazy_vs_prefix_demo_8b.gif}"

mkdir -p slurm analysis
DEMO_RACE_ARGS=()
[[ -n "${DEMO_RECORD_INDEX:-}" ]] && DEMO_RACE_ARGS+=(--record-index "${DEMO_RECORD_INDEX}")

echo "race_gif: model=${DEMO_MODEL} util=${DEMO_GPU_MEM_UTIL} out=${DEMO_OUT}"
bash scripts/demo/record_race_gif.sh "${DEMO_RACE_ARGS[@]}"
echo "DONE. GIF -> $(pwd)/${DEMO_OUT}"
echo "Fetch with: scp <server>:$(pwd)/${DEMO_OUT} ."
