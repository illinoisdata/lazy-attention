#!/bin/bash
# Render the Lazy-vs-Prefix-Caching TTFT GIF (left = Prefix Caching, right =
# Lazy-Attn). Each engine is measured ONE AT A TIME (start -> measure -> tear
# down) so the two vLLM instances never coexist -- coexisting engines trip a
# memory-profiling assertion, and one-at-a-time also halves peak VRAM. The two
# timelines are then merged and rendered (each was measured with the full GPU,
# which is exactly the fair, isolated TTFT to compare).
#
#   bash scripts/demo/record_race_gif.sh                 # -> analysis/lazy_vs_prefix_demo.gif
#   DEMO_OUT=/tmp/race.gif bash scripts/demo/record_race_gif.sh --record-index 3
set -euo pipefail
cd "$(dirname "$0")/../.."
# Local box uses a .venv; on the slurm server the conda `vllm` env is already
# active (slurm_common.sh), so only source the venv if it exists.
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH=.:./lazy_attn

MODEL="${DEMO_MODEL:-hxia7/Llama-3.2-1B-Block-FT}"
LAZY_SUT="${DEMO_LAZY_SUT:-lazyrag}"
BASE_SUT="${DEMO_BASE_SUT:-llmrag}"
PORT="${DEMO_PORT:-8001}"
GPU_MEM_UTIL="${DEMO_GPU_MEM_UTIL:-0.40}"
OUT="${DEMO_OUT:-analysis/lazy_vs_prefix_demo.gif}"
STEM="${OUT%.gif}"
mkdir -p "$(dirname "${OUT}")"

cur_pid=""
cleanup() {
  [ -n "${cur_pid}" ] && kill "${cur_pid}" 2>/dev/null || true
  pkill -9 -f "benchmarks/serve_demo.py" 2>/dev/null || true
  pkill -9 -f "multiprocessing.spawn" 2>/dev/null || true
}
trap cleanup EXIT

wait_health() {  # url
  local i=0
  until curl -sf "$1/health" >/dev/null 2>&1; do
    i=$((i + 1)); [ $i -gt 180 ] && { echo "timeout waiting for $1"; return 1; }
    sleep 2
  done
}

# measure_one <rag_type> <label> <role> <logfile> <out_json>
measure_one() {
  echo "=== starting ${3} server (${1}, :${PORT}) ==="
  python benchmarks/serve_demo.py --rag-type "$1" --port "${PORT}" --label "$2" \
    --model "${MODEL}" --gpu-memory-utilization "${GPU_MEM_UTIL}" --enforce-eager >"$4" 2>&1 &
  cur_pid="$!"
  wait_health "http://127.0.0.1:${PORT}" || { tail -40 "$4"; exit 1; }
  python benchmarks/demo_race.py --role "$3" --url "http://127.0.0.1:${PORT}" \
    --out "${OUT}" --out-json "$5" "${@:6}"
  # tear the engine fully down before starting the other one
  kill "${cur_pid}" 2>/dev/null || true
  pkill -9 -f "benchmarks/serve_demo.py" 2>/dev/null || true
  pkill -9 -f "multiprocessing.spawn" 2>/dev/null || true
  cur_pid=""
  sleep 4
}

measure_one "${LAZY_SUT}" "Lazy-Attn (ours)"      lazy /tmp/demo_lazy.log "${STEM}_lazy.json" "$@"
measure_one "${BASE_SUT}" "Prefix Caching (vLLM)" base /tmp/demo_base.log "${STEM}_base.json" "$@"

echo "=== rendering GIF ==="
python benchmarks/demo_race.py --role render-pair \
  --lazy-json "${STEM}_lazy.json" --base-json "${STEM}_base.json" --out "${OUT}" "$@"

echo "done -> ${OUT}"
