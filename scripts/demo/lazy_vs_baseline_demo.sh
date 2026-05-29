#!/bin/bash
# Live Lazy-vs-baseline RAG demo in a single tmux window:
#
#   +----------------------+--------------------------+
#   |  Lazy-Attn server     |                          |
#   |  (lazyrag, :8001)     |   comparison client      |
#   +-----------------------+   (side-by-side TTFT)    |
#   |  Baseline server      |                          |
#   |  (llmrag,  :8002)     |                          |
#   +-----------------------+--------------------------+
#
# Both servers load the SAME 1B model (one KV cache each), then the client
# pushes a shared doc corpus into both and fires reordered RAG queries. Lazy
# reuses every doc's KV regardless of slot; the prefix-caching baseline must
# recompute once the doc order diverges -> Lazy wins TTFT.
#
# Usage:   bash scripts/demo/lazy_vs_baseline_demo.sh
#          DEMO_NO_ATTACH=1 bash scripts/demo/lazy_vs_baseline_demo.sh   # headless
#          tmux attach -t lazydemo                                       # reattach
#          tmux kill-session -t lazydemo                                 # tear down
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${DEMO_SESSION:-lazydemo}"

MODEL="${DEMO_MODEL:-hxia7/Llama-3.2-1B-Block-FT}"
LAZY_SUT="${DEMO_LAZY_SUT:-lazyrag}"      # Lazy-Attn (ours)
BASE_SUT="${DEMO_BASE_SUT:-llmrag}"       # stock vLLM prefix caching
LAZY_PORT="${DEMO_LAZY_PORT:-8001}"
BASE_PORT="${DEMO_BASE_PORT:-8002}"
GPU_MEM_UTIL="${DEMO_GPU_MEM_UTIL:-0.40}" # two model copies share one GPU
# Client knobs (see demo_client.py --help)
NUM_DOCS="${DEMO_NUM_DOCS:-16}"
K="${DEMO_K:-8}"
QUERIES="${DEMO_QUERIES:-8}"
DOC_LEN="${DEMO_DOC_LEN:-384}"
MAX_TOKENS="${DEMO_MAX_TOKENS:-32}"

# Common preamble each pane runs before its command: enter repo, venv, paths.
PREP="cd '${REPO_ROOT}' && source .venv/bin/activate && export PYTHONPATH=.:./lazy_attn"

SERVE_COMMON="--model '${MODEL}' --gpu-memory-utilization ${GPU_MEM_UTIL} --enforce-eager"
LAZY_CMD="${PREP} && python benchmarks/serve_demo.py --rag-type ${LAZY_SUT} --port ${LAZY_PORT} --label 'Lazy-Attn (ours)' ${SERVE_COMMON}"
BASE_CMD="${PREP} && python benchmarks/serve_demo.py --rag-type ${BASE_SUT} --port ${BASE_PORT} --label 'Prefix Caching' ${SERVE_COMMON}"
CLIENT_CMD="${PREP} && python benchmarks/demo_client.py \
  --lazy-url http://127.0.0.1:${LAZY_PORT} --base-url http://127.0.0.1:${BASE_PORT} \
  --num-docs ${NUM_DOCS} --k ${K} --queries ${QUERIES} --doc-len ${DOC_LEN} --max-tokens ${MAX_TOKENS}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install it (apt-get install -y tmux) or run the 3 commands manually:" >&2
  echo "  1) ${LAZY_CMD}" >&2
  echo "  2) ${BASE_CMD}" >&2
  echo "  3) ${CLIENT_CMD}" >&2
  exit 1
fi

tmux kill-session -t "${SESSION}" 2>/dev/null || true

# Pane 0 (left-top): Lazy server.
tmux new-session -d -s "${SESSION}" -x 220 -y 50 -n demo
tmux send-keys -t "${SESSION}:demo.0" "${LAZY_CMD}" C-m

# Pane 1 (right): client. Split horizontally off pane 0.
tmux split-window -h -t "${SESSION}:demo.0"
# Pane 2 (left-bottom): baseline server. Split vertically off the left pane (0).
tmux split-window -v -t "${SESSION}:demo.0"
tmux send-keys -t "${SESSION}:demo.2" "${BASE_CMD}" C-m

# Give engines time to load before the client starts polling /health.
# (Client also self-polls, so this is just to keep the log readable.)
tmux send-keys -t "${SESSION}:demo.1" "${CLIENT_CMD}" C-m

tmux select-pane -t "${SESSION}:demo.1"

echo "tmux session '${SESSION}' started:"
echo "  pane 0 (top-left)    : Lazy-Attn server  (${LAZY_SUT}) :${LAZY_PORT}"
echo "  pane 2 (bottom-left) : Baseline server   (${BASE_SUT}) :${BASE_PORT}"
echo "  pane 1 (right)       : comparison client"
echo "Reattach: tmux attach -t ${SESSION}   |   Tear down: tmux kill-session -t ${SESSION}"

if [[ "${DEMO_NO_ATTACH:-0}" != "1" ]]; then
  tmux attach -t "${SESSION}"
fi
