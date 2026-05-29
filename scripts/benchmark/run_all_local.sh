#!/bin/bash
# Run the full RQ suite LOCALLY with the 1B model. Local timing is DIRECTIONAL
# (the server with the 8B is authoritative); local accuracy is a RELATIVE
# lazy-vs-baseline quality check. The single local GPU forces stages to run
# serially. Skips mepic per project decision.
#   RQ4  accuracy  (lazyrag/baseline/llmrag on a short LongBench task)
#   RQ1  TTFT-vs-rate (lazyrag/baseline/llmrag x {skewed,uniform})
# RQ2 (hit-ratio) and RQ3 (decode-overhead) have their own scripts and are run
# separately. sm_120 needs --enforce-eager; in-process engine avoids the
# CUDA-fork crash on the stock (llmrag) path.
cd "$(dirname "$0")/../.."            # repo root
source .venv/bin/activate
source scripts/benchmark/bench_lib.sh
benchmark_setup_pythonpath
export GLOBAL_SUT_ARGS=""             # bench_consts.sh references it but never inits it

# Stage/SUT selection so failed pieces can be re-run in isolation:
#   RUN_STAGE = all | rq4 | rq1   (default all)
RUN_STAGE="${RUN_STAGE:-all}"

export SUTS_MODEL="${SUTS_MODEL:-hxia7/Llama-3.2-1B-Block-FT}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN_VLLM_V1}"
export LAZY_ATTENTION_VARIANT="${LAZY_ATTENTION_VARIANT:-lazy}"
LOCAL_SUTS="${LOCAL_SUTS:-lazyrag baseline llmrag}"

# lazy/baseline rely on the in-process engine so the lazy monkeypatch applies
# (it would NOT survive a spawned subprocess). llmrag is stock vLLM and hits the
# CUDA-fork crash, so it needs the spawn start method instead.
set_engine_env() {
  if [[ "$1" == "llmrag" ]]; then
    unset VLLM_ENABLE_V1_MULTIPROCESSING
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
  else
    unset VLLM_WORKER_MULTIPROC_METHOD
    export VLLM_ENABLE_V1_MULTIPROCESSING=0
  fi
}
RQ4_DATASET="${RQ4_DATASET:-2wikimqa}"
mkdir -p results slurm logs_local

if [[ "${RUN_STAGE}" == "all" || "${RUN_STAGE}" == "rq4" ]]; then
echo "############## RQ4 accuracy (dataset=${RQ4_DATASET}) ##############"
BENCH_EXTRA_ARGS=(--enforce-eager --max-model-len 16384 --longbench_out_seq_len 64)
for sut in ${LOCAL_SUTS}; do
  echo "===== RQ4 acc: sut=${sut} dataset=${RQ4_DATASET} ====="
  set_engine_env "${sut}"
  benchmark_run_case "rq4" "${sut}" "${RQ4_DATASET}" 2>&1 | tee "logs_local/rq4_${sut}.log"
done
fi

if [[ "${RUN_STAGE}" == "all" || "${RUN_STAGE}" == "rq1" ]]; then
echo "############## RQ1 TTFT-vs-rate (local, directional) ##############"
for sut in ${LOCAL_SUTS}; do
  set_engine_env "${sut}"
  for tr in skewed uniform; do
    echo "===== RQ1 ttft: sut=${sut} traffic=${tr} ====="
    RATE_SUT="${sut}" TRAFFIC="${tr}" RATE_EXTRA_ARGS="--enforce-eager --max-model-len 16384" \
      bash scripts/benchmark/bench_ttft_rate_scale.sh 2>&1 | tee "logs_local/rq1_${sut}_${tr}.log"
  done
done
fi
echo "RUN_ALL_LOCAL_DONE"
