#!/usr/bin/env bash
# Smoke-test the *unmodified* vLLM backend (no LazyAttention patches).
# For the LazyAttention path use scripts/validate_lazy.py instead.
set -euo pipefail

python - <<'PY'
from vllm import LLM, SamplingParams

llm = LLM(model="hxia7/Llama-3.2-1B-Block-FT", max_model_len=2048,
          gpu_memory_utilization=0.6, enforce_eager=True)
out = llm.generate(["The capital of France is"],
                   SamplingParams(max_tokens=16, temperature=0))
print("GENERATED:", repr(out[0].outputs[0].text))
PY
