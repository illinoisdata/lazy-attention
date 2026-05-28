"""Decode-latency benchmark for the in-kernel cos/sin change.

Generates many tokens over a many-document lazy request (so each decode step
does many per-doc-block Q rotations -> cos/sin work is visible) and reports
decode tokens/sec. Run the same script before/after the kernel change (via
`git stash`) to compare COMPUTE-cos/sin vs LOAD-cos/sin.

Run: .venv/bin/python debug/bench_decode_cossin.py
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("VLLM_USE_LAZY_ATTENTION", "1")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN_VLLM_V1")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import compare_lazy_block_infer as C

# Many short docs -> many cached doc blocks the decode query rotates over.
DOCS = [f"- Fact {i}: item number {i} has value {i * 7 % 97} and tag t{i}.\n"
        for i in range(15)]
SYSTEM = ("<|start_header_id|>system\nAnswer using the facts.\n\n",)
QUERY = "Question: Summarize the facts in one paragraph.\n\n\n"

MAX_TOKENS = 128
ITERS = 5


def main():
    import lazy.__vllm__  # noqa: F401  (applies lazy patches)
    import vllm.transformers_utils.tokenizer as vtok
    from lazy.entrypoints.llm import LazyLLM
    from vllm import SamplingParams

    o = vtok.get_cached_tokenizer
    vtok.get_cached_tokenizer = lambda t: (
        setattr(t, "all_special_tokens_extended", t.all_special_tokens)
        or o(t)) if not hasattr(t, "all_special_tokens_extended") else o(t)

    llm = LazyLLM(model=C.MODEL, gpu_memory_utilization=0.9,
                  enable_prefix_caching=True, trust_remote_code=True,
                  enforce_eager=True)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True)

    def run_once():
        t0 = time.perf_counter()
        out = llm.generate(prompts=[QUERY],
                           sampling_params=sp,
                           document_seqs=[list(SYSTEM) + DOCS])[0].outputs[0]
        dt = time.perf_counter() - t0
        return len(out.token_ids), dt

    run_once()  # warmup (compile kernels)
    times, toks = [], 0
    for _ in range(ITERS):
        n, dt = run_once()
        times.append(dt)
        toks = n
    times.sort()
    median = times[len(times) // 2]
    print(f"tokens={toks} iters={ITERS} median_total={median*1000:.1f}ms "
          f"tok/s={toks/median:.1f} all_ms={[round(t*1000,1) for t in times]}")


if __name__ == "__main__":
    main()
