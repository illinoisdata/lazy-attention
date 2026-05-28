"""High-concurrency decode benchmark for the in-kernel cos/sin change.

At batch=1 the cos_sin_cache fits in cache, so loading it is ~free and computing
cos/sin loses. Under many concurrent decodes the memory subsystem is saturated;
computing cos/sin (SFU, not memory BW) may then win. This runs NREQ concurrent
lazy requests (shared docs, distinct queries) and reports aggregate decode
tok/s + median decode-kernel ms (with LAZY_DECODE_WRAPPER_PROFILE=1).

Run before/after the kernel change (via `git stash`) to compare compute vs load.
Env: NREQ (default 128), MAXTOK (default 64).
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("VLLM_USE_LAZY_ATTENTION", "1")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN_VLLM_V1")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import compare_lazy_block_infer as C

NREQ = int(os.environ.get("NREQ", "128"))
MAXTOK = int(os.environ.get("MAXTOK", "64"))
NDOCS = int(os.environ.get("NDOCS", "15"))
DOCREP = int(os.environ.get("DOCREP", "1"))  # repeat doc text -> longer context

QUERIES = [f"Question {i}: summarize fact {i % NDOCS} in detail please.\n\n\n"
           for i in range(NREQ)]


def main():
    import lazy.__vllm__  # noqa: F401
    import vllm.transformers_utils.tokenizer as vtok
    from lazy.entrypoints.llm import LazyLLM
    from vllm import SamplingParams

    o = vtok.get_cached_tokenizer
    vtok.get_cached_tokenizer = lambda t: (
        setattr(t, "all_special_tokens_extended", t.all_special_tokens)
        or o(t)) if not hasattr(t, "all_special_tokens_extended") else o(t)

    llm = LazyLLM(model=C.MODEL, gpu_memory_utilization=0.9,
                  enable_prefix_caching=True, trust_remote_code=True,
                  enforce_eager=True, max_num_seqs=256)
    sp = SamplingParams(temperature=0.0, max_tokens=MAXTOK, ignore_eos=True)
    # Distinct docs per request: avoids a shared prefix (which would trigger
    # vLLM cascade attention, unsupported by the lazy backend) while still
    # batching NREQ concurrent decodes. DOCREP lengthens each doc (context).
    filler = ("The quick brown fox jumps over the lazy dog near the river. "
              * DOCREP)
    docs = [[f"<|start_header_id|>system\nReq {r}. Answer using the facts.\n\n"]
            + [f"- Req {r} Fact {i}: item {i} value {(i + r) % 97} tag t{i}. "
               f"{filler}\n" for i in range(NDOCS)]
            for r in range(NREQ)]

    def run():
        t0 = time.perf_counter()
        outs = llm.generate(prompts=QUERIES, sampling_params=sp,
                            document_seqs=docs)
        dt = time.perf_counter() - t0
        n = sum(len(o.outputs[0].token_ids) for o in outs)
        return n, dt

    run()  # warmup
    ns, dts = zip(*[run() for _ in range(3)])
    dt = sorted(dts)[len(dts) // 2]
    n = ns[0]
    print(f"NREQ={NREQ} MAXTOK={MAXTOK} total_tokens={n} "
          f"median_wall={dt:.3f}s decode_tok/s={n/dt:.0f}")


if __name__ == "__main__":
    main()
