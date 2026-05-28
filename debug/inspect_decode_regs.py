"""Dump the compiled decode-kernel register usage / spills / occupancy inputs.

Register count + spills are what drive SM occupancy, which is the suspected
reason in-kernel cos/sin compute loses to the cos_sin_cache load on this
decode kernel. Run this in BOTH states (compute working tree, and `git stash`
-> load baseline) and compare n_regs / n_spills for
`kernel_paged_attention_2d_llama`.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_USE_LAZY_ATTENTION", "1")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN_VLLM_V1")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import compare_lazy_block_infer as C


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
                  enforce_eager=True, max_num_seqs=16, max_model_len=4096)
    sp = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    docs = [["<|start_header_id|>system\nAnswer.\n\n"]
            + [f"- Fact {i}: value {i}. " for i in range(6)]]
    llm.generate(prompts=["Question: summarize.\n\n\n"], sampling_params=sp,
                 document_seqs=docs)

    from lazy.attention.ops.models import llama_v1
    for name in ("kernel_paged_attention_2d_llama",
                 "kernel_paged_attention_2d_llama_lazy_only"):
        fn = getattr(llama_v1, name)
        cache = getattr(fn, "cache", {})
        print(f"\n=== {name} : {sum(len(v) for v in cache.values())} variant(s) ===")
        for dev, variants in cache.items():
            for key, ck in variants.items():
                n_regs = getattr(ck, "n_regs", None)
                n_spills = getattr(ck, "n_spills", None)
                md = getattr(ck, "metadata", None)
                num_warps = getattr(md, "num_warps", None) if md else None
                shared = getattr(md, "shared", None) if md else None
                print(f"  dev={dev} n_regs={n_regs} n_spills={n_spills} "
                      f"num_warps={num_warps} shared={shared}")


if __name__ == "__main__":
    main()
