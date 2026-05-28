"""Parity check: default LOAD path vs LAZY_DECODE_COMPUTE_COS_SIN=1 COMPUTE path.

The env var is read per decode-kernel launch, so we flip it between generate()
calls in ONE process and compare the generated token ids. They should match.
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
                  enforce_eager=True, max_num_seqs=16, max_model_len=8192)
    sp = SamplingParams(temperature=0.0, max_tokens=48, ignore_eos=True)
    docs = [["<|start_header_id|>system\nAnswer using the facts.\n\n"]
            + [f"- Fact {i}: item {i} value {(i * 7) % 97} tag t{i}.\n"
               for i in range(12)]]
    prompts = ["Question: summarize the facts in detail please.\n\n\n"]

    def gen(mode):
        os.environ["LAZY_DECODE_COMPUTE_COS_SIN"] = mode
        outs = llm.generate(prompts=prompts, sampling_params=sp,
                            document_seqs=docs)
        return list(outs[0].outputs[0].token_ids)

    load_ids = gen("0")
    comp_ids = gen("1")
    n = min(len(load_ids), len(comp_ids))
    match = sum(int(a == b) for a, b in zip(load_ids, comp_ids))
    print(f"LOAD    tokens[:12]={load_ids[:12]}")
    print(f"COMPUTE tokens[:12]={comp_ids[:12]}")
    print(f"match {match}/{n}  ({'IDENTICAL' if match == n and len(load_ids)==len(comp_ids) else 'DIVERGES'})")


if __name__ == "__main__":
    main()
