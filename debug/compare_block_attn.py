"""Compare block_attn_vllm (rotate K pre-kernel) against the references.

Runs the SAME toy documents+query through:
  - the HF one-shot block-attention reference (compare_lazy_block_infer.run_block)
  - block_attn_vllm (vLLM, K rotated in the runner before a standard kernel)

block_attn_vllm uses the same padded positions as the lazy path but moves the
rotation out of the kernel, so it should reproduce the block-attention result.

Must run with VLLM_ENABLE_V1_MULTIPROCESSING=0 so the (parent-process) lazy +
block-attn monkey-patches are active in the engine.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("VLLM_USE_LAZY_ATTENTION", "1")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN_VLLM_V1")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch

import compare_lazy_block_infer as C


def run_block_attn(model_name: str, max_tokens: int):
    import block_attn_vllm.__vllm__  # noqa: F401  (applies patches)
    import vllm.transformers_utils.tokenizer as vtok
    from lazy.entrypoints.llm import LazyLLM
    from vllm import SamplingParams

    o = vtok.get_cached_tokenizer
    vtok.get_cached_tokenizer = lambda t: (
        setattr(t, "all_special_tokens_extended", t.all_special_tokens)
        or o(t)) if not hasattr(t, "all_special_tokens_extended") else o(t)

    llm = LazyLLM(model=model_name, gpu_memory_utilization=0.9,
                  enable_prefix_caching=True, trust_remote_code=True,
                  enforce_eager=True)
    out = llm.generate(
        prompts=[C.QUERY_BLOCK],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=max_tokens),
        document_seqs=[C.DOC_BLOCKS],
    )[0].outputs[0]
    return list(out.token_ids), out.text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=C.MODEL)
    p.add_argument("--max-tokens", type=int, default=40)
    args = p.parse_args()

    block_ids, block_txt, _ = C.run_block(args.model, args.max_tokens)
    torch.cuda.empty_cache()
    ba_ids, ba_txt = run_block_attn(args.model, args.max_tokens)

    print(f"HF-block   ({len(block_ids)}): {block_txt!r}")
    print(f"block_attn ({len(ba_ids)}): {ba_txt!r}")
    if block_ids == ba_ids:
        print(f"MATCH over {len(block_ids)} tokens")
    else:
        d = next((i for i, (a, b) in enumerate(zip(block_ids, ba_ids))
                  if a != b), min(len(block_ids), len(ba_ids)))
        print(f"first diverge @ {d}")


if __name__ == "__main__":
    main()
