"""Longer block_attn vs HF block-reference check with MULTI-BLOCK documents.

The toy in compare_lazy_block_infer uses single-block docs (from_start == 0
only). This exercises documents longer than block_size (16) so each document
spans several blocks (from_start = 0, 16, 32, ...), validating the per-block
placement (to_start = abs_rot_pos + block_offset * block_size).

Run with VLLM_ENABLE_V1_MULTIPROCESSING=0 so the patches are active in-process.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("VLLM_USE_LAZY_ATTENTION", "1")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN_VLLM_V1")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch

import compare_lazy_block_infer as C

DOC_BLOCKS = [
    "<|start_header_id|>system\n"
    "You are a helpful assistant. Answer the question using only the "
    "documents provided below. Be concise and precise in your answer.\n\n",
    "Document 1: The Eiffel Tower is located in Paris, France. It was "
    "completed in 1889 and stands 330 metres tall. It was the tallest "
    "man-made structure in the world for 41 years.\n",
    "Document 2: The Great Wall of China is over 21000 kilometres long. "
    "Construction began more than 2000 years ago across several dynasties "
    "to protect against northern invasions.\n",
]

QUERY_BLOCK = "Question: In which city is the Eiffel Tower located?\n\n\n"


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
        prompts=[QUERY_BLOCK],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=max_tokens),
        document_seqs=[DOC_BLOCKS],
    )[0].outputs[0]
    return list(out.token_ids), out.text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=C.MODEL)
    p.add_argument("--max-tokens", type=int, default=40)
    args = p.parse_args()

    # Point the shared block reference at our longer documents.
    C.DOC_BLOCKS = DOC_BLOCKS
    C.QUERY_BLOCK = QUERY_BLOCK

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
