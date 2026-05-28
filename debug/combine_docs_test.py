"""Single combined-document test.

Concatenates the system text + all documents into ONE block, followed by the
query. This collapses the multi-block lazy machinery (the previous case had 3
separate per-doc rotation offsets 28/42/45) down to a single document, so we
can see whether divergence from the block reference is reduced.

Reports both greedy block-vs-lazy and teacher-forced per-position agreement,
so we can compare against the multi-doc number (93.5%).
"""

from __future__ import annotations

import argparse

import torch

import compare_lazy_block_infer as C
import teacher_force_validate as T


# Everything before the question, fused into one document block.
COMBINED_DOC = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are an intelligent AI assistant. Answer the user's question using "
    "only the provided documents. Write a detailed, multi-sentence answer.\n\n"
    "- Title: Polish-Russian War (film)\n"
    "Polish-Russian War (Wojna polsko-ruska) is a 2009 Polish film directed by "
    "Xawery Zulawski based on the novel Polish-Russian War under the white-red "
    "flag by Dorota Maslowska.\n"
    "- Title: Xawery Zulawski\n"
    "Xawery Zulawski (born 22 December 1971 in Warsaw) is a Polish film "
    "director. He is the son of actress Malgorzata Braunek and director Andrzej "
    "Zulawski.\n"
    "- Title: Andrzej Zulawski\n"
    "Andrzej Zulawski (1940-2016) was a Polish film director known for "
    "challenging, avant-garde films. He was married to Malgorzata Braunek.\n"
)

DOC_BLOCKS = [COMBINED_DOC]
QUERY_BLOCK = T.QUERY_BLOCK


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=C.MODEL)
    p.add_argument("--max-tokens", type=int, default=80)
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    # Point both the compare machinery and the teacher-forcing helper at the
    # single combined document.
    C.DOC_BLOCKS = DOC_BLOCKS
    C.QUERY_BLOCK = QUERY_BLOCK
    T.DOC_BLOCKS = DOC_BLOCKS

    # 1) plain greedy block vs lazy.
    block_ids, block_txt, _ = C.run_block(args.model, args.max_tokens)
    torch.cuda.empty_cache()
    lazy_ids, lazy_txt, _ = C.run_lazy(args.model, args.max_tokens)
    print(f"block ({len(block_ids)}): {block_txt!r}")
    print(f"lazy  ({len(lazy_ids)}): {lazy_txt!r}")
    if block_ids == lazy_ids:
        print(f"GREEDY MATCH over {len(block_ids)} tokens")
    else:
        d = next((i for i, (a, b) in enumerate(zip(block_ids, lazy_ids))
                  if a != b), min(len(block_ids), len(lazy_ids)))
        print(f"GREEDY diverge @ {d}")

    # 2) teacher-forced agreement on the block-reference sequence.
    ref = block_ids[:-1] if block_ids and block_ids[-1] == tok.eos_token_id \
        else block_ids
    torch.cuda.empty_cache()
    _, lazy_argmax = T.lazy_prompt_argmax(args.model, QUERY_BLOCK, ref, tok)
    n = len(ref)
    ans = lazy_argmax[-n:]
    agree = sum(1 for j in range(n) if ans[j] == ref[j])
    print(f"teacher-forced agreement (single doc): {agree}/{n} "
          f"({100.0 * agree / max(n, 1):.1f}%)")


if __name__ == "__main__":
    main()
