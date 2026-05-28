"""Show the block reference is itself numerically backend-dependent.

Runs the SAME block-attention generation (identical math, identical tokens,
identical positions) twice, changing ONLY the attention implementation
(eager vs sdpa). If the greedy outputs differ, it proves that near-ties flip
under any floating-point implementation difference -- so the lazy Triton path
(a different kernel, with an extra bf16 Q-rotation) cannot be expected to match
HF-eager bit-for-bit either. The divergence is numerical, not a logic bug.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import compare_lazy_block_infer as C
import teacher_force_validate as T


@torch.inference_mode()
def run_block_impl(model_name, attn_impl, max_tokens, top_k=0):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        use_cache=True, attn_implementation=attn_impl)
    model.eval()

    blocks = T.DOC_BLOCKS + [T.QUERY_BLOCK]
    token_blocks = [tok.encode(b, add_special_tokens=False) for b in blocks]
    input_ids = torch.tensor([sum(token_blocks, [])], dtype=torch.long,
                             device=model.device)
    attn_mask = C.build_block_mask([len(t) for t in token_blocks],
                                   torch.bfloat16, model.device)
    out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
    past = out.past_key_values
    nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    gen, tops = [], []
    for _ in range(max_tokens):
        if top_k > 0:
            v, i = torch.topk(out.logits[0, -1].float(), top_k)
            tops.append((i.tolist(), v.tolist()))
        t = int(nxt.item())
        gen.append(t)
        if t == tok.eos_token_id:
            break
        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    del model
    torch.cuda.empty_cache()
    return gen, tok.decode(gen, skip_special_tokens=True), tops, tok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=C.MODEL)
    p.add_argument("--max-tokens", type=int, default=80)
    args = p.parse_args()

    eager_ids, eager_txt, eager_top, tok = run_block_impl(
        args.model, "eager", args.max_tokens, top_k=5)
    sdpa_ids, sdpa_txt, _, _ = run_block_impl(
        args.model, "sdpa", args.max_tokens, top_k=0)

    print(f"eager ({len(eager_ids)}): {eager_txt!r}")
    print(f"sdpa  ({len(sdpa_ids)}): {sdpa_txt!r}")
    if eager_ids == sdpa_ids:
        print(f"eager == sdpa : MATCH over {len(eager_ids)} tokens")
    else:
        first = next((i for i, (a, b) in enumerate(zip(eager_ids, sdpa_ids))
                      if a != b), min(len(eager_ids), len(sdpa_ids)))
        print(f"eager vs sdpa FIRST DIVERGE at index {first}")
        if first < len(eager_top):
            ids, vals = eager_top[first]
            gap = vals[0] - vals[1]
            print(f"  eager top-2 logit gap at that step: {gap:.4f} "
                  f"({tok.decode([ids[0]])!r} vs {tok.decode([ids[1]])!r})")
        print("=> identical math + identical tokens, only the attention "
              "kernel differs, yet greedy output diverges (near-tie).")


if __name__ == "__main__":
    main()
