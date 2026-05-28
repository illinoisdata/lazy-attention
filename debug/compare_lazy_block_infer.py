"""Minimal lazy-vs-block inference check.

The block reference uses a 4D block-attention prefill mask with Transformers.
The lazy path gives the same document blocks to LazyLLM through document_seqs
and uses only the query block as the prompt.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lazy_attn"))

os.environ.setdefault("VLLM_USE_LAZY_ATTENTION", "1")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN_VLLM_V1")


MODEL = "hxia7/Llama-3.2-1B-Block-FT"


DOC_BLOCKS = [
    "<|start_header_id|>system\n"
    "You are a helpful assistant. Answer using the documents.\n\n",
    "Hello world.\n",
    "The cat sat.\n",
]

QUERY_BLOCK = "Question: What sat?\n\n\n"


def build_block_mask(block_lens: list[int], dtype: torch.dtype,
                     device: torch.device) -> torch.Tensor:
    total_len = sum(block_lens)
    query_len = block_lens[-1]
    helper = torch.tril(
        torch.ones(total_len + 64, total_len + 64, dtype=torch.bool))

    mask = torch.zeros(total_len, total_len, dtype=torch.bool)
    offset = 0
    for block_len in block_lens[:-1]:
        mask[offset:offset + block_len, offset:offset + block_len] = (
            helper[:block_len, :block_len])
        offset += block_len
    mask[offset:] = helper[total_len - query_len:total_len, :total_len]

    mask = (1.0 - mask.to(dtype)).masked_fill_(~mask, torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0).to(device)


def format_top_tokens(tokenizer, ids: list[int], values: list[float]) -> str:
    return ", ".join(
        f"{token_id}:{tokenizer.decode([token_id])!r}:{value:.4f}"
        for token_id, value in zip(ids, values)
    )


@torch.inference_mode()
def run_block(model_name: str, max_tokens: int,
              top_k: int = 0) -> tuple[list[int], str, list[dict[str, list]]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=True,
        attn_implementation="eager",
    )
    model.eval()

    blocks = DOC_BLOCKS + [QUERY_BLOCK]
    token_blocks = [tokenizer.encode(b, add_special_tokens=False)
                    for b in blocks]
    input_ids = torch.tensor(
        [sum(token_blocks, [])], dtype=torch.long, device=model.device)
    attn_mask = build_block_mask(
        [len(ids) for ids in token_blocks], torch.bfloat16, model.device)

    out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
    past = out.past_key_values
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    generated: list[int] = []
    topks: list[dict[str, list]] = []
    for _ in range(max_tokens):
        if top_k > 0:
            vals, idxs = torch.topk(out.logits[0, -1].float(), top_k)
            topks.append({
                "ids": idxs.cpu().tolist(),
                "values": vals.cpu().tolist(),
            })
        token_id = int(next_token.item())
        generated.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break
        out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return generated, text, topks


def print_input_tokens(model_name: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    block_token_blocks = [
        tokenizer.encode(b, add_special_tokens=False)
        for b in DOC_BLOCKS + [QUERY_BLOCK]
    ]
    lazy_doc_blocks = [
        tokenizer(doc, add_special_tokens=True)["input_ids"][1:]
        for doc in DOC_BLOCKS
    ]
    lazy_query = tokenizer(
        QUERY_BLOCK, add_special_tokens=True)["input_ids"][1:]

    print("input token check:")
    for idx, (block_ids, lazy_ids) in enumerate(
            zip(block_token_blocks[:-1], lazy_doc_blocks)):
        print(f"  doc{idx} block: {block_ids}")
        print(f"  doc{idx} lazy:  {lazy_ids}")
        print(f"  doc{idx} equal: {block_ids == lazy_ids}")
    print(f"  query block: {block_token_blocks[-1]}")
    print(f"  query lazy:  {lazy_query}")
    print(f"  query equal: {block_token_blocks[-1] == lazy_query}")


def run_lazy(model_name: str, max_tokens: int,
             top_k: int = 0) -> tuple[list[int], str, list]:
    import lazy.__vllm__  # noqa: F401
    import vllm.transformers_utils.tokenizer as vtok
    from lazy.entrypoints.llm import LazyLLM
    from vllm import SamplingParams

    orig_get_cached_tokenizer = vtok.get_cached_tokenizer
    vtok.get_cached_tokenizer = lambda t: (
        setattr(t, "all_special_tokens_extended", t.all_special_tokens)
        or orig_get_cached_tokenizer(t)
    ) if not hasattr(t, "all_special_tokens_extended") else (
        orig_get_cached_tokenizer(t))

    llm = LazyLLM(
        model=model_name,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        trust_remote_code=True,
        enforce_eager=True,
    )
    outputs = llm.generate(
        prompts=[QUERY_BLOCK],
        sampling_params=SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=top_k if top_k > 0 else None,
        ),
        document_seqs=[DOC_BLOCKS],
    )
    sample = outputs[0].outputs[0]
    return list(sample.token_ids), sample.text, (sample.logprobs or [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--print-inputs", action="store_true")
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args()

    if args.print_inputs:
        print_input_tokens(args.model)

    block_ids, block_text, block_topks = run_block(
        args.model, args.max_tokens, args.top_k)
    torch.cuda.empty_cache()
    lazy_ids, lazy_text, lazy_logprobs = run_lazy(
        args.model, args.max_tokens, args.top_k)

    print(f"block ids: {block_ids}")
    print(f"lazy  ids: {lazy_ids}")
    print(f"block text: {block_text!r}")
    print(f"lazy  text: {lazy_text!r}")
    if args.top_k > 0:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        print("block top-k by step:")
        for step, topk in enumerate(block_topks):
            print(f"  step {step}: "
                  f"{format_top_tokens(tokenizer, topk['ids'], topk['values'])}")
        print("lazy logprobs by step:")
        for step, logprob_dict in enumerate(lazy_logprobs):
            items = list(logprob_dict.items())[:args.top_k]
            ids = [int(token_id) for token_id, _ in items]
            vals = [float(logprob.logprob) for _, logprob in items]
            print(f"  step {step}: {format_top_tokens(tokenizer, ids, vals)}")

    if block_ids == lazy_ids:
        print("MATCH")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        first_diff = next(
            i for i, (block_id, lazy_id) in enumerate(zip(block_ids, lazy_ids))
            if block_id != lazy_id)
        print(f"first mismatch index: {first_diff}")
        print(
            "block token: "
            f"{block_ids[first_diff]} "
            f"{tokenizer.decode([block_ids[first_diff]])!r}")
        print(
            "lazy  token: "
            f"{lazy_ids[first_diff]} "
            f"{tokenizer.decode([lazy_ids[first_diff]])!r}")
        print("MISMATCH")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
