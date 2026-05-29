"""Block-Attention inference script following the official Block-Attention pipeline.

Block-Attention uses a 4D attention mask during the prefill phase (not supported by
model.generate()), so we do manual token-by-token generation:
1. Prefill: forward pass with the block-attention mask to fill KV cache
2. Decode: generate tokens one by one with standard causal attention

Reference: https://github.com/TemporaryLoRA/Block-Attention
"""
import argparse
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", use_cache=True,
    )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def block_generate(model, tokenizer, blocks: List[str], max_new_tokens: int = 128) -> str:
    """Generate using Block-Attention: manual prefill + autoregressive decode.

    Prefill phase uses the 4D block-attention mask via model.forward().
    Decode phase uses standard causal attention (each new token sees everything).
    """
    from src.data.block import (build_attention_mask,
                                convert_attention_mask_to_model_required)

    # Tokenize each block separately
    block_token_counts = []
    all_ids = []
    for b in blocks:
        ids = tokenizer.encode(b, add_special_tokens=False)
        all_ids.extend(ids)
        block_token_counts.append(len(ids))

    input_ids = torch.tensor([all_ids], dtype=torch.int64, device=model.device)
    total_len = len(all_ids)

    # Build 4D block-attention mask for prefill
    helper = torch.tril(torch.ones(total_len + 64, total_len + 64, dtype=torch.bool))
    attn_mask = build_attention_mask(
        local_attention_block_tokens=torch.tensor(block_token_counts[:-1], dtype=torch.long),
        global_attention_block_tokens=torch.tensor(block_token_counts[-1], dtype=torch.long),
        lower_triangular_matrix=helper,
    )
    attn_mask = convert_attention_mask_to_model_required(attn_mask)
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).to(model.device)

    # Prefill: forward pass with block-attention mask to populate KV cache
    outputs = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)  # [1, 1]

    generated = []
    for _ in range(max_new_tokens - 1):
        generated.append(next_token.item())

        if next_token.item() == tokenizer.eos_token_id:
            break

        # Decode: each new token sees all previous tokens (standard causal)
        outputs = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

    # Include last token
    if next_token.item() != tokenizer.eos_token_id:
        generated.append(next_token.item())

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_keys(keys: torch.Tensor, delta: int, model) -> torch.Tensor:
    """Apply a uniform RoPE shift R(delta) to every cached key in `keys`.

    `keys` has shape [batch, num_kv_heads, seq, head_dim] and was rotated at
    local positions 0..seq-1. Shifting by `delta` re-rotates it to global
    positions delta..delta+seq-1 (the same uniform rotation for every token,
    which is exactly what block re-positioning needs).
    """
    if delta == 0:
        return keys
    pos = torch.tensor([[delta]], dtype=torch.long, device=keys.device)
    cos, sin = model.model.rotary_emb(keys, pos)  # each [1, 1, head_dim]
    cos = cos.to(keys.dtype).unsqueeze(1)          # [1, 1, 1, head_dim]
    sin = sin.to(keys.dtype).unsqueeze(1)
    return (keys * cos) + (_rotate_half(keys) * sin)


@torch.no_grad()
def block_generate_separated(model, tokenizer, blocks: List[str],
                             max_new_tokens: int = 128) -> str:
    """Block-Attention via the original step-separated flow.

    1. Encode each document block independently at LOCAL positions (0..L-1),
       producing its KV cache (no cross-document attention).
    2. Rotate each document's cached K to its GLOBAL position (uniform RoPE
       shift), then concatenate all documents into one cache.
    3. Prefill the query block against the merged cache (the query attends to
       every document + causally to itself).
    4. Generate token-by-token.

    Unlike `block_generate`, no 4D mask is used: the block structure comes from
    encoding documents separately, and positioning comes from re-rotation.
    """
    from transformers.cache_utils import DynamicCache

    device = model.device
    num_layers = model.config.num_hidden_layers
    *docs, query = blocks

    # Steps 1 + 2: per-document KV at local positions, rotated to global.
    per_layer_keys: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]
    per_layer_vals: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]
    global_offset = 0
    for doc in docs:
        ids = tokenizer.encode(doc, add_special_tokens=False)
        inp = torch.tensor([ids], dtype=torch.int64, device=device)
        pos = torch.arange(len(ids), device=device).unsqueeze(0)
        cache = model(input_ids=inp, position_ids=pos,
                      use_cache=True).past_key_values
        for layer in range(num_layers):
            keys = _rotate_keys(cache.layers[layer].keys, global_offset, model)
            per_layer_keys[layer].append(keys)
            per_layer_vals[layer].append(cache.layers[layer].values)
        global_offset += len(ids)

    merged = DynamicCache()
    for layer in range(num_layers):
        merged.update(torch.cat(per_layer_keys[layer], dim=2),
                      torch.cat(per_layer_vals[layer], dim=2), layer)

    total_doc_len = global_offset

    # Step 3: query prefill against the merged document cache.
    q_ids = tokenizer.encode(query, add_special_tokens=False)
    q_inp = torch.tensor([q_ids], dtype=torch.int64, device=device)
    q_pos = torch.arange(total_doc_len, total_doc_len + len(q_ids),
                         device=device).unsqueeze(0)
    out = model(input_ids=q_inp, position_ids=q_pos,
                past_key_values=merged, use_cache=True)
    past = out.past_key_values
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    # Step 4: decode.
    generated: List[int] = []
    cur_pos = total_doc_len + len(q_ids)
    for _ in range(max_new_tokens):
        generated.append(int(next_token.item()))
        if next_token.item() == tokenizer.eos_token_id:
            break
        pos = torch.tensor([[cur_pos]], dtype=torch.int64, device=device)
        out = model(input_ids=next_token, position_ids=pos,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        cur_pos += 1

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def full_attention_generate(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    """Standard full-attention generation via model.generate()."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3968).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--mode",
        choices=["block", "separated", "full", "both"],
        default="both",
        help="block=one-shot 4D mask, separated=per-doc encode+rotate+query, "
             "full=plain causal, both=block+full")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    print(f"Model loaded: {args.model}\n")

    # Example: multi-document RAG question
    blocks = [
        "<|start_header_id|>system\nYou are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference documents that may help you in answering the user's question.\n\n",
        "- Title: Polish-Russian War (film)\nPolish-Russian War (Wojna polsko-ruska) is a 2009 Polish film directed by Xawery Zulawski based on the novel Polish-Russian War under the white-red flag by Dorota Maslowska.\n",
        "- Title: Xawery Zulawski\nXawery Zulawski (born 22 December 1971 in Warsaw) is a Polish film director. He is the son of actress Malgorzata Braunek and director Andrzej Zulawski.\n",
        "- Title: Viktor Yeliseyev\nViktor Petrovich Yeliseyev (born June 9, 1950) is a Russian general.\n",
        "Please write a high-quality answer for the given question using only the provided search documents.\nQuestion: Who is the mother of the director of film Polish-Russian War (Film)?\n\n\n",
    ]
    prompt = "".join(blocks)

    print("=" * 70)
    print("Question: Who is the mother of the director of film Polish-Russian War?")
    print("=" * 70)

    if args.mode in ("block", "both"):
        answer = block_generate(model, tokenizer, blocks, args.max_new_tokens)
        print(f"\n[Block-Attention (1-shot mask)] {answer}")

    if args.mode == "separated":
        answer = block_generate_separated(model, tokenizer, blocks,
                                          args.max_new_tokens)
        print(f"\n[Block-Attention (separated)]   {answer}")

    if args.mode in ("full", "both"):
        answer = full_attention_generate(model, tokenizer, prompt, args.max_new_tokens)
        print(f"\n[Full-Attention]  {answer}")


if __name__ == "__main__":
    main()
