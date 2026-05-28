"""Utility functions for Block-Attention on vLLM.

Block-Attention re-uses each document's KV cache but re-positions it. We follow
the literal Block-Attention design: documents are encoded (in an earlier step)
at their LOCAL positions 0..L-1, and when a document is needed by a query we
un-rotate every key back to position 0 and then rotate it to the target
contiguous position the document occupies in the assembled sequence. A standard
attention kernel then runs over keys that already sit at their target absolute
positions (the query keeps its own position; the kernel does no Q rotation).

The key cache uses vLLM's paged layout
    key_cache[block] : [num_kv_heads, head_size // x, block_size, x]
where the head dimension d is split as d = (d // x) * x + (d % x). We reconstruct
the contiguous head dimension, apply the rotations, and write it back.

Created by Haocheng at 2025/09/09; literal two-phase rotation 2026-05-28.
"""

from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """transformers.models.llama.modeling_llama.rotate_half (NeoX style)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def cos_sin_rows(cos_sin_cache: torch.Tensor, positions: torch.Tensor,
                 dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position cos/sin taken from the model's own RoPE cache.

    cos_sin_cache has the vLLM layout [max_pos, rotary_dim] where the first half
    of each row is cos and the second half is sin (each rotary_dim // 2). Using
    the model's own cache guarantees the rotation matches exactly how the keys
    were originally rotated (same scaling, same dtype). Returns cos/sin of shape
    [len(positions), head_dim] (each half duplicated for the NeoX convention).
    """
    rows = cos_sin_cache[positions]                    # [P, rotary_dim]
    half = rows.shape[-1] // 2
    cos = rows[:, :half].repeat(1, 2).to(dtype)        # [P, head_dim]
    sin = rows[:, half:].repeat(1, 2).to(dtype)
    return cos, sin


def place_paged_block_keys(key_cache: torch.Tensor, block_id: int,
                           cos_sin_cache: torch.Tensor,
                           from_start: int, to_start: int,
                           dtype: torch.dtype) -> None:
    """Re-position one paged block's keys, the literal Block-Attention way.

    Token at slot ``o`` in the block was stored rotated at position
    ``from_start + o`` (its local position during document prefill) and must end
    up rotated at ``to_start + o`` (its target contiguous position). We do it in
    two explicit steps:

        1. un-rotate to position 0:  R(-(from_start + o))
        2. rotate to the target:     R(+(to_start + o))

    Expressing it as two steps follows the Block-Attention design directly and
    stays correct even when the stored positions are not a simple uniform shift
    of the targets (e.g. if document locality were ever broken).

    key_cache: [num_blocks, num_kv_heads, head_size // x, block_size, x]
    """
    nkv, hdx, bs, x = key_cache.shape[1:]
    head_dim = hdx * x
    blk = key_cache[block_id]                          # [nkv, hdx, bs, x] view
    k = blk.permute(0, 2, 1, 3).reshape(nkv, bs, head_dim).clone()  # [nkv,bs,hd]

    arange = torch.arange(bs, device=k.device)
    from_pos = from_start + arange                     # [bs]
    to_pos = to_start + arange
    cos_f, sin_f = cos_sin_rows(cos_sin_cache, from_pos, dtype)  # [bs, hd]
    cos_t, sin_t = cos_sin_rows(cos_sin_cache, to_pos, dtype)

    # Step 1: un-rotate each token to position 0 -> R(-from) = k*cos - rot*sin.
    k = (k * cos_f) - (rotate_half(k) * sin_f)
    # Step 2: rotate to the target contiguous position -> R(+to).
    k = (k * cos_t) + (rotate_half(k) * sin_t)

    # copy_ persists through the split_kv_cache view to the real paged cache.
    blk.copy_(k.reshape(nkv, bs, hdx, x).permute(0, 2, 1, 3))


def copy_paged_block(key_cache: torch.Tensor, value_cache: torch.Tensor,
                     dst_block: int, src_block: int) -> None:
    """Copy one paged block's K and V (copy-on-write for shared docs)."""
    key_cache[dst_block].copy_(key_cache[src_block])
    value_cache[dst_block].copy_(value_cache[src_block])


def block_key_norm(key_cache: torch.Tensor, block_id: int) -> float:
    """L2 norm of a paged block's keys -- a cheap content sanity check."""
    return float(key_cache[block_id].float().norm())
