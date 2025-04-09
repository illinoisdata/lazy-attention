"""RoPE for Dynamic RAG

Specifically, we return a rotary embedding object that just rotates the query 
while normally key and query are rotated.
"""

import torch
from typing import Optional, Tuple
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.rotary_embedding import _apply_rotary_emb

# @CustomOp.register("rotary_embedding")
# class RotaryEmbedding(CustomOp):
def new_forward_native(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A PyTorch-native implementation of forward()."""
    if offsets is not None:
        positions = positions + offsets
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = self.cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    # only query is rotated
    query_shape = query.shape
    query = query.view(num_tokens, -1, self.head_size)
    query_rot = query[..., :self.rotary_dim]
    query_pass = query[..., self.rotary_dim:]
    query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    return query, key

def new_forward_cuda(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor] = None,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from vllm import _custom_ops as ops
    self.cos_sin_cache = self.cos_sin_cache.to(query.device,
                                                dtype=query.dtype)
    if offsets is not None:
        ops.batched_rotary_embedding_q(positions, query, self.head_size,
                                        self.cos_sin_cache, self.is_neox_style,
                                        self.rotary_dim, offsets)
    else:
        ops.rotary_embedding_q(positions, query, self.head_size,
                                self.cos_sin_cache, self.is_neox_style)
    return query, key

original_forward_cuda = None
original_forward_native = None

def apply_patch():
    import vllm.model_executor.layers.rotary_embedding
    # record the original forward_cuda and forward_native
    global original_forward_cuda
    global original_forward_native
    original_forward_cuda = vllm.model_executor.layers.rotary_embedding.RotaryEmbedding.forward_cuda
    original_forward_native = vllm.model_executor.layers.rotary_embedding.RotaryEmbedding.forward_native

    # apply the new forward_cuda and forward_native
    vllm.model_executor.layers.rotary_embedding.RotaryEmbedding.forward_cuda = new_forward_cuda
    vllm.model_executor.layers.rotary_embedding.RotaryEmbedding.forward_native = new_forward_native

def revert_patch():
    import vllm.model_executor.layers.rotary_embedding
    vllm.model_executor.layers.rotary_embedding.RotaryEmbedding.forward_cuda = original_forward_cuda
    vllm.model_executor.layers.rotary_embedding.RotaryEmbedding.forward_native = original_forward_native
