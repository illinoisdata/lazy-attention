import torch

def rotary_embedding_q(
        positions: torch.Tensor, 
        query: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor, 
        is_neox: bool) -> None:
    torch.ops._C.rotary_embedding_q(positions, query, head_size,
                                    cos_sin_cache, is_neox)


def batched_rotary_embedding_q(positions: torch.Tensor, query: torch.Tensor,
                               head_size: int,
                               cos_sin_cache: torch.Tensor, is_neox: bool,
                               rot_dim: int,
                               cos_sin_cache_offsets: torch.Tensor) -> None:
    torch.ops._C.batched_rotary_embedding_q(positions, query, head_size,
                                            cos_sin_cache, is_neox, rot_dim,
                                            cos_sin_cache_offsets)
    

def apply_patch():
    import vllm._custom_ops
    vllm._custom_ops.rotary_embedding_q = rotary_embedding_q
    vllm._custom_ops.batched_rotary_embedding_q = batched_rotary_embedding_q
