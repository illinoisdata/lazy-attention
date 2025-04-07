# adapted from vllm/tests/kernels/test_pos_encoding.py
from typing import Callable, Optional

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform
from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from tests.kernels.test_pos_encoding import IS_NEOX_STYLE, DTYPES, HEAD_SIZES,\
    ROTARY_DIMS, NUM_HEADS, BATCH_SIZES, SEQ_LENS, SEEDS, CUDA_DEVICES, TENSORS_SHAPES_FN


@pytest.mark.unit
@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("tensor_shape_fn", TENSORS_SHAPES_FN)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rotary_embedding_with_vllm(    
    is_neox_style: bool,
    tensor_shape_fn: Callable[[int, int, int, int], tuple[int]],
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    rotary_dim: Optional[int],
    dtype: torch.dtype,
    seed: int,
    device: str,
    max_position: int = 8192,
    base: int = 10000,
) -> None:
    """
    Test the rotary embedding layer. Compare with original vllm implementation.
    Use native torch functions. (can be tested on CPU or GPU)
    """
    # overwrite device to cpu if not cuda
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        if head_size > HEAD_SIZES[0]:
            pytest.skip("skip large head size on cpu")
        if seq_len > SEQ_LENS[0]:
            pytest.skip("skip large seq len on cpu")

    if rotary_dim is None:
        rotary_dim = head_size

    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    if rotary_dim is None:
        rotary_dim = head_size
    rope = get_rope(head_size, rotary_dim, max_position, base, is_neox_style)
    rope = rope.to(dtype=dtype, device=torch.get_default_device())

    positions = torch.randint(0, max_position, (batch_size, seq_len))
    query_shape = tensor_shape_fn(batch_size, seq_len, num_heads, head_size)
    unrotated_query = torch.randn(query_shape, dtype=dtype)
    unrotated_key = torch.randn_like(unrotated_query)
    # vllm rotate both query and key
    rotated_query, _ = rope.forward_native(positions, unrotated_query, unrotated_key)

    from minidrag.model_executor.layers.rotary_embedding import apply_patch, revert_patch
    # apply patch
    apply_patch()
    rope = get_rope(head_size, rotary_dim, max_position, base, is_neox_style)
    rope = rope.to(dtype=dtype, device=torch.get_default_device())
    # minidrag rotate only query
    drag_rotated_query, drag_unrotated_key = rope.forward_native(positions, unrotated_query, unrotated_key)

    # compare results
    torch.testing.assert_close(rotated_query, 
                               drag_rotated_query, 
                               atol=get_default_atol(rotated_query), 
                               rtol=get_default_rtol(rotated_query))
    torch.testing.assert_close(unrotated_key, 
                               drag_unrotated_key, 
                               atol=get_default_atol(unrotated_key), 
                               rtol=get_default_rtol(unrotated_key))
    # revert patch
    revert_patch()


# /////////////////////////////////////////////////////////////////////////////
# direct copy of vllm/tests/kernels/test_pos_encoding.py
# /////////////////////////////////////////////////////////////////////////////
@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("tensor_shape_fn", TENSORS_SHAPES_FN)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rotary_embedding(
    is_neox_style: bool,
    tensor_shape_fn: Callable[[int, int, int, int], tuple[int]],
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    rotary_dim: Optional[int],
    dtype: torch.dtype,
    seed: int,
    device: str,
    max_position: int = 8192,
    base: int = 10000,
) -> None:
    if rotary_dim is None:
        rotary_dim = head_size

    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    if rotary_dim is None:
        rotary_dim = head_size
    rope = get_rope(head_size, rotary_dim, max_position, base, is_neox_style)
    rope = rope.to(dtype=dtype, device=torch.get_default_device())

    positions = torch.randint(0, max_position, (batch_size, seq_len))
    query_shape = tensor_shape_fn(batch_size, seq_len, num_heads, head_size)
    query = torch.randn(query_shape, dtype=dtype)
    key = torch.randn_like(query)

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_query, ref_key = rope.forward_native(positions, query, key)
    out_query, out_key = rope.forward(positions, query, key)
    # Compare the results.
    torch.testing.assert_close(out_query,
                               ref_query,
                               atol=get_default_atol(out_query),
                               rtol=get_default_rtol(out_query))
    torch.testing.assert_close(out_key,
                               ref_key,
                               atol=get_default_atol(out_key),
                               rtol=get_default_rtol(out_key))


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("tensor_shape_fn", TENSORS_SHAPES_FN)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_batched_rotary_embedding(
    is_neox_style: bool,
    tensor_shape_fn: Callable[[int, int, int, int], tuple[int]],
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    rotary_dim: Optional[int],
    dtype: torch.dtype,
    seed: int,
    device: str,
    max_position: int = 8192,
    base: int = 10000,
) -> None:
    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    if rotary_dim is None:
        rotary_dim = head_size
    rope = get_rope(head_size, rotary_dim, max_position, base, is_neox_style, {
        "rope_type": "linear",
        "factor": (1, )
    })
    rope = rope.to(dtype=dtype, device=torch.get_default_device())

    positions = torch.randint(0, max_position, (batch_size, seq_len))
    query_shape = tensor_shape_fn(batch_size, seq_len, num_heads, head_size)
    query = torch.randn(query_shape, dtype=dtype)
    key = torch.randn_like(query)

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_query, ref_key = rope.forward_native(positions, query, key)
    out_query, out_key = rope.forward(positions,
                                      query,
                                      key,
                                      offsets=torch.zeros(batch_size * seq_len,
                                                          dtype=torch.long,
                                                          device=device))
    # Compare the results.
    torch.testing.assert_close(out_query,
                               ref_query,
                               atol=get_default_atol(out_query),
                               rtol=get_default_rtol(out_query))
    torch.testing.assert_close(out_key,
                               ref_key,
                               atol=get_default_atol(out_key),
                               rtol=get_default_rtol(out_key))