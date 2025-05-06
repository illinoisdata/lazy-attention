# adapted from vllm/tests/kernels/test_prefix_prefill.py

import random
import time
from collections.abc import Callable
from typing import Optional, Tuple

import pytest
import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalFromBottomRightMask

# vllm original functions
from vllm.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode)
from vllm.attention.ops.prefix_prefill import context_attention_fwd
# minidrag custom functions
from minidrag.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode as minidrag_chunked_prefill_paged_decode)
from minidrag.attention.ops.prefix_prefill import (
    context_attention_fwd as minidrag_context_attention_fwd)
from vllm.platforms import current_platform
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from tests.kernels.test_prefix_prefill import (HEAD_SIZES, NUM_HEADS,
                                               NUM_QUERIES_PER_KV,
                                               DTYPES, CUDA_DEVICES, KV_CACHE_DTYPES)
from utils import RoPEPatchContext, DynamicTritonAttnContext
KV_CACHE_DTYPES = ["auto"] #, "fp8", "fp8_e5m2"] # TODO(haocheng): support more data types

OPS = [(chunked_prefill_paged_decode, minidrag_chunked_prefill_paged_decode), 
       (context_attention_fwd, minidrag_context_attention_fwd)]

@pytest.mark.gpu
@pytest.mark.unit
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
# @pytest.mark.parametrize("sliding_window", SLIDING_WINDOW)
@pytest.mark.parametrize("ops", OPS)
@torch.inference_mode()
def test_contexted_kv_attention_with_pos(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    ops: Tuple[Callable],
    is_neox_style: bool = True, # TODO(haocheng): support more styles
    rotary_dim: Optional[int] = None,
    max_position: int = 8192,
    base: int = 10000,
    sliding_window: int = 0,
) -> None:
    op = ops[0]
    dynamic_op = ops[1]
    if 'fp8' in kv_cache_dtype and not current_platform.has_device_capability(
            89):
        pytest.skip(
            'Triton limitation: fp8e4nv data type is not supported on CUDA'
            ' arch < 89')

    current_platform.seed_everything(0)
    torch.set_default_device(device)

    # Need this, otherwise when we capture the graph the process
    # for GPU 1 would run on both GPU0 and GPU1 and things would hang
    #
    # see also similar issue: https://github.com/Dao-AILab/flash-attention/issues/523
    torch.cuda.set_device(device)

    MAX_SEQ_LEN = 1024
    MAX_CTX_LEN = 1024
    BS = 10
    cache_size = 640
    block_size = 32
    max_block_per_request = 64
    query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    # ensure one sequence in batch is a decode
    query_lens[-1] = 1

    ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
    # print(f"query_lens: {query_lens}")
    # print(f"ctx_lens: {ctx_lens}")
    seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
    num_kv_heads = num_heads // num_queries_per_kv

    num_tokens = sum(query_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)

    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1e-3, 1e-3)
    key, value = kv.unbind(dim=1)
    
    # copy the unrotated key  
    unrotated_key = key.clone()
    # then we need to rotate the key and query according to the RoPE
    # |---- context len ----|---- query len ----|
    # |----------------- seq len ---------------|
    # query -> query len part and key -> seq len part 
    with RoPEPatchContext():
        from vllm.model_executor.layers.rotary_embedding import get_rope
        if rotary_dim is None:
            rotary_dim = head_size
        rope = get_rope(head_size, rotary_dim, max_position, base, is_neox_style)
        q_start = 0
        s_start = 0
        for q_len, c_len, s_len in zip(query_lens, ctx_lens, seq_lens):
            q_positions = torch.arange(
                c_len, s_len, dtype=torch.long, device=device)
            s_positions = torch.arange(
                0, s_len, dtype=torch.long, device=device)
            query[q_start:q_start+q_len,:,:], _ = rope.forward(
                q_positions, 
                query[q_start:q_start+q_len,:,:]) # rotate query
            key[s_start:s_start+s_len,:,:], _ = rope.forward(
                s_positions, 
                key[s_start:s_start+s_len,:,:]) # rotate key
            q_start += q_len
            s_start += s_len
    # copy the rotated query
    rotated_query = query.clone()
    
    if kv_cache_dtype == "auto":
        cache_dtype = dtype
    else:
        cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
    k_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=cache_dtype)
    v_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=cache_dtype)
    k = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    values = torch.arange(0, cache_size, dtype=torch.long)
    values = values[torch.randperm(cache_size)]
    block_table = values[:BS * max_block_per_request].view(
        BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.long)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.long)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens,
                                            dtype=torch.long),
                               dim=0)
    max_input_len = MAX_SEQ_LEN
    # copy kv to cache
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1],
                                                dtype=torch.long),
                                   dim=0)
    for i in range(BS):
        for j in range(query_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] +
                                            j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] +
                                              b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads,
                         head_size)[start_slot:end_slot].copy_(
                             key[start_loc:end_loc])
            v_cache.view(-1, num_kv_heads,
                         head_size)[start_slot:end_slot].copy_(
                             value[start_loc:end_loc])
            cur_ctx += block_size
            block_id += 1
    # transpose K_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
    k_cache = k_cache.view(-1, block_size, num_kv_heads, head_size // 8,
                           8).permute(0, 2, 3, 1, 4).contiguous()
    # transpose V_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to V_cache[num_blocks, num_kv_heads, head_size, block_size]
    v_cache = v_cache.view(-1, block_size, num_kv_heads,
                           head_size).permute(0, 2, 3, 1).contiguous()
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Warm up the Triton kernel by calling it once before actually measuring
    # generation time
    op(query,
       k,
       v,
       output,
       kv_cache_dtype,
       k_cache,
       v_cache,
       block_table,
       b_start_loc,
       b_seq_len,
       MAX_CTX_LEN,
       max_input_len,
       k_scale,
       v_scale,
       sliding_window=sliding_window)
    torch.cuda.synchronize()
    start_time = time.time()
    op(query,
       k,
       v,
       output,
       kv_cache_dtype,
       k_cache,
       v_cache,
       block_table,
       b_start_loc,
       b_seq_len,
       MAX_CTX_LEN,
       max_input_len,
       k_scale,
       v_scale,
       sliding_window=sliding_window)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"triton Time: {(end_time - start_time)*1000:.2f} ms")

    scale = float(1.0 / (head_size**0.5))

    attn_op = xops.fmha.cutlass.FwOp()

    if num_kv_heads != num_heads:
        # As of Nov 2023, xformers only supports MHA. For MQA/GQA,
        # project the key and value tensors to the desired number of
        # heads.
        #
        # see also: vllm/model_executor/layers/attention.py
        query = query.view(query.shape[0], num_kv_heads, num_queries_per_kv,
                           query.shape[-1])
        key = key[:, :, None, :].expand(key.shape[0], num_kv_heads,
                                        num_queries_per_kv, key.shape[-1])
        value = value[:, :,
                      None, :].expand(value.shape[0], num_kv_heads,
                                      num_queries_per_kv, value.shape[-1])
    query = query.unsqueeze(0)
    key = key.unsqueeze(0)
    value = value.unsqueeze(0)

    attn_bias = BlockDiagonalCausalFromBottomRightMask.from_seqlens(
        query_lens, seq_lens)
    if sliding_window > 0:
        attn_bias = attn_bias.make_local_attention_from_bottomright(
            sliding_window)
    output_ref = xops.memory_efficient_attention_forward(
        query,
        key,
        value,
        attn_bias=attn_bias,
        p=0.0,
        scale=scale,
        op=attn_op,
    )
    torch.cuda.synchronize()
    start_time = time.time()
    output_ref = xops.memory_efficient_attention_forward(
        query,
        key,
        value,
        attn_bias=attn_bias,
        p=0.0,
        scale=scale,
        op=attn_op,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"xformers Time: {(end_time - start_time)*1000:.2f} ms")
    output_ref = output_ref.reshape(output.shape)
    atol = 1e-3 if "fp8" in kv_cache_dtype else 1e-4
    torch.testing.assert_close(output, output_ref, atol=atol, rtol=0)
    
    # /////////////////////////////////////////////////////////////////////////
    # test our dynamic prefill and decode
    # /////////////////////////////////////////////////////////////////////////
    with DynamicTritonAttnContext():
        dynamic_output = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
        # prepare k cache
        for i in range(BS):
            for j in range(query_lens[i]):
                k[b_start_loc[i] + j].copy_(unrotated_key[b_seq_start_loc[i] + b_ctx_len[i] +
                                                j])
            cur_ctx = 0
            block_id = 0
            while cur_ctx < b_ctx_len[i]:
                start_loc = b_seq_start_loc[i] + cur_ctx
                if cur_ctx + block_size > b_ctx_len[i]:
                    end_loc = b_seq_start_loc[i] + b_ctx_len[i]
                else:
                    end_loc = start_loc + block_size
                start_slot = block_table[i, block_id] * block_size
                end_slot = start_slot + end_loc - start_loc
                k_cache.view(-1, num_kv_heads,
                            head_size)[start_slot:end_slot].copy_(
                                unrotated_key[start_loc:end_loc])
                cur_ctx += block_size
                block_id += 1
        # transpose K_cache[num_blocks, block_size, num_kv_heads, head_size]
        # to K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
        k_cache = k_cache.view(-1, block_size, num_kv_heads, head_size // 8,
                            8).permute(0, 2, 3, 1, 4).contiguous()

        # Warm up the Triton kernel by calling it once before actually measuring
        # generation time
        dynamic_op(rotated_query,
        k,
        v,
        dynamic_output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        MAX_CTX_LEN,
        max_input_len,
        k_scale,
        v_scale,
        sliding_window=sliding_window,
        rotary_dim=rope.rotary_dim,
        cos_sin_cache=rope.cos_sin_cache,
        is_neox_style=rope.is_neox_style,)
        torch.cuda.synchronize()
        start_time = time.time()
        dynamic_op(rotated_query,
        k,
        v,
        dynamic_output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        MAX_CTX_LEN,
        max_input_len,
        k_scale,
        v_scale,
        sliding_window=sliding_window,
        rotary_dim=rope.rotary_dim,
        cos_sin_cache=rope.cos_sin_cache,
        is_neox_style=rope.is_neox_style,)
        torch.cuda.synchronize()
        end_time = time.time()
        print(f"dynamic triton Time: {(end_time - start_time)*1000:.2f} ms")
        torch.testing.assert_close(dynamic_output, output_ref, atol=atol, rtol=0)
        
