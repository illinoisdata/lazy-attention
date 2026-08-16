# Adapted from vllm/attention/ops/chunked_prefill_paged_decode.py

import torch
import triton

from vllm.platforms.rocm import use_rocm_custom_paged_attention

from lazy.attention.ops.prefix_prefill import context_attention_fwd, IS_TURING
from lazy.utils.variants import (
    lazy_decode_compute_cos_sin_enabled,
    lazy_decode_ignore_q_mask_enabled,
    lazy_decode_wrapper_profile_enabled,
    lazy_force_split_decode_enabled,
    no_lazy_enabled,
)

from lazy.attention.ops.models.llama_v1 import (
    kernel_paged_attention_2d_llama,
    kernel_paged_attention_2d_llama_lazy_only,
)


def _cuda_elapsed_ms(start_event, end_event):
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event)


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y



# /////////////////////////////////////////////////////////////////////////////////////

def chunked_prefill_paged_decode(
    query,
    key,
    value,
    output,
    kv_cache_dtype,
    key_cache,
    value_cache,
    block_table,
    query_start_loc,
    seq_lens,
    max_seq_len,
    max_query_len,
    k_scale,
    v_scale,
    alibi_slopes=None,
    sliding_window=None,
    sm_scale=None,
    rotary_dim=None,
    freqs=None,
    cos_sin_cache=None,
    rope_meta=None,
    is_neox_style=True,
    # To rotate the query
    is_lazy=None,
    lazy_variant=None,
    q_offset=None,
    q_mask=None,
    packed_block_table=None,
):
    if no_lazy_enabled():
        is_lazy.fill_(False)

    q_dtype_is_f32 = query.dtype is torch.float32
    IN_PRECISION = 'ieee' if IS_TURING and q_dtype_is_f32 else None
    if sm_scale is None:
        sm_scale = 1.0 / (query.shape[1]**0.5)

    use_alibi_slopes = alibi_slopes is not None

    if sliding_window is None or sliding_window <= 0:
        sliding_window = 0

    if max_query_len > 1:
        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            kv_cache_dtype=kv_cache_dtype,
            k_cache=key_cache,
            v_cache=value_cache,
            b_loc=block_table,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_seq_len=max_seq_len,
            max_input_len=max_query_len,
            k_scale=k_scale,
            v_scale=v_scale,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
            skip_decode=True,
            rotary_dim=rotary_dim,
            cos_sin_cache=cos_sin_cache,
            is_neox_style=is_neox_style,
            # To rotate the query
            is_lazy=is_lazy,
            q_offset=q_offset,
            q_mask=q_mask,
        )

    block_size = value_cache.shape[3]
    num_seqs = len(seq_lens)
    num_query_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    num_queries_per_kv = query.shape[1] // key.shape[1]
    head_size = query.shape[2]

    # Conversion of FP8 Tensor from uint8 storage to
    # appropriate torch.dtype for interpretation by Triton
    if "fp8" in kv_cache_dtype:
        assert key_cache.dtype == torch.uint8
        assert value_cache.dtype == torch.uint8

        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            target_dtype = torch.float8_e4m3fn
        elif kv_cache_dtype == "fp8_e5m2":
            target_dtype = torch.float8_e5m2
        else:
            raise ValueError("Unsupported FP8 dtype:", kv_cache_dtype)

        key_cache = key_cache.view(target_dtype)
        value_cache = value_cache.view(target_dtype)

    num_queries_per_kv_padded = max(triton.next_power_of_2(num_queries_per_kv),
                                    16)

    # vLLM 0.9.x added kv_cache_dtype to this predicate.
    use_custom = use_rocm_custom_paged_attention(query.dtype, head_size,
                                                 block_size,
                                                 num_queries_per_kv,
                                                 max_seq_len, sliding_window,
                                                 kv_cache_dtype)
    
    if use_custom:
        raise NotImplementedError("Custom paged attention is not implemented")
    else:
        profile_decode = lazy_decode_wrapper_profile_enabled()
        force_split_decode = lazy_force_split_decode_enabled()
        ignore_q_mask = lazy_decode_ignore_q_mask_enabled()
        # Default: LOAD cos/sin from cos_sin_cache. Computing them in-kernel
        # spills registers, and measured slower everywhere except large-batch
        # decode on head_size=128 -- see docs/design.md 4.3 and
        # benchmarks/bench_rope_cos_sin.py.
        compute_cos_sin = lazy_decode_compute_cos_sin_enabled()
        rope_kw = rope_meta if rope_meta is not None else dict(
            ROPE_TYPE=0, BASE=10000.0, SCALING_FACTOR=1.0, LOW_FACTOR=1.0,
            HIGH_FACTOR=1.0, ORIG_MAX_POSITION=8192,
            PI_VALUE=3.141592653589793)
        all_lazy = (is_lazy is not None) and bool(torch.all(is_lazy).item())
        if profile_decode:
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            kernel_start = torch.cuda.Event(enable_timing=True)
            kernel_end = torch.cuda.Event(enable_timing=True)
        decode_kernel = (kernel_paged_attention_2d_llama_lazy_only
                         if force_split_decode and all_lazy
                         else kernel_paged_attention_2d_llama)
        decode_block_table = (
            packed_block_table if packed_block_table is not None else block_table
        )
        if profile_decode:
            total_start.record()
            kernel_start.record()
        decode_kernel[(
                num_seqs,
                num_kv_heads,
            )](
                output_ptr=output,
                query_ptr=query,
                key_cache_ptr=key_cache,
                value_cache_ptr=value_cache,
                block_tables_ptr=decode_block_table,
                seq_lens_ptr=seq_lens,
                alibi_slopes_ptr=alibi_slopes,
                scale=sm_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                num_query_heads=num_query_heads,
                num_queries_per_kv=num_queries_per_kv,
                num_queries_per_kv_padded=num_queries_per_kv_padded,
                block_table_stride=decode_block_table.stride(0),
                query_stride_0=query.stride(0),
                query_stride_1=query.stride(1),
                output_stride_0=output.stride(0),
                output_stride_1=output.stride(1),
                IN_PRECISION=IN_PRECISION,
                BLOCK_SIZE=block_size,
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
                USE_ALIBI_SLOPES=use_alibi_slopes,
                SLIDING_WINDOW=sliding_window,
                x=key_cache.shape[4],
                stride_k_cache_0=key_cache.stride(0),
                stride_k_cache_1=key_cache.stride(1),
                stride_k_cache_2=key_cache.stride(2),
                stride_k_cache_3=key_cache.stride(3),
                stride_k_cache_4=key_cache.stride(4),
                stride_v_cache_0=value_cache.stride(0),
                stride_v_cache_1=value_cache.stride(1),
                stride_v_cache_2=value_cache.stride(2),
                stride_v_cache_3=value_cache.stride(3),
                filter_by_query_len=True,
                query_start_len_ptr=query_start_loc,
                rotary_dim=rotary_dim,
                rotary_dim_pow2=triton.next_power_of_2(rotary_dim),  # padded
                is_neox_style=is_neox_style,
                is_lazy_ptr=is_lazy,
                q_offset_ptr=q_offset,
                q_mask_ptr=q_mask,
                cos_sin_cache_ptr=cos_sin_cache,
                IGNORE_Q_MASK=ignore_q_mask,
                COMPUTE_COS_SIN=compute_cos_sin,
                **rope_kw,
            )
        if profile_decode:
            kernel_end.record()
            total_end.record()
            kernel_ms = _cuda_elapsed_ms(kernel_start, kernel_end)
            total_ms = _cuda_elapsed_ms(total_start, total_end)
            other_ms = max(total_ms - kernel_ms, 0.0)
            print(
                f"LazyDecodeProfile num_seqs={num_seqs} max_seq_len={max_seq_len} "
                f"heads={num_query_heads}/{num_kv_heads} split={int(force_split_decode and all_lazy)} "
                f"pack_ms={0.0:.3f} kernel_ms={kernel_ms:.3f} "
                f"unpack_ms={0.0:.3f} total_ms={total_ms:.3f} other_ms={other_ms:.3f}",
                flush=True,
            )
