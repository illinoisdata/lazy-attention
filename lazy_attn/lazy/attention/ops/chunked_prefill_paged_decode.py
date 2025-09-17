# Adapted from vllm/attention/ops/chunked_prefill_paged_decode.py

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from vllm import _custom_ops as ops
from vllm.platforms.rocm import use_rocm_custom_paged_attention
from vllm.attention.ops.chunked_prefill_paged_decode import context_attention_fwd as context_attention_fwd_orig
from vllm.attention.ops.chunked_prefill_paged_decode import kernel_paged_attention_2d as kernel_paged_attention_2d_orig


from lazy.attention.ops.prefix_prefill import context_attention_fwd, IS_TURING

from lazy.attention.ops.models.llama import kernel_paged_attention_2d_llama


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def kernel_paged_attention_2d(
        output_ptr,  # [num_tokens, num_query_heads, head_size]
        query_ptr,  # [num_tokens, num_query_heads, head_size]
        key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
        value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
        block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
        seq_lens_ptr,  # [num_seqs]
        alibi_slopes_ptr,  # [num_query_heads]
        scale,  # float32
        k_scale,  # float32
        v_scale,  # float32
        num_query_heads: tl.constexpr,  # int
        num_queries_per_kv: tl.constexpr,  # int
        num_queries_per_kv_padded: tl.constexpr,  # int
        block_table_stride: tl.int64,  # int
        query_stride_0: tl.int64,  # int
        query_stride_1: tl.int64,  # int, should be equal to head_size
        output_stride_0: tl.int64,  # int
        output_stride_1: tl.int64,  # int, should be equal to head_size
        IN_PRECISION: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,  # int
        HEAD_SIZE: tl.constexpr,  # int
        HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
        USE_ALIBI_SLOPES: tl.constexpr,  # bool
        SLIDING_WINDOW: tl.constexpr,  # int
        x: tl.constexpr,  # int
        stride_k_cache_0: tl.int64,  # int
        stride_k_cache_1: tl.int64,  # int
        stride_k_cache_2: tl.int64,  # int
        stride_k_cache_3: tl.int64,  # int
        stride_k_cache_4: tl.int64,  # int
        stride_v_cache_0: tl.int64,  # int
        stride_v_cache_1: tl.int64,  # int
        stride_v_cache_2: tl.int64,  # int
        stride_v_cache_3: tl.int64,  # int
        filter_by_query_len: tl.constexpr,  # bool
        query_start_len_ptr,  # [num_seqs+1]
        rotary_dim: tl.constexpr,  # int
        rotary_dim_pow2: tl.constexpr,  # int
        # cos_sin_cache_ptr,
        freqs_ptr,
        is_neox_style: tl.constexpr,  # bool
        # To rotate the query
        is_lazy_ptr,
        q_offset_ptr,
        q_mask_ptr,
):
    # TODO(haocheng): consider the case rot dim != head size
    seq_idx = tl.program_id(0)
    # Get the condition as early as possible
    is_lazy = tl.load(is_lazy_ptr + seq_idx)
    # new_freq = tl.load(freqs_ptr + tl.arange(0, HEAD_SIZE_PADDED))

# /////////////////////////////////////////////////////////////////////////////////////////
# Rotate the query
    if is_lazy:
        kv_head_idx = tl.program_id(1)
        # # Load freq in advance
        # freqs: (HEAD_SIZE_PADDED,) repeated two times to match the padded size
        # freq = tl.load(freqs_ptr + tl.arange(0, HEAD_SIZE_PADDED))
        # freq = tl.load(query_ptr + tl.arange(0, HEAD_SIZE_PADDED)) 
        # new_freq = tl.load(freqs_ptr + tl.arange(0, HEAD_SIZE_PADDED))
        # query_ptr = query_ptr + query_stride_0
        # new_freq = tl.load(freqs_ptr + tl.arange(0, HEAD_SIZE_PADDED))
        if filter_by_query_len:
            cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
            cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx +
                                                1)
            cur_batch_query_len = cur_batch_in_all_stop_index \
                - cur_batch_in_all_start_index
            if cur_batch_query_len > 1:
                return
        else:
            cur_batch_in_all_start_index = seq_idx

        query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(
            0, num_queries_per_kv_padded)

        query_offset = (cur_batch_in_all_start_index * query_stride_0 +
                        query_head_idx[:, None] * query_stride_1)

        head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
        head_mask = head_mask & (query_head_idx < num_query_heads)

        dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1,
                            0).to(tl.int1)
        dim_mask_half = tl.where(tl.arange(0, HEAD_SIZE_PADDED // 2) < HEAD_SIZE // 2, 1,
                            0).to(tl.int1)


        # Key optimization: sparse rotation, we keep Q_full as the main representation, Q_1/Q_2 only created when rotating
        Q_full = tl.load(
            query_ptr + query_offset + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
            mask=dim_mask[None, :] & head_mask[:, None],
            other=0.0,
        )

        # tl.device_print("pid=%d, BLOCK_SIZE=%d, vector_size=%d\n", BLOCK_SIZE, head_mask.shape[0])
        
        # We use Q full by default
        Q_rotated = Q_full
        
        block_table_offset = seq_idx * block_table_stride

        M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
        L = tl.full([num_queries_per_kv_padded], 1.0, dtype=tl.float32)
        acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED],
                    dtype=tl.float32)

        # sequence len for this particular sequence
        seq_len = tl.load(seq_lens_ptr + seq_idx)

        # alibi slope for this head
        if USE_ALIBI_SLOPES:
            alibi_slope = tl.load(alibi_slopes_ptr + query_head_idx,
                                  mask=head_mask,
                                  other=0.0)

        num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)
        prev_offset = 0
        
        # iterate through tiles - 稀疏优化版本
        for j in range(0, num_blocks):
            physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)
            rot_offset_val = tl.load(q_offset_ptr + block_table_offset + j)
            q_mask_val = tl.load(q_mask_ptr + block_table_offset + j)

            offs_n = tl.arange(0, BLOCK_SIZE)
            offs_d = tl.arange(0, HEAD_SIZE_PADDED)

            v_offset = (physical_block_idx * stride_v_cache_0 +
                        kv_head_idx * stride_v_cache_1 +
                        offs_d[None, :] * stride_v_cache_2 +
                        offs_n[:, None] * stride_v_cache_3)

            k_offset = (physical_block_idx * stride_k_cache_0 +
                        kv_head_idx * stride_k_cache_1 +
                        (offs_d[:, None] // x) * stride_k_cache_2 +
                        offs_n[None, :] * stride_k_cache_3 +
                        (offs_d[:, None] % x) * stride_k_cache_4)

            K_load_full = tl.load(key_cache_ptr + k_offset,
                                mask=dim_mask[:, None],
                                other=0.0)

            if K_load_full.dtype.is_fp8():
                K_full = (K_load_full.to(tl.float32) * tl.load(k_scale)).to(Q_rotated.dtype)
            else:
                K_full = K_load_full

            V_load = tl.load(value_cache_ptr + v_offset,
                            mask=dim_mask[None, :],
                            other=0.0)

            if V_load.dtype.is_fp8():
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q_rotated.dtype)
            else:
                V = V_load

            seq_offset = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            boundary = tl.full([BLOCK_SIZE], seq_len, dtype=tl.int32)
            seq_mask = seq_offset[None, :] < boundary
            
            # if q_mask_val != 0:
            seq_mask = seq_mask & (tl.arange(0, BLOCK_SIZE) < (BLOCK_SIZE - q_mask_val))
            
            needs_rotation = (prev_offset != rot_offset_val)

            if needs_rotation:
                # Only rotate when necessary
                relative_rot = rot_offset_val - prev_offset
                cols = tl.arange(0, HEAD_SIZE)
                # new_freq = tl.load(freqs_ptr + tl.arange(0, HEAD_SIZE_PADDED))
                
                # new_freq = new_freq.to(Q_rotated.dtype)
                # -------
                # offs = tl.arange(0, HEAD_SIZE) % (HEAD_SIZE // 2)

                # new_freq = 1 / libdevice.pow(10000.0, 2 * offs / HEAD_SIZE)
                # -------
                low_factor, high_factor, scaling_factor, pi_value = 1.0, 4.0, 8.0, 3.14159265358979323846
                orig_max_position, low_wave, high_wave = 8192, 8192, 2048

                offs = tl.arange(0, HEAD_SIZE) % (HEAD_SIZE // 2)

                inv_freq = 1 / libdevice.pow(10000.0, 2 * offs / HEAD_SIZE)

                # wave_len = 2π / inv_freq
                # 所以 orig_max_position / wave_len = orig_max_position * inv_freq / (2π)
                smooth = tl.where(
                    high_factor != low_factor,
                    (orig_max_position * inv_freq / (2 * pi_value) - low_factor) / (high_factor - low_factor),
                    0.0
                )

                # 条件 wave_len < high_wave <=> inv_freq > 2π / high_wave
                # 条件 wave_len > low_wave <=> inv_freq < 2π / low_wave
                new_freq = tl.where(
                    inv_freq > (2 * pi_value / high_wave),
                    inv_freq,
                    tl.where(
                        inv_freq < (2 * pi_value / low_wave),
                        inv_freq / scaling_factor,
                        inv_freq * (smooth + (1 - smooth) / scaling_factor)
                    )
                )
                # -------
                theta = relative_rot * new_freq # / libdevice.pow(10000.0, 2 * offs / HEAD_SIZE)
                cos_val = libdevice.cos(theta)
                sin_val = libdevice.sin(theta)

                # mask for前半 / 后半
                mask_q1 = cols < HEAD_SIZE // 2
                mask_q2 = ~mask_q1

                # 从Q_rotated里挑出对应部分，确保类型一致
                q1 = tl.where(mask_q1[None, :], Q_rotated, 0.0)
                q2 = tl.where(mask_q2[None, :], Q_rotated, 0.0)
                
                # 向量化旋转计算，保持原始数据类型
                q1_new = q1 * cos_val - q2 * sin_val
                q2_new = q1 * sin_val + q2 * cos_val

                # 重建Q_rotated，保持原始数据类型
                Q_rotated = tl.where(mask_q1[None, :], 
                q1_new.to(Q_rotated.dtype), 
                q2_new.to(Q_rotated.dtype))
                
                prev_offset = rot_offset_val
            
            # 统一路径：都使用完整GEMM
            S = tl.where(head_mask[:, None] & seq_mask, 0.0,
                        float("-inf")).to(tl.float32)
            
            # 关键优化：所有blocks都使用高效的完整GEMM
            qk = tl.dot(Q_rotated, K_full, input_precision=IN_PRECISION)
            
            S += scale * qk

            context_len = seq_len - 1

            if SLIDING_WINDOW > 0:
                S = tl.where((context_len - seq_offset) < SLIDING_WINDOW, S,
                            -10000)

            if USE_ALIBI_SLOPES:
                S += alibi_slope[:, None] * (seq_offset - context_len)

            # compute running maximum
            m_j = tl.maximum(M, tl.max(S, axis=1))
            P = tl.exp(S - m_j[:, None])
            l_j = tl.sum(P, axis=1)
            alpha = tl.exp(M - m_j)
            acc = acc * alpha[:, None]
            L = L * alpha + l_j
            M = m_j
            acc += tl.dot(P.to(V.dtype), V)

        # epilogue
        acc = acc / L[:, None]

        output_offset = (cur_batch_in_all_start_index * output_stride_0 +
                        query_head_idx * output_stride_1)

        tl.store(
            output_ptr + output_offset[:, None] +
            tl.arange(0, HEAD_SIZE_PADDED)[None, :],
            acc,
            mask=dim_mask[None, :] & head_mask[:, None],
        )
    

# /////////////////////////////////////////////////////////////////////////////////////////
# Return to the original code
    else:
        kv_head_idx = tl.program_id(1)

        if filter_by_query_len:
            cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
            cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx +
                                                1)
            cur_batch_query_len = cur_batch_in_all_stop_index \
                - cur_batch_in_all_start_index
            if cur_batch_query_len > 1:
                return
        else:
            cur_batch_in_all_start_index = seq_idx

        query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(
            0, num_queries_per_kv_padded)

        query_offset = (cur_batch_in_all_start_index * query_stride_0 +
                        query_head_idx[:, None] * query_stride_1)

        head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
        head_mask = head_mask & (query_head_idx < num_query_heads)

        dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1,
                            0).to(tl.int1)

        # Q : (num_queries_per_kv, HEAD_SIZE,)
        Q = tl.load(
            query_ptr + query_offset + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
            mask=dim_mask[None, :] & head_mask[:, None],
            other=0.0,
        )

        block_table_offset = seq_idx * block_table_stride

        M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
        L = tl.full([num_queries_per_kv_padded], 1.0, dtype=tl.float32)
        acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED],
                    dtype=tl.float32)

        # sequence len for this particular sequence
        seq_len = tl.load(seq_lens_ptr + seq_idx)

        # alibi slope for this head
        if USE_ALIBI_SLOPES:
            alibi_slope = tl.load(alibi_slopes_ptr + query_head_idx,
                                mask=head_mask,
                                other=0.0)

        num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)

        # iterate through tiles
        for j in range(0, num_blocks):

            physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

            offs_n = tl.arange(0, BLOCK_SIZE)
            offs_d = tl.arange(0, HEAD_SIZE_PADDED)

            v_offset = (physical_block_idx * stride_v_cache_0 +
                        kv_head_idx * stride_v_cache_1 +
                        offs_d[None, :] * stride_v_cache_2 +
                        offs_n[:, None] * stride_v_cache_3)

            k_offset = (physical_block_idx * stride_k_cache_0 +
                        kv_head_idx * stride_k_cache_1 +
                        (offs_d[:, None] // x) * stride_k_cache_2 +
                        offs_n[None, :] * stride_k_cache_3 +
                        (offs_d[:, None] % x) * stride_k_cache_4)

            # K : (HEAD_SIZE, BLOCK_SIZE)
            K_load = tl.load(key_cache_ptr + k_offset,
                            mask=dim_mask[:, None],
                            other=0.0)

            if K_load.dtype.is_fp8():
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
            else:
                K = K_load

            # V : (BLOCK_SIZE, HEAD_SIZE)
            V_load = tl.load(value_cache_ptr + v_offset,
                            mask=dim_mask[None, :],
                            other=0.0)

            if V_load.dtype.is_fp8():
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
            else:
                V = V_load

            seq_offset = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            boundary = tl.full([BLOCK_SIZE], seq_len, dtype=tl.int32)
            seq_mask = seq_offset[None, :] < boundary

            # S : (num_queries_per_kv, BLOCK_SIZE,)
            S = tl.where(head_mask[:, None] & seq_mask, 0.0,
                        float("-inf")).to(tl.float32)
            S += scale * tl.dot(Q, K)

            context_len = seq_len - 1

            if SLIDING_WINDOW > 0:
                S = tl.where((context_len - seq_offset) < SLIDING_WINDOW, S,
                            -10000)

            if USE_ALIBI_SLOPES:
                S += alibi_slope[:, None] * (seq_offset - context_len)

            # compute running maximum
            # m_j : (num_queries_per_kv,)
            m_j = tl.maximum(M, tl.max(S, axis=1))

            # P : (num_queries_per_kv, BLOCK_SIZE,)
            P = tl.exp(S - m_j[:, None])

            # l_j : (num_queries_per_kv,)
            l_j = tl.sum(P, axis=1)

            # alpha : (num_queries_per_kv, )
            alpha = tl.exp(M - m_j)

            # acc : (num_queries_per_kv, BLOCK_SIZE,)
            acc = acc * alpha[:, None]

            # update constants
            L = L * alpha + l_j
            M = m_j

            # acc : (num_queries_per_kv, BLOCK_SIZE,)
            acc += tl.dot(P.to(V.dtype), V)

        # epilogue
        acc = acc / L[:, None]

        output_offset = (cur_batch_in_all_start_index * output_stride_0 +
                        query_head_idx * output_stride_1)

        tl.store(
            output_ptr + output_offset[:, None] +
            tl.arange(0, HEAD_SIZE_PADDED)[None, :],
            acc,
            mask=dim_mask[None, :] & head_mask[:, None],
        )


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
    is_neox_style=True,
    # To rotate the query
    is_lazy=None,
    q_offset=None,
    q_mask=None,
):

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

    use_custom = use_rocm_custom_paged_attention(query.dtype, head_size,
                                                 block_size,
                                                 num_queries_per_kv,
                                                 max_seq_len, sliding_window)
    
    if use_custom:
        raise NotImplementedError("Custom paged attention is not implemented")
    else:
        kernel_paged_attention_2d_llama[(
            num_seqs,
            num_kv_heads,
        )](
            output_ptr=output,
            query_ptr=query,
            key_cache_ptr=key_cache,
            value_cache_ptr=value_cache,
            block_tables_ptr=block_table,
            seq_lens_ptr=seq_lens,
            alibi_slopes_ptr=alibi_slopes,
            scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            num_queries_per_kv_padded=num_queries_per_kv_padded,
            block_table_stride=block_table.stride(0),
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
            # cos_sin_cache_ptr=cos_sin_cache,
            # freqs_ptr=new_freqs,
            is_neox_style=is_neox_style,
            # To rotate the query
            is_lazy_ptr=is_lazy,
            q_offset_ptr=q_offset,
            q_mask_ptr=q_mask,
        )