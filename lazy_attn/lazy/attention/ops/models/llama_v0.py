"""
Llama attention kernel with paging support.

Changed by Haocheng at 2025/09/15
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def llama_cos_sin(
    position: tl.int32,
    HEAD_SIZE: tl.constexpr,
    ORIG_MAX_POSITION: tl.constexpr,
    LOW_FACTOR: tl.constexpr,
    HIGH_FACTOR: tl.constexpr,
    SCALING_FACTOR: tl.constexpr,
    PI_VALUE: tl.constexpr,
    BASE: tl.constexpr,
):
    """Compute the rotary embedding cos and sin values for Llama 3.1."""
    
    low_freq_wavelen: tl.constexpr = ORIG_MAX_POSITION / LOW_FACTOR
    high_freq_wavelen: tl.constexpr = ORIG_MAX_POSITION / HIGH_FACTOR
    half_head_size: tl.constexpr = HEAD_SIZE // 2

    inv_freqs = 1.0 / libdevice.pow(BASE, 2 * (tl.arange(0, HEAD_SIZE) % half_head_size) / HEAD_SIZE)
    wave_len = 2 * PI_VALUE / inv_freqs

    smooth = (ORIG_MAX_POSITION / wave_len - LOW_FACTOR
                      ) / (HIGH_FACTOR - LOW_FACTOR)
    # NOTE(haocheng): Llama 3.1 8B needs smooth, so we add it here. (not zero)

    new_freqs = tl.where(
        wave_len < high_freq_wavelen,
            inv_freqs,
            tl.where(
                wave_len > low_freq_wavelen,
                inv_freqs / SCALING_FACTOR,
                (1 - smooth) * inv_freqs / SCALING_FACTOR +
                smooth * inv_freqs,
            ),
        )
    # tl.device_print("llama new_freqs:", new_freqs * 1000000)

    theta = position * new_freqs
    cos_val = libdevice.cos(theta)
    sin_val = libdevice.sin(theta)
    return cos_val, sin_val


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y

@triton.jit
def kernel_paged_attention_2d_llama(
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
        # freqs_ptr,
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
                cos_val, sin_val = llama_cos_sin(
                    relative_rot,
                    HEAD_SIZE=rotary_dim,
                    ORIG_MAX_POSITION=2048,
                    LOW_FACTOR=1.0,
                    HIGH_FACTOR=32.0,
                    SCALING_FACTOR=4.0,
                    PI_VALUE=3.14159265358979323846,
                    BASE=10000.0,
                )

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