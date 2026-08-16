"""
Llama attention kernel with paging support.

Changed by Haocheng at 2025/09/15
"""

import triton
import triton.language as tl

from lazy.model_executor.rope import rope_cos_sin_from_freqs, rope_freqs


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
        is_neox_style: tl.constexpr,  # bool
        # To rotate the query
        is_lazy_ptr,
        q_offset_ptr,
        q_mask_ptr,
        cos_sin_cache_ptr,
        # RoPE scaling constexprs for the optional in-kernel compute path
        # (COMPUTE_COS_SIN=True). Compiled out / unused in the default load path.
        ROPE_TYPE: tl.constexpr = 0,
        BASE: tl.constexpr = 10000.0,
        SCALING_FACTOR: tl.constexpr = 1.0,
        LOW_FACTOR: tl.constexpr = 1.0,
        HIGH_FACTOR: tl.constexpr = 1.0,
        ORIG_MAX_POSITION: tl.constexpr = 8192,
        PI_VALUE: tl.constexpr = 3.141592653589793,
        IGNORE_Q_MASK: tl.constexpr = False,
        # Default: LOAD cos/sin from the cos_sin_cache table. Set True to
        # COMPUTE them in-kernel instead: that trades the table loads for
        # registers, and only pays off in one regime (large-batch decode at
        # head_size=128). See docs/design.md 4.3 and
        # benchmarks/bench_rope_cos_sin.py.
        COMPUTE_COS_SIN: tl.constexpr = False,
):
    # TODO(haocheng): Consider the case rot dim != head size in the future.
    # Now we assume rot dim == head size
    # assert rotary_dim == HEAD_SIZE, "Currently only support rotary_dim == head_size"
    
    seq_idx = tl.program_id(0)
    # Get the condition as early as possible
    is_lazy = tl.load(is_lazy_ptr + seq_idx)

# /////////////////////////////////////////////////////////////////////////////////////////
# Rotate the query instead of rotating K/V
    if is_lazy:
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
        embed_dim: tl.constexpr = HEAD_SIZE // 2
        cols = tl.arange(0, HEAD_SIZE)
        mask_q1 = cols < embed_dim
        # Q : (num_queries_per_kv, HEAD_SIZE,)
        Q_full = tl.load(
            query_ptr + query_offset + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
            mask=dim_mask[None, :] & head_mask[:, None],
            other=0.0,
        )
        Q_rotated = Q_full
        block_table_offset = seq_idx * block_table_stride

        if COMPUTE_COS_SIN:
            # Position-independent: hoisted out of the block loop so the pow()
            # and the Llama-3 smoothing are paid once per program instead of
            # once per rotation change.
            rope_freq = rope_freqs(ROPE_TYPE, HEAD_SIZE, BASE, SCALING_FACTOR,
                                   LOW_FACTOR, HIGH_FACTOR, ORIG_MAX_POSITION,
                                   PI_VALUE)

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
        prev_rot_offset = tl.full([], -1, dtype=tl.int32)
        # iterate through tiles - sparse rotation version
        for j in range(0, num_blocks):
            # physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)
            # rot_offset_val = 0 # tl.load(q_offset_ptr + block_table_offset + j)
            # q_mask_val = 0 #tl.load(q_mask_ptr + block_table_offset + j)
            
            packed_val = tl.load(block_tables_ptr + block_table_offset + j)
            physical_block_idx = (packed_val >> 32).to(tl.int32)
            rot_offset_val = ((packed_val >> 16) & 0xFFFF).to(tl.int32)
            q_mask_val = (packed_val & 0xFFFF).to(tl.int32)
            if IGNORE_Q_MASK:
                q_mask_val = 0

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

            K_load = tl.load(key_cache_ptr + k_offset,
                                mask=dim_mask[:, None],
                                other=0.0)

            if K_load.dtype.is_fp8():
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q_full.dtype)
            else:
                K = K_load

            V_load = tl.load(value_cache_ptr + v_offset,
                            mask=dim_mask[None, :],
                            other=0.0)

            if V_load.dtype.is_fp8():
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q_full.dtype)
            else:
                V = V_load

            seq_offset = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            boundary = tl.full([BLOCK_SIZE], seq_len, dtype=tl.int32)
            seq_mask = seq_offset[None, :] < boundary
            
            # Apply q_mask_val
            # TODO(haocheng): Now, q_mask_val is 0 means no padding, 1 means padding one token.
            # We can use BLOCK_SIZE to represent the no padding case in the future.
            # Then we save one subtraction operation here.
            seq_mask = seq_mask & (tl.arange(0, BLOCK_SIZE) < (BLOCK_SIZE - q_mask_val))

            if rot_offset_val != prev_rot_offset:
                if rot_offset_val == 1:
                    Q_rotated = Q_full
                elif rot_offset_val != 0:
                    # q_offset stores the absolute rotation position with a +1 bias.
                    # Rebuild from Q_full so numerical error does not accumulate.
                    if COMPUTE_COS_SIN:
                        cos_val, sin_val = rope_cos_sin_from_freqs(
                            rot_offset_val - 1, rope_freq)
                    else:
                        cache_cols = tl.arange(0, HEAD_SIZE) % embed_dim
                        cache_base = (rot_offset_val - 1) * rotary_dim
                        cos_val = tl.load(
                            cos_sin_cache_ptr + cache_base + cache_cols)
                        sin_val = tl.load(
                            cos_sin_cache_ptr + cache_base + embed_dim + cache_cols)
                    rev = (tl.arange(0, HEAD_SIZE_PADDED) +
                           (HEAD_SIZE_PADDED // 2)) % HEAD_SIZE_PADDED
                    Q_rev = tl.load(
                        query_ptr + query_offset + rev[None, :],
                        mask=dim_mask[None, :] & head_mask[:, None],
                        other=0.0,
                    )
                    q1 = tl.where(mask_q1[None, :], Q_full, Q_rev)
                    q2 = tl.where(mask_q1[None, :], Q_rev, Q_full)

                    q1_new = q1 * cos_val + q2 * sin_val
                    q2_new = -q1 * sin_val + q2 * cos_val

                    Q_rotated = tl.where(mask_q1[None, :],
                                         q1_new.to(Q_full.dtype),
                                         q2_new.to(Q_full.dtype))
                prev_rot_offset = rot_offset_val
            qk = tl.dot(Q_rotated, K, input_precision=IN_PRECISION)

            S = tl.where(head_mask[:, None] & seq_mask, 0.0,
                        float("-inf")).to(tl.float32)
            S += scale * qk

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
            packed_val = tl.load(block_tables_ptr + block_table_offset + j)
            physical_block_idx = (packed_val >> 32).to(tl.int32)

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


# Lazy-only decode kernel used for A/B testing against the unified kernel.
@triton.jit
def kernel_paged_attention_2d_llama_lazy_only(
        output_ptr,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        block_tables_ptr,
        seq_lens_ptr,
        alibi_slopes_ptr,
        scale,
        k_scale,
        v_scale,
        num_query_heads: tl.constexpr,
        num_queries_per_kv: tl.constexpr,
        num_queries_per_kv_padded: tl.constexpr,
        block_table_stride: tl.int64,
        query_stride_0: tl.int64,
        query_stride_1: tl.int64,
        output_stride_0: tl.int64,
        output_stride_1: tl.int64,
        IN_PRECISION: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_SIZE: tl.constexpr,
        HEAD_SIZE_PADDED: tl.constexpr,
        USE_ALIBI_SLOPES: tl.constexpr,
        SLIDING_WINDOW: tl.constexpr,
        x: tl.constexpr,
        stride_k_cache_0: tl.int64,
        stride_k_cache_1: tl.int64,
        stride_k_cache_2: tl.int64,
        stride_k_cache_3: tl.int64,
        stride_k_cache_4: tl.int64,
        stride_v_cache_0: tl.int64,
        stride_v_cache_1: tl.int64,
        stride_v_cache_2: tl.int64,
        stride_v_cache_3: tl.int64,
        filter_by_query_len: tl.constexpr,
        query_start_len_ptr,
        rotary_dim: tl.constexpr,
        rotary_dim_pow2: tl.constexpr,
        is_neox_style: tl.constexpr,
        is_lazy_ptr,
        q_offset_ptr,
        q_mask_ptr,
        cos_sin_cache_ptr,
        # RoPE scaling constexprs for the optional in-kernel compute path
        # (COMPUTE_COS_SIN=True). Compiled out / unused in the default load path.
        ROPE_TYPE: tl.constexpr = 0,
        BASE: tl.constexpr = 10000.0,
        SCALING_FACTOR: tl.constexpr = 1.0,
        LOW_FACTOR: tl.constexpr = 1.0,
        HIGH_FACTOR: tl.constexpr = 1.0,
        ORIG_MAX_POSITION: tl.constexpr = 8192,
        PI_VALUE: tl.constexpr = 3.141592653589793,
        IGNORE_Q_MASK: tl.constexpr = False,
        # Default: LOAD cos/sin from the cos_sin_cache table. Set True to
        # COMPUTE them in-kernel instead: that trades the table loads for
        # registers, and only pays off in one regime (large-batch decode at
        # head_size=128). See docs/design.md 4.3 and
        # benchmarks/bench_rope_cos_sin.py.
        COMPUTE_COS_SIN: tl.constexpr = False,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(0, num_queries_per_kv_padded)
    query_offset = (cur_batch_in_all_start_index * query_stride_0 +
                    query_head_idx[:, None] * query_stride_1)
    head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)

    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)
    embed_dim: tl.constexpr = HEAD_SIZE // 2
    cols = tl.arange(0, HEAD_SIZE)
    mask_q1 = cols < embed_dim

    Q_full = tl.load(
        query_ptr + query_offset + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        mask=dim_mask[None, :] & head_mask[:, None],
        other=0.0,
    )
    Q_rotated = Q_full
    block_table_offset = seq_idx * block_table_stride

    if COMPUTE_COS_SIN:
        # Position-independent -- hoisted out of the block loop, as above.
        rope_freq = rope_freqs(ROPE_TYPE, HEAD_SIZE, BASE, SCALING_FACTOR,
                               LOW_FACTOR, HIGH_FACTOR, ORIG_MAX_POSITION,
                               PI_VALUE)

    M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
    L = tl.full([num_queries_per_kv_padded], 1.0, dtype=tl.float32)
    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED], dtype=tl.float32)
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_head_idx, mask=head_mask, other=0.0)

    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)
    prev_rot_offset = tl.full([], -1, dtype=tl.int32)
    for j in range(0, num_blocks):
        packed_val = tl.load(block_tables_ptr + block_table_offset + j)
        physical_block_idx = (packed_val >> 32).to(tl.int32)
        rot_offset_val = ((packed_val >> 16) & 0xFFFF).to(tl.int32)
        q_mask_val = (packed_val & 0xFFFF).to(tl.int32)
        if IGNORE_Q_MASK:
            q_mask_val = 0

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

        K_load = tl.load(key_cache_ptr + k_offset, mask=dim_mask[:, None], other=0.0)
        if K_load.dtype.is_fp8():
            K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q_full.dtype)
        else:
            K = K_load

        V_load = tl.load(value_cache_ptr + v_offset, mask=dim_mask[None, :], other=0.0)
        if V_load.dtype.is_fp8():
            V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q_full.dtype)
        else:
            V = V_load

        seq_offset = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        boundary = tl.full([BLOCK_SIZE], seq_len, dtype=tl.int32)
        seq_mask = seq_offset[None, :] < boundary
        seq_mask = seq_mask & (tl.arange(0, BLOCK_SIZE) < (BLOCK_SIZE - q_mask_val))

        if rot_offset_val != prev_rot_offset:
            if rot_offset_val == 1:
                Q_rotated = Q_full
            elif rot_offset_val != 0:
                if COMPUTE_COS_SIN:
                    cos_val, sin_val = rope_cos_sin_from_freqs(
                        rot_offset_val - 1, rope_freq)
                else:
                    cache_cols = tl.arange(0, HEAD_SIZE) % embed_dim
                    cache_base = (rot_offset_val - 1) * rotary_dim
                    cos_val = tl.load(cos_sin_cache_ptr + cache_base + cache_cols)
                    sin_val = tl.load(
                        cos_sin_cache_ptr + cache_base + embed_dim + cache_cols)
                rev = (tl.arange(0, HEAD_SIZE_PADDED) + (HEAD_SIZE_PADDED // 2)) % HEAD_SIZE_PADDED
                Q_rev = tl.load(
                    query_ptr + query_offset + rev[None, :],
                    mask=dim_mask[None, :] & head_mask[:, None],
                    other=0.0,
                )
                q1 = tl.where(mask_q1[None, :], Q_full, Q_rev)
                q2 = tl.where(mask_q1[None, :], Q_rev, Q_full)
                q1_new = q1 * cos_val + q2 * sin_val
                q2_new = -q1 * sin_val + q2 * cos_val
                Q_rotated = tl.where(mask_q1[None, :], q1_new.to(Q_full.dtype), q2_new.to(Q_full.dtype))
            prev_rot_offset = rot_offset_val

        qk = tl.dot(Q_rotated, K, input_precision=IN_PRECISION)
        S = tl.where(head_mask[:, None] & seq_mask, 0.0, float("-inf")).to(tl.float32)
        S += scale * qk

        context_len = seq_len - 1
        if SLIDING_WINDOW > 0:
            S = tl.where((context_len - seq_offset) < SLIDING_WINDOW, S, -10000)
        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)

        m_j = tl.maximum(M, tl.max(S, axis=1))
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += tl.dot(P.to(V.dtype), V)

    acc = acc / L[:, None]
    output_offset = (cur_batch_in_all_start_index * output_stride_0 +
                     query_head_idx * output_stride_1)
    tl.store(
        output_ptr + output_offset[:, None] + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        acc,
        mask=dim_mask[None, :] & head_mask[:, None],
    )
