# TODO(haocheng): modify it later

# SPDX-License-Identifier: Apache-2.0

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

import triton
import triton.language as tl

from vllm.logger import init_logger

logger = init_logger(__name__)


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def kernel_unified_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.int64,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.int64,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    # //////////////////////////////////
    cos_sin_cache_ptr,
    # ROTARY_DIM: tl.constexpr, # int not used
    # IS_NEOX_STYLE: tl.constexpr, # not used
    is_lazy_req_ptr, # [num_seqs], used to skip
    # if is a document query, the query is start after the documents
    # e.g., two docs, one [16, 16, 14] and one [16]
    # [-2, 0, 0, -46]
    lazy_offset_ptr, # [num_seqs, max_num_blocks_per_seq] ~ same a block_tables_ptr
    # [16, 16, 14, 16], here 14 means 2 tokens in the block will be masked
    lazy_mask_ptr, # [num_seqs, max_num_blocks_per_seq] ~ same a block_tables_ptr
):  
    # Note(haocheng): assume rotary_dim is always equal to head_size
    # assert ROTARY_DIM == HEAD_SIZE, "rotary_dim should be equal to head_size"

    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        mid_val = tl.load(query_start_len_ptr + mid) // BLOCK_Q + mid
        if mid_val <= q_block_global_idx:
            left = mid + 1
        else:
            right = mid

    seq_idx = left - 1
    q_block_start_idx = tl.load(query_start_len_ptr +
                                seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    
    # Note(haocheng): If is_lazy_req is false, it means that the query is not a 
    # document query, in this case, no need to do anything.
    # Otherwise, we need to rotate the query according to the document layout.
    is_lazy = tl.load(is_lazy_req_ptr + seq_idx)

    cur_batch_query_len = cur_batch_in_all_stop_index \
        - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_Q * num_queries_per_kv)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    
    embed_dim: tl.constexpr = HEAD_SIZE // 2
    offs_d_0 = tl.arange(0, HEAD_SIZE_PADDED // 2)
    offs_d_1 = offs_d_0 + embed_dim
    
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + \
        offs_m % num_queries_per_kv

    # query_offset = (query_offset_0[:, None] * query_stride_0 +
    #                 query_offset_1[:, None] * query_stride_1 + offs_d[None, :])
    query_load_offset_0 = (query_offset_0[:, None] * query_stride_0 + 
                           query_offset_1[:, None] * query_stride_1 + offs_d_0[None, :])
    # query_load_offset_1 = query_load_offset_0 + embed_dim

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    dim_mask_half = tl.where(offs_d_0 < embed_dim, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # # Q : (BLOCK_Q * num_queries_per_kv, HEAD_SIZE,)
    # Q = tl.load(
    #     query_ptr + query_offset,
    #     mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    #     other=0.0,
    # )
    
    Q_0 = tl.load(
        query_ptr + query_load_offset_0,
        mask=dim_mask_half[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )
    
    Q_1 = tl.load(
        query_ptr + query_load_offset_0 + embed_dim,
        mask=dim_mask_half[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    M = tl.full([BLOCK_Q * num_queries_per_kv],
                float("-inf"),
                dtype=tl.float32)
    L = tl.full([BLOCK_Q * num_queries_per_kv], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q * num_queries_per_kv, HEAD_SIZE_PADDED],
                   dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1,
                              mask=query_mask_1,
                              other=0.0)

    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)

    # iterate through tiles
    for j in range(0, num_blocks):
        
        if is_lazy:
            # try to rotate the query according to the document layout
            rotate_offset = tl.load(lazy_offset_ptr + block_table_offset + j)
            if rotate_offset > 0:
                # rotate backwards
                # clockwise rotation: reverse position embedding
                # [ cos sin]
                # [-sin cos]
                position = tl.full([BLOCK_Q * num_queries_per_kv], 
                                   rotate_offset, dtype=tl.int32)
                cos_val = tl.load(cos_sin_cache_ptr + 
                                  (position[:, None] * HEAD_SIZE +
                                  offs_d_0[None,:]),
                                  mask=dim_mask_half[None,:],
                                  other=0.0)
                sin_val = tl.load(cos_sin_cache_ptr + 
                                  (position[:, None] * HEAD_SIZE +
                                  offs_d_1[None,:]),
                                  mask=dim_mask_half[None,:],
                                  other=0.0)
                # Note(haocheng): it is reverse rotation
                temp_Q_0 = Q_0 * cos_val + Q_1 * sin_val
                temp_Q_1 = -Q_0 * sin_val + Q_1 * cos_val
                Q_0 = temp_Q_0
                Q_1 = temp_Q_1
                
            elif rotate_offset < 0:
                # reload
                Q_0 = tl.load(
                    query_ptr + query_load_offset_0,
                    mask=dim_mask_half[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
                    other=0.0,
                )
                    
                Q_1 = tl.load(
                    query_ptr + query_load_offset_0 + embed_dim,
                    mask=dim_mask_half[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
                    other=0.0,
                )

        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

        offs_n = tl.arange(0, BLOCK_SIZE)

        v_offset = (physical_block_idx * stride_v_cache_0 +
                    kv_head_idx * stride_v_cache_2 +
                    offs_d[None, :] * stride_v_cache_3 +
                    offs_n[:, None] * stride_v_cache_1)

        # k_offset = (physical_block_idx * stride_k_cache_0 +
        #             kv_head_idx * stride_k_cache_2 +
        #             offs_d[:, None] * stride_k_cache_3 +
        #             offs_n[None, :] * stride_k_cache_1)

        # # K : (HEAD_SIZE, BLOCK_SIZE)
        # K_load = tl.load(key_cache_ptr + k_offset,
        #                  mask=dim_mask[:, None],
        #                  other=0.0)

        # if K_load.dtype.is_fp8():
        #     if Q.dtype.is_fp8():
        #         K = K_load
        #     else:
        #         K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        # else:
        #     K = K_load
        
        # Note(haocheng): load K in two parts as well
        k_offset_0 = (physical_block_idx * stride_k_cache_0 +
                      kv_head_idx * stride_k_cache_2 +
                      offs_d_0[:, None] * stride_k_cache_3 +
                      offs_n[None, :] * stride_k_cache_1)
        
        K_load_0 = tl.load(key_cache_ptr + k_offset_0,
                           mask=dim_mask_half[:,None],
                           other=0.0)
        K_load_1 = tl.load(key_cache_ptr + k_offset_0 + embed_dim * stride_k_cache_3,
                           mask=dim_mask_half[:,None],
                           other=0.0)
        
        if K_load_0.dtype.is_fp8():
            if Q_0.dtype.is_fp8():
                K_0 = K_load_0
                K_1 = K_load_1
            else:
                K_0 = (K_load_0.to(tl.float32) * tl.load(k_scale)).to(Q_0.dtype)
                K_1 = (K_load_1.to(tl.float32) * tl.load(k_scale)).to(Q_1.dtype)
        else:
            K_0 = K_load_0
            K_1 = K_load_1
        
        # V : (BLOCK_SIZE, HEAD_SIZE)
        V_load = tl.load(value_cache_ptr + v_offset,
                         mask=dim_mask[None, :],
                         other=0.0)

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        seq_offset = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        # S : (BLOCK_Q * num_queries_per_kv, BLOCK_SIZE,)
        S = tl.zeros(shape=(BLOCK_Q * num_queries_per_kv, BLOCK_SIZE),
                     dtype=tl.float32)

        # S += scale * tl.dot(Q, K)
        S += scale * (tl.dot(Q_0, K_0) + tl.dot(Q_1, K_1))

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        # Note(haocheng): here we use mask to mask out padding tokens
        lazy_padding_mask = (tl.arange(0, BLOCK_SIZE) < 
            tl.load(lazy_mask_ptr + block_table_offset + j))
        S = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & 
                     lazy_padding_mask[None, :] & seq_mask,
                     S, float("-inf"))
        # S = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
        #              S, float("-inf"))


        if SLIDING_WINDOW > 0:
            S = tl.where((context_len + query_pos[:, None] - seq_offset)
                         < SLIDING_WINDOW, S, float("-inf"))

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)

        # compute running maximum
        # m_j : (BLOCK_Q * num_queries_per_kv,)
        m_j = tl.maximum(M, tl.max(S, axis=1))
        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_Q * num_queries_per_kv, BLOCK_SIZE,)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_Q * num_queries_per_kv,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_Q * num_queries_per_kv, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_Q * num_queries_per_kv, BLOCK_SIZE,)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_Q * num_queries_per_kv, BLOCK_SIZE,)
        acc += tl.dot(P.to(V.dtype), V)

    # epilogue
    acc = acc / L[:, None]

    output_offset = (query_offset_0[:, None] * output_stride_0 +
                     query_offset_1[:, None] * output_stride_1 +
                     offs_d[None, :])

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    alibi_slopes=None,
    # ////////////////////////////////////
    cos_sin_cache=None,
    is_lazy_req=None,
    lazy_offset=None,
    lazy_mask=None,
    # TODO(haocheng): not used
    rotary_dim=None,
    is_neox_style=None,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    use_alibi_slopes = alibi_slopes is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = 16
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    kernel_unified_attention_2d[(
        total_num_q_blocks,
        num_kv_heads,
    )](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        scale=softmax_scale,
        k_scale=k_descale,
        v_scale=v_descale,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_ALIBI_SLOPES=use_alibi_slopes,
        USE_SOFTCAP=(softcap > 0),
        SLIDING_WINDOW=(1 + window_size[0]),
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        # //////////////////////////////////
        cos_sin_cache_ptr=cos_sin_cache,
        is_lazy_req_ptr=is_lazy_req,
        lazy_offset_ptr=lazy_offset,
        lazy_mask_ptr=lazy_mask,
    )
