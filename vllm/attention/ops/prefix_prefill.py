# The kernels in this file are adapted from LightLLM's context_attention_fwd:
# https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py

# Modified by DynamicRAG team to include DynamicRAG's Rotary Encoding for prefilling stage

import torch
import triton
import triton.language as tl

from vllm.platforms import current_platform


if triton.__version__ >= "2.1.0":

    # **************************** drag begins ****************************
    """
    v2 means we load the query in two parts, and for k, we have two parts
    qk = qk_1 + qk_2
    that is, a deeper fusion of rotary embedding and qk computation
    """
    @triton.jit
    def _fwd_kernel_dynamic_rag_v2(
        Q,
        K,
        V,
        K_cache,
        V_cache,
        B_Loc,
        sm_scale,
        k_scale,
        v_scale,
        B_Start_Loc,
        B_Seqlen,
        B_Ctxlen,
        block_size,
        x,
        Out,
        stride_b_loc_b,
        stride_b_loc_s,
        stride_qbs,
        stride_qh,
        stride_qd,
        stride_kbs,
        stride_kh,
        stride_kd,
        stride_vbs,
        stride_vh,
        stride_vd,
        stride_obs,
        stride_oh,
        stride_od,
        stride_k_cache_bs,
        stride_k_cache_h,
        stride_k_cache_d,
        stride_k_cache_bl,
        stride_k_cache_x,
        stride_v_cache_bs,
        stride_v_cache_h,
        stride_v_cache_d,
        stride_v_cache_bl,
        num_queries_per_kv: int,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,  # head size
        BLOCK_DMODEL_PADDED: tl.constexpr,  # head size padded to a power of 2
        BLOCK_N: tl.constexpr,
        SLIDING_WINDOW: tl.constexpr,
        cos_sin_cache,
        rotary_dim: tl.constexpr,
    ):
        # grid: (batch_size, num_head, num_blocks) -> triton instances for parallel execution
        # ceil(max_input_len / BLOCK) = num_blocks
        # from grid get id of batch, head, and start_m

        cur_batch = tl.program_id(0)  
        cur_head = tl.program_id(1)
        start_m = tl.program_id(2)

        cur_kv_head = cur_head // num_queries_per_kv  # share kv for query

        cur_batch_ctx_len = tl.load(B_Ctxlen + cur_batch)  # get context length for current request
        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)  # get sequence length for current request
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)  # get start index for current request
        cur_batch_query_len = cur_batch_seq_len - cur_batch_ctx_len  # get query length for current request

        # start position inside of the query
        # generally, N goes over kv, while M goes over query_len
        block_start_loc = BLOCK_M * start_m

        # initialize offsets
        # [N]; starts at 0
        offs_n = tl.arange(0, BLOCK_N)
        # [D]; starts at 0
        offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
        # [M]; starts at current position in query
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

        dim_mask = tl.where(
            tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1,
            0).to(tl.int1)  # [D]
        dim_mask_half = tl.where(
            tl.arange(0, BLOCK_DMODEL_PADDED // 2) < BLOCK_DMODEL // 2, 1,
            0).to(tl.int1)  # [D//2]

        # divide query into two parts
        embed_dim: tl.constexpr = rotary_dim // 2
        offs_d1 = tl.arange(0, embed_dim)
        offs_d2 = offs_d1 + embed_dim
        # [M, D//2]
        off_q1 = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs +
            cur_head * stride_qh + offs_d1[None, :] * stride_qd)
        # [M, D//2]
        off_q2 = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs +
            cur_head * stride_qh + offs_d2[None, :] * stride_qd)
        
        q_1 = tl.load(Q + off_q1,
                    mask=dim_mask_half[None, :] &
                    (offs_m[:, None] < cur_batch_query_len),
                    other=0.0)  # [M,D//2]
        q_2 = tl.load(Q + off_q2,
                    mask=dim_mask_half[None, :] &
                    (offs_m[:, None] < cur_batch_query_len),
                    other=0.0)  # [M,D//2]

        # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # [M]
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # [M]
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED],
                       dtype=tl.float32)  # [M,D]
            
        # compute query against context (no causal mask here)
        for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            bn = tl.load(B_Loc + cur_batch * stride_b_loc_b +
                         ((start_n + offs_n) // block_size) * stride_b_loc_s,
                         mask=(start_n + offs_n) < cur_batch_ctx_len,
                         other=0)  # [N]
            # [N,D]
            off_v = (
                bn[:, None] * stride_v_cache_bs +
                cur_kv_head * stride_v_cache_h +
                offs_d[None, :] * stride_v_cache_d +
                (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl)
            
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)  # [M,N]

            # [D//2,N]
            off_k_1 = (bn[None, :] * stride_k_cache_bs +  # block
                     cur_kv_head * stride_k_cache_h +         # head
                     (offs_d1[:, None] // x) * stride_k_cache_d + 
                     ((start_n + offs_n[None, :]) % block_size) *
                     stride_k_cache_bl +
                     (offs_d1[:, None] % x) * stride_k_cache_x)
                
            off_k_2 = (bn[None, :] * stride_k_cache_bs +
                     cur_kv_head * stride_k_cache_h +
                     (offs_d2[:, None] // x) * stride_k_cache_d +
                     ((start_n + offs_n[None, :]) % block_size) *
                     stride_k_cache_bl +
                     (offs_d2[:, None] % x) * stride_k_cache_x)
            
            k_load_1 = tl.load(K_cache + off_k_1,
                                mask=dim_mask_half[:, None] &
                                ((start_n + offs_n[None, :]) < cur_batch_ctx_len),
                                other=0.0)
            k_load_2 = tl.load(K_cache + off_k_2,
                                mask=dim_mask_half[:, None] &
                                ((start_n + offs_n[None, :]) < cur_batch_ctx_len),
                                other=0.0)

            if k_load_1.dtype.is_fp8():
                k_load_1 = (k_load_1.to(tl.float32) * k_scale).to(q_1.dtype)
            else:
                k_load_1 = k_load_1

            if k_load_2.dtype.is_fp8():
                k_load_2 = (k_load_2.to(tl.float32) * k_scale).to(q_1.dtype)
            else:
                k_load_2 = k_load_2

            # fuse rotary embedding
            positions = start_n + tl.arange(0, BLOCK_N)  # [N]
            # [D//2, N]
            offs_cache_1 = (positions[None, :] * rotary_dim +
                            offs_d1[:, None])
            offs_cache_2 = (positions[None, :] * rotary_dim +
                            offs_d2[:, None])
                
            cos_val = tl.load(cos_sin_cache + offs_cache_1,
                               mask=dim_mask_half[:, None] &
                                ((start_n + offs_n[None, :]) < cur_batch_ctx_len),
                                other=0.0)
            sin_val = tl.load(cos_sin_cache + offs_cache_2, 
                              mask=dim_mask_half[:, None] &
                                ((start_n + offs_n[None, :]) < cur_batch_ctx_len),
                                other=0.0)
                
            qk += tl.dot(q_1, (k_load_1 * cos_val - k_load_2 * sin_val))
            qk += tl.dot(q_2, (k_load_1 * sin_val + k_load_2 * cos_val))
            qk = tl.where((start_n + offs_n[None, :]) < cur_batch_ctx_len, qk,
                            float("-inf"))
            qk *= sm_scale
            if SLIDING_WINDOW > 0:
                # (cur_batch_ctx_len + offs_m[:, None]) are the positions of
                # Q entries in sequence
                # (start_n + offs_n[None, :]) are the positions of
                # KV entries in sequence
                # So the condition makes sure each entry in Q only attends
                # to KV entries not more than SLIDING_WINDOW away.
                #
                # We can't use -inf here, because the
                # sliding window may lead to the entire row being masked.
                # This then makes m_ij contain -inf, which causes NaNs in
                # exp().
                qk = tl.where((cur_batch_ctx_len + offs_m[:, None]) -
                              (start_n + offs_n[None, :]) < SLIDING_WINDOW, qk,
                              -10000)

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)  # [M]
            p = tl.exp(qk - m_ij[:, None])  # [M,N]
            l_ij = tl.sum(p, 1)  # [M]
            # -- update m_i and l_i
            m_i_new = tl.maximum(m_i, m_ij)  # [M]
            alpha = tl.exp(m_i - m_i_new)  # [M]
            beta = tl.exp(m_ij - m_i_new)  # [M]
            l_i_new = alpha * l_i + beta * l_ij  # [M]

            # -- update output accumulator --
            # scale p
            p_scale = beta / l_i_new
            p = p * p_scale[:, None]
            # scale acc
            acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc
            v_load = tl.load(V_cache + off_v,
                             mask=dim_mask[None, :] &
                             ((start_n + offs_n[:, None]) < cur_batch_ctx_len),
                             other=0.0)  # [N,D]
            if v_load.dtype.is_fp8():
                v = (v_load.to(tl.float32) * v_scale).to(q.dtype)
            else:
                v = v_load
            p = p.to(v.dtype)

            acc += tl.dot(p, v)
            # # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new

        off_k = (offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh +
                 offs_d[:, None] * stride_kd)
        off_v = (offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh +
                 offs_d[None, :] * stride_vd)

       
        k_ptrs = K + off_k
        v_ptrs = V + off_v

        # block_mask is 0 when we're already past the current query length
        block_mask = tl.where(block_start_loc < cur_batch_query_len, 1, 0)

        # compute query against itself (with causal mask)
        for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            # Apply rotary embedding to the key tensor `k` for the new part

            # [D//2,N]
            off_k_1 = (offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh +
                       offs_d1[:, None] * stride_kd)
            off_k_2 = (offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh +
                       offs_d2[:, None] * stride_kd)
                
            k_load_1 = tl.load(K + off_k_1 + 
                (cur_batch_in_all_start_index + start_n) * stride_kbs,
                        mask=dim_mask_half[:, None] &
                        ((start_n + offs_n[None, :]) < cur_batch_query_len),
                        other=0.0)
            k_load_2 = tl.load(K + off_k_2 + 
                (cur_batch_in_all_start_index + start_n) * stride_kbs,
                        mask=dim_mask_half[:, None] &
                        ((start_n + offs_n[None, :]) < cur_batch_query_len),
                        other=0.0)

            # [D//2, N]
            positions = cur_batch_ctx_len + start_n + tl.arange(0, BLOCK_N)  # [N]
            offs_cache_1 = (positions[None, :] * rotary_dim +
                                offs_d1[:, None])
            offs_cache_2 = (positions[None, :] * rotary_dim +
                                offs_d2[:, None])
                
            cos_val = tl.load(cos_sin_cache + offs_cache_1, 
                              mask=dim_mask_half[:, None] &
                        ((start_n + offs_n[None, :]) < cur_batch_query_len),
                        other=0.0)
            sin_val = tl.load(cos_sin_cache + offs_cache_2, 
                              mask=dim_mask_half[:, None] &
                        ((start_n + offs_n[None, :]) < cur_batch_query_len),
                        other=0.0)
                
            qk += tl.dot(q_1, (k_load_1 * cos_val - k_load_2 * sin_val))
            qk += tl.dot(q_2, (k_load_1 * sin_val + k_load_2 * cos_val))

            qk *= sm_scale

            # apply causal mask
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
                          float("-inf"))
            if SLIDING_WINDOW > 0:
                qk = tl.where(
                    offs_m[:, None] -
                    (start_n + offs_n[None, :]) < SLIDING_WINDOW, qk, -10000)

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            beta = tl.exp(m_ij - m_i_new)
            l_i_new = alpha * l_i + beta * l_ij
            # -- update output accumulator --
            # scale p
            p_scale = beta / l_i_new
            p = p * p_scale[:, None]
            # scale acc
            acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc
            v = tl.load(v_ptrs +
                        (cur_batch_in_all_start_index + start_n) * stride_vbs,
                        mask=dim_mask[None, :] &
                        ((start_n + offs_n[:, None]) < cur_batch_query_len),
                        other=0.0)
            p = p.to(v.dtype)

            acc += tl.dot(p, v)
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new
        # initialize pointers to output
        off_o = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs +
            cur_head * stride_oh + offs_d[None, :] * stride_od)
        out_ptrs = Out + off_o
        tl.store(out_ptrs,
                 acc,
                 mask=dim_mask[None, :] &
                 (offs_m[:, None] < cur_batch_query_len))
        return
    
    @triton.jit
    def _fwd_kernel_dynamic_rag(
        Q,
        K,
        V,
        K_cache,
        V_cache,
        B_Loc,
        sm_scale,
        k_scale,
        v_scale,
        B_Start_Loc,
        B_Seqlen,
        B_Ctxlen,
        block_size,
        x,
        Out,
        stride_b_loc_b,
        stride_b_loc_s,
        stride_qbs,
        stride_qh,
        stride_qd,
        stride_kbs,
        stride_kh,
        stride_kd,
        stride_vbs,
        stride_vh,
        stride_vd,
        stride_obs,
        stride_oh,
        stride_od,
        stride_k_cache_bs,
        stride_k_cache_h,
        stride_k_cache_d,
        stride_k_cache_bl,
        stride_k_cache_x,
        stride_v_cache_bs,
        stride_v_cache_h,
        stride_v_cache_d,
        stride_v_cache_bl,
        num_queries_per_kv: int,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,  # head size
        BLOCK_DMODEL_PADDED: tl.constexpr,  # head size padded to a power of 2
        BLOCK_N: tl.constexpr,
        SLIDING_WINDOW: tl.constexpr,
        cos_sin_cache,
        rotary_dim: tl.constexpr,
    ):
        # grid: (batch_size, num_head, num_blocks) -> triton instances for parallel execution
        # ceil(max_input_len / BLOCK) = num_blocks
        # from grid get id of batch, head, and start_m
        # For debug:
        debug_cached_part = False
        debug_new_part = False

        cur_batch = tl.program_id(0)  
        cur_head = tl.program_id(1)
        start_m = tl.program_id(2)

        cur_kv_head = cur_head // num_queries_per_kv  # share kv for query

        cur_batch_ctx_len = tl.load(B_Ctxlen + cur_batch)  # get context length for current request
        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)  # get sequence length for current request
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)  # get start index for current request
        cur_batch_query_len = cur_batch_seq_len - cur_batch_ctx_len  # get query length for current request

        # start position inside of the query
        # generally, N goes over kv, while M goes over query_len
        block_start_loc = BLOCK_M * start_m

        # initialize offsets
        # [N]; starts at 0
        offs_n = tl.arange(0, BLOCK_N)
        # [D]; starts at 0
        offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
        # [M]; starts at current position in query
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # [M,D]
        off_q = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs +
            cur_head * stride_qh + offs_d[None, :] * stride_qd)

        dim_mask = tl.where(
            tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1,
            0).to(tl.int1)  # [D]

        q = tl.load(Q + off_q,
                    mask=dim_mask[None, :] &
                    (offs_m[:, None] < cur_batch_query_len),
                    other=0.0)  # [M,D]

        # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # [M]
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # [M]
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED],
                       dtype=tl.float32)  # [M,D]
            
        # compute query against context (no causal mask here)
        for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            bn = tl.load(B_Loc + cur_batch * stride_b_loc_b +
                         ((start_n + offs_n) // block_size) * stride_b_loc_s,
                         mask=(start_n + offs_n) < cur_batch_ctx_len,
                         other=0)  # [N]
            '''
            bn: e.g. [0,0,.., 0,1,1,..,1,0...], 0 for block 0, 1 for block 1, etc.
            '''
            # [D,N]
            # off_k = (bn[None, :] * stride_k_cache_bs +
            #          cur_kv_head * stride_k_cache_h +  # [1, N]
            #          (offs_d[:, None] // x) * stride_k_cache_d +  # offs_d = tl.arange(0, BLOCK_DMODEL_PADDED) [D, 1]
            #          ((start_n + offs_n[None, :]) % block_size) *
            #          stride_k_cache_bl +
            #          (offs_d[:, None] % x) * stride_k_cache_x)
            # [N,D]
            off_v = (
                bn[:, None] * stride_v_cache_bs +
                cur_kv_head * stride_v_cache_h +
                offs_d[None, :] * stride_v_cache_d +
                (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl)
            
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)  # [M,N]

                # Apply rotary embedding to the key tensor `k` for the cached part
                # k [head_dim, block_length]
                # if ((tl.program_id(0) == 0) and (tl.program_id(1) == 0)) and (tl.program_id(2) == 0):
                #     tl.static_print('rotary_dim: ', rotary_dim)

                # second block wrong
                # TODO(haocheng): better rotary_dim support different style neox and gptj
            embed_dim: tl.constexpr = rotary_dim // 2
            dim_mask = tl.where(
                    tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1,
                    0).to(tl.int1)  # [D]
                # TODO(haocheng): robust to the number of BLOCK_DMODEL_PADDED
            offs_d1 = tl.arange(0, rotary_dim) % embed_dim
            offs_d2 = offs_d1 + embed_dim

            # [D,N]
            off_k_1 = (bn[None, :] * stride_k_cache_bs +  # block
                     cur_kv_head * stride_k_cache_h +         # head
                     (offs_d1[:, None] // x) * stride_k_cache_d + 
                     ((start_n + offs_n[None, :]) % block_size) *
                     stride_k_cache_bl +
                     (offs_d1[:, None] % x) * stride_k_cache_x)
                
            off_k_2 = (bn[None, :] * stride_k_cache_bs +
                     cur_kv_head * stride_k_cache_h +
                     (offs_d2[:, None] // x) * stride_k_cache_d +
                     ((start_n + offs_n[None, :]) % block_size) *
                     stride_k_cache_bl +
                     (offs_d2[:, None] % x) * stride_k_cache_x)
                
            k_load_1 = tl.load(K_cache + off_k_1,
                                mask=dim_mask[:, None] &
                                ((start_n + offs_n[None, :]) < cur_batch_ctx_len),
                                other=0.0)
            k_load_2 = tl.load(K_cache + off_k_2,
                                mask=dim_mask[:, None] &
                                ((start_n + offs_n[None, :]) < cur_batch_ctx_len),
                                other=0.0)

            if k_load_1.dtype.is_fp8():
                k_load_1 = (k_load_1.to(tl.float32) * k_scale).to(q.dtype)
            else:
                k_load_1 = k_load_1

            if k_load_2.dtype.is_fp8():
                k_load_2 = (k_load_2.to(tl.float32) * k_scale).to(q.dtype)
            else:
                k_load_2 = k_load_2

            index_d = tl.arange(0, rotary_dim)
            k_load = tl.where(index_d[:, None] < embed_dim, k_load_1, k_load_2)

                # get corresponding cos_sin_cache
                # [D, N]
                # TODO(haocheng): mask
                # if ((tl.program_id(0) == 0) and (tl.program_id(1) == 0)):
                #     tl.device_print('compute qk for cached k', start_n)
            positions = start_n + tl.arange(0, BLOCK_N)  # [N]
            offs_cache_1 = (positions[None, :] * rotary_dim +
                                offs_d1[:, None])
            offs_cache_2 = (positions[None, :] * rotary_dim +
                                offs_d2[:, None])
                
            cos_val = tl.load(cos_sin_cache + offs_cache_1)
            sin_val = tl.load(cos_sin_cache + offs_cache_2)
                
            rotated_k_load_1 = k_load_1 * cos_val - k_load_2 * sin_val
            rotated_k_load_2 = k_load_1 * sin_val + k_load_2 * cos_val

            index_d = tl.arange(0, rotary_dim)
            k = tl.where(index_d[:, None] < embed_dim, rotated_k_load_1, rotated_k_load_2)
            #     ###########################################
            # if debug_cached_part:
            #     tl.debug_barrier()
            #     if ((tl.program_id(0) == 0) and (tl.program_id(1) == 0)) and (tl.program_id(2) == 0):
            #         int_val = tl.cast(k_load * 1000, tl.int32)
            #         tl.device_print('cached k after custom rotary 1k', int_val)
            #     ###########################################
            qk += tl.dot(q, k)
            qk = tl.where((start_n + offs_n[None, :]) < cur_batch_ctx_len, qk,
                            float("-inf"))
            qk *= sm_scale
            if SLIDING_WINDOW > 0:
                # (cur_batch_ctx_len + offs_m[:, None]) are the positions of
                # Q entries in sequence
                # (start_n + offs_n[None, :]) are the positions of
                # KV entries in sequence
                # So the condition makes sure each entry in Q only attends
                # to KV entries not more than SLIDING_WINDOW away.
                #
                # We can't use -inf here, because the
                # sliding window may lead to the entire row being masked.
                # This then makes m_ij contain -inf, which causes NaNs in
                # exp().
                qk = tl.where((cur_batch_ctx_len + offs_m[:, None]) -
                              (start_n + offs_n[None, :]) < SLIDING_WINDOW, qk,
                              -10000)

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)  # [M]
            p = tl.exp(qk - m_ij[:, None])  # [M,N]
            l_ij = tl.sum(p, 1)  # [M]
            # -- update m_i and l_i
            m_i_new = tl.maximum(m_i, m_ij)  # [M]
            alpha = tl.exp(m_i - m_i_new)  # [M]
            beta = tl.exp(m_ij - m_i_new)  # [M]
            l_i_new = alpha * l_i + beta * l_ij  # [M]

            # -- update output accumulator --
            # scale p
            p_scale = beta / l_i_new
            p = p * p_scale[:, None]
            # scale acc
            acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc
            v_load = tl.load(V_cache + off_v,
                             mask=dim_mask[None, :] &
                             ((start_n + offs_n[:, None]) < cur_batch_ctx_len),
                             other=0.0)  # [N,D]
            if v_load.dtype.is_fp8():
                v = (v_load.to(tl.float32) * v_scale).to(q.dtype)
            else:
                v = v_load
            p = p.to(v.dtype)

            acc += tl.dot(p, v)
            # # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new

        off_k = (offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh +
                 offs_d[:, None] * stride_kd)
        off_v = (offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh +
                 offs_d[None, :] * stride_vd)

       
        k_ptrs = K + off_k
        v_ptrs = V + off_v

        # block_mask is 0 when we're already past the current query length
        block_mask = tl.where(block_start_loc < cur_batch_query_len, 1, 0)

        # compute query against itself (with causal mask)
        for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            # Apply rotary embedding to the key tensor `k` for the new part
            if True:
                # Apply rotary embedding to the key tensor `k` for the new part
                embed_dim: tl.constexpr = rotary_dim // 2
                dim_mask = tl.where(
                    tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1,
                    0).to(tl.int1)  # [D]
                # TODO(haocheng): robust to the number of BLOCK_DMODEL_PADDED
                offs_d1 = tl.arange(0, rotary_dim) % embed_dim
                offs_d2 = offs_d1 + embed_dim

                # [D,N]
                off_k_1 = (offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh +
                 offs_d1[:, None] * stride_kd)
                
                off_k_2 = (offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh +
                 offs_d2[:, None] * stride_kd)
                
                k_load_1 = tl.load(K + off_k_1 + 
                (cur_batch_in_all_start_index + start_n) * stride_kbs,
                        mask=dim_mask[:, None] &
                        ((start_n + offs_n[None, :]) < cur_batch_query_len),
                        other=0.0)
                k_load_2 = tl.load(K + off_k_2 + 
                (cur_batch_in_all_start_index + start_n) * stride_kbs,
                        mask=dim_mask[:, None] &
                        ((start_n + offs_n[None, :]) < cur_batch_query_len),
                        other=0.0)

                index_d = tl.arange(0, rotary_dim)
                k_load = tl.where(index_d[:, None] < embed_dim, k_load_1, k_load_2)
                #########################################
                if debug_new_part:
                    tl.debug_barrier()
                    if ((tl.program_id(0) == 0) and (tl.program_id(1) == 0)) and (tl.program_id(2) == 0):
                        int_val = tl.cast(k_load * 1000, tl.int32)
                        tl.device_print('new k before custom rotary 1k', int_val)
                #########################################
                # # get corresponding cos_sin_cache
                # # [D//2, N]
                # # TODO(haocheng): mask
                # # if ((tl.program_id(0) == 0) and (tl.program_id(1) == 0)):
                # #     tl.device_print('compute qk for query itself', cur_batch_ctx_len + start_n)
                positions = cur_batch_ctx_len + start_n + tl.arange(0, BLOCK_N)  # [N]
                offs_cache_1 = (positions[None, :] * rotary_dim +
                                offs_d1[:, None])
                offs_cache_2 = (positions[None, :] * rotary_dim +
                                offs_d2[:, None])
                
                cos_val = tl.load(cos_sin_cache + offs_cache_1)
                sin_val = tl.load(cos_sin_cache + offs_cache_2)
                
                rotated_k_load_1 = k_load_1 * cos_val - k_load_2 * sin_val
                rotated_k_load_2 = k_load_1 * sin_val + k_load_2 * cos_val

                index_d = tl.arange(0, rotary_dim)
                k_load = tl.where(index_d[:, None] < embed_dim, rotated_k_load_1, rotated_k_load_2)
                qk += tl.dot(q, k_load)
                qk *= sm_scale
            # if ((tl.program_id(0) == 0) and (tl.program_id(1) == 0)) and (tl.program_id(2) == 0):
            #     int_val = tl.cast(qk * 1000, tl.int32)
            #     tl.device_print('after new qk', int_val)

            # apply causal mask
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
                          float("-inf"))
            if SLIDING_WINDOW > 0:
                qk = tl.where(
                    offs_m[:, None] -
                    (start_n + offs_n[None, :]) < SLIDING_WINDOW, qk, -10000)

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            beta = tl.exp(m_ij - m_i_new)
            l_i_new = alpha * l_i + beta * l_ij
            # -- update output accumulator --
            # scale p
            p_scale = beta / l_i_new
            p = p * p_scale[:, None]
            # scale acc
            acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc
            v = tl.load(v_ptrs +
                        (cur_batch_in_all_start_index + start_n) * stride_vbs,
                        mask=dim_mask[None, :] &
                        ((start_n + offs_n[:, None]) < cur_batch_query_len),
                        other=0.0)
            p = p.to(v.dtype)

            acc += tl.dot(p, v)
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new
        # initialize pointers to output
        off_o = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs +
            cur_head * stride_oh + offs_d[None, :] * stride_od)
        out_ptrs = Out + off_o
        tl.store(out_ptrs,
                 acc,
                 mask=dim_mask[None, :] &
                 (offs_m[:, None] < cur_batch_query_len))
        return

    # **************************** drag ends ******************************

    @triton.jit
    def _fwd_kernel(
        Q,
        K,
        V,
        K_cache,
        V_cache,
        B_Loc,
        sm_scale,
        k_scale,
        v_scale,
        B_Start_Loc,
        B_Seqlen,
        B_Ctxlen,
        block_size,
        x,
        Out,
        stride_b_loc_b,
        stride_b_loc_s,
        stride_qbs,
        stride_qh,
        stride_qd,
        stride_kbs,
        stride_kh,
        stride_kd,
        stride_vbs,
        stride_vh,
        stride_vd,
        stride_obs,
        stride_oh,
        stride_od,
        stride_k_cache_bs,
        stride_k_cache_h,
        stride_k_cache_d,
        stride_k_cache_bl,
        stride_k_cache_x,
        stride_v_cache_bs,
        stride_v_cache_h,
        stride_v_cache_d,
        stride_v_cache_bl,
        num_queries_per_kv: int,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,  # head size
        BLOCK_DMODEL_PADDED: tl.constexpr,  # head size padded to a power of 2
        BLOCK_N: tl.constexpr,
        SLIDING_WINDOW: tl.constexpr,
    ):
        cur_batch = tl.program_id(0)
        cur_head = tl.program_id(1)
        start_m = tl.program_id(2)

        cur_kv_head = cur_head // num_queries_per_kv

        cur_batch_ctx_len = tl.load(B_Ctxlen + cur_batch)
        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
        cur_batch_query_len = cur_batch_seq_len - cur_batch_ctx_len

        # start position inside of the query
        # generally, N goes over kv, while M goes over query_len
        block_start_loc = BLOCK_M * start_m

        # initialize offsets
        # [N]; starts at 0
        offs_n = tl.arange(0, BLOCK_N)
        # [D]; starts at 0
        offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
        # [M]; starts at current position in query
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # [M,D]
        off_q = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs +
            cur_head * stride_qh + offs_d[None, :] * stride_qd)

        dim_mask = tl.where(
            tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1,
            0).to(tl.int1)  # [D]

        q = tl.load(Q + off_q,
                    mask=dim_mask[None, :] &
                    (offs_m[:, None] < cur_batch_query_len),
                    other=0.0)  # [M,D]

        # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # [M]
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # [M]
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED],
                       dtype=tl.float32)  # [M,D]
            
        # compute query against context (no causal mask here)
        for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            bn = tl.load(B_Loc + cur_batch * stride_b_loc_b +
                         ((start_n + offs_n) // block_size) * stride_b_loc_s,
                         mask=(start_n + offs_n) < cur_batch_ctx_len,
                         other=0)  # [N]
            # [D,N]
            off_k = (bn[None, :] * stride_k_cache_bs +
                     cur_kv_head * stride_k_cache_h +
                     (offs_d[:, None] // x) * stride_k_cache_d +
                     ((start_n + offs_n[None, :]) % block_size) *
                     stride_k_cache_bl +
                     (offs_d[:, None] % x) * stride_k_cache_x)
            # [N,D]
            off_v = (
                bn[:, None] * stride_v_cache_bs +
                cur_kv_head * stride_v_cache_h +
                offs_d[None, :] * stride_v_cache_d +
                (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl)
            
            k_load = tl.load(K_cache + off_k,
                             mask=dim_mask[:, None] &
                             ((start_n + offs_n[None, :]) < cur_batch_ctx_len),
                             other=0.0)  # [D,N]

            if k_load.dtype.is_fp8():
                k = (k_load.to(tl.float32) * k_scale).to(q.dtype)
            else:
                k = k_load

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)  # [M,N]
            qk += tl.dot(q, k)
            qk = tl.where((start_n + offs_n[None, :]) < cur_batch_ctx_len, qk,
                          float("-inf"))
            qk *= sm_scale
            if SLIDING_WINDOW > 0:
                # (cur_batch_ctx_len + offs_m[:, None]) are the positions of
                # Q entries in sequence
                # (start_n + offs_n[None, :]) are the positions of
                # KV entries in sequence
                # So the condition makes sure each entry in Q only attends
                # to KV entries not more than SLIDING_WINDOW away.
                #
                # We can't use -inf here, because the
                # sliding window may lead to the entire row being masked.
                # This then makes m_ij contain -inf, which causes NaNs in
                # exp().
                qk = tl.where((cur_batch_ctx_len + offs_m[:, None]) -
                              (start_n + offs_n[None, :]) < SLIDING_WINDOW, qk,
                              -10000)

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)  # [M]
            p = tl.exp(qk - m_ij[:, None])  # [M,N]
            l_ij = tl.sum(p, 1)  # [M]
            # -- update m_i and l_i
            m_i_new = tl.maximum(m_i, m_ij)  # [M]
            alpha = tl.exp(m_i - m_i_new)  # [M]
            beta = tl.exp(m_ij - m_i_new)  # [M]
            l_i_new = alpha * l_i + beta * l_ij  # [M]

            # -- update output accumulator --
            # scale p
            p_scale = beta / l_i_new
            p = p * p_scale[:, None]
            # scale acc
            acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc
            v_load = tl.load(V_cache + off_v,
                             mask=dim_mask[None, :] &
                             ((start_n + offs_n[:, None]) < cur_batch_ctx_len),
                             other=0.0)  # [N,D]
            if v_load.dtype.is_fp8():
                v = (v_load.to(tl.float32) * v_scale).to(q.dtype)
            else:
                v = v_load
            p = p.to(v.dtype)

            acc += tl.dot(p, v)
            # # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new

        off_k = (offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh +
                 offs_d[:, None] * stride_kd)
        off_v = (offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh +
                 offs_d[None, :] * stride_vd)

       
        k_ptrs = K + off_k
        v_ptrs = V + off_v

        # block_mask is 0 when we're already past the current query length
        block_mask = tl.where(block_start_loc < cur_batch_query_len, 1, 0)

        # compute query against itself (with causal mask)
        for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            k = tl.load(k_ptrs +
                        (cur_batch_in_all_start_index + start_n) * stride_kbs,
                        mask=dim_mask[:, None] &
                        ((start_n + offs_n[None, :]) < cur_batch_query_len),
                        other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk *= sm_scale
            # apply causal mask
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
                          float("-inf"))
            if SLIDING_WINDOW > 0:
                qk = tl.where(
                    offs_m[:, None] -
                    (start_n + offs_n[None, :]) < SLIDING_WINDOW, qk, -10000)

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            beta = tl.exp(m_ij - m_i_new)
            l_i_new = alpha * l_i + beta * l_ij
            # -- update output accumulator --
            # scale p
            p_scale = beta / l_i_new
            p = p * p_scale[:, None]
            # scale acc
            acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc
            v = tl.load(v_ptrs +
                        (cur_batch_in_all_start_index + start_n) * stride_vbs,
                        mask=dim_mask[None, :] &
                        ((start_n + offs_n[:, None]) < cur_batch_query_len),
                        other=0.0)
            p = p.to(v.dtype)

            acc += tl.dot(p, v)
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new
        # initialize pointers to output
        off_o = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs +
            cur_head * stride_oh + offs_d[None, :] * stride_od)
        out_ptrs = Out + off_o
        tl.store(out_ptrs,
                 acc,
                 mask=dim_mask[None, :] &
                 (offs_m[:, None] < cur_batch_query_len))
        return

    @triton.jit
    def _fwd_kernel_flash_attn_v2(
        Q,
        K,
        V,
        K_cache,
        V_cache,
        B_Loc,
        sm_scale,
        B_Start_Loc,
        B_Seqlen,
        B_Ctxlen,
        block_size,
        x,
        Out,
        stride_b_loc_b,
        stride_b_loc_s,
        stride_qbs,
        stride_qh,
        stride_qd,
        stride_kbs,
        stride_kh,
        stride_kd,
        stride_vbs,
        stride_vh,
        stride_vd,
        stride_obs,
        stride_oh,
        stride_od,
        stride_k_cache_bs,
        stride_k_cache_h,
        stride_k_cache_d,
        stride_k_cache_bl,
        stride_k_cache_x,
        stride_v_cache_bs,
        stride_v_cache_h,
        stride_v_cache_d,
        stride_v_cache_bl,
        num_queries_per_kv: int,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        cur_batch = tl.program_id(0)
        cur_head = tl.program_id(1)
        start_m = tl.program_id(2)

        cur_kv_head = cur_head // num_queries_per_kv

        cur_batch_ctx_len = tl.load(B_Ctxlen + cur_batch)
        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

        block_start_loc = BLOCK_M * start_m

        # initialize offsets
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_q = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs +
            cur_head * stride_qh + offs_d[None, :] * stride_qd)

        q = tl.load(
            Q + off_q,
            mask=offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len,
            other=0.0)

        # # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            bn = tl.load(B_Loc + cur_batch * stride_b_loc_b +
                         ((start_n + offs_n) // block_size) * stride_b_loc_s,
                         mask=(start_n + offs_n) < cur_batch_ctx_len,
                         other=0)
            off_k = (bn[None, :] * stride_k_cache_bs +
                     cur_kv_head * stride_k_cache_h +
                     (offs_d[:, None] // x) * stride_k_cache_d +
                     ((start_n + offs_n[None, :]) % block_size) *
                     stride_k_cache_bl +
                     (offs_d[:, None] % x) * stride_k_cache_x)
            off_v = (
                bn[:, None] * stride_v_cache_bs +
                cur_kv_head * stride_v_cache_h +
                offs_d[None, :] * stride_v_cache_d +
                (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl)
            k = tl.load(K_cache + off_k,
                        mask=(start_n + offs_n[None, :]) < cur_batch_ctx_len,
                        other=0.0)
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk = tl.where((start_n + offs_n[None, :]) < cur_batch_ctx_len, qk,
                          float("-inf"))
            qk *= sm_scale

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)
            m_i_new = tl.maximum(m_i, m_ij)
            p = tl.math.exp(qk - m_i_new[:, None])
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i

            alpha = tl.math.exp(m_i - m_i_new)
            l_i_new = alpha * l_i + l_ij
            # -- update output accumulator --
            # scale p
            # scale acc
            acc_scale = alpha
            # acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc
            v = tl.load(V_cache + off_v,
                        mask=(start_n + offs_n[:, None]) < cur_batch_ctx_len,
                        other=0.0)

            p = p.to(v.dtype)
            acc += tl.dot(p, v)
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new

        off_k = (offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh +
                 offs_d[:, None] * stride_kd)
        off_v = (offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh +
                 offs_d[None, :] * stride_vd)
        k_ptrs = K + off_k
        v_ptrs = V + off_v

        block_mask = tl.where(
            block_start_loc < cur_batch_seq_len - cur_batch_ctx_len, 1, 0)

        for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            k = tl.load(k_ptrs +
                        (cur_batch_in_all_start_index + start_n) * stride_kbs,
                        mask=(start_n + offs_n[None, :]) <
                        cur_batch_seq_len - cur_batch_ctx_len,
                        other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk *= sm_scale
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
                          float("-inf"))

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)
            m_i_new = tl.maximum(m_i, m_ij)
            p = tl.math.exp(qk - m_i_new[:, None])
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i

            alpha = tl.math.exp(m_i - m_i_new)
            l_i_new = alpha * l_i + l_ij
            # -- update output accumulator --
            # scale p
            # scale acc
            acc_scale = alpha
            # acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc
            v = tl.load(v_ptrs +
                        (cur_batch_in_all_start_index + start_n) * stride_vbs,
                        mask=(start_n + offs_n[:, None]) <
                        cur_batch_seq_len - cur_batch_ctx_len,
                        other=0.0)

            p = p.to(v.dtype)
            acc += tl.dot(p, v)
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new

        # acc /= l_i[:, None]
        # initialize pointers to output
        off_o = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs +
            cur_head * stride_oh + offs_d[None, :] * stride_od)
        out_ptrs = Out + off_o
        tl.store(out_ptrs,
                 acc,
                 mask=offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len)
        return

    @triton.jit
    def _fwd_kernel_alibi(
        Q,
        K,
        V,
        K_cache,
        V_cache,
        B_Loc,
        sm_scale,
        k_scale,
        v_scale,
        B_Start_Loc,
        B_Seqlen,
        B_Ctxlen,
        Alibi_slopes,
        block_size,
        x,
        Out,
        stride_b_loc_b,
        stride_b_loc_s,
        stride_qbs,
        stride_qh,
        stride_qd,
        stride_kbs,
        stride_kh,
        stride_kd,
        stride_vbs,
        stride_vh,
        stride_vd,
        stride_obs,
        stride_oh,
        stride_od,
        stride_k_cache_bs,
        stride_k_cache_h,
        stride_k_cache_d,
        stride_k_cache_bl,
        stride_k_cache_x,
        stride_v_cache_bs,
        stride_v_cache_h,
        stride_v_cache_d,
        stride_v_cache_bl,
        num_queries_per_kv: int,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,  # head size
        BLOCK_DMODEL_PADDED: tl.constexpr,  # head size padded to a power of 2
        BLOCK_N: tl.constexpr,
    ):
        # attn_bias[]
        cur_batch = tl.program_id(0)
        cur_head = tl.program_id(1)
        start_m = tl.program_id(2)

        cur_kv_head = cur_head // num_queries_per_kv

        # cur_batch_seq_len: the length of prompts
        # cur_batch_ctx_len: the length of prefix
        # cur_batch_in_all_start_index: the start id of the dim=0
        cur_batch_ctx_len = tl.load(B_Ctxlen + cur_batch)
        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

        block_start_loc = BLOCK_M * start_m

        # initialize offsets
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_q = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs +
            cur_head * stride_qh + offs_d[None, :] * stride_qd)

        dim_mask = tl.where(
            tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1, 0).to(tl.int1)

        q = tl.load(Q + off_q,
                    mask=dim_mask[None, :] &
                    (offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len),
                    other=0.0)

        # # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)

        alibi_slope = tl.load(Alibi_slopes + cur_head)
        alibi_start_q = tl.arange(
            0, BLOCK_M) + block_start_loc + cur_batch_ctx_len
        alibi_start_k = 0
        for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            bn = tl.load(B_Loc + cur_batch * stride_b_loc_b +
                         ((start_n + offs_n) // block_size) * stride_b_loc_s,
                         mask=(start_n + offs_n) < cur_batch_ctx_len,
                         other=0)
            off_k = (bn[None, :] * stride_k_cache_bs +
                     cur_kv_head * stride_k_cache_h +
                     (offs_d[:, None] // x) * stride_k_cache_d +
                     ((start_n + offs_n[None, :]) % block_size) *
                     stride_k_cache_bl +
                     (offs_d[:, None] % x) * stride_k_cache_x)
            off_v = (
                bn[:, None] * stride_v_cache_bs +
                cur_kv_head * stride_v_cache_h +
                offs_d[None, :] * stride_v_cache_d +
                (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl)
            k_load = tl.load(K_cache + off_k,
                             mask=dim_mask[:, None] &
                             ((start_n + offs_n[None, :]) < cur_batch_ctx_len),
                             other=0.0)  # [D,N]

            if k_load.dtype.is_fp8():
                k = (k_load.to(tl.float32) * k_scale).to(q.dtype)
            else:
                k = k_load

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk = tl.where((start_n + offs_n[None, :]) < cur_batch_ctx_len, qk,
                          float("-inf"))
            qk *= sm_scale

            # load alibi
            alibi = (tl.arange(0, BLOCK_N)[None, :] + alibi_start_k -
                     alibi_start_q[:, None]) * alibi_slope
            alibi = tl.where(
                (alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len),
                alibi, float("-inf"))
            qk += alibi
            alibi_start_k += BLOCK_N

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)
            m_i_new = tl.maximum(m_i, m_ij)
            p = tl.math.exp(qk - m_i_new[:, None])
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i

            alpha = tl.math.exp(m_i - m_i_new)
            l_i_new = alpha * l_i + l_ij
            # -- update output accumulator --
            # scale p
            # scale acc
            acc_scale = alpha
            # acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc
            v_load = tl.load(V_cache + off_v,
                             mask=dim_mask[None, :] &
                             ((start_n + offs_n[:, None]) < cur_batch_ctx_len),
                             other=0.0)
            if v_load.dtype.is_fp8():
                v = (v_load.to(tl.float32) * v_scale).to(q.dtype)
            else:
                v = v_load
            p = p.to(v.dtype)

            acc += tl.dot(p, v, allow_tf32=False)
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new

        off_k = (offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh +
                 offs_d[:, None] * stride_kd)
        off_v = (offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh +
                 offs_d[None, :] * stride_vd)
        k_ptrs = K + off_k
        v_ptrs = V + off_v

        block_mask = tl.where(
            block_start_loc < cur_batch_seq_len - cur_batch_ctx_len, 1, 0)

        # init alibi
        alibi_slope = tl.load(Alibi_slopes + cur_head)
        alibi_start_q = tl.arange(
            0, BLOCK_M) + block_start_loc + cur_batch_ctx_len
        alibi_start_k = cur_batch_ctx_len
        # # init debugger
        # offset_db_q = tl.arange(0, BLOCK_M) + block_start_loc
        # offset_db_k = tl.arange(0, BLOCK_N)
        # calc q[BLOCK_M, BLOCK_MODEL] mul k[prefix_len: , BLOCK_DMODEL]
        for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            k = tl.load(k_ptrs +
                        (cur_batch_in_all_start_index + start_n) * stride_kbs,
                        mask=dim_mask[:, None] &
                        ((start_n + offs_n[None, :]) <
                         cur_batch_seq_len - cur_batch_ctx_len),
                        other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k, allow_tf32=False)
            qk *= sm_scale
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
                          float("-inf"))

            # load alibi
            alibi = (tl.arange(0, BLOCK_N)[None, :] + alibi_start_k -
                     alibi_start_q[:, None]) * alibi_slope
            alibi = tl.where(
                (alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len),
                alibi, float("-inf"))
            qk += alibi
            alibi_start_k += BLOCK_N

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)
            m_i_new = tl.maximum(m_i, m_ij)
            p = tl.math.exp(qk - m_i_new[:, None])
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i

            alpha = tl.math.exp(m_i - m_i_new)
            l_i_new = alpha * l_i + l_ij
            # -- update output accumulator --
            # scale p
            # scale acc
            acc_scale = alpha
            # acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc
            v = tl.load(v_ptrs +
                        (cur_batch_in_all_start_index + start_n) * stride_vbs,
                        mask=dim_mask[None, :] &
                        ((start_n + offs_n[:, None]) <
                         cur_batch_seq_len - cur_batch_ctx_len),
                        other=0.0)
            p = p.to(v.dtype)

            acc += tl.dot(p, v, allow_tf32=False)
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new

        acc = acc / l_i[:, None]

        # initialize pointers to output
        off_o = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs +
            cur_head * stride_oh + offs_d[None, :] * stride_od)
        out_ptrs = Out + off_o
        tl.store(out_ptrs,
                 acc,
                 mask=dim_mask[None, :] &
                 (offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len))
        return

    @torch.inference_mode()
    def context_attention_fwd(q,
                              k,  # hint: unrotated key when use DynamicRAG, otherwise rotated key
                              v,
                              o,
                              kv_cache_dtype: str,
                              k_cache,
                              v_cache,
                              b_loc,
                              b_start_loc,
                              b_seq_len,
                              b_ctx_len,
                              max_input_len,
                              k_scale: float = 1.0,
                              v_scale: float = 1.0,
                              alibi_slopes=None,
                              sliding_window=None,
                              # **************************** drag begins ****************************
                              cos_sin_cache=None,
                              rotary_dim=None,
                              provider='triton', # for benchmark
                              # **************************** drag ends ******************************
                              ):
        BLOCK = 128 if current_platform.has_device_capability(80) else 64
        NUM_WARPS = 8

        # need to reduce num. blocks when using fp32
        # due to increased use of GPU shared memory
        if q.dtype is torch.float32:
            BLOCK = BLOCK // 2

        # Conversion of FP8 Tensor from uint8 storage to
        # appropriate torch.dtype for interpretation by Triton
        if "fp8" in kv_cache_dtype:
            assert (k_cache.dtype == torch.uint8)
            assert (v_cache.dtype == torch.uint8)

            if kv_cache_dtype in ("fp8", "fp8_e4m3"):
                target_dtype = torch.float8_e4m3fn
            elif kv_cache_dtype == "fp8_e5m2":
                target_dtype = torch.float8_e5m2
            else:
                raise ValueError("Unsupported FP8 dtype:", kv_cache_dtype)

            k_cache = k_cache.view(target_dtype)
            v_cache = v_cache.view(target_dtype)

        if (k_cache.dtype == torch.uint8
                or v_cache.dtype == torch.uint8 and kv_cache_dtype == "auto"):
            raise ValueError("kv_cache_dtype='auto' unsupported for\
                FP8 KV Cache prefill kernel")

        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        # round up Lk to a power of 2 - this is required for Triton block size
        Lk_padded = triton.next_power_of_2(Lk)
        sm_scale = 1.0 / (Lq**0.5)
        batch, head = b_seq_len.shape[0], q.shape[1]
        num_queries_per_kv = q.shape[1] // k.shape[1]

        grid = (batch, head, triton.cdiv(max_input_len, BLOCK))  # batch, head,

        # 0 means "disable"
        if sliding_window is None or sliding_window <= 0:
            sliding_window = 0

        if alibi_slopes is not None:
            _fwd_kernel_alibi[grid](
                q,
                k,
                v,
                k_cache,
                v_cache,
                b_loc,
                sm_scale,
                k_scale,
                v_scale,
                b_start_loc,
                b_seq_len,
                b_ctx_len,
                alibi_slopes,
                v_cache.shape[3],
                k_cache.shape[4],
                o,
                b_loc.stride(0),
                b_loc.stride(1),
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                o.stride(0),
                o.stride(1),
                o.stride(2),
                k_cache.stride(0),
                k_cache.stride(1),
                k_cache.stride(2),
                k_cache.stride(3),
                k_cache.stride(
                    4
                ),  #[num_blocks, num_kv_heads, head_size/x, block_size, x]
                v_cache.stride(0),
                v_cache.stride(1),
                v_cache.stride(2),
                v_cache.stride(
                    3),  #[num_blocks, num_kv_heads, head_size, block_size]
                num_queries_per_kv=num_queries_per_kv,
                BLOCK_M=BLOCK,
                BLOCK_DMODEL=Lk,
                BLOCK_DMODEL_PADDED=Lk_padded,
                BLOCK_N=BLOCK,
                num_warps=NUM_WARPS,
                num_stages=1,
            )
            return
        
        # **************************** drag begins ****************************
        # for debug
        # print(f'-------------------------------- prefill, context len is {b_ctx_len}, seq len is {b_seq_len}')
        if (cos_sin_cache is not None) and (rotary_dim is not None) and (provider == 'triton'):
            _fwd_kernel_dynamic_rag_v2[grid](
                q,
                k,
                v,
                k_cache,
                v_cache,
                b_loc,  # block tables
                sm_scale,  # scores = (Q @ K.T) * sm_scale
                k_scale,
                v_scale,
                b_start_loc,  # query start loc
                b_seq_len,  # sequence length
                b_ctx_len,  # context length
                v_cache.shape[3],  # block_size, e.g., 16 from [9596, 8, 128, 16]
                k_cache.shape[4],  # x, e.g., 8 from [9596, 8, 16, 16, 8]
                o,  # output
                b_loc.stride(0),
                b_loc.stride(1),
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                o.stride(0),
                o.stride(1),
                o.stride(2),
                k_cache.stride(0),
                k_cache.stride(1),
                k_cache.stride(2),
                k_cache.stride(3),
                k_cache.stride(
                    4),  #[num_blocks, num_kv_heads, head_size/x, block_size, x]
                v_cache.stride(0),
                v_cache.stride(1),
                v_cache.stride(2),
                v_cache.stride(
                    3),  #[num_blocks, num_kv_heads, head_size, block_size]
                num_queries_per_kv=num_queries_per_kv,  # share kv for each query
                BLOCK_M=BLOCK,
                BLOCK_DMODEL=Lk,
                BLOCK_DMODEL_PADDED=Lk_padded,
                BLOCK_N=BLOCK,
                SLIDING_WINDOW=sliding_window,
                num_warps=NUM_WARPS,
                num_stages=1,
                cos_sin_cache=cos_sin_cache,  # [max_position, rot_dim]
                rotary_dim=rotary_dim,
            )
            return
        # **************************** drag ends ******************************

        # grid: (batch_size, num_head, num_blocks) -> triton instances for parallel execution
        # ceil(max_input_len / BLOCK) = num_blocks
        _fwd_kernel[grid](
            q,
            k,
            v,
            k_cache,
            v_cache,
            b_loc,  # block tables
            sm_scale,  # scores = (Q @ K.T) * sm_scale
            k_scale,
            v_scale,
            b_start_loc,  # query start loc
            b_seq_len,  # sequence length
            b_ctx_len,  # context length
            v_cache.shape[3],  # block_size, e.g., 16 from [9596, 8, 128, 16]
            k_cache.shape[4],  # x, e.g., 8 from [9596, 8, 16, 16, 8]
            o,  # output
            b_loc.stride(0),
            b_loc.stride(1),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            k_cache.stride(
                4),  #[num_blocks, num_kv_heads, head_size/x, block_size, x]
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(
                3),  #[num_blocks, num_kv_heads, head_size, block_size]
            num_queries_per_kv=num_queries_per_kv,  # share kv for each query
            BLOCK_M=BLOCK,
            BLOCK_DMODEL=Lk,
            BLOCK_DMODEL_PADDED=Lk_padded,
            BLOCK_N=BLOCK,
            SLIDING_WINDOW=sliding_window,
            num_warps=NUM_WARPS,
            num_stages=1,
        )
        return
