# The kernels in this file are adapted from LightLLM's context_attention_fwd:
# https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py

# This v1,
# We want to use the bigger gemm rather than the smaller one.
# To see whether this is faster, we need to benchmark it.

# Here the extra cost is only the loading of q twice, and extra load of cos/sin.


import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from vllm.platforms import current_platform

# Static kernels parameters
BASE_BLOCK = 128 if current_platform.has_device_capability(80) else 64
NUM_WARPS = 4 if current_platform.is_rocm() else 8

# To check compatibility
IS_TURING = current_platform.get_device_capability() == (7, 5)


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

    inv_freqs = 1.0 / libdevice.pow(BASE, 2 * (tl.arange(0, HEAD_SIZE) % half_head_size) / HEAD_SIZE) # repeat twice
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
    cos_val = libdevice.cos(theta).reshape(128, 1)
    sin_val = libdevice.sin(theta).reshape(128, 1)
    return cos_val, sin_val

@triton.jit
def _fwd_kernel(Q,
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
                x: tl.constexpr,
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
                stride_k_cache_bl: tl.constexpr,
                stride_k_cache_x,
                stride_v_cache_bs,
                stride_v_cache_h,
                stride_v_cache_d,
                stride_v_cache_bl,
                num_queries_per_kv: tl.constexpr,
                IN_PRECISION: tl.constexpr,
                BLOCK_M: tl.constexpr,
                BLOCK_DMODEL: tl.constexpr,
                BLOCK_DMODEL_PADDED: tl.constexpr,
                BLOCK_SIZE: tl.constexpr,
                BLOCK_N: tl.constexpr,
                SLIDING_WINDOW: tl.constexpr,
                num_unroll_cache: tl.constexpr,
                num_unroll_request: tl.constexpr,
                SKIP_DECODE: tl.constexpr,
                # /////////////////
                cos_sin_cache,
                rotary_dim: tl.constexpr,
                rotary_dim_pow2: tl.constexpr,
                is_neox_style: tl.constexpr,
                # ///////////////
                is_lazy_ptr,
                q_offset_ptr,
                q_mask_ptr,
                # /////////////////
                MAX_Q_LEN: tl.constexpr = 0,
                MAX_CTX_LEN: tl.constexpr = 0
):
    cur_batch = tl.program_id(0)
    
    # Get the condition as early as possible
    is_lazy = tl.load(is_lazy_ptr + cur_batch)

# ////////////////////////////////////////////////////////
    if is_lazy:
        cur_head = tl.program_id(1)
        start_m = tl.program_id(2)

        cur_kv_head = cur_head // num_queries_per_kv

        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
        cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)
        cur_batch_query_len = (cur_batch_in_all_stop_index -
                            cur_batch_in_all_start_index)
        cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len

        if SKIP_DECODE and cur_batch_query_len == 1:
            return

        # start position inside of the query
        # generally, N goes over kv, while M goes over query_len
        block_start_loc = BLOCK_M * start_m

        # initialize offsets
        # [BLOCK_SIZE]; starts at 0
        offs_bs_n = tl.arange(0, BLOCK_SIZE)
        # [N]; starts at 0
        offs_n = tl.arange(0, BLOCK_N)
        # [D]; starts at 0 ~ D is 128 for tulu3
        offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
        offs_d_rev = (tl.arange(0, BLOCK_DMODEL_PADDED) + 
               (BLOCK_DMODEL_PADDED // 2)) % BLOCK_DMODEL_PADDED
        # [M]; starts at current position in query
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # [M,D]
        off_q = ((cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs +
                 cur_head * stride_qh + offs_d[None, :] * stride_qd)
        # off_q_rev = ((cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs +
        #              cur_head * stride_qh + offs_d_rev[None, :] * stride_qd)

        dim_mask = tl.where(
            tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1,
            0).to(tl.int1)  # [D]

        q = tl.load(Q + off_q,
                    mask=dim_mask[None, :] &
                    (offs_m[:, None] < cur_batch_query_len),
                    other=0.0)  # [M,D]
        # q_rev = tl.load(Q + off_q_rev,
        #             mask=dim_mask[None, :] &
        #             (offs_m[:, None] < cur_batch_query_len),
        #             other=0.0)  # [M,D]
        q_rotated = q # Use q as the default value for the rotated q
        embed_dim: tl.constexpr = BLOCK_DMODEL // 2

        # initialize pointer to m and l
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)  # [M,D]

        # compute query against context (no causal mask here)
        for start_n in tl.range(0, cur_batch_ctx_len, BLOCK_SIZE, \
                                loop_unroll_factor=num_unroll_cache):
            start_n = tl.multiple_of(start_n, BLOCK_SIZE)
            # -- compute qk ----
            bn = tl.load(B_Loc + cur_batch * stride_b_loc_b +
                        (start_n // BLOCK_SIZE) * stride_b_loc_s)
            
            # Prepare the lazy metadata
            # First we load the q_offset, typically it is an array like [-3, 0, -30, 0, -1, 0, ]
            # Here if one value is 0, it means no update is needed for this KV block, just use the q from previous steps
            rot_offset_val = tl.load(
                q_offset_ptr + cur_batch * stride_b_loc_b +
                (start_n // BLOCK_SIZE) * stride_b_loc_s)
            q_mask_val = tl.load(q_mask_ptr + cur_batch * stride_b_loc_b +
                (start_n // BLOCK_SIZE) * stride_b_loc_s)
            
            # Load KV as the full blocks
            # [D,BLOCK_SIZE]
            off_k = (
                bn[None, :] * stride_k_cache_bs + cur_kv_head * stride_k_cache_h +
                (offs_d[:, None] // x) * stride_k_cache_d +
                ((start_n + offs_bs_n[None, :]) % BLOCK_SIZE) * stride_k_cache_bl +
                (offs_d[:, None] % x) * stride_k_cache_x)

            off_k_rev = (
                bn[None, :] * stride_k_cache_bs + cur_kv_head * stride_k_cache_h +
                (offs_d_rev[:, None] // x) * stride_k_cache_d +
                ((start_n + offs_bs_n[None, :]) % BLOCK_SIZE) * stride_k_cache_bl +
                (offs_d_rev[:, None] % x) * stride_k_cache_x)

            # [BLOCK_SIZE,D]
            off_v = (bn[:, None] * stride_v_cache_bs +
                    cur_kv_head * stride_v_cache_h +
                    offs_d[None, :] * stride_v_cache_d +
                    offs_bs_n[:, None] * stride_v_cache_bl)

            if start_n + BLOCK_SIZE > cur_batch_ctx_len or \
                BLOCK_DMODEL != BLOCK_DMODEL_PADDED:
                k_load = tl.load(
                    K_cache + off_k,
                    mask=dim_mask[:, None] &
                    ((start_n + offs_bs_n[None, :]) < cur_batch_ctx_len),
                    other=0.0)  # [D,N]
                k_load_rev = tl.load(
                    K_cache + off_k_rev,
                    mask=dim_mask[:, None] &
                    ((start_n + offs_bs_n[None, :]) < cur_batch_ctx_len),
                    other=0.0)  # [D,N]
            else:
                k_load = tl.load(K_cache + off_k)
                k_load_rev = tl.load(K_cache + off_k_rev)

            if k_load.dtype.is_fp8():
                k = (k_load.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
                k_rev = (k_load_rev.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
            else:
                k = k_load
                k_rev = k_load_rev
            
            k_rotated = k
            # Then we always rotate key
            abs_rot_pos = 0
            if rot_offset_val != 0:
                abs_rot_pos = -(rot_offset_val + 1) # this means we need to rotate the q back by abs_rot_pos positions
            # positions = tl.full([1], abs_rot_pos, dtype=tl.int32)
            
            # off_cos = ((positions[None, :]) * rotary_dim +
            #                 (offs_d[:, None] % (BLOCK_DMODEL_PADDED // 2))) # repeat twice as well
            # cos_val = tl.load(cos_sin_cache + off_cos)
            # sin_val = tl.load(cos_sin_cache + off_cos + embed_dim)
            cos_val, sin_val = llama_cos_sin(rot_offset_val + 1,
                                                 HEAD_SIZE=128,
                                                 ORIG_MAX_POSITION=8192,
                                                 LOW_FACTOR=1.0,
                                                 HIGH_FACTOR=4.0,
                                                 SCALING_FACTOR=8.0,
                                                 PI_VALUE=3.141592653589793,
                                                 BASE=500000.0)
                
            mask_k1 = offs_d < BLOCK_DMODEL_PADDED // 2
            mask_k2 = ~mask_k1
            k1 = tl.where(mask_k1[:, None], k, k_rev) # k1 repeats twice
            k2 = tl.where(mask_k2[:, None], k, k_rev) # k2 repeats twice

                # Then we get the rotated q, by default we since the rotation direction is always -1
            k1_new = k1 * cos_val - k2 * sin_val
            k2_new = k1 * sin_val + k2 * cos_val
            
            k_rotated = tl.where(mask_k1[:, None], 
                k1_new.to(k_rotated.dtype), 
                k2_new.to(k_rotated.dtype))
                
            qk = tl.zeros([BLOCK_M, BLOCK_SIZE], dtype=tl.float32)  # [M,N]
            qk = tl.dot(q, k_rotated, acc=qk, input_precision=IN_PRECISION)
            seq_bound = min(cur_batch_ctx_len, start_n + BLOCK_SIZE - q_mask_val)            
            qk = tl.where((start_n + offs_bs_n[None, :]) < seq_bound, qk,
                            float("-inf"))
            qk *= sm_scale
            
            if SLIDING_WINDOW > 0:
                # (cur_batch_ctx_len + offs_m[:, None]) are the positions of
                # Q entries in sequence
                # (start_n + offs_bs_n[None, :]) are the positions of
                # KV entries in sequence
                # So the condition makes sure each entry in Q only attends
                # to KV entries not more than SLIDING_WINDOW away.
                #
                # We can't use -inf here, because the
                # sliding window may lead to the entire row being masked.
                # This then makes m_ij contain -inf, which causes NaNs in
                # exp().
                qk = tl.where((cur_batch_ctx_len + offs_m[:, None]) -
                            (start_n + offs_bs_n[None, :]) < SLIDING_WINDOW, qk,
                            -10000)

            # compute running maximum
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            alpha = tl.exp(m_i - m_ij)
            acc = acc * alpha[:, None]

            # update acc
            if start_n + BLOCK_SIZE > cur_batch_ctx_len or \
                BLOCK_DMODEL != BLOCK_DMODEL_PADDED:
                v_load = tl.load(
                    V_cache + off_v,
                    mask=dim_mask[None, :] &
                    ((start_n + offs_bs_n[:, None]) < cur_batch_ctx_len),
                    other=0.0)  # [N,D]
            else:
                v_load = tl.load(V_cache + off_v)

            if v_load.dtype.is_fp8():
                v = (v_load.to(tl.float32) * tl.load(v_scale)).to(q.dtype)
            else:
                v = v_load
            p = p.to(v.dtype)

            acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
            # # update m_i and l_i
            l_i = l_i * alpha + l_ij
            m_i = m_ij

        off_k = (offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh +
                offs_d[:, None] * stride_kd)
        off_v = (offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh +
                offs_d[None, :] * stride_vd)
        k_ptrs = K + off_k
        v_ptrs = V + off_v

        # block_mask is 0 when we're already past the current query length
        block_mask = tl.where(block_start_loc < cur_batch_query_len, 1, 0)

        # compute query against itself (with causal mask)
        for start_n in tl.range(0, \
                            block_mask * (start_m + 1) * BLOCK_M, BLOCK_N, \
                            loop_unroll_factor=num_unroll_request):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            k = tl.load(k_ptrs +
                        (cur_batch_in_all_start_index + start_n) * stride_kbs,
                        mask=dim_mask[:, None] &
                        ((start_n + offs_n[None, :]) < cur_batch_query_len),
                        other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
            qk *= sm_scale
            # apply causal mask
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
                        float("-inf"))
            if SLIDING_WINDOW > 0:
                qk = tl.where(
                    offs_m[:, None] - (start_n + offs_n[None, :]) < SLIDING_WINDOW,
                    qk, -10000)

            # compute running maximum
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            alpha = tl.exp(m_i - m_ij)
            acc = acc * alpha[:, None]

            # update acc
            v = tl.load(v_ptrs +
                        (cur_batch_in_all_start_index + start_n) * stride_vbs,
                        mask=dim_mask[None, :] &
                        ((start_n + offs_n[:, None]) < cur_batch_query_len),
                        other=0.0)
            p = p.to(v.dtype)

            acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
            # update m_i and l_i
            l_i = l_i * alpha + l_ij
            m_i = m_ij

        acc = acc / l_i[:, None]

        # initialize pointers to output
        off_o = ((cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs +
                cur_head * stride_oh + offs_d[None, :] * stride_od)
        out_ptrs = Out + off_o
        tl.store(out_ptrs,
                acc,
                mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len))
        return
# /////////////////////////////////////////////////////////////////////////////////
    else:
        cur_head = tl.program_id(1)
        start_m = tl.program_id(2)

        cur_kv_head = cur_head // num_queries_per_kv

        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
        cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)
        cur_batch_query_len = (cur_batch_in_all_stop_index -
                            cur_batch_in_all_start_index)
        cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len

        if SKIP_DECODE and cur_batch_query_len == 1:
            return

        # start position inside of the query
        # generally, N goes over kv, while M goes over query_len
        block_start_loc = BLOCK_M * start_m

        # initialize offsets
        # [BLOCK_SIZE]; starts at 0
        offs_bs_n = tl.arange(0, BLOCK_SIZE)
        # [N]; starts at 0
        offs_n = tl.arange(0, BLOCK_N)
        # [D]; starts at 0
        offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
        # [M]; starts at current position in query
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # [M,D]
        off_q = ((cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs +
                cur_head * stride_qh + offs_d[None, :] * stride_qd)

        dim_mask = tl.where(
            tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1,
            0).to(tl.int1)  # [D]

        q = tl.load(Q + off_q,
                    mask=dim_mask[None, :] &
                    (offs_m[:, None] < cur_batch_query_len),
                    other=0.0)  # [M,D]

        # initialize pointer to m and l
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)  # [M,D]

        # compute query against context (no causal mask here)
        for start_n in tl.range(0, cur_batch_ctx_len, BLOCK_SIZE, \
                                loop_unroll_factor=num_unroll_cache):
            start_n = tl.multiple_of(start_n, BLOCK_SIZE)
            # -- compute qk ----
            bn = tl.load(B_Loc + cur_batch * stride_b_loc_b +
                        (start_n // BLOCK_SIZE) * stride_b_loc_s)
            # [D,BLOCK_SIZE]
            off_k = (
                bn[None, :] * stride_k_cache_bs + cur_kv_head * stride_k_cache_h +
                (offs_d[:, None] // x) * stride_k_cache_d +
                ((start_n + offs_bs_n[None, :]) % BLOCK_SIZE) * stride_k_cache_bl +
                (offs_d[:, None] % x) * stride_k_cache_x)

            # [BLOCK_SIZE,D]
            off_v = (bn[:, None] * stride_v_cache_bs +
                    cur_kv_head * stride_v_cache_h +
                    offs_d[None, :] * stride_v_cache_d +
                    offs_bs_n[:, None] * stride_v_cache_bl)

            if start_n + BLOCK_SIZE > cur_batch_ctx_len or \
                BLOCK_DMODEL != BLOCK_DMODEL_PADDED:
                k_load = tl.load(
                    K_cache + off_k,
                    mask=dim_mask[:, None] &
                    ((start_n + offs_bs_n[None, :]) < cur_batch_ctx_len),
                    other=0.0)  # [D,N]
            else:
                k_load = tl.load(K_cache + off_k)

            if k_load.dtype.is_fp8():
                k = (k_load.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
            else:
                k = k_load

            qk = tl.zeros([BLOCK_M, BLOCK_SIZE], dtype=tl.float32)  # [M,N]
            qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
            qk = tl.where((start_n + offs_bs_n[None, :]) < cur_batch_ctx_len, qk,
                        float("-inf"))
            qk *= sm_scale
            if SLIDING_WINDOW > 0:
                # (cur_batch_ctx_len + offs_m[:, None]) are the positions of
                # Q entries in sequence
                # (start_n + offs_bs_n[None, :]) are the positions of
                # KV entries in sequence
                # So the condition makes sure each entry in Q only attends
                # to KV entries not more than SLIDING_WINDOW away.
                #
                # We can't use -inf here, because the
                # sliding window may lead to the entire row being masked.
                # This then makes m_ij contain -inf, which causes NaNs in
                # exp().
                qk = tl.where((cur_batch_ctx_len + offs_m[:, None]) -
                            (start_n + offs_bs_n[None, :]) < SLIDING_WINDOW, qk,
                            -10000)

            # compute running maximum
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            alpha = tl.exp(m_i - m_ij)
            acc = acc * alpha[:, None]

            # update acc
            if start_n + BLOCK_SIZE > cur_batch_ctx_len or \
                BLOCK_DMODEL != BLOCK_DMODEL_PADDED:
                v_load = tl.load(
                    V_cache + off_v,
                    mask=dim_mask[None, :] &
                    ((start_n + offs_bs_n[:, None]) < cur_batch_ctx_len),
                    other=0.0)  # [N,D]
            else:
                v_load = tl.load(V_cache + off_v)

            if v_load.dtype.is_fp8():
                v = (v_load.to(tl.float32) * tl.load(v_scale)).to(q.dtype)
            else:
                v = v_load
            p = p.to(v.dtype)

            acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
            # # update m_i and l_i
            l_i = l_i * alpha + l_ij
            m_i = m_ij

        off_k = (offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh +
                offs_d[:, None] * stride_kd)
        off_v = (offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh +
                offs_d[None, :] * stride_vd)
        k_ptrs = K + off_k
        v_ptrs = V + off_v

        # block_mask is 0 when we're already past the current query length
        block_mask = tl.where(block_start_loc < cur_batch_query_len, 1, 0)

        # compute query against itself (with causal mask)
        for start_n in tl.range(0, \
                            block_mask * (start_m + 1) * BLOCK_M, BLOCK_N, \
                            loop_unroll_factor=num_unroll_request):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            k = tl.load(k_ptrs +
                        (cur_batch_in_all_start_index + start_n) * stride_kbs,
                        mask=dim_mask[:, None] &
                        ((start_n + offs_n[None, :]) < cur_batch_query_len),
                        other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
            qk *= sm_scale
            # apply causal mask
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
                        float("-inf"))
            if SLIDING_WINDOW > 0:
                qk = tl.where(
                    offs_m[:, None] - (start_n + offs_n[None, :]) < SLIDING_WINDOW,
                    qk, -10000)

            # compute running maximum
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            alpha = tl.exp(m_i - m_ij)
            acc = acc * alpha[:, None]

            # update acc
            v = tl.load(v_ptrs +
                        (cur_batch_in_all_start_index + start_n) * stride_vbs,
                        mask=dim_mask[None, :] &
                        ((start_n + offs_n[:, None]) < cur_batch_query_len),
                        other=0.0)
            p = p.to(v.dtype)

            acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
            # update m_i and l_i
            l_i = l_i * alpha + l_ij
            m_i = m_ij

        acc = acc / l_i[:, None]

        # initialize pointers to output
        off_o = ((cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs +
                cur_head * stride_oh + offs_d[None, :] * stride_od)
        out_ptrs = Out + off_o
        tl.store(out_ptrs,
                acc,
                mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len))
        return


# ////////////////////////////////////////////////////////

    

@torch.inference_mode()
def context_attention_fwd(q,
                          k,
                          v,
                          o,
                          kv_cache_dtype: str,
                          k_cache,
                          v_cache,
                          b_loc,
                          b_start_loc,
                          b_seq_len,
                          max_seq_len,
                          max_input_len,
                          k_scale: torch.Tensor,
                          v_scale: torch.Tensor,
                          alibi_slopes=None,
                          sliding_window=None,
                          sm_scale=None,
                          skip_decode=False,
                          cos_sin_cache=None,
                          rotary_dim=None,
                          is_neox_style=True,
                          # ///////
                          is_lazy=None,
                          q_offset=None,
                          q_mask=None,
                          ):

    q_dtype_is_f32 = q.dtype is torch.float32

    # Turing does have tensor core for float32 multiplication
    # use ieee as fallback for triton kernels work. There is also
    # warning on vllm/config.py to inform users this fallback
    # implementation
    IN_PRECISION = 'ieee' if IS_TURING and q_dtype_is_f32 else None

    # Conversion of FP8 Tensor from uint8 storage to
    # appropriate torch.dtype for interpretation by Triton
    if "fp8" in kv_cache_dtype:
        assert (k_cache.dtype == torch.uint8)
        assert (v_cache.dtype == torch.uint8)

        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            target_dtype = current_platform.fp8_dtype()
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

    if sm_scale is None:
        sm_scale = 1.0 / (Lq**0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]
    num_queries_per_kv = q.shape[1] // k.shape[1]

    assert batch + 1 == len(b_start_loc)

    # 0 means "disable"
    if sliding_window is None or sliding_window <= 0:
        sliding_window = 0

    if alibi_slopes is not None:
        raise NotImplementedError("Alibi slopes are not implemented for prefix prefill")

    max_seq_len = 0 if max_seq_len is None else max_seq_len
    extra_kargs = {}
    if current_platform.is_rocm():
        extra_kargs = {"kpack": 2, "waves_per_eu": 2}

    grid = lambda META: (batch, head,
                         triton.cdiv(max_input_len, META["BLOCK_M"]))
    _fwd_kernel[grid](
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
            4),  #[num_blocks, num_kv_heads, head_size/x, block_size, x]
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),  #[num_blocks, num_kv_heads, head_size, block_size]
        BLOCK_SIZE=v_cache.shape[3],
        num_queries_per_kv=num_queries_per_kv,
        IN_PRECISION=IN_PRECISION,
        BLOCK_DMODEL=Lk,
        BLOCK_DMODEL_PADDED=Lk_padded,
        SLIDING_WINDOW=sliding_window,
        SKIP_DECODE=skip_decode,
        # /////////////////
        cos_sin_cache=cos_sin_cache,  # [max_position, rot_dim]
        rotary_dim=rotary_dim,
        rotary_dim_pow2=triton.next_power_of_2(rotary_dim),
        is_neox_style=is_neox_style,
        is_lazy_ptr=is_lazy,
        q_offset_ptr=q_offset,
        q_mask_ptr=q_mask,
        # /////////////////
        BLOCK_M=128,
        BLOCK_N=64,
        num_unroll_cache=4,
        num_unroll_request=1,
        num_warps=4,
        num_stages=1,
        **extra_kargs)    
    return