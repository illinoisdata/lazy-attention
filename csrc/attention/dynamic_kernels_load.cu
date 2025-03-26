// a impl fused the rotary embedding and dot product and change the load method (best performance)

#include <cuda_bf16.h>

template <typename scalar_t, int VEC_SIZE>
inline __device__ void apply_neox_rotary_embedding(
    scalar_t* __restrict__ arr1, scalar_t* __restrict__ arr2,
    const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr) {
  scalar_t x;
  scalar_t y;

#pragma unroll
  for (int i = 0; i < VEC_SIZE; i++){
    x = arr1[i];
    y = arr2[i];
    arr1[i] = x * cos_ptr[i] - y * sin_ptr[i];
    arr2[i] = y * cos_ptr[i] + x * sin_ptr[i];
  }
}

//////////////////////
template <typename T>
__device__ __forceinline__ T negate(const T& a) {
    return -a;
}

// 特化版本 - 对于 float2 类型
template <>
__device__ __forceinline__ float2 negate<float2>(const float2& a) {
    return make_float2(-a.x, -a.y);
}

// 特化版本 - 对于 float4 类型
template <>
__device__ __forceinline__ float4 negate<float4>(const float4& a) {
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}

// 特化版本 - 对于 uint2 类型
template <>
__device__ __forceinline__ uint2 negate<uint2>(const uint2& a) {
    // 注意：对无符号整数取负会导致溢出，这里我们假设这是预期行为
    return make_uint2(-a.x, -a.y);
}

// 特化版本 - 对于 uint4 类型
template <>
__device__ __forceinline__ uint4 negate<uint4>(const uint4& a) {
    // 注意：对无符号整数取负会导致溢出，这里我们假设这是预期行为
    return make_uint4(-a.x, -a.y, -a.z, -a.w);
}

// 为 __nv_bfloat16 类型特化
template <>
__device__ __forceinline__ __nv_bfloat16 negate<__nv_bfloat16>(const __nv_bfloat16& a) {
    return __hneg(a);
}

// 为 __nv_bfloat162 类型特化
template <>
__device__ __forceinline__ __nv_bfloat162 negate<__nv_bfloat162>(const __nv_bfloat162& a) {
    return __hneg2(a);
}

// 为 vllm::bf16_4_t 类型特化
template <>
__device__ __forceinline__ vllm::bf16_4_t negate<vllm::bf16_4_t>(const vllm::bf16_4_t& a) {
    vllm::bf16_4_t result;
    
    // 假设 bf16_4_t 可以被视为 __nv_bfloat162 数组
    const __nv_bfloat162* a_bf162 = reinterpret_cast<const __nv_bfloat162*>(&a);
    __nv_bfloat162* result_bf162 = reinterpret_cast<__nv_bfloat162*>(&result);
    
    result_bf162[0] = __hneg2(a_bf162[0]);
    result_bf162[1] = __hneg2(a_bf162[1]);
    
    return result;
}

// 为 vllm::bf16_8_t 类型特化
template <>
__device__ __forceinline__ vllm::bf16_8_t negate<vllm::bf16_8_t>(const vllm::bf16_8_t& a) {
    vllm::bf16_8_t result;
    
    // 假设 bf16_8_t 可以被视为 __nv_bfloat162 数组
    const __nv_bfloat162* a_bf162 = reinterpret_cast<const __nv_bfloat162*>(&a);
    __nv_bfloat162* result_bf162 = reinterpret_cast<__nv_bfloat162*>(&result);
    
    result_bf162[0] = __hneg2(a_bf162[0]);
    result_bf162[1] = __hneg2(a_bf162[1]);
    result_bf162[2] = __hneg2(a_bf162[2]);
    result_bf162[3] = __hneg2(a_bf162[3]);
    
    return result;
}

//////////////////////

template <typename scalar_t, int THREAD_GROUP_SIZE, typename Vec, int N>
inline __device__ float apply_neox_rotary_embedding_and_dot(
    const Vec (&q)[N],
    const Vec (&k)[N],
    const Vec (&cos_sin)[N]
) {
    using A_vec = typename FloatVec<Vec>::Type;
    // Compute the parallel products for Q*K^T (treat vector lanes separately).

    Vec k_front = mul<Vec, Vec, Vec>(k[0], cos_sin[0]);
    k_front = vllm::fma(k[0+N/2], negate(cos_sin[0+N/2]), k_front);

    Vec k_back = mul<Vec, Vec, Vec>(k[0+N/2], cos_sin[0]);
    k_back = vllm::fma(k[0], cos_sin[0+N/2], k_back);

    A_vec qk_vec = mul<A_vec, Vec, Vec>(q[0], k_front);
    qk_vec = vllm::fma(q[0+N/2], k_back, qk_vec);

    #pragma unroll
  for (int ii = 1; ii < N / 2; ++ii) {
    Vec k_front_ii = mul<Vec, Vec, Vec>(k[ii], cos_sin[ii]);
    k_front_ii = vllm::fma(k[ii+N/2], negate(cos_sin[ii+N/2]), k_front_ii);

    Vec k_back_ii = mul<Vec, Vec, Vec>(k[ii+N/2], cos_sin[ii]);
    k_back_ii = vllm::fma(k[ii], cos_sin[ii+N/2], k_back_ii);

    qk_vec = vllm::fma(q[ii], k_front_ii, qk_vec);
    qk_vec = vllm::fma(q[ii+N/2], k_back_ii, qk_vec);
  }
  float qk = sum(qk_vec);
#pragma unroll
  for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
    qk += VLLM_SHFL_XOR_SYNC(qk, mask);
  }
  return qk;
}


// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE, bool IS_NEOX,
          int PARTITION_SIZE = 0>  // Zero means no partitioning.
__device__ void dynamic_paged_attention_kernel(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                                // max_num_partitions]
    scalar_t* __restrict__ out,  // [num_seqs, num_heads, max_num_partitions,
                                                // head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                                // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                                // head_size, block_size]
    const cache_t* __restrict__ cos_sin_cache,  // [max_position, rot_dim]
                                                // e.g.
                                                //[
                                                // [cos(1 * theta1), cos(1 * theta2), cos(1 * theta3), ... cos(1 * theta(d/2)), sin(1 * theta1), sin(1 * theta2), sin(1 * theta3), ... sin(1 * theta(d/2))]
                                                // ......
                                                // [cos(max_pos * theta1), cos(max_pos * theta2), cos(max_pos * theta3), ... cos(max_pos * theta(d/2)), sin(max_pos * theta1), sin(max_pos * theta2), sin(max_pos * theta3), ... sin(max_pos * theta(d/2))]
                                                // ]
    const int rot_dim,                    // dimensions used for rotary embedding, for llama, all dimensions will be used.
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float k_scale, const float v_scale, const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step) {
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int max_num_partitions = gridDim.z;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const int seq_len = seq_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= seq_len) {
    // No work to do. Terminate the thread block.
    return;
  }

  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int num_blocks_per_partition =
      USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_seq_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx =
      USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx =
      MIN(start_block_idx + num_blocks_per_partition, num_seq_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx =
      MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS =
      NUM_THREADS / THREAD_GROUP_SIZE;  // Note: This assumes THREAD_GROUP_SIZE
                                        // divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);

  constexpr int NUM_TOKENS_PER_THREAD_GROUP =
      DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / num_kv_heads; // In some algorithm, query head(num_head) maybe more than key-value head
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope =
      alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread
  // group fetch or compute 16 bytes at a time. For example, if the size of a
  // thread group is 4 and the data type is half, then the vector size is 16
  // constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 2); // change vec size to 2 for RoPE

  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Quant_vec = typename Vec<cache_t, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE; //global index
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE; // this thread's offset  in the thread group

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in
  // the group has 0, 4, 8, ... th vectors of the query, and the second thread
  // has 1, 5, 9, ... th vectors of the query, and so on. NOTE(woosuk): Because
  // q is split from a qkv tensor, it may not be contiguous.
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
       i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads();  // TODO(naed90): possible speedup if this is replaced with a
                    // memory wall right before we use q_vecs

  // Memory planning.
  // The extern keyword indicates that the size of shared_mem is not specified at compile time but rather defined at kernel launch.
  // For example, if you call the kernel with <<<numBlocks, numThreads, sharedMemSize>>>, the sharedMemSize parameter specifies the number of bytes allocated to shared_mem.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(cache_t);
  float qk_max = -FLT_MAX;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  // blocksparse specific vars
  int bs_block_offset;
  int q_bs_block_id;
  if constexpr (IS_BLOCK_SPARSE) {
    // const int num_blocksparse_blocks = DIVIDE_ROUND_UP(seq_len,
    // blocksparse_block_size);
    q_bs_block_id = (seq_len - 1) / blocksparse_block_size;
    if (blocksparse_head_sliding_step >= 0)
      // sliding on q heads
      bs_block_offset =
          (tp_rank * num_heads + head_idx) * blocksparse_head_sliding_step + 1;
    else
      // sliding on kv heads
      bs_block_offset = (tp_rank * num_kv_heads + kv_head_idx) *
                            (-blocksparse_head_sliding_step) +
                        1;
  }

///////////////////////////////////////////////////////////////////////////////////////////////
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    //loop 1: locate the block this warp is dealing with

    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    // For blocksparse attention: skip computation on blocks that are not
    // attended
    if constexpr (IS_BLOCK_SPARSE) {
      const int k_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
      const bool is_remote =
          ((k_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0);
      const bool is_local =
          (k_bs_block_id > q_bs_block_id - blocksparse_local_blocks);
      if (!is_remote && !is_local) {
        for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
          const int physical_block_offset =
              (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
          const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;

          if (thread_group_offset == 0) {
            // NOTE(linxihui): assign very large number to skipped tokens to
            // avoid contribution to the sumexp softmax normalizer. This will
            // not be used at computing sum(softmax*v) as the blocks will be
            // skipped.
            logits[token_idx - start_token_idx] = -FLT_MAX;
          }
        }
        continue;
      }
    }

    // TODO(haocheng): we can support token skip here
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the the thread group size is 4, then the first thread in
    // the group has 0, 4, 8, ... th vectors of the key, and the second thread
    // has 1, 5, 9, ... th vectors of the key, and so on.
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      //loop 2: locate the token this thread group is dealing with

      //physical_block_offset: the token's offset within the block
      const int physical_block_offset =
          (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE; //% to make sure won't exceed the block size
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset; // the token's index in the whole sequence
      K_vec k_vecs[NUM_VECS_PER_THREAD]; // k_vecs for this thread to deal with.
      K_vec cos_sin_vecs[NUM_VECS_PER_THREAD];

      const cache_t* cache_ptr = cos_sin_cache + token_idx * rot_dim;
      const int embed_dim = rot_dim / 2;
      const cache_t* cos_ptr = cache_ptr;
      const cache_t* sin_ptr = cache_ptr + embed_dim;

      // k: num_blocks, num_kv_heads, head_size/x, block_size, x]
      const cache_t* k_ptr =
        k_cache + physical_block_number * kv_block_stride +
        kv_head_idx * kv_head_stride + physical_block_offset * x;
///////////////////////////////////////////////////////////////////////////////////////////////
/* deal with one token by this thread group
 */
      constexpr int HALF_NUM_VECS_PER_HEAD = HEAD_SIZE / VEC_SIZE / 2;
      // constexpr int BYTES_PER_VEC = VEC_SIZE * sizeof(scalar_t);
      if (IS_NEOX) {
      #pragma unroll
        for (int j = 0; j < NUM_VECS_PER_THREAD / 2; j++) { // load 2 vec for each thread, one for front half, one for back half
              //loop 3: locate the vecs this thread group is dealing with.
              //loop for each thread to load the vec one by one
              const int vec_idx_front = thread_group_offset + j * THREAD_GROUP_SIZE;
              const int vec_idx_back = vec_idx_front + HALF_NUM_VECS_PER_HEAD;
              // [num_blocks, num_kv_heads, head_size/x, block_size, x]
              //                            |- offest1 
              //                                                     |- offset2
              const int offset1_front = (vec_idx_front * VEC_SIZE) / x;
              const int offset2_front = (vec_idx_front * VEC_SIZE) % x;

              const int offset1_back = (vec_idx_back * VEC_SIZE) / x;
              const int offset2_back = (vec_idx_back * VEC_SIZE) % x;

              // scalar_t* current_k_vec_front = reinterpret_cast<scalar_t*>(&k_vecs[j]);
              // scalar_t* current_k_vec_back = reinterpret_cast<scalar_t*>(&k_vecs[j + NUM_VECS_PER_THREAD / 2]);
              
              // scalar_t* current_q_vec_front = reinterpret_cast<scalar_t*>(&q_vecs[thread_group_offset][j]);
              // scalar_t* current_q_vec_back = reinterpret_cast<scalar_t*>(&q_vecs[thread_group_offset][j + NUM_VECS_PER_THREAD / 2]);
              
              // scalar_t* current_sin_vec = reinterpret_cast<scalar_t*>(&sin_vecs[j]);
              // scalar_t* current_cos_vec = reinterpret_cast<scalar_t*>(&cos_vecs[j]);

              if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
                k_vecs[j] = *reinterpret_cast<const K_vec*>(
                  k_ptr + offset1_front * BLOCK_SIZE * x + offset2_front);
                k_vecs[j + NUM_VECS_PER_THREAD / 2] = *reinterpret_cast<const K_vec*>(
                  k_ptr + offset1_back * BLOCK_SIZE * x + offset2_back);
                
                cos_sin_vecs[j] = *reinterpret_cast<const K_vec*>(
                  cos_ptr + vec_idx_front * VEC_SIZE);
                cos_sin_vecs[j + NUM_VECS_PER_THREAD / 2] = *reinterpret_cast<const K_vec*>(
                  sin_ptr + vec_idx_front * VEC_SIZE);
              } else {
                // TODO(haocheng): change this to fp8
                // Vector conversion from Quant_vec to K_vec.
                assert(false);
                Quant_vec k_vec_quant = *reinterpret_cast<const Quant_vec*>(
                    k_ptr + offset1_back * BLOCK_SIZE * x + offset2_back);
                k_vecs[j] = fp8::scaled_convert<K_vec, Quant_vec, KV_DTYPE>(
                    k_vec_quant, k_scale);
              }
        } 
      } // IS_NEOX
      else {
        // TODO(haocheng): implement gpt-j style rotation
        assert(false);
      } // IS_GPTJ

      // Compute rotation anddot product.
      // This includes a reduction across the threads in the same thread group.
      float qk = scale * 
      apply_neox_rotary_embedding_and_dot<scalar_t, THREAD_GROUP_SIZE>(
          q_vecs[thread_group_offset], k_vecs,
          cos_sin_vecs);
      // float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(
      //                        q_vecs[thread_group_offset], k_vecs);
      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - seq_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= seq_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }
///////////////////////////////////////////////////////////////////////////////////////////////

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = VLLM_SHFL_SYNC(qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0) {
    float* max_logits_ptr = max_logits +
                            seq_idx * num_heads * max_num_partitions +
                            head_idx * max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions +
                          head_idx * max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using V_quant_vec = typename Vec<cache_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD =
      DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  scalar_t zero_value;
  zero(zero_value);
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    // For blocksparse attention: skip computation on blocks that are not
    // attended
    if constexpr (IS_BLOCK_SPARSE) {
      int v_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
      if (!((v_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0) &&
          !((v_bs_block_id > q_bs_block_id - blocksparse_local_blocks))) {
        continue;
      }
    }
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx -
                                                           start_token_idx));

    const cache_t* v_ptr = v_cache + physical_block_number * kv_block_stride +
                           kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec;

        if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
          v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        } else {
          V_quant_vec v_quant_vec =
              *reinterpret_cast<const V_quant_vec*>(v_ptr + offset);
          // Vector conversion from V_quant_vec to V_vec.
          v_vec = fp8::scaled_convert<V_vec, V_quant_vec, KV_DTYPE>(v_quant_vec,
                                                                    v_scale);
        }
        if (block_idx == num_seq_blocks - 1) {
          // NOTE(woosuk): When v_vec contains the tokens that are out of the
          // context, we should explicitly zero out the values since they may
          // contain NaNs. See
          // https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec_ptr[j] = token_idx + j < seq_len ? v_vec_ptr[j] : zero_value;
          }
        }
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }

  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += VLLM_SHFL_XOR_SYNC(acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE(woosuk): A barrier is required because the shared memory space for
  // logits is reused for the output.
  __syncthreads();

  // Perform reduction across warps.
  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    __syncthreads();

    // Lower warps update the output.
    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    __syncthreads();
  }

  // Write the final output.
  if (warp_idx == 0) {
    scalar_t* out_ptr =
        out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}