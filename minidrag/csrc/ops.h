void rotary_embedding_q(torch::Tensor& positions, torch::Tensor& query,
                        int64_t head_size,
                        torch::Tensor& cos_sin_cache, bool is_neox);

void batched_rotary_embedding_q(torch::Tensor& positions, torch::Tensor& query,
                                int64_t head_size,
                                torch::Tensor& cos_sin_cache, bool is_neox,
                                int64_t rot_dim,
                                torch::Tensor& cos_sin_cache_offsets);