  ops.def(
      "rotary_embedding_q(Tensor positions, Tensor! query,"
      "                   int head_size,"
      "                   Tensor cos_sin_cache, bool is_neox) -> ()");
  ops.impl("rotary_embedding_q", torch::kCUDA, &rotary_embedding_q);

  ops.def(
      "batched_rotary_embedding_q(Tensor positions, Tensor! query,"
      "                           int head_size,"
      "                           Tensor cos_sin_cache, bool is_neox,"
      "                           int rot_dim,"
      "                           Tensor cos_sin_cache_offsets) -> ()");
  ops.impl("batched_rotary_embedding_q", torch::kCUDA, &batched_rotary_embedding_q);