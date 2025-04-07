def get_attn_backend_cls(cls, selected_backend, head_size, dtype,
                             kv_cache_dtype, block_size, use_v1,
                             use_mla) -> str:
    return ("vllm.v1.attention.backends."
            "triton_attn.TritonAttentionBackend")

original_get_attn_backend_cls = None

def apply_patch():
    import vllm.platforms.cuda
    global original_get_attn_backend_cls
    original_get_attn_backend_cls = vllm.platforms.cuda.CudaPlatformBase.get_attn_backend_cls
    vllm.platforms.cuda.CudaPlatformBase.get_attn_backend_cls = get_attn_backend_cls

def revert_patch():
    import vllm.platforms.cuda
    vllm.platforms.cuda.CudaPlatformBase.get_attn_backend_cls = original_get_attn_backend_cls