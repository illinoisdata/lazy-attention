class MiniDynamicRAG:
    @classmethod
    def apply_patches(cls):
        from minidrag.platforms.cuda import apply_patch as apply_cuda_patch
        from minidrag.attention.layer import apply_patch as apply_attn_layer_patch
        from minidrag.attention.backends.triton_attn import apply_patch as apply_triton_attn_patch
        from minidrag.core.kv_cache_utils import apply_patch as apply_kv_cache_utils_patch
        from minidrag.model_executor.layers.rotary_embedding import apply_patch as apply_rotary_embedding_patch
        from minidrag.model_executor.models.llama import apply_patch as apply_llama_patch
        from minidrag.minidrag._custom_ops import apply_patch as apply_custom_ops_patch

        # step 1: use triton attention backend
        apply_cuda_patch()
        apply_kv_cache_utils_patch()
        apply_custom_ops_patch()
        apply_rotary_embedding_patch()
        apply_triton_attn_patch()
        apply_attn_layer_patch()
        apply_llama_patch()
        

    @classmethod
    def revert_patches(cls):
        from minidrag.platforms.cuda import revert_patch as revert_cuda_patch
        from minidrag.attention.layer import revert_patch as revert_attn_layer_patch
        from minidrag.attention.backends.triton_attn import revert_patch as revert_triton_attn_patch
        from minidrag.core.kv_cache_utils import revert_patch as revert_kv_cache_utils_patch
        from minidrag.model_executor.layers.rotary_embedding import revert_patch as revert_rotary_embedding_patch
        from minidrag.model_executor.models.llama import revert_patch as revert_llama_patch

        revert_cuda_patch()
        revert_attn_layer_patch()
        revert_triton_attn_patch()
        revert_kv_cache_utils_patch()
        revert_rotary_embedding_patch()
        revert_llama_patch()

    @classmethod
    def apply_triton_backend(cls):
        from minidrag.platforms.cuda import apply_patch as apply_cuda_patch
        apply_cuda_patch()

    @classmethod
    def revert_triton_backend(cls):
        from minidrag.platforms.cuda import revert_patch as revert_cuda_patch   
        revert_cuda_patch()

    def __init__(self, config=None):
        pass
        
    def __enter__(self):
        pass
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
