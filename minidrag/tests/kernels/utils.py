class RoPEPatchContext:
    """Context manager for safely applying and reverting RoPE patches."""
    
    def __init__(self):
        self.patched = False
        
    def __enter__(self):
        # torch.cuda.empty_cache()
        from minidrag.model_executor.layers.rotary_embedding import apply_patch as apply_rope_patch
        apply_rope_patch()
        self.patched = True
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.patched:
            try:
                from minidrag.model_executor.layers.rotary_embedding import revert_patch as revert_rope_patch
                revert_rope_patch()
            except Exception as e:
                print(f"Warning: Failed to revert patch: {e}")
            # finally:
            #     torch.cuda.empty_cache()  # clear GPU memory after reverting the patch
            

class DynamicTritonAttnContext:
    """Context manager for safely applying and reverting TritonAttn patches."""
    
    def __init__(self):
        self.patched = False
        
    def __enter__(self):
        # torch.cuda.empty_cache()
        from minidrag.attention.backends.triton_attn import apply_patch as apply_triton_attn_patch
        apply_triton_attn_patch()
        self.patched = True
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.patched:
            try:
                from minidrag.attention.backends.triton_attn import revert_patch as revert_triton_attn_patch
                revert_triton_attn_patch()
            except Exception as e:
                print(f"Warning: Failed to revert patch: {e}")
            # finally:
            #     torch.cuda.empty_cache()  # clear GPU memory after reverting the patch