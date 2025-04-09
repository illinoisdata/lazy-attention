import torch


class RoPEPatchContext:
    """Context manager for safely applying and reverting RoPE patches."""
    
    def __init__(self):
        self.patched = False
        
    def __enter__(self):
        from minidrag._custom_ops import apply_patch as apply_custom_patch
        from minidrag.model_executor.layers.rotary_embedding import apply_patch as apply_rope_patch
        # torch.cuda.empty_cache()  # clear GPU memory before applying the patch
        apply_custom_patch()
        apply_rope_patch()
        self.patched = True
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.patched:
            try:
                from minidrag.model_executor.layers.rotary_embedding import revert_patch as revert_rope_patch
                from minidrag._custom_ops import revert_patch as revert_custom_patch
                revert_rope_patch()
                revert_custom_patch()
            except Exception as e:
                print(f"Warning: Failed to revert patch: {e}")
            finally:
                # torch.cuda.empty_cache()  # clear GPU memory after reverting the patch
                pass