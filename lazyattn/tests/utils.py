import os

import torch

from lazy.ctxmgr import LazyAttentionContextManager


def setup_deterministic_env():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    

class LazyAttentionContext:
    """Context manager for safely applying and reverting RoPE patches."""
    
    def __init__(self):
        self.patched = False
        
    def __enter__(self):
        LazyAttentionContextManager.apply_patches()
        torch.cuda.synchronize()
        LazyAttentionContextManager.apply_patches_subprocess()
        torch.cuda.synchronize()
        self.patched = True
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.patched:
            try:
                LazyAttentionContextManager.revert_patches()
                torch.cuda.synchronize()
                LazyAttentionContextManager.revert_patches_subprocess
                torch.cuda.synchronize()
            except Exception as e:
                print(f"Warning: Failed to revert patch: {e}")
                

class TritonAttnBackendContext:
    """Context manager for safely applying and reverting TritonAttn patches."""
    
    def __init__(self):
        self.patched = False
        
    def __enter__(self):
        LazyAttentionContextManager.apply_triton_backend()
        self.patched = True
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.patched:
            try:
                LazyAttentionContextManager.revert_triton_backend()
            except Exception as e:
                print(f"Warning: Failed to revert patch: {e}")
                
                
def set_seed(seed):
    # for reproducibility
    from vllm.model_executor.utils import set_random_seed
    set_random_seed(seed)