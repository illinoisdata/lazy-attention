import torch

# monkey patch functions
from lazy.attention.layer import apply_patch as apply_attn_layer_patch
from lazy.attention.backends.triton_attn import apply_patch as apply_triton_attn_patch
from lazy.model_executor.models.llama import apply_patch as apply_llama_patch

from lazy.attention.layer import revert_patch as revert_attn_layer_patch
from lazy.attention.backends.triton_attn import revert_patch as revert_triton_attn_patch
from lazy.model_executor.models.llama import revert_patch as revert_llama_patch

def patched_run_engine_core(*args, dp_rank=0, local_dp_rank=0, ready_pipe, **kwargs):
    # patch all subprocesses
    LazyAttentionContextManager.apply_patches_curproc()
    torch.cuda.synchronize()
    import vllm.v1.engine.core
    return vllm.v1.engine.core.EngineCoreProc.run_engine_core(
                *args,
                dp_rank=dp_rank, 
                local_dp_rank=local_dp_rank, 
                ready_pipe=ready_pipe, 
                **kwargs)


class LazyAttentionContextManager:
    @classmethod
    def apply_patches_subproc(cls):
        import vllm.v1.engine.core
        vllm.v1.engine.core.EngineCoreProc.run_engine_core = patched_run_engine_core
        torch.cuda.synchronize()
    
    @classmethod
    def revert_patches_subproc(cls):
        pass
    
    @classmethod
    def apply_patches_curproc(cls):
        """Patch current process with LazyAttention patches.
        """
        LazyAttentionContextManager.apply_triton_backend()
        apply_triton_attn_patch()
        apply_attn_layer_patch()
        apply_llama_patch()
        torch.cuda.synchronize()
        
    @classmethod
    def revert_patches_curproc(cls):
        revert_llama_patch()
        revert_attn_layer_patch()
        revert_triton_attn_patch()
        LazyAttentionContextManager.revert_triton_backend()        
        torch.cuda.synchronize()
        

    @classmethod
    def apply_triton_backend(cls):
        import os
        os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1" 

    @classmethod
    def revert_triton_backend(cls):
        import os
        os.environ["VLLM_ATTENTION_BACKEND"] = "" 

    def __init__(self, config=None):
        pass
        
    def __enter__(self):
        LazyAttentionContextManager.apply_patches_curproc()
        LazyAttentionContextManager.apply_patches_subproc()

    def __exit__(self, exc_type, exc_val, exc_tb):
        LazyAttentionContextManager.revert_patches_subproc()
        LazyAttentionContextManager.revert_patches_curproc()
        
