import torch

from lazy.vllm_patch import (
    VLLM_TRITON_BACKEND,
    apply_attention_patches,
    clear_attention_backend,
    revert_attention_patches,
)

_ORIGINAL_RUN_ENGINE_CORE = None

def patched_run_engine_core(*args, dp_rank=0, local_dp_rank=0, ready_pipe, **kwargs):
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
        global _ORIGINAL_RUN_ENGINE_CORE
        if _ORIGINAL_RUN_ENGINE_CORE is None:
            _ORIGINAL_RUN_ENGINE_CORE = vllm.v1.engine.core.EngineCoreProc.run_engine_core
        vllm.v1.engine.core.EngineCoreProc.run_engine_core = patched_run_engine_core
        torch.cuda.synchronize()
    
    @classmethod
    def revert_patches_subproc(cls):
        import vllm.v1.engine.core
        if _ORIGINAL_RUN_ENGINE_CORE is not None:
            vllm.v1.engine.core.EngineCoreProc.run_engine_core = _ORIGINAL_RUN_ENGINE_CORE
        torch.cuda.synchronize()
    
    @classmethod
    def apply_patches_curproc(cls):
        LazyAttentionContextManager.apply_triton_backend()
        apply_attention_patches()
        torch.cuda.synchronize()
        
    @classmethod
    def revert_patches_curproc(cls):
        revert_attention_patches()
        LazyAttentionContextManager.revert_triton_backend()        
        torch.cuda.synchronize()
        

    @classmethod
    def apply_triton_backend(cls):
        import os
        os.environ["VLLM_ATTENTION_BACKEND"] = VLLM_TRITON_BACKEND

    @classmethod
    def revert_triton_backend(cls):
        clear_attention_backend()

    def __init__(self, config=None):
        pass
        
    def __enter__(self):
        LazyAttentionContextManager.apply_patches_curproc()
        LazyAttentionContextManager.apply_patches_subproc()

    def __exit__(self, exc_type, exc_val, exc_tb):
        LazyAttentionContextManager.revert_patches_subproc()
        LazyAttentionContextManager.revert_patches_curproc()

    @classmethod
    def apply_patches(cls):
        cls.apply_patches_curproc()

    @classmethod
    def revert_patches(cls):
        cls.revert_patches_curproc()

    @classmethod
    def apply_patches_subprocess(cls):
        cls.apply_patches_subproc()

    @classmethod
    def revert_patches_subprocess(cls):
        cls.revert_patches_subproc()
