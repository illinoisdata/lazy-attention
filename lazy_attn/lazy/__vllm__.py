from __future__ import annotations  # isort:skip

import os

# backend (model -> attention layer -> attention impl)
from lazy.attention.backends.triton_attn import forward as triton_attn_forward
from lazy.attention.layer import (
    forward as attn_layer_forward,
    set_splitting_ops_for_v1
)
from lazy.model_executor.models.llama import forward as llama_attn_forward

# frontend
from lazy.request import LazyRequest
from lazy.entrypoints.llm import LazyLLM
from lazy.engine.llm_engine import LazyLLMEngine
from lazy.engine.processor import LazyProcessor
from lazy.engine import EngineCoreRequest as LazyEngineCoreRequest
from lazy.engine.core import LazyEngineCoreProc

# scheduler
from lazy.core.sched.scheduler import LazyScheduler

# async
from lazy.engine.async_llm import AsyncLazyLLM

# ///////////////////////////////
from lazy.worker.gpu_input_batch import CachedRequestState
from lazy.worker.gpu_model_runner import LazyGPUModelRunner
# //////////////////////////////

# Step 0: Set environment variable for Triton backend
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1" 


def proc_patch():
    import vllm
    # TODO: fix the routering problem, when reuse and when not reuse
    
    # Step 1.1: Patch triton attention forward function
    import vllm.v1.attention.backends.triton_attn
    vllm.v1.attention.backends.triton_attn.TritonAttentionImpl.forward = triton_attn_forward
    
    # Step 1.2: Patch attention layer
    vllm.attention.layer.Attention.forward = attn_layer_forward
    vllm.config.CompilationConfig.set_splitting_ops_for_v1 = set_splitting_ops_for_v1
    
    # Step 1.3: Patch llama attention forward function
    import vllm.model_executor.models.llama
    vllm.model_executor.models.llama.LlamaAttention.forward = llama_attn_forward

    # Step 2: We need to patch the frontend of the vllm to send our dynamic requests to the backend (real model executor)
    # vllm.v1.engine.__init__.EngineCoreRequest = LazyEngineCoreRequest
    import vllm.v1.engine
    vllm.v1.engine.EngineCoreRequest = LazyEngineCoreRequest
    vllm.v1.engine.core.EngineCoreRequest = LazyEngineCoreRequest
    vllm.v1.engine.core_client.EngineCoreRequest = LazyEngineCoreRequest
    vllm.v1.engine.core.Request = LazyRequest

    import vllm.v1.request
    vllm.v1.request.Request = LazyRequest
    import vllm.entrypoints.llm
    vllm.entrypoints.llm.LLM = LazyLLM
    import vllm.v1.engine.processor
    vllm.v1.engine.processor.Processor = LazyProcessor
    import vllm.v1.engine.llm_engine
    vllm.v1.engine.llm_engine.LLMEngine = LazyLLMEngine
    import vllm.v1.engine.core
    vllm.v1.engine.core.EngineCoreProc = LazyEngineCoreProc
    import vllm.v1.engine.core_client
    vllm.v1.engine.core_client.EngineCoreProc = LazyEngineCoreProc
    
    # Patch the model runner
    import vllm.v1.worker
    vllm.v1.worker.gpu_input_batch.CachedRequestState = CachedRequestState
    vllm.v1.worker.gpu_model_runner.GPUModelRunner = LazyGPUModelRunner
    
    # Step 3: finally, we patch the backend for scehduling
    import vllm.v1.core.sched.scheduler
    vllm.v1.core.sched.scheduler.Scheduler = LazyScheduler
    
    # Step 4: patch for async mode
    import vllm.v1.engine.async_llm
    vllm.v1.engine.async_llm.AsyncLLM = AsyncLazyLLM

    vllm.LLM = LazyLLM

proc_patch()
