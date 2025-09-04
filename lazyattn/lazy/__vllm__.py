from __future__ import annotations  # isort:skip

import os

# backend
from lazy.attention.backends.triton_attn import forward as triton_attn_forward
from lazy.attention.layer import (
    forward as attn_layer_forward,
    set_splitting_ops_for_v1
)
from lazy.model_executor.models.llama import forward as llama_attn_forward

# frontend
from lazy.request import _Request
from lazy.entrypoints.llm import (
    generate as llm_generate,
    _validate_and_add_requests as llm_validate_and_add_requests, 
    _add_request as llm_add_request,
)
from lazy.engine.llm_engine import add_request as llm_engine_add_request
from lazy.engine.processor import process_inputs as llm_engine_process_inputs
from lazy.engine import EngineCoreRequest, EngineCoreEventType
from lazy.engine.core import LazyEngineCoreProc

# scheduler
from lazy.core.sched.scheduler import MiniDynamicRAGScheduler

# async
from lazy.engine.async_llm import add_request, generate, _add_request

# ///////////////////////////////
from lazy.worker.gpu_input_batch import CachedRequestState
from lazy.worker.gpu_model_runner import LazyGPUModelRunner
# //////////////////////////////

# Step 0: Set environment variable for Triton backend
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1" 


def proc_patch():
    import vllm as _vllm
    # TODO: fix the routering problem, when reuse and when not reuse
    # Step 1.4: Patch triton attention forward function
    import vllm.v1.attention.backends.triton_attn
    vllm.v1.attention.backends.triton_attn.TritonAttentionImpl.forward = triton_attn_forward
    # Step 1.5: Patch attention layer
    _vllm.attention.layer.Attention.forward = attn_layer_forward
    _vllm.config.CompilationConfig.set_splitting_ops_for_v1 = set_splitting_ops_for_v1
    # Step 1.6: Patch llama attention forward function
    import vllm.model_executor.models.llama
    vllm.model_executor.models.llama.LlamaAttention.forward = llama_attn_forward

    # Step 2: We need to patch the frontend of the vllm to send our dynamic requests to the backend (real model executor)
    # vllm.v1.engine.__init__.EngineCoreRequest = EngineCoreRequest
    import vllm.v1.engine
    global EngineCoreRequest
    vllm.v1.engine.EngineCoreRequest = EngineCoreRequest
    vllm.v1.engine.EngineCoreEventType = EngineCoreEventType
    vllm.v1.engine.core.EngineCoreRequest = EngineCoreRequest
    vllm.v1.engine.core.Request = _Request
    vllm.v1.engine.core_client.EngineCoreRequest = EngineCoreRequest
    
    import vllm.v1.request
    vllm.v1.request.Request = _Request
    
    import vllm.entrypoints.llm
    vllm.entrypoints.llm.LLM.generate = llm_generate
    vllm.entrypoints.llm.LLM._validate_and_add_requests = llm_validate_and_add_requests
    vllm.entrypoints.llm.LLM._add_request = llm_add_request
    import vllm.v1.engine.llm_engine
    vllm.v1.engine.llm_engine.LLMEngine.add_request = llm_engine_add_request
    import vllm.v1.engine.processor
    vllm.v1.engine.processor.Processor.process_inputs = llm_engine_process_inputs
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
    vllm.v1.core.sched.scheduler.Scheduler = MiniDynamicRAGScheduler
    
    # Step 4: patch for async mode
    import vllm.v1.engine.async_llm
    vllm.v1.engine.async_llm.AsyncLLM.add_request = add_request
    vllm.v1.engine.async_llm.AsyncLLM._add_request = _add_request
    vllm.v1.engine.async_llm.AsyncLLM.generate = generate

proc_patch()
