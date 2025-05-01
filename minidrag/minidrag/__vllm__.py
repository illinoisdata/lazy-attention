from __future__ import annotations  # isort:skip

import os

from minidrag.core.kv_cache_utils import hash_request_tokens_no_prefix
from minidrag._custom_ops import (
    rotary_embedding_q, 
    batched_rotary_embedding_q,
)
from minidrag.model_executor.layers.rotary_embedding import (
    forward_cuda as rotary_embedding_forward_cuda,
    forward_native as rotary_embedding_forward_native,
)
from minidrag.attention.backends.triton_attn import forward as triton_attn_forward
from minidrag.attention.layer import (
    forward as attn_layer_forward,
    set_splitting_ops_for_v1
)
from minidrag.model_executor.models.llama import forward as llama_attn_forward

# frontend
from minidrag.request import _Request
from minidrag.entrypoints.llm import (
    generate as llm_generate, 
    _validate_and_add_requests as llm_validate_and_add_requests, 
    _add_request as llm_add_request,
)
from minidrag.engine.llm_engine import add_request as llm_engine_add_request
from minidrag.engine.processor import process_inputs as llm_engine_process_inputs
from minidrag.engine import EngineCoreRequest
from minidrag.engine.core import process_input_socket

# scheduler
from minidrag.core.sched.scheduler import MiniDynamicRAGScheduler


# Step 0: Set environment variable for Triton backend
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1" 


def proc_patch():
    import vllm as _vllm
    # TODO: fix the routering problem, when reuse and when not reuse
    # # Step 1.1: Patch hash function for block content
    # _vllm.v1.core.kv_cache_utils.hash_request_tokens = hash_request_tokens_no_prefix
    # Step 1.2: Patch custom ops
    _vllm._custom_ops.rotary_embedding_q = rotary_embedding_q
    _vllm._custom_ops.batched_rotary_embedding_q = batched_rotary_embedding_q
    # Step 1.3: Patch RotaryEmbedding forward functions
    _vllm.model_executor.layers.rotary_embedding.RotaryEmbedding.forward_cuda = rotary_embedding_forward_cuda
    _vllm.model_executor.layers.rotary_embedding.RotaryEmbedding.forward_native = rotary_embedding_forward_native
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
    import vllm.v1.engine
    # vllm.v1.engine.__init__.EngineCoreRequest = EngineCoreRequest
    _vllm.v1.engine.EngineCoreRequest = EngineCoreRequest
    # import vllm.v1.request
    _vllm.v1.request.Request = _Request
    _vllm.v1.engine.core.Request = _Request
    import vllm.entrypoints.llm
    vllm.entrypoints.llm.LLM.generate = llm_generate
    vllm.entrypoints.llm.LLM._validate_and_add_requests = llm_validate_and_add_requests
    vllm.entrypoints.llm.LLM._add_request = llm_add_request
    import vllm.v1.engine.llm_engine
    vllm.v1.engine.llm_engine.LLMEngine.add_request = llm_engine_add_request
    import vllm.v1.engine.processor
    vllm.v1.engine.processor.Processor.process_inputs = llm_engine_process_inputs
    import vllm.v1.engine.core
    vllm.v1.engine.core.EngineCoreProc.process_input_socket = process_input_socket
    
    # Step 3: finally, we patch the backend for scehduling
    import vllm.v1.core.sched.scheduler
    vllm.v1.core.sched.scheduler.Scheduler = MiniDynamicRAGScheduler


proc_patch()
