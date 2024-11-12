import io
import sys

import numpy as np

from vllm import LLM, SamplingParams
from vllm.utils import Device


def get_block_size(llm: LLM) -> int:
    return llm.llm_engine.cache_config.block_size


def get_tokenizer_grp(llm: LLM):
    return llm.llm_engine.tokenizer


def get_tokenizer(llm: LLM):
    return llm.llm_engine.tokenizer.tokenizer


def get_ctx(llm: LLM):
    return llm.llm_engine.scheduler_contexts[0]


def get_llm_engine(llm: LLM):
    """
    Get the llm engine.
    """
    return llm.llm_engine


def get_model_executor(llm: LLM):
    """
    Get the model executor.
    """
    return get_llm_engine(llm).model_executor


def get_worker(llm: LLM):
    """
    Get the worker.
    """
    return get_model_executor(llm).driver_worker


def get_model_runner(llm: LLM):
    """
    Get the model runner.
    """
    return get_worker(llm).model_runner


def get_model(llm: LLM):
    return get_model_runner(llm).model


def get_attn_backend(llm: LLM):
    """
    Get the attention backend.
    """
    return get_model_runner(llm).attn_backend


def get_scheduler(llm: LLM, scheduler_id: int = 0):
    """
    Get the scheduler for the given id.
    """
    return llm.llm_engine.scheduler[scheduler_id]


def get_scheduler_context(llm: LLM, context_id: int = 0):
    """
    Get the seq group metadata list.
    """
    return get_llm_engine(llm).scheduler_contexts[context_id]


def get_seq_grp_metadata(llm: LLM, context_id: int = 0):
    """
    Get the seq group metadata list.
    """
    return get_scheduler_context(llm, context_id).seq_grp_metadata


def get_gpu_cache(llm: LLM):
    """
    Get the whole gpu cache.
    """
    return llm.llm_engine.model_executor.driver_worker.gpu_cache


def get_block_allocator(llm: LLM):
    """
    Get the block allocator.
    """
    return llm.llm_engine.scheduler[0].block_manager.block_allocator._allocators[Device.GPU]


def get_evictor(llm: LLM):
    """
    Get the evictor.
    """
    return get_block_allocator(llm).evictor


def get_gpu_block(llm: LLM, block_id: int):
    """
    Get the gpu block for the given block id.
    """
    return get_gpu_cache(llm)[:, :, block_id, ...]  # attn_layers, kv, num blocks, block size, head size


def slot_id_to_block_id(slot_id, block_size: int = 16):
    # block id starts from 0
    block_id = np.floor(slot_id / block_size)
    return block_id


def get_sampling_param(stage: str = "prefill", max_tokens: int = 20) -> SamplingParams:
    if stage == "prefill":
        # max_tokens=1 for prefilling, 0 is illegal for vllm
        sampling_param = SamplingParams(max_tokens=1, seed=2024, stop_token_ids=[128008, 128001])
    elif stage == "decode":
        sampling_param = SamplingParams(max_tokens=max_tokens, seed=2024, stop_token_ids=[128008, 128001])
    else:
        raise ValueError("Invalid stage")
    return sampling_param


def capture_output(func, *args, **kwargs):
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        func(*args, **kwargs)
    finally:
        sys.stdout = sys.__stdout__

    return captured_output.getvalue()
