from __future__ import annotations

import importlib
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from lazy.attention.layer import forward as attn_layer_forward
from lazy.attention.layer import set_splitting_ops_for_v1
from lazy.attention.backends.triton_attn import forward as triton_attn_forward
from lazy.core.sched.scheduler import LazyScheduler
from lazy.engine import EngineCoreRequest as LazyEngineCoreRequest
from lazy.engine.async_llm import AsyncLazyLLM
from lazy.engine.core import LazyEngineCoreProc
from lazy.engine.llm_engine import LazyLLMEngine
from lazy.engine.processor import LazyProcessor
from lazy.entrypoints.llm import LazyLLM
from lazy.model_executor.layers.rotary_embedding import (Llama3RotaryEmbedding,
                                                         LazyRotaryEmbedding)
from lazy.model_executor.models.llama import forward as llama_attn_forward
from lazy.request import LazyRequest
from lazy.utils.triton_compat import apply_triton_launch_hook_shim
from lazy.worker.gpu_input_batch import CachedRequestState
from lazy.worker.gpu_model_runner import LazyGPUModelRunner

VLLM_TRITON_BACKEND = "TRITON_ATTN_VLLM_V1"


@dataclass(frozen=True)
class PatchTarget:
    module_name: str
    attribute_name: str
    replacement: Any

    @property
    def key(self) -> str:
        return f"{self.module_name}.{self.attribute_name}"


_ORIGINALS: dict[str, Any] = {}


def _set_attention_backend() -> None:
    os.environ["VLLM_ATTENTION_BACKEND"] = VLLM_TRITON_BACKEND


def clear_attention_backend() -> None:
    os.environ.pop("VLLM_ATTENTION_BACKEND", None)


def _resolve_owner(module: Any, attribute_name: str) -> tuple[Any, str]:
    owner = module
    attr_parts = attribute_name.split(".")
    for part in attr_parts[:-1]:
        owner = getattr(owner, part)
    return owner, attr_parts[-1]


def _apply_targets(targets: Iterable[PatchTarget]) -> None:
    for target in targets:
        module = importlib.import_module(target.module_name)
        owner, leaf_attr = _resolve_owner(module, target.attribute_name)
        if target.key not in _ORIGINALS:
            _ORIGINALS[target.key] = getattr(owner, leaf_attr)
        setattr(owner, leaf_attr, target.replacement)


def _revert_targets(targets: Iterable[PatchTarget]) -> None:
    for target in reversed(tuple(targets)):
        original = _ORIGINALS.pop(target.key, None)
        if original is None:
            continue
        module = importlib.import_module(target.module_name)
        owner, leaf_attr = _resolve_owner(module, target.attribute_name)
        setattr(owner, leaf_attr, original)


def attention_targets() -> tuple[PatchTarget, ...]:
    return (
        PatchTarget(
            "vllm.model_executor.layers.rotary_embedding",
            "Llama3RotaryEmbedding",
            Llama3RotaryEmbedding,
        ),
        # Checkpoints with `rope_scaling: null` (e.g. the block-fine-tuned
        # models) get the plain RotaryEmbedding, which must also expose
        # inv_freq for the deferred key rotation.
        PatchTarget(
            "vllm.model_executor.layers.rotary_embedding",
            "RotaryEmbedding",
            LazyRotaryEmbedding,
        ),
        PatchTarget(
            "vllm.v1.attention.backends.triton_attn",
            "TritonAttentionImpl.forward",
            triton_attn_forward,
        ),
        PatchTarget("vllm.attention.layer", "Attention.forward", attn_layer_forward),
        PatchTarget(
            "vllm.config",
            "CompilationConfig.set_splitting_ops_for_v1",
            set_splitting_ops_for_v1,
        ),
        PatchTarget(
            "vllm.model_executor.models.llama",
            "LlamaAttention.forward",
            llama_attn_forward,
        ),
    )


def frontend_targets() -> tuple[PatchTarget, ...]:
    return (
        PatchTarget("vllm.v1.engine", "EngineCoreRequest", LazyEngineCoreRequest),
        PatchTarget("vllm.v1.engine.core", "EngineCoreRequest", LazyEngineCoreRequest),
        PatchTarget(
            "vllm.v1.engine.core_client",
            "EngineCoreRequest",
            LazyEngineCoreRequest,
        ),
        PatchTarget("vllm.v1.engine.core", "Request", LazyRequest),
        PatchTarget("vllm.v1.request", "Request", LazyRequest),
        PatchTarget("vllm.entrypoints.llm", "LLM", LazyLLM),
        PatchTarget("vllm.v1.engine.processor", "Processor", LazyProcessor),
        PatchTarget("vllm.v1.engine.llm_engine", "LLMEngine", LazyLLMEngine),
        PatchTarget("vllm.v1.engine.core", "EngineCoreProc", LazyEngineCoreProc),
        PatchTarget(
            "vllm.v1.engine.core_client",
            "EngineCoreProc",
            LazyEngineCoreProc,
        ),
        PatchTarget(
            "vllm.v1.worker.gpu_input_batch",
            "CachedRequestState",
            CachedRequestState,
        ),
        PatchTarget(
            "vllm.v1.worker.gpu_model_runner",
            "GPUModelRunner",
            LazyGPUModelRunner,
        ),
        PatchTarget("vllm.v1.engine.async_llm", "AsyncLLM", AsyncLazyLLM),
        PatchTarget("vllm", "LLM", LazyLLM),
    )


def scheduler_targets(scheduler_cls: type[Any]) -> tuple[PatchTarget, ...]:
    return (
        PatchTarget("vllm.v1.core.sched.scheduler", "Scheduler", scheduler_cls),
    )


def apply_attention_patches() -> None:
    _set_attention_backend()
    apply_triton_launch_hook_shim()
    _apply_targets(attention_targets())


def revert_attention_patches() -> None:
    _revert_targets(attention_targets())


def apply_all_patches(
    scheduler_cls: type[Any] = LazyScheduler,
) -> None:
    _set_attention_backend()
    apply_triton_launch_hook_shim()
    _apply_targets(attention_targets())
    _apply_targets(frontend_targets())
    _apply_targets(scheduler_targets(scheduler_cls))


def revert_all_patches(
    scheduler_cls: type[Any] = LazyScheduler,
    *,
    clear_backend: bool = False,
) -> None:
    _revert_targets(scheduler_targets(scheduler_cls))
    _revert_targets(frontend_targets())
    _revert_targets(attention_targets())
    if clear_backend:
        clear_attention_backend()
