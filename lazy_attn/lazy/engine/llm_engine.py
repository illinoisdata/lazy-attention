"""
We change the default vLLM engine to our customized engine so that it can accept
lazy attention requests with document sequences.

Changed by Haocheng at 2025/09/03
"""

from collections.abc import Mapping
from copy import copy
from typing import Any, Callable, Optional, Union

from typing_extensions import TypeVar

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer_group import (
    TokenizerGroup, init_tokenizer_from_configs)
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Device
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
# from vllm.v1.engine.processor import Processor
from vllm.v1.executor.abstract import Executor
from vllm.v1.metrics.loggers import StatLoggerFactory

logger = init_logger(__name__)

from vllm.v1.engine.llm_engine import LLMEngine
from collections.abc import Sequence
from lazy.engine.processor import LazyProcessor

class LazyLLMEngine(LLMEngine):
    def __init__(self, *args, **kwargs):
        mm_registry = kwargs.get("mm_registry")
        super().__init__(*args, **kwargs)
        # Use customized processor to process prompt(query) and documents
        self.processor = LazyProcessor(vllm_config=self.vllm_config,
                                       tokenizer=self.tokenizer,
                                       mm_registry=mm_registry)

    def add_request(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
        # For lazy attention
        document_seq: Optional[Sequence[PromptType]] = None,
    ) -> None:
        if document_seq is None:
            # Nothing lazy about this request. Delegating keeps upstream
            # behaviour exactly -- including parallel sampling, which the
            # document path below cannot do and used to disable for every
            # request the patched engine saw.
            return super().add_request(
                request_id,
                prompt,
                params,
                arrival_time=arrival_time,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority)

        n = params.n if isinstance(params, SamplingParams) else 1
        if n != 1:
            # Not an assert: this is a user-facing input error, and asserts
            # vanish under `python -O`.
            raise ValueError(
                f"LazyAttention does not support n > 1 with documents (got "
                f"n={n}). The document path spawns one prefill request per "
                "document, which the parallel-sampling fan-out does not model.")

        logger.debug("Using customized process_inputs for lazy attention.")
        # NOTE: pass by keyword. vLLM has inserted new positional parameters
        # into Processor.process_inputs across releases (e.g. tokenization_kwargs
        # in 0.9.x), which silently shifts positional arguments.
        prompt_str, request = self.processor.process_inputs(
            request_id,
            prompt,
            params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority,
            # For lazy attention
            document_seq=document_seq,
            block_size=self.cache_config.block_size)

        # Make a new RequestState and queue.
        self.output_processor.add_request(request, prompt_str, None, 0)
        # Add the request to EngineCore.
        self.engine_core.add_request(request)


def apply_patch():
    import vllm.v1.engine.llm_engine
    vllm.v1.engine.llm_engine.LLMEngine.add_request = add_request
