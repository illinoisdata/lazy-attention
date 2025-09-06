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
        # Use custmoized processor to process prompt(query) and documents
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
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
        # For lazy attention
        document_seq: Optional[Sequence[PromptType]] = None,
    ) -> None:
        if document_seq is None:
            logger.debug("Using default process_inputs.")
            # Process raw inputs into the request.
            prompt_str, request = self.processor.process_inputs(
                request_id, prompt, params, arrival_time, lora_request,
                trace_headers, prompt_adapter_request, priority)
        else:
            logger.debug("Using customized process_inputs for lazy attention.")
            block_size = self.cache_config.block_size
            prompt_str, request = self.processor.process_inputs(
                request_id, prompt, params, arrival_time, lora_request,
                trace_headers, prompt_adapter_request, priority,
                # For lazy attention
                document_seq=document_seq,
                block_size=block_size)

        n = params.n if isinstance(params, SamplingParams) else 1
        assert n == 1, "n > 1 is not supported in customized engine for lazy attention now."

        if n == 1:
            # Make a new RequestState and queue.
            self.output_processor.add_request(request, prompt_str, None, 0)
            # Add the request to EngineCore.
            self.engine_core.add_request(request)
            return

        # TODO(haocheng): support child requests.
        # Fan out child requests (for n>1).
        parent_req = ParentRequest(request_id, params)
        for idx in range(n):
            request_id, params = parent_req.get_child_info(idx)
            child_request = request if idx == n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = params

            # Make a new RequestState and queue.
            self.output_processor.add_request(child_request, prompt_str,
                                              parent_req, idx)
            # Add the request to EngineCore.
            self.engine_core.add_request(child_request)


def apply_patch():
    import vllm.v1.engine.llm_engine
    vllm.v1.engine.llm_engine.LLMEngine.add_request = add_request
