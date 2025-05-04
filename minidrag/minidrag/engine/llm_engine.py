# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from copy import copy
from typing import Any, Callable, Optional, Union

from typing_extensions import TypeVar

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics_types import StatLoggerBase
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer_group import (
    BaseTokenizerGroup, init_tokenizer_from_configs)
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Device
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.abstract import Executor
from collections.abc import Sequence

# class LLMEngine:
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
    # For dynamic rag
    document_seq: Optional[Sequence[PromptType]] = None,
) -> None:
    # Process raw inputs into the request.
    if document_seq is None:
        # Fall back to the default behavior.
        request = self.processor.process_inputs(request_id, prompt, params,
                                                arrival_time, lora_request,
                                                trace_headers,
                                                prompt_adapter_request,
                                                priority)
    else:
        # Use customized behavior.
        block_size = self.cache_config.block_size
        # Ctor EngineCoreRequest
        request = self.processor.process_inputs(request_id, prompt, params,
                                                arrival_time, lora_request,
                                                trace_headers,
                                                prompt_adapter_request,
                                                priority,
                                                document_seq=document_seq,
                                                block_size=block_size)

    # print(f"after preprocess: {request}")
    n = params.n if isinstance(params, SamplingParams) else 1
    assert n == 1, "n > 1 is not supported in customized engine now."
    if n == 1:
        # Make a new RequestState and queue.
        self.output_processor.add_request(request, None, 0)
        # Add the request to EngineCore.
        # print(f"after preprocess, when add to engien core: {request}")
        self.engine_core.add_request(request)  # Send by socket.
        return
    
    # -------------------------------------------------------------------------
    # Expect not to reach here.
    # TODO(haocheng): support child requests.
    # Fan out child requests (for n>1).
    parent_req = ParentRequest(request_id, params)
    for idx in range(n):
        request_id, params = parent_req.get_child_info(idx)
        child_request = request if idx == n - 1 else copy(request)
        child_request.request_id = request_id
        child_request.sampling_params = params
        # Make a new RequestState and queue.
        self.output_processor.add_request(child_request, parent_req, idx)
        # Add the request to EngineCore.
        self.engine_core.add_request(child_request)
        

def apply_patch():
    import vllm.v1.engine.llm_engine
    vllm.v1.engine.llm_engine.LLMEngine.add_request = add_request