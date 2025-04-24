"""
Here we patch the LLM class to allow the formatting of dynamic requests.
Compared to the original requests, dynamic requests allow a new field `documents`,
which is a list of strings, and a new field `documents_token_ids`, which is a list of lists of integers.
"""

import itertools
import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Optional, Union, cast, overload


from vllm import SamplingParams
from vllm.inputs import PromptType
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding.guided_fields import (
    GuidedDecodingRequest, LLMGuidedOptions)
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import (RequestOutputKind, SamplingParams)

# class LLM:
def generate(
    self,
    prompts: Union[Union[PromptType, Sequence[PromptType]],
                       Optional[Union[str, list[str]]]] = None,
    sampling_params: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
    prompt_token_ids: Optional[Union[list[int], list[list[int]]]] = None,
    use_tqdm: bool = True,
    lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
    prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    guided_options_request: Optional[Union[LLMGuidedOptions,
                                               GuidedDecodingRequest]] = None,
    priority: Optional[list[int]] = None,
    # new optional argument to support document sequences for dynamic rag
    document_seqs: Union[Union[Sequence[PromptType], Sequence[Sequence[PromptType]]],
                       Optional[Union[list[str], list[list[str]]]]] = None,
    ) -> list[RequestOutput]:
        assert prompt_token_ids is None, "[deprecated] prompt_token_ids is not supported in LLM.generate()"

        parsed_prompts = cast(Union[PromptType, Sequence[PromptType]], 
                              prompts)
        parsed_document_seqs = cast(Union[Sequence[PromptType], Sequence[Sequence[PromptType]]], 
                                    document_seqs)

        if sampling_params is None:
            # Use default sampling params.
            sampling_params = self.get_default_sampling_params()

        self._validate_and_add_requests(
            prompts=parsed_prompts,
            params=sampling_params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            guided_options=guided_options_request,
            priority=priority,
            # new optional argument to support document sequences for dynamic rag
            document_seqs=parsed_document_seqs,)

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return self.engine_class.validate_outputs(outputs, RequestOutput)


def _validate_and_add_requests(
    self,
    prompts: Union[PromptType, Sequence[PromptType]],
    params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams,
                      Sequence[PoolingParams]],
    lora_request: Optional[Union[Sequence[LoRARequest], LoRARequest]],
    prompt_adapter_request: Optional[PromptAdapterRequest],
    guided_options: Optional[GuidedDecodingRequest] = None,
    priority: Optional[list[int]] = None,
    # new optional argument to support document sequences for dynamic rag
    document_seqs :Union[Sequence[PromptType], Sequence[Sequence[PromptType]]] = None,
) -> None:
    if guided_options is not None:
        warnings.warn(
                "guided_options_request is deprecated, use "
                "SamplingParams.guided_decoding instead",
                DeprecationWarning,
                stacklevel=2,
            )

    if isinstance(prompts, (str, dict)):
        # Convert a single prompt to a list.
        prompts = [prompts]
        
    if document_seqs is not None:
        if isinstance(document_seqs[0], (str, dict)):
            # Convert a document sequence to a nested list.
            document_seqs = [document_seqs]
        assert len(prompts) == len(document_seqs), \
            "The lengths of prompts and document_seqs must be the same."

    num_requests = len(prompts)
    if isinstance(params, list) and len(params) != num_requests:
        raise ValueError("The lengths of prompts and params "
                             "must be the same.")
    if isinstance(lora_request,
                      list) and len(lora_request) != num_requests:
        raise ValueError("The lengths of prompts and lora_request "
                             "must be the same.")

    for sp in params if isinstance(params, list) else (params, ):
        if isinstance(sp, SamplingParams):
            self._add_guided_params(sp, guided_options)

                # We only care about the final output
            sp.output_kind = RequestOutputKind.FINAL_ONLY

    # Add requests to the engine.
    for i, prompt in enumerate(prompts):
        self._add_request(
                prompt,
                params[i] if isinstance(params, Sequence) else params,
                lora_request=lora_request[i] if isinstance(
                    lora_request, Sequence) else lora_request,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority[i] if priority else 0,
                # new optional argument to support document sequences for dynamic rag
                document_seq=document_seqs[i] if document_seqs else None,
            )
        
        
def _add_request(
    self,
    prompt: PromptType,
    params: Union[SamplingParams, PoolingParams],
    lora_request: Optional[LoRARequest] = None,
    prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    priority: int = 0,
    document_seq: Optional[Sequence[PromptType]] = None,
) -> None:
    request_id = str(next(self.request_counter))
    self.llm_engine.add_request(
        request_id,
        prompt,
        params,
        lora_request=lora_request,
        prompt_adapter_request=prompt_adapter_request,
        priority=priority,
        # new optional argument to support document sequences for dynamic rag
        document_seq=document_seq,
    )