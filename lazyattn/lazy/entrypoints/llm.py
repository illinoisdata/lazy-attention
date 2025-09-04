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

import itertools
import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Callable, ClassVar, Optional, Union, cast, overload

import cloudpickle
import torch.nn as nn
from tqdm.auto import tqdm
from typing_extensions import TypeVar, deprecated

from vllm.beam_search import (BeamSearchInstance, BeamSearchOutput,
                              BeamSearchSequence, get_beam_search_score)
from vllm.engine.arg_utils import (EngineArgs, HfOverrides, PoolerConfig,
                                   TaskOption)
from vllm.engine.llm_engine import LLMEngine
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         ChatTemplateContentFormatOption,
                                         apply_hf_chat_template,
                                         apply_mistral_chat_template,
                                         parse_chat_messages,
                                         resolve_chat_template_content_format)
from vllm.entrypoints.score_utils import (_cosine_similarity,
                                          _validate_score_input_lens)
from vllm.inputs import PromptType, SingletonPrompt, TextPrompt, TokensPrompt
from vllm.inputs.parse import parse_and_batch_prompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding.guided_fields import (
    GuidedDecodingRequest, LLMGuidedOptions)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.outputs import (ClassificationRequestOutput, EmbeddingRequestOutput,
                          PoolingRequestOutput, RequestOutput,
                          ScoringRequestOutput)
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import (BeamSearchParams, GuidedDecodingParams,
                                  RequestOutputKind, SamplingParams)
from vllm.transformers_utils.tokenizer import (AnyTokenizer, MistralTokenizer,
                                               get_cached_tokenizer)
from vllm.usage.usage_lib import UsageContext
from vllm.utils import (Counter, Device, deprecate_args, deprecate_kwargs,
                        is_list_of)
# class LLM:
@deprecate_kwargs(
    "prompt_token_ids",
    is_deprecated=lambda: True,
    additional_message="Please use the 'prompts' parameter instead.",
)
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
    # For LazyAttn, new optional argument to support document sequences for dynamic rag
    document_seqs: Union[Union[Sequence[PromptType], Sequence[Sequence[PromptType]]],
                       Optional[Union[list[str], list[list[str]]]]] = None,
) -> list[RequestOutput]:
    """Generates the completions for the input prompts.
    This class automatically batches the given prompts, considering
    the memory constraint. For the best performance, put all of your prompts
    into a single list and pass it to this method.
    Args:
        prompts: The prompts to the LLM. You may pass a sequence of prompts
            for batch inference. See {class}`~vllm.inputs.PromptType`
            for more details about the format of each prompts.
        sampling_params: The sampling parameters for text generation. If
            None, we use the default sampling parameters.
            When it is a single value, it is applied to every prompt.
            When it is a list, the list must have the same length as the
            prompts and it is paired one by one with the prompt.
        use_tqdm: Whether to use tqdm to display the progress bar.
        lora_request: LoRA request to use for generation, if any.
        prompt_adapter_request: Prompt Adapter request to use for
            generation, if any.
        priority: The priority of the requests, if any.
            Only applicable when priority scheduling policy is enabled.
    Returns:
        A list of `RequestOutput` objects containing the
        generated completions in the same order as the input prompts.
    :::{note}
    Using `prompts` and `prompt_token_ids` as keyword parameters is
    considered legacy and may be deprecated in the future. You should
    instead pass them via the `inputs` parameter.
    :::
    """
    runner_type = self.llm_engine.model_config.runner_type
    if runner_type not in ["generate", "transcription"]:
        messages = [
            "LLM.generate() is only supported for (conditional) generation "
            "models (XForCausalLM, XForConditionalGeneration).",
        ]
        supported_runner_types = self.llm_engine.model_config \
            .supported_runner_types
        if "generate" in supported_runner_types:
            messages.append(
                "Your model supports the 'generate' runner, but is "
                f"currently initialized for the '{runner_type}' runner. "
                "Please initialize vLLM using `--task generate`.")
        raise ValueError(" ".join(messages))
    if prompt_token_ids is not None:
        parsed_prompts = self._convert_v1_inputs(
            prompts=cast(Optional[Union[str, list[str]]], prompts),
            prompt_token_ids=prompt_token_ids,
        )
    else:
        parsed_prompts = cast(Union[PromptType, Sequence[PromptType]],
                              prompts)
        parsed_document_seqs = cast(Union[Sequence[PromptType], Sequence[Sequence[PromptType]]], 
                                    document_seqs)
    if isinstance(guided_options_request, dict):
        if len(guided_options_request) > 1:
            raise ValueError(
                "You can only use one guided decoding but multiple is "
                f"specified: {guided_options_request}")
        guided_options_request = GuidedDecodingRequest(
            **guided_options_request)
    if sampling_params is None:
        # Use default sampling params.
        sampling_params = self.get_default_sampling_params()
    self._validate_and_add_requests(
        prompts=parsed_prompts,
        params=sampling_params,
        use_tqdm=use_tqdm,
        lora_request=lora_request,
        prompt_adapter_request=prompt_adapter_request,
        guided_options=guided_options_request,
        priority=priority,
        # /////
        document_seqs=parsed_document_seqs,
    )
    outputs = self._run_engine(use_tqdm=use_tqdm)
    return self.engine_class.validate_outputs(outputs, RequestOutput)


def _validate_and_add_requests(
    self,
    prompts: Union[PromptType, Sequence[PromptType]],
    params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams,
                  Sequence[PoolingParams]],
    *,
    use_tqdm: bool,
    lora_request: Optional[Union[Sequence[LoRARequest], LoRARequest]],
    prompt_adapter_request: Optional[PromptAdapterRequest],
    tokenization_kwargs: Optional[dict[str, Any]] = None,
    guided_options: Optional[GuidedDecodingRequest] = None,
    priority: Optional[list[int]] = None,
    # ///////////////
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
            "The lengths of prompts and document_seqs must be the same." \
            f" {len(prompts)} != {len(document_seqs)}"
            
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
    it = prompts
    if use_tqdm:
        it = tqdm(it, desc="Adding requests")
    for i, prompt in enumerate(it):
        self._add_request(
            prompt,
            params[i] if isinstance(params, Sequence) else params,
            tokenization_kwargs=tokenization_kwargs,
            lora_request=lora_request[i] if isinstance(
                lora_request, Sequence) else lora_request,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority[i] if priority else 0,
            # ////////
            document_seq=document_seqs[i] if document_seqs else None,
        )
        
def _add_request(
    self,
    prompt: PromptType,
    params: Union[SamplingParams, PoolingParams],
    tokenization_kwargs: Optional[dict[str, Any]] = None,
    lora_request: Optional[LoRARequest] = None,
    prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    priority: int = 0,
    # /////////
    document_seq: Optional[Sequence[PromptType]] = None,
) -> None:
    request_id = str(next(self.request_counter))
    self.llm_engine.add_request(
        request_id,
        prompt,
        params,
        lora_request=lora_request,
        tokenization_kwargs=tokenization_kwargs,
        prompt_adapter_request=prompt_adapter_request,
        priority=priority,
        # //////////
        document_seq=document_seq,
    )