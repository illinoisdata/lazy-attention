""" In this file, we inject several functions to InputPreprocessor class to 
support the preprocessing of document_seq ([doc1, ..., docn]). 

Specifically, we need to do the following:
1. Convert each document sequence to a list of token ids.
2. Padding the document sequences to be a multiple of the block size.
"""

import time
from collections.abc import Mapping, Sequence
from typing import Optional, Union, Any

from vllm.config import VllmConfig
from vllm.inputs import ProcessorInputs, PromptType
from vllm.inputs.parse import split_enc_dec_inputs
from vllm.inputs.preprocess import InputPreprocessor
from vllm.lora.request import LoRARequest
from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalKwargs,
                             MultiModalRegistry)
from vllm.multimodal.inputs import PlaceholderRange
from vllm.multimodal.utils import merge_and_sort_multimodal_metadata
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.utils import cdiv, sha256

from minidrag.engine import EngineCoreRequest

# class Processor:


def process_inputs(
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
    # For dynamic rag
    document_seq: Optional[Sequence[PromptType]] = None,
    block_size: Optional[int] = None,
) -> EngineCoreRequest:
    # TODO(woosuk): Support pooling models.
    # TODO(woosuk): Support encoder-decoder models.
    self._validate_lora(lora_request)
    self._validate_params(params, lora_request)
    if priority != 0:
        raise ValueError("V1 does not support priority yet.")
    if trace_headers is not None:
        raise ValueError("V1 does not support tracing yet.")
    if prompt_adapter_request is not None:
        raise ValueError("V1 does not support prompt_adapter_request.")
    
    if arrival_time is None:
        arrival_time = time.time()
        
    # Process inputs, which includes:
    # 1. Tokenize text prompt, with LoRA request if one exists.
    # 2. For multimodal models with a merged preprocessor, preprocess
    #   multimodal data and expand prompt token ids accordingly.
    # 3. Apply prompt adapter to prompt token ids if one exists.
    processed_inputs: ProcessorInputs = self.input_preprocessor.preprocess(
        prompt,
        tokenization_kwargs=tokenization_kwargs,
        lora_request=lora_request,
        prompt_adapter_request=prompt_adapter_request,
        return_mm_hashes=self.use_hash,
    )
    from vllm.platforms import current_platform
    current_platform.validate_request(
        prompt=prompt,
        params=params,
        processed_inputs=processed_inputs,
    )
    
    # ////////////////////////////////////////
    documents_token_ids = None
    documents_hash = None
    document_seq_hash = None
    real_doc_lens = None
    if document_seq is not None:
        documents_token_ids = []
        documents_hash = []
        real_doc_lens = []
        for doc in document_seq:
            processed_document_seq: ProcessorInputs = self.input_preprocessor.preprocess(
                doc,
                lora_request=lora_request,
                prompt_adapter_request=prompt_adapter_request,
                return_mm_hashes=self.use_hash,
            )
            encoder_inputs, decoder_inputs = split_enc_dec_inputs(processed_document_seq)
            # check the length of the document sequence
            doc_token_ids = decoder_inputs["prompt_token_ids"]
            real_doc_lens.append(len(doc_token_ids))
            nearest_multiple = ((len(doc_token_ids) + block_size - 1) // block_size) * block_size
            # pad the document sequence
            # left padding
            pad_length = nearest_multiple - len(doc_token_ids)
            if pad_length > 0:
                try:
                    pad_token = self.tokenizer.tokenizer.pad_token
                except AttributeError:
                    print("Warning: no pad token in the tokenizer, use '<pad>'.")
                if pad_token is None: #if tokenizer.tokenizer.pad_token is None
                    pad_token = "<pad>"
                pad_token_ids = self.tokenizer.tokenizer(pad_token, 
                                                         add_special_tokens=False)["input_ids"]  # skip the first token
                if len(pad_token_ids) > 1:
                    print(f"Warning: the pad token id is not a single token. {pad_token_ids}")
                pad_token_id = pad_token_ids[-1]
                # left padding
                # doc_token_ids = [pad_token_id] * pad_length + doc_token_ids
                # right padding
                doc_token_ids = doc_token_ids + [pad_token_id] * pad_length
            assert len(doc_token_ids) % block_size == 0, "The length of the document sequence is not a multiple of the block size."

            documents_token_ids.append(doc_token_ids)
            # hash the document sequence
            documents_hash.append(str(sha256(tuple(doc_token_ids))))
            document_seq_hash = str(sha256(tuple(documents_hash)))
    # ////////////////////////////////////////

    eos_token_id = self.input_preprocessor.get_eos_token_id(lora_request)
    
    self._validate_model_inputs(processed_inputs, lora_request)
    
    encoder_inputs, decoder_inputs = split_enc_dec_inputs(processed_inputs)
    
    # TODO: Impl encoder-decoder
    if encoder_inputs is not None:
        raise NotImplementedError
    
    assert isinstance(params, SamplingParams)
    # TODO: can we avoid cloning here in multiproc case?
    sampling_params = params.clone()
    # If unset max tokens, then generate up to the max_model_len.
    if sampling_params.max_tokens is None:
        sampling_params.max_tokens = (
            self.model_config.max_model_len -
            len(decoder_inputs["prompt_token_ids"]))
    sampling_params.update_from_generation_config(
        self.generation_config_fields, eos_token_id)
    sampling_params.update_from_tokenizer(
        self.tokenizer.get_lora_tokenizer(lora_request))
    
    # Multimodal related.
    sorted_mm_inputs: Optional[Sequence[Optional[MultiModalKwargs]]] = None
    sorted_mm_positions: Optional[list[PlaceholderRange]] = None
    sorted_mm_hashes: Optional[list[str]] = None
    if decoder_inputs["type"] == "multimodal":
        decoder_mm_inputs = decoder_inputs["mm_kwargs"]
        # Merge and flatten multimodal placeholders, hashes and inputs
        # from dictionaries to lists, and sort them by each item's position
        # in the input sequence.
        (
            sorted_item_modalities,
            sorted_mm_positions,
            sorted_mm_hashes,
        ) = merge_and_sort_multimodal_metadata(
            decoder_inputs["mm_placeholders"],
            decoder_inputs["mm_hashes"] if self.use_hash else None,
        )
        
        # The output of merged multi-modal processor (`decoder_mm_inputs`)
        # is a single MultiModalKwargs for all items from all modalities.
        # This code flattens kwargs for individual items in a list and
        # sorts them by each item's position in the input sequence if there
        # are multiple modalities.
        unique_modalities = set(sorted_item_modalities)
        if len(unique_modalities) > 1:
            orig_sorted_mm_inputs = []
            used_indices = {modality: 0 for modality in unique_modalities}
            
            for modality in sorted_item_modalities:
                items = decoder_mm_inputs.get_items(modality)
                item = items[used_indices[modality]]
                orig_sorted_mm_inputs.append(
                    MultiModalKwargs.from_items([item]))
                used_indices[modality] += 1
        else:
            orig_sorted_mm_inputs = [
                MultiModalKwargs.from_items([item]) for item in
                decoder_mm_inputs.get_items(sorted_item_modalities[0])
            ]
            
        if sorted_mm_hashes is not None:
            sorted_mm_inputs = self.mm_input_cache_client.get_and_update_p0(
                orig_sorted_mm_inputs, sorted_mm_hashes)
        else:
            sorted_mm_inputs = orig_sorted_mm_inputs
            
    return decoder_inputs.get("prompt"), EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=decoder_inputs["prompt_token_ids"],
        mm_inputs=sorted_mm_inputs,
        mm_hashes=sorted_mm_hashes,
        mm_placeholders=sorted_mm_positions,
        sampling_params=sampling_params,
        eos_token_id=eos_token_id,
        arrival_time=arrival_time,
        lora_request=lora_request,
        cache_salt=decoder_inputs.get("cache_salt"),
        # For dynamic rag
        real_doc_lens=real_doc_lens,
        documents_token_ids=documents_token_ids,
        document_seq_hash=document_seq_hash,
    )


def apply_patch():
    """Apply the patch to the InputPreprocessor class."""
    import vllm.v1.engine.processor
    vllm.v1.engine.processor.Processor.process_inputs = process_inputs