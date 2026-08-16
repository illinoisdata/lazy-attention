"""
In this file, we inject several functions to InputPreprocessor class to 
support the preprocessing of document_seq ([doc1, ..., docn]). 

Specifically, we need to do the following:
1. Convert each document sequence to a list of token ids.
2. Padding the document sequences to be a multiple of the block size.
3. Generate a hash value for the document sequence (it is used to identify 
the query following the document sequence), using sha256.

Changed by Haocheng at 2025/09/03
"""

import time
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional, Union

from vllm.config import VllmConfig
from vllm.inputs import ProcessorInputs, PromptType, SingletonInputs
from vllm.inputs.parse import split_enc_dec_inputs
from vllm.inputs.preprocess import InputPreprocessor
from vllm.lora.request import LoRARequest
from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalKwargs,
                             MultiModalRegistry)
from vllm.multimodal.inputs import PlaceholderRange
from vllm.multimodal.processing import EncDecMultiModalProcessor
from vllm.multimodal.utils import merge_and_sort_multimodal_metadata
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.mm_input_cache import MirroredProcessingCache
from vllm.v1.structured_output.backend_guidance import (
    validate_guidance_grammar)
from vllm.v1.structured_output.backend_xgrammar import (
    validate_xgrammar_grammar)

from vllm.utils import cdiv, sha256  # to distinguish document_seq
from vllm.v1.engine.processor import Processor

from lazy.engine import EngineCoreRequest
from lazy.utils.rotation import (MAX_PACKED_Q_OFFSET,
                                 max_rotation_offset)
import logging
import os

logger = logging.getLogger(__name__)


def _hash_document_seq(
    documents_token_ids_padded: Optional[list[list[int]]], ) -> Optional[str]:
    """The seed for the query blocks' hash chain: the documents *and* the split.

    The boundaries are part of the identity. Two document sets that flatten to
    the same token sequence -- one 32-token document, or two 16-token ones --
    are encoded block-diagonally into different KV, so seeding the same chain
    would let a query block computed against one be reused for the other.

    Hashing a tuple of tuples encodes that explicitly. The previous
    `chain.from_iterable(...)` did distinguish them in practice, but only
    because `sha256` pickles the unconsumed iterator and the pickle happens to
    embed the nesting -- not a property to leave a correctness claim resting on.
    """
    if not documents_token_ids_padded:
        return None
    return hex(sha256(tuple(tuple(doc)
                            for doc in documents_token_ids_padded)))[2:]


def _validate_document_request(
    params: Union[SamplingParams, PoolingParams],
    lora_request: Optional[LoRARequest],
) -> None:
    """Reject the request shapes the document path cannot serve correctly.

    Both are rejected here, at the one point every entry -- sync, async, and
    whatever calls `add_request` directly -- passes through.

    Pooling: a document request is spawned by deep-copying the parent's
    `sampling_params` and setting `max_tokens = 1`. A pooling parent has none,
    so the copy is `None` and the spawn raises `AttributeError` from inside the
    scheduler, long after the request was accepted.

    LoRA: the failure is silent, which is worse. Documents are hashed and
    populated through `LazyRequest.document_request`, which does not forward
    `lora_request` -- so the two sides agree and the parent *does* see its
    documents as ready. But those blocks were computed by the base model, while
    the parent's query and decoding run with the adapter, mixing incompatible
    layer states into one attention. Nothing errors; the answer is just wrong.
    """
    if not isinstance(params, SamplingParams):
        raise ValueError(
            "LazyAttention does not support documents with pooling requests: "
            "a document request is a prefill that needs sampling_params. Send "
            "the documents inline in the prompt instead.")
    if lora_request is not None:
        raise ValueError(
            "LazyAttention does not support documents with LoRA: document KV "
            "is computed by the base model, so reusing it under an adapter "
            "would silently mix base and adapter states. Send the documents "
            "inline in the prompt, or drop the LoRA request.")


def _validate_rotation_offsets(document_lens: list[int],
                               document_lens_padded: list[int]) -> None:
    """Reject document sets whose rotation offsets the packed table cannot hold.

    The packed block table gives q_offset 16 bits; past that the shifted value
    carries into the physical-block field and the request is answered against
    the wrong blocks, silently. The model runner keeps the same check as a last
    line of defence, but by then the engine is committed -- raising there takes
    down every concurrent request, so the decision belongs here, where it is
    one request's error.
    """
    needed = max_rotation_offset(document_lens, document_lens_padded)
    if needed > MAX_PACKED_Q_OFFSET:
        raise ValueError(
            f"LazyAttention cannot serve these documents: they need a rotation "
            f"offset of {needed}, past the {MAX_PACKED_Q_OFFSET} the packed "
            f"block table's 16-bit field holds. The bound is the total padding "
            f"plus the lengths of every document but the last "
            f"(a single document is always offset 1, however long it is), so "
            f"send fewer or shorter leading documents, or inline them in the "
            f"prompt.")


class LazyProcessor(Processor):

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
        data_parallel_rank: Optional[int] = None,
        # For lazy attention
        document_seq: Optional[Sequence[PromptType]] = None,
        block_size: Optional[int] = None,
    ) -> tuple[Optional[str], EngineCoreRequest]:

        # TODO(woosuk): Support pooling models.
        # TODO(woosuk): Support encoder-decoder models.
        self._validate_lora(lora_request)
        self._validate_params(params, lora_request)
        if trace_headers is not None:
            raise ValueError("V1 does not support tracing yet.")
        if prompt_adapter_request is not None:
            raise ValueError("V1 does not support prompt_adapter_request.")

        data_parallel_size = self.vllm_config.parallel_config.data_parallel_size
        if data_parallel_rank is not None and not (0 <= data_parallel_rank <
                                                   data_parallel_size):
            raise ValueError(f"data_parallel_rank {data_parallel_rank} "
                             f"is out of range [0, {data_parallel_size}).")

        if arrival_time is None:
            arrival_time = time.time()

        # NOTE(haocheng):
        # for lazy attention, we pad the document sequence to be a
        # multiple of the block size, and hash the document sequence.
        documents_token_ids_padded = None
        document_lens = None # the original lengths of each document sequence before padding
        document_lens_padded = None
        if document_seq is not None:
            _validate_document_request(params, lora_request)
            documents_token_ids_padded = []
            document_lens = []
            document_lens_padded = []
            for document in document_seq:
                processed_document_seq: ProcessorInputs = self.input_preprocessor.preprocess(
                    document,
                    lora_request=lora_request,
                    prompt_adapter_request=prompt_adapter_request,
                    return_mm_hashes=self.use_hash,
                )
                encoder_inputs, decoder_inputs = split_enc_dec_inputs(processed_document_seq)
                
                # check the length of the document sequence
                # NOTE!!!(haocheng): Exclude special tokens? It only used by test hqa accuracy now.
                doc_token_ids = decoder_inputs["prompt_token_ids"]
                doc_token_ids  = doc_token_ids[1:] # remove the starting bos token
                document_lens.append(len(doc_token_ids))
                nearest_multiple = ((document_lens[-1] + block_size - 1) // block_size) * block_size
                document_lens_padded.append(nearest_multiple)
                pad_length = nearest_multiple - document_lens[-1]

                if pad_length > 0:
                    try:
                        pad_token = self.tokenizer.tokenizer.pad_token
                    except AttributeError:
                        logger.warning("Warning: AttributeError - no tokenizer.pad_token, use '<pad>'.")
                        pad_token = "<pad>"
                        
                    if pad_token is None:
                        logger.warning("Warning: has tokenizer.pad_token but it is None, use '<pad>'.")
                        pad_token = "<pad>"
                    
                    pad_token_ids = self.tokenizer.tokenizer(pad_token,
                                                             add_special_tokens=False)["input_ids"]  # skip the first token
                    if len(pad_token_ids) > 1:
                        logger.warning(f"Warning: the pad token id is not a single token. {pad_token_ids}")
                    pad_token_id = pad_token_ids[-1]
                    # We use right padding
                    doc_token_ids = doc_token_ids + [pad_token_id] * pad_length
                assert len(doc_token_ids) % block_size == 0, "The length of the document sequence is not a multiple of the block size."

                documents_token_ids_padded.append(doc_token_ids)

            _validate_rotation_offsets(document_lens, document_lens_padded)
        
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
        eos_token_id = self.input_preprocessor.get_eos_token_id(lora_request)

        self._validate_model_inputs(processed_inputs, lora_request)

        encoder_inputs, decoder_inputs = split_enc_dec_inputs(processed_inputs)
        # NOTE!!!(haocheng): Exclude special tokens? It only used by test hqa accuracy now.
        if document_seq is not None:
            decoder_inputs["prompt_token_ids"] = decoder_inputs["prompt_token_ids"][1:]

        # TODO: Impl encoder-decoder
        if encoder_inputs is not None:
            raise NotImplementedError

        sampling_params = None
        pooling_params = None
        if isinstance(params, SamplingParams):
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
        else:
            pooling_params = params.clone()

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
        
        # For DEBUGGING
        if os.environ.get("DEBUG_LAZY") == "1":
            logger.info(f"Document seq token ids padded: {documents_token_ids_padded}")
            logger.info(f"Document lens: {document_lens}")
            logger.info(f"Document lens padded: {document_lens_padded}")
        

        return decoder_inputs.get("prompt"), EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=decoder_inputs["prompt_token_ids"],
            mm_inputs=sorted_mm_inputs,
            mm_hashes=sorted_mm_hashes,
            mm_placeholders=sorted_mm_positions,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            eos_token_id=eos_token_id,
            arrival_time=arrival_time,
            lora_request=lora_request,
            cache_salt=decoder_inputs.get("cache_salt"),
            data_parallel_rank=data_parallel_rank,
            priority=priority,
            # For lazy attention
            documents_token_ids_padded=documents_token_ids_padded, # (num_documents, seq_len)
            document_lens=document_lens, # (num_documents,)
            document_lens_padded=document_lens_padded, # (num_documents,)
            document_seq_hash=_hash_document_seq(documents_token_ids_padded),
        )


def apply_patch():
    """Apply the patch to the InputPreprocessor class."""
    import vllm.v1.engine.processor
    vllm.v1.engine.processor.Processor.process_inputs = process_inputs
