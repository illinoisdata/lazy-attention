from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict, Set

import torch
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SequenceData
from vllm.model_executor.sampling_metadata import SamplingMetadata, SequenceGroupToSample
from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

from drag.logging import logger
try:
    from vllm.attention.backends.xformers import XFormersMetadata
except Exception as e:
    logger.error(f"Failed to import XFormersMetadata: {e}")


@dataclass
class SeqResource:
    """Record the kv-cache resources for each sequence
    block_table: the block table for the sequence
    slot_mapping: the slot mapping in the last block

    e.g., when prefilling,
    block_table = [0, 1], slot_mapping = [16] and the block size is 16,
    then the slots for prefilling are [0, 1, 2, ..., 16]

    when decoding,
    block_table = [0, 1], slot_mapping = [17] and the block size is 16,
    the slot for decoding is [17]
    """
    seq_id: int
    block_table: torch.Tensor  # device=cuda, dtype=torch.int32
    slot_mapping: torch.Tensor  # device=cuda


@dataclass
class PreModelInput:
    """Use to simplify the model input for the model"""
    batch_size: int
    query_lens: List[int]
    context_lens: List[int]
    seq_lens: List[int]
    query_token_lists: List[torch.Tensor]  # device=cuda
    prompt_token_lists: List[torch.Tensor]
    seq_resources: List[SeqResource]
    sampling_params: SamplingParams
    # not used for prefill
    output_token_lists: Optional[List[torch.Tensor]] = None


@dataclass
class ModelInput:
    # device=cuda
    input_tokens: Optional[torch.Tensor] = None
    # device=cuda
    input_positions: Optional[torch.Tensor] = None
    seq_lens: Optional[torch.Tensor] = None
    query_lens: Optional[torch.Tensor] = None
    attn_metadata: Optional[Dict[str, Any]] = None
    sampling_metadata: Optional[Dict[str, Any]] = None
    request_ids_to_seq_ids: Optional[Dict[str, int]] = None
    is_prompt: Optional[bool] = None

    # ignored fields
    finished_requests_ids: List[str] = field(default_factory=list)
    lora_mapping: Optional[Dict[str, Any]] = None
    lora_requests: Set[str] = field(default_factory=set)
    prompt_adapter_mapping: Optional[Dict[str, Any]] = None
    prompt_adapter_requests: Set[str] = field(default_factory=set)
    multi_modal_kwargs: Dict[str, Any] = field(default_factory=dict)
    virtual_engine: int = 0
    async_callback: Optional[callable] = None
    seq_group_metadata_list: Optional[List[Any]] = None
    scheduler_outputs: Optional[Any] = None


class ModelInputFactory:
    """The role of this class is to build the model input for the model"""

    def __init__(self, default_config: Optional[Dict[str, Any]] = None):
        """
        initialize the factory class

        Args:
            default_config: optional default config dict
        """
        self.default_config = default_config or {}

    def _build_default_model_input(self) -> ModelInput:
        """
        create a model input with default values

        Returns:
            ModelInput: with default values
        """
        model_input = ModelInput()

        # apply any custom default config
        for key, value in self.default_config.items():
            if hasattr(model_input, key):
                setattr(model_input, key, value)

        return model_input

    def _build_attention_metadata(self, input: PreModelInput, is_prefill: bool) -> XFormersMetadata:
        """Build attention metadata with different configurations based on prefill/decode phase"""
        # Note: suffix "_tensor" and "_start_loc" are on cuda device and dtype=torch.int32
        device = input.query_token_lists[0].device
        # Prepare common data
        slot_mapping = torch.cat(
            [seq_res.slot_mapping.clone() for seq_res in input.seq_resources])
        # TODO(haocheng): check if the block table is scalar, optimize the logic
        block_values = [seq_res.block_table.clone().detach().tolist() for seq_res in input.seq_resources]
        block_tables = torch.tensor(block_values, device=device, dtype=torch.int32)

        # Create base metadata
        metadata = {
            "slot_mapping": slot_mapping,
            "seq_lens": input.seq_lens,
            "seq_lens_tensor": torch.tensor(input.seq_lens, dtype=torch.int32, device=device),
            "context_lens_tensor": torch.tensor(input.context_lens, dtype=torch.int32, device=device),
            "block_tables": block_tables,
            "use_cuda_graph": False,
            # Encoder and cross-attention fields
            "encoder_seq_lens": None,
            "encoder_seq_lens_tensor": None,
            "max_encoder_seq_len": None,
            "cross_slot_mapping": None,
            "cross_block_tables": None,
        }

        query_lens = input.query_lens
        seq_lens = input.seq_lens
        prefill_tokens = sum(min(q_len, 512) for q_len in query_lens)

        # Calculate query start positions
        # device=cuda, dtype=torch.int32
        query_start = torch.zeros(
            input.batch_size + 1, dtype=torch.int32, device=device)
        query_lens_tensor = torch.tensor(query_lens, dtype=torch.int32)
        query_start[1:] = torch.cumsum(query_lens_tensor, dim=0)
        # device=cuda, dtype=torch.int32
        seq_start = torch.zeros(input.batch_size + 1,
                                dtype=torch.int32, device=device)
        seq_start_tensor = torch.tensor(seq_lens, dtype=torch.int32)
        seq_start[1:] = torch.cumsum(seq_start_tensor, dim=0)

        # Add phase-specific fields
        if is_prefill:
            # Prefill phase
            metadata.update({
                "num_prefills": input.batch_size,
                "num_prefill_tokens": prefill_tokens,
                "num_decode_tokens": 0,
                "max_query_len": max(query_lens) if query_lens else 0,
                "max_prefill_seq_len": max(input.seq_lens) if input.seq_lens else 0,
                "max_decode_seq_len": 0,
                "seq_start_loc": seq_start,
                "query_start_loc": query_start,
            })
        else:
            # Decode phase
            metadata.update({
                "num_prefills": 0,
                "num_prefill_tokens": 0,
                "num_decode_tokens": input.batch_size,
                "max_query_len": 1,
                "max_prefill_seq_len": 0,
                "max_decode_seq_len": max(input.seq_lens) if input.seq_lens else 0,
                "seq_start_loc": seq_start,
                "query_start_loc": torch.arange(input.batch_size + 1, device=device, dtype=torch.int32),
            })

        return XFormersMetadata(**metadata)

    def _build_sampling_metadata(self, input: PreModelInput, is_prefill: bool) -> SamplingMetadata:
        device = input.query_token_lists[0].device
        metadata = {
            "selected_token_indices": torch.cumsum(torch.tensor(input.query_lens, device=device), dim=0) - 1,
            "categorized_sample_indices": {
                SamplingType.GREEDY: torch.arange(input.batch_size, dtype=torch.int32, device=device),
                SamplingType.RANDOM: torch.tensor([], dtype=torch.int32, device=device),
                SamplingType.RANDOM_SEED: torch.tensor([], dtype=torch.int32, device=device),
            },
            "num_prompts": input.batch_size,
        }

        metadata["seq_groups"] = []
        for i, seq_res in enumerate(input.seq_resources):
            # Create generator on CPU first, then set device
            generator = torch.Generator(device=device)
            generator.manual_seed(2024)

            seq_group = SequenceGroupToSample(
                seq_ids=[seq_res.seq_id],
                sampling_params=input.sampling_params,
                seq_data={seq_res.seq_id: SequenceData.from_seqs(
                    prompt_token_ids=input.prompt_token_lists[i],
                )
                },
                seq_len=input.seq_lens[i],
                query_len=input.query_lens[i],
                generator=generator,
                is_prompt=True,
                prompt_logprob_indices=[],
                sample_indices=[i],
            )
            if not is_prefill:
                # decode phase
                seq_group.seq_data[seq_res.seq_id].output_token_ids = input.output_token_lists[i]
                seq_group.seq_data[seq_res.seq_id]._cumulative_logprob = torch.inf
                seq_group.seq_data[seq_res.seq_id]._num_computed_tokens = input.seq_lens[i] - 1
                seq_group.seq_len = None
                seq_group.is_prompt = False

            metadata["seq_groups"].append(seq_group)
        return SamplingMetadata(**metadata)

    def is_decode(self, input: PreModelInput) -> bool:
        if input.output_token_lists is None:
            return False
        is_decodes = [len(output_token_ids) >
                      0 for output_token_ids in input.output_token_lists]
        is_decode = all(is_decodes)
        is_prefill = all(not is_decode for is_decode in is_decodes)
        if not (is_decode or is_prefill):
            raise ValueError("different stages for batched requests")
        return is_decode

    def build(self, input: PreModelInput) -> ModelInput:
        """
        accept the pre-processed model input and build the full model input

        Args:
            pre_model_input: pre-processed model input

        Returns:
            ModelInput: configured model input
        """
        # TODO(haocheng): optimize the logic, prefill or decode
        is_prefill = not self.is_decode(input)
        device = input.query_token_lists[0].device

        model_input = self._build_default_model_input()

        model_input.input_tokens = torch.cat(
            input.query_token_lists, dim=0)  # device=cuda
        input_positions = [torch.arange(query_len, device=device) + context_len
                           for query_len, context_len
                           in zip(input.query_lens, input.context_lens)]  # device=cuda
        model_input.input_positions = torch.cat(input_positions, dim=0)

        model_input.seq_lens = input.seq_lens
        model_input.query_lens = input.query_lens
        # TODO(haocheng): support beam search, one request may have multiple seq_ids
        model_input.request_ids_to_seq_ids = {str(seq_res.seq_id): [seq_res.seq_id]
                                              for seq_res in input.seq_resources}
        model_input.is_prompt = is_prefill
        # key fields
        model_input.attn_metadata = self._build_attention_metadata(
            input, is_prefill=is_prefill)
        model_input.sampling_metadata = self._build_sampling_metadata(
            input, is_prefill=is_prefill)

        self._validate_model_input(model_input)

        return model_input

    def build_vllm_model_input(self, input: PreModelInput) -> ModelInputForGPUWithSamplingMetadata:
        model_input = self.build(input)
        return ModelInputForGPUWithSamplingMetadata(**model_input.__dict__)

    def _validate_model_input(self, model_input: ModelInput) -> None:
        pass

    def build_batch(self, pre_model_inputs: List[Any]) -> List[ModelInput]:
        return [self.build(pre_input) for pre_input in pre_model_inputs]
