from typing import Dict, List

import numpy as np
import torch

from vllm.attention.backends.xformers import XFormersMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata, SequenceGroupToSample
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SequenceData
from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata


class VllmMetadataBuilder:
    @staticmethod
    def build_batch_model_input(batch_size: int,
                                 batch_query_lens: List[int],#for encoding, query's token length. # for decoding, all 1 (because only decode 1 at a time)
                                 batch_context_lens: List[int],#for encoding, documents' token lengths. # for decoding, length of kv_cache
                                 batch_seq_lens: List[int],#for encoding, query_lens + documents' length. # for decoding, query_lens + documents' length + generated tokens length
                                 batch_prompt_token_ids: List[List[int]], #? only prompt, or ctx?
                                 batch_query_token_ids: List[List[int]],#? the query?
                                 batch_num_prefill_tokens: List[int],# for decode, all 0. for prefill, the number of tokens to be prefilled.
                                 batch_block_tables: List[List[int]],
                                 batch_slot_mapping: List[List[int]],
                                 batch_output_token_ids: List[List[int]], #not added by scheduler # used for batch decode to get the input token id
                                 req_ids: List[int],#seq id
                                 seq_ids: List[int],#seq id
                                 generator: torch.Generator,
                                 sampling_params: SamplingParams) -> ModelInputForGPUWithSamplingMetadata:
        # function copied from Haocheng's drag.py
        flag_prefill = all(np.array(batch_num_prefill_tokens) > 0)

        # check
        if not flag_prefill:
            assert all(np.array(batch_num_prefill_tokens) ==
                       0), "Support continuous batching only"

        # build a tensor dict for model input
        tensor_dict = {}
        if flag_prefill:
            # concat batch_query_token_ids
            input_tokens = torch.cat(batch_query_token_ids, dim=0).cuda()
            batch_input_positions = [torch.arange(query_len) + context_len
                                     for query_len, context_len
                                     in zip(batch_query_lens, batch_context_lens)]#positions of query in the sequence

            input_positions = torch.cat(batch_input_positions, dim=0).cuda()
            slot_mapping = torch.cat(batch_slot_mapping, dim=0).cuda()
            num_prefill_tokens = sum(batch_num_prefill_tokens)
            batch_seq_data = []
            for batch_id in range(batch_size):
                seq_data = SequenceData.from_seqs(
                    prompt_token_ids=batch_prompt_token_ids[batch_id],
                )
                batch_seq_data.append({seq_ids[batch_id]: seq_data})

            tensor_dict = {
                "input_tokens": input_tokens,
                "input_positions": input_positions,
                "seq_lens": batch_seq_lens,
                "query_lens": batch_query_lens,
                "lora_requests": set(),
                "lora_mapping": None,
                "attn_metadata": VllmMetadataBuilder._build_attn_metadata(
                    num_prefill_tokens=num_prefill_tokens,
                    num_decode_tokens=0,
                    slot_mapping=slot_mapping,
                    ctx_lens=batch_context_lens,
                    seq_lens=batch_seq_lens,
                    block_tables=batch_block_tables,
                ),

                "multi_modal_kwargs": dict(),
                "prompt_adapter_mapping": None,
                "prompt_adapter_requests": set(),
                "virtual_engine": 0,
                # TODO(haocheng): used for beam search, now do nothing
                "request_ids_to_seq_ids": {str(req_id): seq_id
                                           for req_id, seq_id in zip(req_ids, seq_ids)},
                "finished_requests_ids": [],
                "virtual_engine": 0,
                "async_callback": None,
                "seq_group_metadata_list": None,
                "scheduler_outputs": None,
                "sampling_metadata": VllmMetadataBuilder._build_sampling_metadata_batch(
                    batch_seq_ids=seq_ids,
                    sampling_params=sampling_params,
                    batch_seq_data=batch_seq_data,
                    seq_lens=batch_seq_lens,
                    query_lens=batch_query_lens,
                    generator=generator,
                    is_prompt=True,
                ),
                "is_prompt": True,
            }
        else:
            batch_size = len(batch_seq_lens)
            input_tokens = []
            for batch_id in range(batch_size):
                input_tokens.append(batch_output_token_ids[batch_id][-1])
            input_tokens = torch.tensor(input_tokens).cuda()
            input_positions = (np.array(batch_seq_lens) - np.array(batch_query_lens)).tolist()
            input_positions = torch.tensor(input_positions).cuda()
            slot_mapping = torch.cat(batch_slot_mapping, dim=0).cuda()
            batch_seq_data = []
            for batch_id in range(batch_size):
                seq_data = SequenceData.from_seqs(
                    prompt_token_ids=batch_prompt_token_ids[batch_id],
                    output_token_ids=batch_output_token_ids[batch_id],
                )
                batch_seq_data.append({seq_ids[batch_id]: seq_data})
            tensor_dict = {
                "input_tokens": input_tokens,
                "input_positions": input_positions,
                "seq_lens": batch_seq_lens,
                "query_lens": batch_query_lens,
                "lora_requests": set(),
                "lora_mapping": None,
                "attn_metadata": VllmMetadataBuilder._build_attn_metadata(
                    num_prefill_tokens=0,
                    num_decode_tokens=batch_size, # each request decode one token
                    slot_mapping=slot_mapping,
                    ctx_lens=batch_context_lens,
                    seq_lens=batch_seq_lens,
                    block_tables=batch_block_tables,
                ),

                "multi_modal_kwargs": dict(),
                "prompt_adapter_mapping": None,
                "prompt_adapter_requests": set(),
                "virtual_engine": 0,
                # TODO(haocheng): used for beam search, now do nothing
                "request_ids_to_seq_ids": {str(req_id): seq_id
                                           for req_id, seq_id in zip(req_ids, seq_ids)},
                "finished_requests_ids": [],
                "virtual_engine": 0,
                "async_callback": None,
                "seq_group_metadata_list": None,
                "scheduler_outputs": None,
                "sampling_metadata": VllmMetadataBuilder._build_sampling_metadata_batch(
                    batch_seq_ids=seq_ids,
                    sampling_params=sampling_params,
                    batch_seq_data=batch_seq_data,
                    seq_lens=batch_seq_lens,
                    query_lens=batch_query_lens,
                    generator=generator,
                    is_prompt=False,
                ),
                "is_prompt": False,
            }

        model_input = ModelInputForGPUWithSamplingMetadata.from_broadcasted_tensor_dict(
            tensor_dict=tensor_dict,
            attn_backend="XFormersBackend",
        )
        return model_input
    
    @staticmethod
    def _build_sampling_metadata_batch(
        batch_seq_ids: List[List[int]],
        sampling_params: SamplingParams,
        batch_seq_data: List[Dict[int, SequenceData]],
        seq_lens: List[int],
        query_lens: List[int],
        generator: torch.Generator,
        is_prompt: bool,
    ) -> SamplingMetadata:
        query_start_loc = (np.array([sum(query_lens[:i+1]) for i in range(0, len(query_lens))]) - 1).tolist()
        selected_token_indices = torch.tensor(query_start_loc, dtype=torch.int32).cuda()

        seq_groups = []
        for i, (seq_ids, seq_data, seq_len, query_len) in enumerate(zip(batch_seq_ids, batch_seq_data, seq_lens, query_lens)):
            seq_groups.append(
                SequenceGroupToSample(
                    seq_ids=seq_ids,
                    sampling_params=sampling_params,
                    seq_data=seq_data,
                    seq_len=seq_len,
                    query_len=query_len,
                    generator=generator,
                    is_prompt=is_prompt,
                    prompt_logprob_indices=[],
                    sample_indices=[i],
                )
            )

        return SamplingMetadata(
            seq_groups=seq_groups,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices={
                SamplingType.GREEDY: torch.tensor([], dtype=torch.int32).cuda(),
                SamplingType.RANDOM: torch.tensor([], dtype=torch.int32).cuda(),
                SamplingType.RANDOM_SEED: torch.tensor([], dtype=torch.int32).cuda(),
            },
            num_prompts=len(seq_groups),
        )

    @staticmethod
    def _build_attn_metadata(
        num_prefill_tokens: int,
        num_decode_tokens: int,
        slot_mapping: List[int],
        ctx_lens: List[int],
        seq_lens: List[int],
        block_tables: List[List[int]],
    ):
        batch_size = len(seq_lens)

        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64).cuda()
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32).cuda()
        ctx_lens_tensor = torch.tensor(ctx_lens, dtype=torch.int32).cuda()
        block_tables = torch.tensor(block_tables, dtype=torch.int32).cuda()

        query_lens = (np.array(seq_lens) - np.array(ctx_lens)).tolist()
        query_start_loc = [0, ] + [sum(query_lens[:i+1]) for i in range(0, len(query_lens))]
        seq_start_loc = [0, ] + [sum(seq_lens[:i+1]) for i in range(0, len(seq_lens))]

        if num_prefill_tokens > 0:
            attn_metadata = XFormersMetadata(
                num_prefills=batch_size,
                num_prefill_tokens=num_prefill_tokens,
                num_decode_tokens=0,
                slot_mapping=slot_mapping,
                seq_lens=seq_lens,
                seq_lens_tensor=seq_lens_tensor,
                seq_start_loc=torch.tensor(seq_start_loc).cuda(),
                max_query_len=max(query_lens),
                max_prefill_seq_len=max(seq_lens),
                max_decode_seq_len=0,
                query_start_loc=torch.tensor(query_start_loc).cuda(),
                context_lens_tensor=ctx_lens_tensor,
                block_tables=block_tables,
                use_cuda_graph=False,
                # Begin encoder & cross attn fields below...
                encoder_seq_lens=None,
                encoder_seq_lens_tensor=None,
                max_encoder_seq_len=None,
                cross_slot_mapping=None,
                cross_block_tables=None,
            )
        else:
            assert num_decode_tokens > 0
            attn_metadata = XFormersMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decode_tokens=num_decode_tokens,
                slot_mapping=slot_mapping,
                seq_lens=seq_lens,
                seq_lens_tensor=seq_lens_tensor,
                max_query_len=max(query_lens),
                max_prefill_seq_len=0,
                max_decode_seq_len=max(seq_lens),
                query_start_loc=torch.tensor(query_start_loc).cuda(),
                context_lens_tensor=ctx_lens_tensor,
                block_tables=block_tables,
                use_cuda_graph=False,
                # Begin encoder & cross attn fields below...
                encoder_seq_lens=None,
                encoder_seq_lens_tensor=None,
                max_encoder_seq_len=None,
                cross_slot_mapping=None,
                cross_block_tables=None,
            )
        return attn_metadata
