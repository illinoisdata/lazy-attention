import pytest
import torch

from drag.factory import *

sampling_param = SamplingParams(max_tokens=2,
                                seed=2024,
                                temperature=0,
                                stop_token_ids=[128008, 128001])
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

"""
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
"""
query_token_lists = [
    torch.tensor([128000, 9906, 11, 856, 836, 374],
                 device=device),
    torch.tensor([128000, 791, 4872, 315, 279, 3723, 4273, 374],
                 device=device),
    torch.tensor([128000, 791, 6864, 315, 9822, 374],
                 device=device),
    torch.tensor([128000, 791, 3938, 315, 15592, 374],
                 device=device)
]

prompt_token_lists = [
    torch.tensor([128000, 9906, 11, 856, 836, 374]),
    torch.tensor([128000, 791, 4872, 315, 279, 3723, 4273, 374]),
    torch.tensor([128000, 791, 6864, 315, 9822, 374]),
    torch.tensor([128000, 791, 3938, 315, 15592, 374])
]

prefill_input = PreModelInput(
    batch_size=4,
    query_lens=[6, 8, 6, 6],
    context_lens=[0, 0, 0, 0],
    seq_lens=[6, 8, 6, 6],
    query_token_lists=query_token_lists,
    prompt_token_lists=prompt_token_lists,
    seq_resources=[SeqResource(seq_id=0,
                               block_table=torch.tensor(
                                   [0], device=device, dtype=torch.int32),
                               slot_mapping=torch.tensor([0, 1, 2, 3, 4, 5], device=device)),
                   SeqResource(seq_id=1,
                               block_table=torch.tensor(
                                   [1], device=device, dtype=torch.int32),
                               slot_mapping=torch.tensor([16, 17, 18, 19, 20, 21, 22, 23], device=device)),
                   SeqResource(seq_id=2,
                               block_table=torch.tensor(
                                   [2], device=device, dtype=torch.int32),
                               slot_mapping=torch.tensor([32, 33, 34, 35, 36, 37], device=device)),
                   SeqResource(seq_id=3,
                               block_table=torch.tensor(
                                   [3], device=device, dtype=torch.int32),
                               slot_mapping=torch.tensor([48, 49, 50, 51, 52, 53], device=device))
                   ],
    sampling_params=sampling_param,
)

# then decode one token
decode_input = PreModelInput(
    batch_size=4,
    query_lens=[1, 1, 1, 1],
    context_lens=[6, 8, 6, 6],
    seq_lens=[7, 9, 7, 7],
    query_token_lists=[torch.tensor([35266], device=device),
                       torch.tensor([279], device=device),
                       torch.tensor([264], device=device),
                       torch.tensor([10107], device=device)],
    prompt_token_lists=prompt_token_lists,
    output_token_lists=[torch.tensor([35266]),
                        torch.tensor([279]),
                        torch.tensor([264]),
                        torch.tensor([10107])],
    seq_resources=[SeqResource(seq_id=0,
                               block_table=torch.tensor(
                                   [0], device=device, dtype=torch.int32),
                               slot_mapping=torch.tensor([6], device=device)),
                   SeqResource(seq_id=1,
                               block_table=torch.tensor(
                                   [1], device=device, dtype=torch.int32),
                               slot_mapping=torch.tensor([24], device=device)),
                   SeqResource(seq_id=2,
                               block_table=torch.tensor(
                                   [2], device=device, dtype=torch.int32),
                               slot_mapping=torch.tensor([38], device=device)),
                   SeqResource(seq_id=3,
                               block_table=torch.tensor(
                                   [3], device=device, dtype=torch.int32),
                               slot_mapping=torch.tensor([54], device=device))
                   ],
    sampling_params=sampling_param,
)
# -------------------------------------------------------------------------------------------------


def assert_list_equal(actual_list, expected_list):
    assert len(actual_list) == len(expected_list)
    for i, (actual, expected) in enumerate(zip(actual_list, expected_list)):
        assert actual == expected, f"seq_lens[{i}] expected {expected}, got {actual}"


def test_factory_vllm_prefill():
    factory = ModelInputFactory()
    model_input = factory.build_vllm_model_input(prefill_input)
    # /////////////////////////////////////////////////////////////////////////
    expected_input_tokens = torch.tensor([
        128000, 9906, 11, 856, 836, 374,
        128000, 791, 4872, 315, 279, 3723, 4273, 374,
        128000, 791, 6864, 315, 9822, 374,
        128000, 791, 3938, 315, 15592, 374
    ], device=device)
    assert model_input.input_tokens.shape == expected_input_tokens.shape, \
        f"Token shape mismatch: {model_input.input_tokens.shape} vs {expected_input_tokens.shape}"
    assert torch.all(model_input.input_tokens == expected_input_tokens), \
        f"Token mismatch: \nActual: {model_input.input_tokens}\nExpected: {expected_input_tokens}"

    expected_input_positions = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3,
                                             4, 5], device=device)
    assert torch.all(model_input.input_positions == expected_input_positions), \
        f"Input positions mismatch: \nActual: {model_input.input_positions}\nExpected: {expected_input_positions}"

    assert_list_equal(model_input.seq_lens, [6, 8, 6, 6])
    assert_list_equal(model_input.query_lens, [6, 8, 6, 6])

    # check attention and sampling metadata
    # print("-" * 80)
    # print(model_input)
    # print("-" * 80)


def test_factory_vllm_decode():
    factory = ModelInputFactory()
    model_input = factory.build_vllm_model_input(decode_input)
    # /////////////////////////////////////////////////////////////////////////
    # print("-" * 80)
    # print(model_input)
    # print("-" * 80)


def test_output():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    from vllm import LLM
    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
              gpu_memory_utilization=0.9,
              enforce_eager=True,
              enable_prefix_caching=True,
              )
    outputs = llm.generate(prompts, sampling_param)

if __name__ == "__main__":
    test_factory_vllm_prefill()
    test_factory_vllm_decode()
    test_output()
    from torch import tensor

# import array

# a = SequenceGroupToSample(seq_ids=[3], sampling_params=SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0, top_p=1.0, top_k=-1, min_p=0.0, seed=2024, stop=[], stop_token_ids=[128008, 128001], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=2, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), guided_decoding=None, seq_data={3: SequenceData(prompt_token_ids=array('l', [128000, 791, 3938, 315, 15592, 374]), output_token_ids=(), cumulative_logprob=0.0, get_num_computed_tokens=0)}, seq_len=6, query_len=6, is_prompt=True, prompt_logprob_indices=[], sample_indices=[3])
                      
    # sampling_metadata=SamplingMetadata(seq_groups=[SequenceGroupToSample(seq_ids=[0], sampling_params=SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0, top_p=1.0, top_k=-1, min_p=0.0, seed=2024, stop=[], stop_token_ids=[128008, 128001], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=2, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), guided_decoding=None, seq_data={0: SequenceData(prompt_token_ids=array('l', [128000, 9906, 11, 856, 836, 374]), output_token_ids=(), cumulative_logprob=0.0, get_num_computed_tokens=0}, seq_len=6, query_len=6, generator=<torch._C.Generator object at 0x7fcf3824e370>, is_prompt=True, prompt_logprob_indices=[], sample_indices=[0]), SequenceGroupToSample(seq_ids=[1], sampling_params=SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0, top_p=1.0, top_k=-1, min_p=0.0, seed=2024, stop=[], stop_token_ids=[128008, 128001], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=2, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), guided_decoding=None, seq_data={1: SequenceData(prompt_token_ids=array('l', [128000, 791, 4872, 315, 279, 3723, 4273, 374]), output_token_ids=(), cumulative_logprob=0.0, get_num_computed_tokens=0}, seq_len=8, query_len=8, generator=<torch._C.Generator object at 0x7fcf3824e390>, is_prompt=True, prompt_logprob_indices=[], sample_indices=[1]), SequenceGroupToSample(seq_ids=[2], sampling_params=SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0, top_p=1.0, top_k=-1, min_p=0.0, seed=2024, stop=[], stop_token_ids=[128008, 128001], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=2, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), guided_decoding=None, seq_data={2: SequenceData(prompt_token_ids=array('l', [128000, 791, 6864, 315, 9822, 374]), output_token_ids=(), cumulative_logprob=0.0, get_num_computed_tokens=0}, seq_len=6, query_len=6, generator=<torch._C.Generator object at 0x7fcf3824e350>, is_prompt=True, prompt_logprob_indices=[], sample_indices=[2]), SequenceGroupToSample(seq_ids=[3], sampling_params=SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0, top_p=1.0, top_k=-1, min_p=0.0, seed=2024, stop=[], stop_token_ids=[128008, 128001], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=2, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), guided_decoding=None, seq_data={3: SequenceData(prompt_token_ids=array('l', [128000, 791, 3938, 315, 15592, 374]), output_token_ids=(), cumulative_logprob=0.0, get_num_computed_tokens=0}, seq_len=6, query_len=6, generator=<torch._C.Generator object at 0x7fcf3824e330>, is_prompt=True, prompt_logprob_indices=[], sample_indices=[3])], )
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              

    a = ModelInputForGPUWithSamplingMetadata(input_tokens=tensor([35266,   279,   264, 10107], device='cuda:0'), input_positions=tensor([6, 8, 6, 6], device='cuda:0'), seq_lens=[7, 9, 7, 7], query_lens=[1, 1, 1, 1], lora_mapping=None, lora_requests=set(), attn_metadata=XFormersMetadata(seq_lens_tensor=tensor([7, 9, 7, 7], device='cuda:0', dtype=torch.int32), max_decode_seq_len=9, block_tables=tensor([[0],
        [1],
        [2],
        [3]], device='cuda:0', dtype=torch.int32), num_prefills=0, num_prefill_tokens=0, num_decode_tokens=4, slot_mapping=tensor([ 6, 24, 38, 54], device='cuda:0'), max_prefill_seq_len=0, use_cuda_graph=False, seq_lens=[7, 9, 7, 7], seq_start_loc=tensor([ 0,  7, 16, 23, 30], device='cuda:0', dtype=torch.int32), context_lens_tensor=tensor([6, 8, 6, 6], device='cuda:0', dtype=torch.int32), max_query_len=1, max_decode_query_len=None, query_start_loc=tensor([0, 1, 2, 3, 4], device='cuda:0', dtype=torch.int32), _cached_prefill_metadata=None, _cached_decode_metadata=None, encoder_seq_lens=None, encoder_seq_lens_tensor=None, max_encoder_seq_len=None, num_encoder_tokens=None, cross_slot_mapping=None, cross_block_tables=None), prompt_adapter_mapping=None, prompt_adapter_requests=set(), multi_modal_kwargs={}, request_ids_to_seq_ids={'0': [0], '1': [1], '2': [2], '3': [3]}, finished_requests_ids=[], virtual_engine=0, async_callback=None, seq_group_metadata_list=None, scheduler_outputs=None, sampling_metadata=SamplingMetadata(seq_groups=[SequenceGroupToSample(seq_ids=[0], sampling_params=SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0, top_p=1.0, top_k=-1, min_p=0.0, seed=2024, stop=[], stop_token_ids=[128008, 128001], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=2, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), guided_decoding=None, seq_data={0: SequenceData(prompt_token_ids=array('l', [128000, 9906, 11, 856, 836, 374]), output_token_ids=(35266,), cumulative_logprob=inf, get_num_computed_tokens=6)}, seq_len=None, query_len=1, is_prompt=False, prompt_logprob_indices=[], sample_indices=[0]), SequenceGroupToSample(seq_ids=[1], sampling_params=SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0, top_p=1.0, top_k=-1, min_p=0.0, seed=2024, stop=[], stop_token_ids=[128008, 128001], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=2, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), guided_decoding=None, seq_data={1: SequenceData(prompt_token_ids=array('l', [128000, 791, 4872, 315, 279, 3723, 4273, 374]), output_token_ids=(279,), cumulative_logprob=inf, get_num_computed_tokens=8)}, seq_len=None, query_len=1, is_prompt=False, prompt_logprob_indices=[], sample_indices=[1]), SequenceGroupToSample(seq_ids=[2], sampling_params=SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0, top_p=1.0, top_k=-1, min_p=0.0, seed=2024, stop=[], stop_token_ids=[128008, 128001], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=2, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), guided_decoding=None, seq_data={2: SequenceData(prompt_token_ids=array('l', [128000, 791, 6864, 315, 9822, 374]), output_token_ids=(264,), cumulative_logprob=inf, get_num_computed_tokens=6)}, seq_len=None, query_len=1,  is_prompt=False, prompt_logprob_indices=[], sample_indices=[2]), SequenceGroupToSample(seq_ids=[3], sampling_params=SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0, top_p=1.0, top_k=-1, min_p=0.0, seed=2024, stop=[], stop_token_ids=[128008, 128001], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=2, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), guided_decoding=None, seq_data={3: SequenceData(prompt_token_ids=array('l', [128000, 791, 3938, 315, 15592, 374]), output_token_ids=(10107,), cumulative_logprob=inf, get_num_computed_tokens=6)}, seq_len=None, query_len=1,  is_prompt=False, prompt_logprob_indices=[], sample_indices=[3])], selected_token_indices=tensor([0, 1, 2, 3], device='cuda:0'),), is_prompt=False)