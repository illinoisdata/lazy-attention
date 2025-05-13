r"""Benchmark online serving throughput.

Extracted from VLLM/benchmark/benchmark_serving.py

Example
    python3 benchmarks/benchmark_rag_serving.py \
        --exp parrot \
        --rag_type=parrot \
        --dataset-name random \
        --tokenizer facebook/opt-125m
"""

import argparse
import asyncio
import dataclasses
import json
import random
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import chatragbench
import longbench
import numpy as np
from simple_parsing import ArgumentParser
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from rag.logging import logger
from rag.rag import RAG, DocumentId, RAGArgs, make_rag
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.transformers_utils.tokenizer import get_tokenizer

MILLISECONDS_TO_SECONDS_CONVERSION = 1000


@dataclasses.dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: List[Tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: List[Tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: List[Tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: List[Tuple[float, float]]


@dataclasses.dataclass
class RAGRequest:
    prompt: str
    prompt_len: int
    output_len: int
    document_len: int
    documents: List[DocumentId]
    sampling_params: SamplingParams


@dataclasses.dataclass
class RAGDataset:
    requests: List[RAGRequest]
    documents: List[str]


@dataclasses.dataclass
class RAGRequestFuncInput:
    rag: RAG
    request: RAGRequest


@dataclasses.dataclass
class RAGRequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = dataclasses.field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""


def sample_random_requests(
    rag: RAG,
    prefix_len: int,
    input_len: int,
    output_len: int,
    document_len: int,
    num_prompts: int,
    num_documents: int,
    num_documents_per_prompt: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    seed: int = 1111,
) -> List[RAGRequest]:
    assert num_documents_per_prompt <= num_documents

    rng = np.random.default_rng(seed=seed)
    prefix_token_ids = rng.integers(0, tokenizer.vocab_size, size=prefix_len).tolist()
    input_lens = rng.integers(
        int(input_len * range_ratio),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = rng.integers(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )
    document_lens = rng.integers(
        int(document_len * range_ratio),
        document_len + 1,
        size=num_documents,
    )
    document_offsets = rng.integers(0, tokenizer.vocab_size, size=num_documents)
    prompt_document_lens = rng.integers(
        int(num_documents_per_prompt * range_ratio),
        num_documents_per_prompt + 1,
        size=num_prompts,
    )
    prompt_document_offsets = rng.integers(0, num_documents, size=num_prompts)
    offsets = rng.integers(0, tokenizer.vocab_size, size=num_prompts)
    documents: List[str] = []
    for i in range(num_documents):
        document = tokenizer.decode([(document_offsets[i] + i + j) % tokenizer.vocab_size for j in range(document_lens[i])])
        documents.append(document)
    doc_ids = rag.add_cache(documents)
    input_requests = []
    for i in range(num_prompts):
        prompt_doc_ids = [
            doc_ids[(prompt_document_offsets[i] + i + j) % num_documents] for j in range(prompt_document_lens[i])
        ]
        prompt = tokenizer.decode(
            prefix_token_ids + [(offsets[i] + i + j) % tokenizer.vocab_size for j in range(input_lens[i])]
        )
        output_len = int(output_lens[i])
        document_len = sum(
            document_lens[(prompt_document_offsets[i] + i + j) % num_documents] for j in range(prompt_document_lens[i])
        )
        input_requests.append(
            RAGRequest(
                prompt=prompt,
                prompt_len=int(prefix_len + input_lens[i]),
                output_len=output_len,
                document_len=document_len,
                documents=prompt_doc_ids,
                sampling_params=SamplingParams(max_tokens=output_len, seed=42, temperature=0.0, repetition_penalty=1.0, stop_token_ids=[]),
            )
        )

    return input_requests


def sample_chatragbench_requests(
    args: chatragbench.ChatRAGBenchArgs,
    rag: RAG,
    tokenizer: PreTrainedTokenizerBase,
) -> List[RAGRequest]:
    # Get prompt_list
    data_list, prompt_without_context_list = chatragbench.get_prompt_list(args)
    logger.info(f"Loaded {len(prompt_without_context_list)} ChatRAG-Bench prompts")

    # Fill document cache and collect prompt document IDs.
    doc_hash_to_id: Dict[int, DocumentId] = {}
    doc_ids_by_prompt: List[List[DocumentId]] = []
    document_tokens_by_prompt: List[List[int]] = []
    for item in data_list:
        prompt_doc_ids: List[DocumentId] = []
        document_tokens = []
        for ctx in item["ctxs"][: args.num_ctx]:
            document = ctx["text"]
            doc_hash = hash(document)
            if doc_hash not in doc_hash_to_id:
                doc_ids = rag.add_cache([document])
                assert len(doc_ids) == 1
                doc_hash_to_id[doc_hash] = doc_ids[0]
            prompt_doc_ids.append(doc_hash_to_id[doc_hash])
            document_tokens.append(len(tokenizer.encode(document)))
        doc_ids_by_prompt.append(prompt_doc_ids)
        document_tokens_by_prompt.append(document_tokens)
    num_documents_by_prompt = np.array([len(document_tokens) for document_tokens in document_tokens_by_prompt])
    num_document_tokens_by_prompt = np.array([sum(document_tokens) for document_tokens in document_tokens_by_prompt])
    logger.info(f"{len(doc_hash_to_id)} unique documents")
    logger.info(
        "Per-prompt number of documents, "
        f"min= {num_documents_by_prompt.min()}, "
        f"max= {num_documents_by_prompt.max()}, "
        f"mean= {num_documents_by_prompt.mean()}"
    )
    logger.info(
        "Per-prompt document tokens, "
        f"min= {num_document_tokens_by_prompt.min()}, "
        f"max= {num_document_tokens_by_prompt.max()}, "
        f"mean= {num_document_tokens_by_prompt.mean()}"
    )

    # Generate input requests.
    input_requests = []
    max_len = 0
    for prompt, prompt_doc_ids, document_tokens in zip(
        prompt_without_context_list, doc_ids_by_prompt, document_tokens_by_prompt
    ):
        prompt_len = len(tokenizer.encode(prompt))
        output_len = args.out_seq_len
        document_len = sum(document_tokens)
        input_requests.append(
            RAGRequest(
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=output_len,
                document_len=document_len,
                documents=prompt_doc_ids,
                sampling_params=SamplingParams(max_tokens=output_len, ignore_eos=True, 
                                               temperature=0, seed=42),
            )
        )
        max_len = max(max_len, prompt_len + document_len)
    logger.info(f"max_len= {max_len} tokens")
    return input_requests


def sample_longbench_requests(
    args: longbench.LongBenchArgs,
    rag: RAG,
    tokenizer: PreTrainedTokenizerBase,
) -> List[RAGRequest]:
    # Get prompt_list
    longbench_dataset = longbench.load_dataset(args.longbench_dataset_name)
    logger.info(f"Loaded {len(longbench_dataset.rows)} LongBench prompts")

    # Fill document cache and collect prompt document IDs.
    doc_hash_to_id: Dict[int, DocumentId] = {}
    doc_ids_by_prompt: List[List[DocumentId]] = []
    document_len_by_prompt: List[int] = []
    sum_tokens: int = 0
    for row in longbench_dataset.rows:
        document = row.context  # One document per LongBench prompt.
        # make sure is list
        if isinstance(document, str):
            document = [document]
            
        # print(f"Document: {document}")
        document_token = 0
        context_doc_ids = []
        for doc in document:
            document_token += len(tokenizer.encode(doc))
            doc_hash = hash(doc)
            if doc_hash not in doc_hash_to_id:
                doc_ids = rag.add_cache([doc])
                doc_id = doc_ids[0]
                doc_hash_to_id[doc_hash] = doc_id
            else:
                doc_id = doc_hash_to_id[doc_hash]
            context_doc_ids.append(doc_id)
        assert len(context_doc_ids) == len(document), f"doc_ids: {len(doc_ids)}, document: {len(document)}"
        sum_tokens += document_token
        doc_ids_by_prompt.append(context_doc_ids)
        document_len_by_prompt.append(document_token)
    logger.info(f"{len(doc_hash_to_id)} unique documents, sum tokens= {sum_tokens}")

    # Generate input requests.
    input_requests = []
    max_len = 0
    for row, prompt_doc_ids, document_len in zip(longbench_dataset.rows, doc_ids_by_prompt, document_len_by_prompt):
        prompt = row.input
        prompt_len = len(tokenizer.encode(prompt))
        output_len = args.longbench_out_seq_len
        input_requests.append(
            RAGRequest(
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=output_len,
                document_len=document_len,
                documents=prompt_doc_ids,
                sampling_params=SamplingParams(max_tokens=output_len, ignore_eos=True, temperature=0, seed=42),
            )
        )
        max_len = max(max_len, prompt_len + document_len)
    logger.info(f"max_len= {max_len} tokens")
    return input_requests

def sample_2wikimqa_block_requests(
    args,
    rag,
    tokenizer: PreTrainedTokenizerBase,
) -> List:
    
    jsonl_path = '/u/mpamnani/vllm/minidrag/scripts/block-attn-bench-datahub/processed_data/2wiki_eval/dataset'
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    logger.info(f"Loaded {len(data)} 2WikiMultiHopQA prompts")

    doc_hash_to_id: Dict[int, str] = {}
    doc_ids_by_prompt: List[List[str]] = []
    document_len_by_prompt: List[int] = []
    sum_tokens = 0

    for sample in data:
        doc_ids = []
        doc_token_count = 0
        for doc in sample["documents"]:
            doc_str = f"Title: {doc['title']}\n{doc['text'].strip()}"
            doc_hash = hash(doc_str)
            if doc_hash not in doc_hash_to_id:
                new_ids = rag.add_cache([doc_str])
                assert len(new_ids) == 1
                doc_hash_to_id[doc_hash] = new_ids[0]
                doc_token_count += len(tokenizer.encode(doc_str))
            doc_ids.append(doc_hash_to_id[doc_hash])
        doc_ids_by_prompt.append(doc_ids)
        document_len_by_prompt.append(doc_token_count)
        sum_tokens += doc_token_count
    logger.info(f"{len(doc_hash_to_id)} unique documents, sum tokens={sum_tokens}")

    input_requests = []
    max_len = 0
    for sample, prompt_doc_ids, document_len in zip(data, doc_ids_by_prompt, document_len_by_prompt):
        prompt = sample["prompt"]
        prompt_len = len(tokenizer.encode(prompt))
        output_len = 32 # args.longbench_out_seq_len
        input_requests.append(
            RAGRequest(
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=output_len,
                document_len=document_len,
                documents=prompt_doc_ids,
                sampling_params=SamplingParams(max_tokens=output_len, ignore_eos=True, temperature=0, seed=42),
            )
        )
        max_len = max(max_len, prompt_len + document_len)
    logger.info(f"max_len= {max_len} tokens")

    return input_requests

def sample_hotpotqa_block_requests(
    args,
    rag,
    tokenizer: PreTrainedTokenizerBase,
) -> List:
    
    jsonl_path = '/u/mpamnani/vllm/minidrag/scripts/block-attn-bench-datahub/processed_data/hqa_eval/dataset'
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    logger.info(f"Loaded {len(data)} HotpotQA prompts")

    doc_hash_to_id: Dict[int, str] = {}
    doc_ids_by_prompt: List[List[str]] = []
    document_len_by_prompt: List[int] = []
    sum_tokens = 0

    for sample in data:
        doc_ids = []
        doc_token_count = 0
        for doc in sample["documents"]:
            doc_str = f"Title: {doc['title']}\n{doc['text'].strip()}"
            doc_hash = hash(doc_str)
            if doc_hash not in doc_hash_to_id:
                new_ids = rag.add_cache([doc_str])
                assert len(new_ids) == 1
                doc_hash_to_id[doc_hash] = new_ids[0]
                doc_token_count += len(tokenizer.encode(doc_str))
            doc_ids.append(doc_hash_to_id[doc_hash])
        doc_ids_by_prompt.append(doc_ids)
        document_len_by_prompt.append(doc_token_count)
        sum_tokens += doc_token_count
    logger.info(f"{len(doc_hash_to_id)} unique documents, sum tokens={sum_tokens}")

    input_requests = []
    max_len = 0
    for sample, prompt_doc_ids, document_len in zip(data, doc_ids_by_prompt, document_len_by_prompt):
        prompt = sample["prompt"]
        prompt_len = len(tokenizer.encode(prompt))
        output_len = 32 # args.longbench_out_seq_len
        input_requests.append(
            RAGRequest(
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=output_len,
                document_len=document_len,
                documents=prompt_doc_ids,
                sampling_params=SamplingParams(max_tokens=output_len, ignore_eos=True, temperature=0, seed=42),
            )
        )
        max_len = max(max_len, prompt_len + document_len)
    logger.info(f"max_len= {max_len} tokens")

    return input_requests

def sample_2wikimqa_cacheblend_requests(
    args,
    rag,
    tokenizer: PreTrainedTokenizerBase,
) -> List:
    json_path = '/u/mpamnani/CacheBlend/inputs/wikimqa_s.json' 
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} 2wikimqa-cache-blend prompts")

    doc_hash_to_id: Dict[int, str] = {}
    doc_ids_by_prompt: List[List[str]] = []
    document_len_by_prompt: List[int] = []
    sum_tokens = 0

    for sample in data:
        doc_ids = []
        doc_token_count = 0
        for doc in sample["ctxs"]:
            doc_str = f"Title: {doc['title']}\n{doc['text'].strip()}"
            doc_hash = hash(doc_str)
            if doc_hash not in doc_hash_to_id:
                new_ids = rag.add_cache([doc_str])
                assert len(new_ids) == 1
                doc_hash_to_id[doc_hash] = new_ids[0]
                doc_token_count += len(tokenizer.encode(doc_str))
            doc_ids.append(doc_hash_to_id[doc_hash])
        doc_ids_by_prompt.append(doc_ids)
        document_len_by_prompt.append(doc_token_count)
        sum_tokens += doc_token_count
    logger.info(f"{len(doc_hash_to_id)} unique documents, sum tokens={sum_tokens}")

    input_requests = []
    max_len = 0
    for sample, prompt_doc_ids, document_len in zip(data, doc_ids_by_prompt, document_len_by_prompt):
        prompt = sample["question"]
        prompt_len = len(tokenizer.encode(prompt))
        output_len = 32  # Change as needed (args.longbench_out_seq_len)
        input_requests.append(
            RAGRequest(
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=output_len,
                document_len=document_len,
                documents=prompt_doc_ids,
                sampling_params=SamplingParams(max_tokens=output_len, ignore_eos=True, temperature=0, seed=42),
            )
        )
        max_len = max(max_len, prompt_len + document_len)
    logger.info(f"max_len= {max_len} tokens")

    return input_requests

def sample_samsum_cacheblend_requests(
    args,
    rag,
    tokenizer: PreTrainedTokenizerBase,
) -> List:
    json_path = '/u/mpamnani/CacheBlend/inputs/samsum.json'
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} Samsum-cache-blend prompts")

    doc_hash_to_id: Dict[int, str] = {}
    doc_ids_by_prompt: List[List[str]] = []
    document_len_by_prompt: List[int] = []
    sum_tokens = 0

    for sample in data:
        doc_ids = []
        doc_token_count = 0
        for doc in sample["ctxs"]:
            doc_str = f"Title: {doc['title']}\n{doc['text'].strip()}"
            doc_hash = hash(doc_str)
            if doc_hash not in doc_hash_to_id:
                new_ids = rag.add_cache([doc_str])
                assert len(new_ids) == 1
                doc_hash_to_id[doc_hash] = new_ids[0]
                doc_token_count += len(tokenizer.encode(doc_str))
            doc_ids.append(doc_hash_to_id[doc_hash])
        doc_ids_by_prompt.append(doc_ids)
        document_len_by_prompt.append(doc_token_count)
        sum_tokens += doc_token_count
    logger.info(f"{len(doc_hash_to_id)} unique documents, sum tokens={sum_tokens}")

    input_requests = []
    max_len = 0
    for sample, prompt_doc_ids, document_len in zip(data, doc_ids_by_prompt, document_len_by_prompt):
        prompt = sample["question"]
        prompt_len = len(tokenizer.encode(prompt))
        output_len = 32  # Change as needed (args.longbench_out_seq_len)
        input_requests.append(
            RAGRequest(
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=output_len,
                document_len=document_len,
                documents=prompt_doc_ids,
                sampling_params=SamplingParams(max_tokens=output_len, ignore_eos=True, temperature=0, seed=42),
            )
        )
        max_len = max(max_len, prompt_len + document_len)
    logger.info(f"max_len= {max_len} tokens")

    return input_requests

def sample_musique_cacheblend_requests(
    args,
    rag,
    tokenizer: PreTrainedTokenizerBase,
) -> List:
    json_path = '/u/mpamnani/CacheBlend/inputs/musique_s.json'
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)


    logger.info(f"Loaded {len(data)} Musique-cache-blend prompts")

    doc_hash_to_id: Dict[int, str] = {}
    doc_ids_by_prompt: List[List[str]] = []
    document_len_by_prompt: List[int] = []
    sum_tokens = 0

    for sample in data:
        doc_ids = []
        doc_token_count = 0
        for doc in sample["ctxs"]:
            doc_str = f"Title: {doc['title']}\n{doc['text'].strip()}"
            doc_hash = hash(doc_str)
            if doc_hash not in doc_hash_to_id:
                new_ids = rag.add_cache([doc_str])
                assert len(new_ids) == 1
                doc_hash_to_id[doc_hash] = new_ids[0]
                doc_token_count += len(tokenizer.encode(doc_str))
            doc_ids.append(doc_hash_to_id[doc_hash])
        doc_ids_by_prompt.append(doc_ids)
        document_len_by_prompt.append(doc_token_count)
        sum_tokens += doc_token_count
    logger.info(f"{len(doc_hash_to_id)} unique documents, sum tokens={sum_tokens}")

    input_requests = []
    max_len = 0
    for sample, prompt_doc_ids, document_len in zip(data, doc_ids_by_prompt, document_len_by_prompt):
        prompt = sample["question"]
        prompt_len = len(tokenizer.encode(prompt))
        output_len = 32  # Change as needed (args.longbench_out_seq_len)
        input_requests.append(
            RAGRequest(
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=output_len,
                document_len=document_len,
                documents=prompt_doc_ids,
                sampling_params=SamplingParams(max_tokens=output_len, ignore_eos=True, temperature=0, seed=42),
            )
        )
        max_len = max(max_len, prompt_len + document_len)
    logger.info(f"max_len= {max_len} tokens")

    return input_requests

async def get_request(
    input_requests: List[RAGRequest],
    request_rate: float,
    sample_requests: Optional[int],
    seed: int = 1111,
) -> AsyncGenerator[Tuple[int, RAGRequest], None]:
    request_ids = list(range(len(input_requests)))
    if sample_requests is not None:
        rng = np.random.default_rng(seed=seed)
        request_ids = list(rng.integers(0, high=len(input_requests), size=sample_requests))
    for request_id in request_ids:
        request = input_requests[request_id]
        yield int(request_id), request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def rag_request_func(
    request_func_input: RAGRequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RAGRequestFuncOutput:
    output = RAGRequestFuncOutput()
    output.prompt_len = request_func_input.request.prompt_len

    generated_texts: List[str] = []
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    try:
        async for response in request_func_input.rag.iter_generate(
            doc_ids=request_func_input.request.documents,
            query=request_func_input.request.prompt,
            sampling_params=request_func_input.request.sampling_params,
        ):
            timestamp = time.perf_counter()
            if ttft == 0.0:
                # First token.
                ttft = time.perf_counter() - st
                output.ttft = ttft
            else:
                output.itl.append(timestamp - most_recent_timestamp)

            most_recent_timestamp = timestamp
            generated_texts.append(response)

        latency = time.perf_counter() - st
        output.generated_text = "".join(generated_texts)
        output.success = True
        output.latency = latency
        if len(output.generated_text) <= 0:
            output.success = False
            logger.error(f"Found empty response for {request_func_input}")
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


ASYNC_REQUEST_FUNCS = {
    "rag": rag_request_func,
}


def calculate_metrics(
    input_requests: List[RAGRequest],
    input_request_ids: List[int],
    outputs: List[RAGRequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[float],
    gootput_config_dict: Dict[str, float],
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    all_tpots: List[float] = []
    ttfts: List[float] = []
    e2els: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note : this may inflate the output token count slightly
            output_len = len(tokenizer(outputs[i].generated_text, add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            total_input += input_requests[input_request_ids[i]].prompt_len
            tpot = 0.0
            if output_len > 1:
                tpot = (outputs[i].latency - outputs[i].ttft) / (output_len - 1)
                tpots.append(tpot)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if gootput_config_dict:
        valid_metrics = []
        slo_values = []

        if "ttft" in gootput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(gootput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION)
        if "tpot" in gootput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(gootput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION)
        if "e2el" in gootput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(gootput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION)

        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration " "on the benchmark arguments.", stacklevel=2
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=float(np.mean(ttfts or 0) * 1000),  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=float(np.std(ttfts or 0) * 1000),
        median_ttft_ms=float(np.median(ttfts or 0) * 1000),
        percentiles_ttft_ms=[(p, float(np.percentile(ttfts or 0, p) * 1000)) for p in selected_percentiles],
        mean_tpot_ms=float(np.mean(tpots or 0) * 1000),
        std_tpot_ms=float(np.std(tpots or 0) * 1000),
        median_tpot_ms=float(np.median(tpots or 0) * 1000),
        percentiles_tpot_ms=[(p, float(np.percentile(tpots or 0, p) * 1000)) for p in selected_percentiles],
        mean_itl_ms=float(np.mean(itls or 0) * 1000),
        std_itl_ms=float(np.std(itls or 0) * 1000),
        median_itl_ms=float(np.median(itls or 0) * 1000),
        percentiles_itl_ms=[(p, float(np.percentile(itls or 0, p) * 1000)) for p in selected_percentiles],
        mean_e2el_ms=float(np.median(e2els or 0) * 1000),
        std_e2el_ms=float(np.std(e2els or 0) * 1000),
        median_e2el_ms=float(np.mean(e2els or 0) * 1000),
        percentiles_e2el_ms=[(p, float(np.percentile(e2els or 0, p) * 1000)) for p in selected_percentiles],
    )

    return metrics, actual_output_lens


async def benchmark(
    backend: str,
    rag: RAG,
    input_requests: List[RAGRequest],
    tokenizer: PreTrainedTokenizerBase,
    sample_requests: Optional[int],
    request_rate: float,
    disable_tqdm: bool,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[float],
    gootput_config_dict: Dict[str, float],
    max_concurrency: Optional[int],
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    logger.info("Starting initial single prompt test run...")
    test_request = input_requests[0]
    test_input = RAGRequestFuncInput(
        rag=rag,
        request=test_request,
    )
    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}"
        )
    else:
        logger.info("Initial test run completed. Starting main benchmark run...")

    logger.info(f"Traffic request rate: {request_rate}")
    logger.info(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=sample_requests if sample_requests is not None else len(input_requests))

    # This can be used once the minimum Python version is 3.10 or higher,
    # and it will simplify the code in limited_request_func.
    #    semaphore = (asyncio.Semaphore(max_concurrency)
    #                 if max_concurrency else contextlib.nullcontext())
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)

    benchmark_start_time = time.perf_counter()
    input_request_ids: List[int] = []
    tasks: List[asyncio.Task] = []
    async for request_id, request in get_request(input_requests, request_rate, sample_requests=sample_requests):
        request_func_input = RAGRequestFuncInput(
            rag=rag,
            request=request,
        )
        input_request_ids.append(request_id)
        tasks.append(asyncio.create_task(limited_request_func(request_func_input=request_func_input, pbar=pbar)))
    outputs: List[RAGRequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        input_request_ids=input_request_ids,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
        gootput_config_dict=gootput_config_dict,
    )

    benchmark_result_strs = [
        "",
        "{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="),
        "{:<40} {:<10}".format("Successful requests:", metrics.completed),
        "{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration),
        "{:<40} {:<10}".format("Total input tokens:", metrics.total_input),
        "{:<40} {:<10}".format("Total generated tokens:", metrics.total_output),
        "{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput),
        "{:<40} {:<10.2f}".format("Request goodput (req/s):", metrics.request_goodput if gootput_config_dict else -1),
        "{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput),
        "{:<40} {:<10.2f}".format("Total Token throughput (tok/s):", metrics.total_token_throughput),
    ]

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput:": metrics.request_goodput if gootput_config_dict else None,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_request_ids": input_request_ids,
        "input_lens": [output.prompt_len for output in outputs],
        "input_texts": [input_requests[i].prompt for i in input_request_ids],
        "document_lens": [input_requests[i].document_len for i in input_request_ids],
        "documents_list": [input_requests[i].documents for i in input_request_ids],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        benchmark_result_strs.append("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        benchmark_result_strs.append(
            "{:<40} {:<10.2f}".format(f"Mean {metric_name} (ms):", getattr(metrics, f"mean_{metric_attribute_name}_ms"))
        )
        benchmark_result_strs.append(
            "{:<40} {:<10.2f}".format(f"Median {metric_name} (ms):", getattr(metrics, f"median_{metric_attribute_name}_ms"))
        )
        result[f"mean_{metric_attribute_name}_ms"] = getattr(metrics, f"mean_{metric_attribute_name}_ms")
        result[f"median_{metric_attribute_name}_ms"] = getattr(metrics, f"median_{metric_attribute_name}_ms")
        result[f"std_{metric_attribute_name}_ms"] = getattr(metrics, f"std_{metric_attribute_name}_ms")
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            benchmark_result_strs.append("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    benchmark_result_strs.append("=" * 50)
    logger.info("\n".join(benchmark_result_strs))

    return result


def check_goodput_args(args):
    # Check and parse goodput arguments
    gootput_config_dict = {}
    VALID_NAMES = ["ttft", "tpot", "e2el"]
    if args.goodput:
        gootput_config_dict = parse_goodput(args.goodput)
        for slo_name, slo_val in gootput_config_dict.items():
            if slo_name not in VALID_NAMES:
                raise ValueError(
                    f"Invalid metric name found, {slo_name}: {slo_val}. "
                    "The service level objective name should be one of "
                    f"{str(VALID_NAMES)}. "
                )
            if slo_val < 0:
                raise ValueError(
                    f"Invalid value found, {slo_name}: {slo_val}. "
                    "The service level objective value should be "
                    "non-negative."
                )
    return gootput_config_dict


def parse_goodput(slo_pairs):
    gootput_config_dict = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            gootput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Invalid format found for service level objectives. "
            'Specify service level objectives for goodput as "KEY:VALUE" '
            "pairs, where the key is a metric name, and the value is a "
            "number in milliseconds."
        ) from err
    return gootput_config_dict


def load_dataset(
    args: argparse.Namespace,
    rag: RAG,
    tokenizer: PreTrainedTokenizerBase,
) -> List[RAGRequest]:
    if args.dataset_name == "random":
        return sample_random_requests(
            rag=rag,
            prefix_len=args.random_prefix_len,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            document_len=args.random_document_len,
            num_prompts=args.random_num_prompts,
            num_documents=args.random_num_documents,
            num_documents_per_prompt=args.random_num_documents_per_prompt,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
        )
    elif args.dataset_name == "chatragbench":
        return sample_chatragbench_requests(
            args=args.chatragbench,
            rag=rag,
            tokenizer=tokenizer,
        )
    elif args.dataset_name == "longbench":
        return sample_longbench_requests(
            args=args.longbench,
            rag=rag,
            tokenizer=tokenizer,
        )
    elif args.dataset_name == "2wikimqa_block":
        return sample_2wikimqa_block_requests(args=args,
                                        rag=rag, 
                                        tokenizer=tokenizer,
                                       )
    elif args.dataset_name == "hotpotqa_block":
        return sample_hotpotqa_block_requests(args=args,
                                        rag=rag, 
                                        tokenizer=tokenizer,
                                        )
    elif args.dataset_name == "2wikimqa_cacheblend":
        return sample_2wikimqa_cacheblend_requests(args=args,
                                        rag=rag, 
                                        tokenizer=tokenizer,
                                        )    
    elif args.dataset_name == "samsum_cacheblend":
        return sample_samsum_cacheblend_requests(args=args,
                                        rag=rag, 
                                        tokenizer=tokenizer,
                                        )
    elif args.dataset_name == "musique_cacheblend":
        return sample_musique_cacheblend_requests(args=args,
                                        rag=rag, 
                                        tokenizer=tokenizer,
                                        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")


def main(args: argparse.Namespace):
    args.seed = 42
    logger.info(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    exp = args.exp
    tokenizer_id = args.tokenizer

    rag = make_rag(args.rag, engine_args=EngineArgs.from_cli_args(args))
    tokenizer = get_tokenizer(tokenizer_id, trust_remote_code=args.trust_remote_code)

    input_requests = load_dataset(args, rag, tokenizer)

    gootput_config_dict = check_goodput_args(args)

    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            rag=rag,
            input_requests=input_requests,
            tokenizer=tokenizer,
            sample_requests=args.sample_requests,
            request_rate=args.request_rate,
            disable_tqdm=args.disable_tqdm,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
            gootput_config_dict=gootput_config_dict,
            max_concurrency=args.max_concurrency,
        )
    )

    # Save config and results to json
    result_json: Dict[str, Any] = {}

    # Setup
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_json["date"] = current_dt
    result_json["backend"] = backend
    result_json["exp"] = exp
    result_json["tokenizer_id"] = tokenizer_id
    result_json["sample_requests"] = args.sample_requests

    # Metadata
    if args.metadata:
        for item in args.metadata:
            if "=" in item:
                kvstring = item.split("=")
                result_json[kvstring[0].strip()] = kvstring[1].strip()
            else:
                raise ValueError("Invalid metadata format. Please use KEY=VALUE format.")

    # Traffic
    result_json["request_rate"] = args.request_rate if args.request_rate < float("inf") else "inf"
    result_json["max_concurrency"] = args.max_concurrency

    # RAG stats
    result_json["rag_stats"] = rag.get_stats_dict()

    # Merge with benchmark result
    result_json = {**result_json, **benchmark_result}

    # Save to file
    result_path = Path(args.result_dir) / f"{exp}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f)
    logger.info(f"Save results to {result_path}")
    logger.info(f"RAG: {rag}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="rag",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--exp",
        type=str,
        help="Experiment name for naming output files.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        choices=["random", "chatragbench", "longbench","2wikimqa_block", "hotpotqa_block", "2wikimqa_cacheblend", "samsum_cacheblend", "musique_cacheblend"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the sharegpt/sonnet dataset. " "Or the huggingface dataset ID if using HF dataset.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )
    # parser.add_argument(
    #     "--tokenizer",
    #     type=str,
    #     help="Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    # )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--sample-requests",
        type=int,
        default=200,
        help="IF set, randomly sample this many requests with replacement to test.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    # parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument(
    #     "--trust-remote-code",
    #     action="store_true",
    #     help="Trust remote code from huggingface",
    # )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="results/",
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl,e2el",
        help="Comma-seperated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        'Allowed metric names are "ttft", "tpot", "itl", "e2el". '
        'Default value is "ttft,tpot,itl,e2el".',
    )
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-seperated list of percentiles for selected metrics. "
        'To report 25-th, 50-th, and 75-th percentiles, use "25,50,75". '
        'Default value is "99". '
        'Use "--percentile-metrics" to select metrics.',
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help='Specify service level objectives for goodput as "KEY:VALUE" '
        "pairs, where the key is a metric name, and the value is in "
        'milliseconds. Multiple "KEY:VALUE" pairs can be provided, '
        "separated by spaces. Allowed request level metric names are "
        '"ttft", "tpot", "e2el". For more context on the definition of '
        "goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 "
        "and the blog: https://hao-ai-lab.github.io/blogs/distserve",
    )
    #block-bench
    parser.add_argument(
        "--json-path",
        type=str,
        default=None,
        help="Path to the preprocessed block-bench dataset JSON file.",
    )
    random_group = parser.add_argument_group("random dataset options")
    parser.add_argument(
        "--random-num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to generate.",
    )
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-document-len",
        type=int,
        default=64,
        help="Number of tokens per document, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-num-documents",
        type=int,
        default=4,
        help="Number of documents to generate, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-num-documents-per-prompt",
        type=int,
        default=4,
        help="Number of documents included in each prompt, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=1.0,
        help="Range of sampled ratio of input/output length, " "used only for random sampling.",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help="Number of fixed prefix tokens before random "
        " context. The length range of context in a random "
        " request is [random-prefix-len, "
        " random-prefix-len + random-prefix-len * random-range-ratio).",
    )

    EngineArgs.add_cli_args(parser)
    parser.add_arguments(RAGArgs, "rag")
    parser.add_arguments(chatragbench.ChatRAGBenchArgs, "chatragbench")
    parser.add_arguments(longbench.LongBenchArgs, "longbench")

    args = parser.parse_args()
    main(args)
