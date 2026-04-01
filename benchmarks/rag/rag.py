"""Unified RAGs"""
from __future__ import annotations

import os
if os.environ.get("VLLM_USE_LAZY_ATTENTION", "0") == "1":
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1"
    import lazy.__vllm__
    
import uuid
import asyncio
import dataclasses
import sys
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, AsyncGenerator, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import promptcache
import torch
import transformers
from transformers.cache_utils import DynamicCache

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaConfig, LlamaForCausalLM
from transformers import AsyncTextIteratorStreamer
from threading import Thread


from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer as vllm_get_tokenizer

from rag.logging import logger

try:
    from vllm.v1.engine.async_llm import AsyncLLM
except ImportError:
    logger.warning("Missing vllm.v1.engine.async_llm.AsyncLLM (ok for cacheblend)")

DocumentId = int


class RAG(ABC):

    @abstractmethod
    def add_cache(self, docs: List[str]) -> List[DocumentId]:
        pass

    @abstractmethod
    async def iter_generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: SamplingParams,
        position_ids: Optional[List[int]] = None,
    ) -> AsyncGenerator[str, None]:
        yield ""

    def generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: SamplingParams,
        position_ids: Optional[List[int]] = None,
    ) -> List[str]:

        async def collect_generate():
            outputs = []
            async for output in self.iter_generate(
                doc_ids=doc_ids,
                query=query,
                sampling_params=sampling_params,
                position_ids=position_ids,
            ):
                outputs.append(output)
            return outputs

        return asyncio.run(collect_generate())

    @abstractmethod
    def destroy_cache(self, doc_ids: Optional[List[str]] = None) -> None:
        pass

    def get_stats_dict(self) -> Dict[str, Any]:
        return {}


class ParrotRAG(RAG):

    def __init__(self) -> None:
        RAG.__init__(self)
        self._doc_counter: int = 0

    def add_cache(self, docs: List[str]) -> List[int]:
        doc_ids = []
        for _ in docs:
            doc_ids.append(self._doc_counter)
            self._doc_counter += 1
        return doc_ids

    async def iter_generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: SamplingParams,
        position_ids: Optional[List[int]] = None,
    ) -> AsyncGenerator[str, None]:
        yield query

    def destroy_cache(self, doc_ids: Optional[List[str]] = None) -> None:
        pass


class CacheManager:
    def __init__(self):
        self.cache_hit = 0
        self.cache_miss = 0

    def put(self, item_ids: List[Any], item_weights: List[int]) -> Tuple[bool, List[Any], List[int], List[int]]:
        raise NotImplementedError("Abstract method")

    def get_cache_hit(self) -> int:
        return self.cache_hit

    def get_cache_miss(self) -> int:
        return self.cache_miss


class NoCacheManager(CacheManager):
    def __init__(self):
        CacheManager.__init__(self)

    def put(self, item_ids: List[Any], item_weights: List[int]) -> Tuple[bool, List[Any], List[int], List[int]]:
        assert len(item_ids) == len(item_weights)
        self.cache_miss += len(item_ids)
        return False, [], [], item_weights


class SequenceCacheManager(CacheManager):
    """Cache of exact sequence. A cache key is the strict sequence of item IDs."""

    def __init__(self, capacity: int):
        CacheManager.__init__(self)
        self.capacity = capacity
        self.cache: OrderedDict[Any, List[int]] = OrderedDict()
        self.current_weight = 0

    def put(self, item_ids: List[Any], item_weights: List[int]) -> Tuple[bool, List[Any], List[int], List[int]]:
        evicted_items: List[Any] = []
        evicted_item_weights: List[int] = []
        fill_items_weights: List[int] = []
        total_weight = sum(item_weights)
        if total_weight > self.capacity:
            # logger.error(f"Cannot cache all items ({sum(item_weights)}) within the capacity ({self.capacity})")
            return True, evicted_items, evicted_item_weights, fill_items_weights

        # If item already exists, remove it to update its position
        cache_key = tuple(item_ids)
        if cache_key in self.cache:
            self.current_weight -= sum(self.cache[cache_key])
            del self.cache[cache_key]
            self.cache_hit += len(item_ids)
        else:
            fill_items_weights.extend(item_weights)
            self.cache_miss += len(item_ids)

        # Evict items if necessary to make space for the new item
        while self.current_weight + total_weight > self.capacity:
            # Evict the least recently used item
            evicted_item_ids, evicted_item_weight = self.cache.popitem(last=False)
            evicted_items.extend(evicted_item_ids)
            evicted_item_weights.extend(evicted_item_weight)
            self.current_weight -= sum(evicted_item_weight)

        # Add the new item
        self.cache[cache_key] = item_weights
        self.current_weight += total_weight

        return False, evicted_items, evicted_item_weights, fill_items_weights


class PrefixTreeCacheManager(CacheManager):
    """Cache of prefix tree.

    LRU tracks prefix usage (which uses prefixes of prefix). Evict suffix of prefix one by one.

    Needs to fill suffix (e.g., BC in ABC) when the prefix is cached but not entire string (e.g., A is cached but not AB).
    """

    def __init__(self, capacity: int):
        CacheManager.__init__(self)
        self.capacity = capacity
        self.prefix_cache: OrderedDict[Any, int] = OrderedDict()  # Prefix --> suffix weight
        self.current_weight = 0

    def put(self, item_ids: List[Any], item_weights: List[int]) -> Tuple[bool, List[Any], List[int], List[int]]:
        evicted_items: List[Any] = []
        evicted_item_weights: List[int] = []
        fill_items_weights: List[int] = []
        if sum(item_weights) > self.capacity:
            # logger.error(f"Cannot cache all items ({sum(item_weights)}) within the capacity ({self.capacity})")
            return True, evicted_items, evicted_item_weights, fill_items_weights

        # Find largest prefix.
        largest_prefix_rdx: Optional[int] = None
        for rdx in range(len(item_ids), 0, -1):
            if tuple(item_ids[:rdx]) in self.prefix_cache:
                largest_prefix_rdx = rdx
                break
            fill_items_weights.append(item_weights[rdx - 1])

        # If a prefix already exists, remove all prefixes to update its position
        if largest_prefix_rdx is not None:
            for sub_rdx in range(1, largest_prefix_rdx + 1):
                prefix_ids = tuple(item_ids[:sub_rdx])
                self.current_weight -= self.prefix_cache[prefix_ids]
                del self.prefix_cache[prefix_ids]
                self.cache_hit += 1
        else:
            largest_prefix_rdx = 0
        self.cache_miss += len(item_ids) - largest_prefix_rdx

        # Evict items if necessary to make space for the new item
        while self.current_weight + sum(item_weights) > self.capacity:
            # Evict the least recently used item
            evicted_prefix_ids, evicted_item_weight = self.prefix_cache.popitem(last=False)
            evicted_items.append(evicted_prefix_ids[-1])
            evicted_item_weights.append(evicted_item_weight)
            self.current_weight -= evicted_item_weight

        # Add all prefixes in reverse order so that shorter prefixes are evicted later.
        for rdx in range(len(item_ids), 0, -1):
            # Add the new item
            self.prefix_cache[tuple(item_ids[:rdx])] = item_weights[rdx - 1]
            self.current_weight += item_weights[rdx - 1]

        return False, evicted_items, evicted_item_weights, fill_items_weights


class LRUCacheManager(CacheManager):
    def __init__(self, capacity: int):
        CacheManager.__init__(self)
        self.capacity = capacity
        self.cache: OrderedDict[Any, int] = OrderedDict()
        self.current_weight = 0

    def put(self, item_ids: List[Any], item_weights: List[int]) -> Tuple[bool, List[Any], List[int], List[int]]:
        evicted_items: List[Any] = []
        evicted_item_weights: List[int] = []
        fill_items_weights: List[int] = []
        if sum(item_weights) > self.capacity:
            # logger.error(f"Cannot cache all items ({sum(item_weights)}) within the capacity ({self.capacity})")
            return True, evicted_items, evicted_item_weights, fill_items_weights

        for item_id, item_weight in zip(item_ids, item_weights):

            # If item already exists, remove it to update its position
            if item_id in self.cache:
                self.current_weight -= self.cache[item_id]
                del self.cache[item_id]
                self.cache_hit += 1
            else:
                fill_items_weights.append(item_weight)
                self.cache_miss += 1

            # Evict items if necessary to make space for the new item
            while self.current_weight + item_weight > self.capacity:
                # Evict the least recently used item
                evicted_item_id, evicted_item_weight = self.cache.popitem(last=False)
                evicted_items.append(evicted_item_id)
                evicted_item_weights.append(evicted_item_weight)
                self.current_weight -= evicted_item_weight

            # Add the new item
            self.cache[item_id] = item_weight
            self.current_weight += item_weight

        return False, evicted_items, evicted_item_weights, fill_items_weights


class CacheParrotRAG(ParrotRAG):
    """Only use for cache estimation."""

    def __init__(self, tokenizer_id: str, cache_manager: CacheManager) -> None:
        ParrotRAG.__init__(self)
        self._tokenizer = vllm_get_tokenizer(tokenizer_id)
        self._doc_tokens: Dict[DocumentId, int] = {}

        # Need to be thread-safe.
        self._sync_lock = asyncio.Lock()
        self._cache_manager = cache_manager
        self._is_fails: List[bool] = []
        self._evicted_doc_ids: List[DocumentId] = []
        self._evicted_doc_tokens: List[int] = []
        self._fill_doc_tokens: List[int] = []

    def add_cache(self, docs: List[str]) -> List[int]:
        doc_ids = ParrotRAG.add_cache(self, docs)
        for doc, doc_id in zip(docs, doc_ids):
            self._doc_tokens[doc_id] = len(self._tokenizer.encode(doc))
        return doc_ids

    async def iter_generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: SamplingParams,
        position_ids: Optional[List[int]] = None,
    ) -> AsyncGenerator[str, None]:
        # Simulate allocating for document cache.
        doc_tokens = [self._doc_tokens[doc_id] for doc_id in doc_ids]
        async with self._sync_lock:
            is_fail, evicted_doc_ids, evicted_doc_tokens, fill_doc_tokens = self._cache_manager.put(doc_ids, doc_tokens)
            self._is_fails.append(is_fail)
            self._evicted_doc_ids.extend(evicted_doc_ids)
            self._evicted_doc_tokens.extend(evicted_doc_tokens)
            self._fill_doc_tokens.extend(fill_doc_tokens)
        yield "cache!"

    def destroy_cache(self, doc_ids: Optional[List[str]] = None) -> None:
        pass

    def get_stats_dict(self) -> Dict[str, Any]:
        return {
            "hit": self._cache_manager.get_cache_hit(),
            "miss": self._cache_manager.get_cache_miss(),
            "num_evict": len(self._evicted_doc_tokens),
            "num_fill": len(self._fill_doc_tokens),
            "evict_toks": sum(self._evicted_doc_tokens),
            "fill_toks": sum(self._fill_doc_tokens),
            "fails": sum(self._is_fails),
        }

    def __str__(self):
        return (
            "CacheParrotRAG("
            f"hit= {self._cache_manager.get_cache_hit()}, "
            f"miss= {self._cache_manager.get_cache_miss()}, "
            f"num_evict= {len(self._evicted_doc_tokens)}, "
            f"num_fill= {len(self._fill_doc_tokens)}, "
            f"evict_toks= {sum(self._evicted_doc_tokens)}, "
            f"fill_toks= {sum(self._fill_doc_tokens)}, "
            f"fails= {sum(self._is_fails)}"
            ")"
        )


class LLMRAG(RAG):

    def __init__(self, llm: "AsyncLLM") -> None:
        RAG.__init__(self)
        self._llm = llm
        self._docs: Dict[DocumentId, str] = {}
        self._last_request_id: int = 0

    def add_cache(self, docs: List[str]) -> List[int]:
        doc_ids = []
        for doc in docs:
            doc_id = len(self._docs)
            self._docs[doc_id] = doc
            doc_ids.append(doc_id)
        return doc_ids
    
    async def add_doc_async(self, request_id: str, docs_ids: List[int]) -> None:
        for idx, doc_id in enumerate(docs_ids):
            doc = self._docs[doc_id]
            # 使用 async for 来迭代异步生成器
            async for _ in self._llm.generate(
                prompt=doc,
                sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
                request_id=f"cache_{request_id}_{idx}_{uuid.uuid4().hex}",
            ):
                # 我们只需要等待生成完成，不需要使用生成的内容
                pass

    async def iter_generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: SamplingParams,
        position_ids: Optional[List[int]] = None,
    ) -> AsyncGenerator[str, None]:
        # print(sampling_params)
        request_id = self._next_request_id()
        document = [self._docs[doc_id] for doc_id in doc_ids]
        if isinstance(document, str):
            context = document
        else:
            context = "".join(document)
        prompt = context + query
        
        latest_idx = 0
        async for generate_output in self._llm.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            if len(generate_output.outputs) > 1:
                logger.warning(f"Found {len(generate_output.outputs)} outputs, yielding first one.")
            prev_latest_idx = latest_idx
            latest_idx = len(generate_output.outputs[0].text)
            if prev_latest_idx < latest_idx:
                yield generate_output.outputs[0].text[prev_latest_idx:]

    def destroy_cache(self, doc_ids: Optional[List[str]] = None) -> None:
        pass

    def _next_request_id(self) -> str:
        request_id = str(self._last_request_id)
        self._last_request_id += 1
        return request_id

class RecomLLMRAG(RAG):

    def __init__(self, llm: "AsyncLLM") -> None:
        RAG.__init__(self)
        self._llm = llm
        self._docs: Dict[DocumentId, str] = {}
        self._last_request_id: int = 0

    def add_cache(self, docs: List[str]) -> List[int]:
        doc_ids = []
        for doc in docs:
            doc_id = len(self._docs)
            self._docs[doc_id] = doc
            doc_ids.append(doc_id)
        return doc_ids
    
    async def add_doc_async(self, request_id: str, docs_ids: List[int]) -> None:
        pass

    async def iter_generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: SamplingParams,
        position_ids: Optional[List[int]] = None,
    ) -> AsyncGenerator[str, None]:
        # print(sampling_params)
        request_id = self._next_request_id()
        preamble = f"{request_id}<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference documents that may help you in answering the user's question.\n\n"
        query = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nPlease write a high-quality answer for the given question using only the provided search documents (some of which might be irrelevant)" + query
        document = [self._docs[doc_id] for doc_id in doc_ids]
        document = [preamble] + document
        if isinstance(document, str):
            context = document
        else:
            context = "\n".join(document)
        prompt = context + "\n\n" + query
        
        latest_idx = 0
        async for generate_output in self._llm.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            if len(generate_output.outputs) > 1:
                logger.warning(f"Found {len(generate_output.outputs)} outputs, yielding first one.")
            prev_latest_idx = latest_idx
            latest_idx = len(generate_output.outputs[0].text)
            if prev_latest_idx < latest_idx:
                yield generate_output.outputs[0].text[prev_latest_idx:]

    def destroy_cache(self, doc_ids: Optional[List[str]] = None) -> None:
        pass

    def _next_request_id(self) -> str:
        request_id = str(self._last_request_id)
        self._last_request_id += 1
        return request_id


class ReuseLLMRAG(RAG):

    def __init__(self, llm: "AsyncLLM") -> None:
        RAG.__init__(self)
        self._llm = llm
        self._docs: Dict[DocumentId, str] = {}
        self._last_request_id: int = 0

    def add_cache(self, docs: List[str]) -> List[int]:
        doc_ids = []
        for doc in docs:
            doc_id = len(self._docs)
            self._docs[doc_id] = doc
            doc_ids.append(doc_id)
        return doc_ids
    
    async def add_doc_async(self, request_id: str, docs_ids: List[int]) -> None:
        latest_idx = 0
        preamble = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference documents that may help you in answering the user's question.\n\n"
        document = [self._docs[doc_id] for doc_id in docs_ids]
        document = [preamble] + document
        if isinstance(document, str):
            context = document
        else:
            context = "\n".join(document)

        # 使用 async for 来迭代异步生成器
        async for _ in self._llm.generate(
            prompt=context,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
            request_id=f"cache_{request_id}_{uuid.uuid4().hex}",
        ):
            pass

    async def iter_generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: SamplingParams,
        position_ids: Optional[List[int]] = None,
    ) -> AsyncGenerator[str, None]:
        # print(sampling_params)
        request_id = self._next_request_id()
        preamble = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference documents that may help you in answering the user's question.\n\n"
        query = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nPlease write a high-quality answer for the given question using only the provided search documents (some of which might be irrelevant)" + query
        document = [self._docs[doc_id] for doc_id in doc_ids]
        document = [preamble] + document
        if isinstance(document, str):
            context = document
        else:
            context = "\n".join(document)
        prompt = context + "\n\n" + query
        
        latest_idx = 0
        async for generate_output in self._llm.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            if len(generate_output.outputs) > 1:
                logger.warning(f"Found {len(generate_output.outputs)} outputs, yielding first one.")
            prev_latest_idx = latest_idx
            latest_idx = len(generate_output.outputs[0].text)
            if prev_latest_idx < latest_idx:
                yield generate_output.outputs[0].text[prev_latest_idx:]

    def destroy_cache(self, doc_ids: Optional[List[str]] = None) -> None:
        pass

    def _next_request_id(self) -> str:
        request_id = str(self._last_request_id)
        self._last_request_id += 1
        return request_id

"""
WARNING: Only works with CacheBlend's vllm
"""
class CacheBlendRAG(RAG):
    def __init__(self, llm: "LLM") -> None:
        RAG.__init__(self)
        self._llm = llm
        self._docs: Dict[DocumentId, str] = {}
        self._last_request_id: int = 0

        # Enable CacheBlend feature
        cache_fuse_metadata = (
            self._llm
                .llm_engine
                .model_executor
                .driver_worker
                .model_runner
                .model
                .model
                .cache_fuse_metadata
        )
        cache_fuse_metadata['collect'] = True
        cache_fuse_metadata['check'] = False

    def add_cache(self, docs: List[str]) -> List[int]:
        doc_ids = []
        for doc in docs:
            doc_id = len(self._docs)
            self._docs[doc_id] = doc
            doc_ids.append(doc_id)
        return doc_ids
    
    async def add_doc_async(self, request_id: str, docs_ids: List[int]) -> None:
        pass

    async def iter_generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: SamplingParams,
        position_ids: Optional[List[int]] = None,
    ) -> AsyncGenerator[str, None]:
        request_id = self._next_request_id()
        preamble = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference documents that may help you in answering the user's question.\n\n"
        query = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nPlease write a high-quality answer for the given question using only the provided search documents (some of which might be irrelevant)" + query
        document = [self._docs[doc_id] for doc_id in doc_ids]
        document = [preamble] + document
        if isinstance(document, str):
            context = document
        else:
            context = "\n".join(document)
        prompt = context + "\n\n" + query
        
        generate_output = self._llm.generate(
            prompts=[prompt],
            sampling_params=sampling_params,
        )
        for output in generate_output:
            yield output.outputs[0].text

    def destroy_cache(self, doc_ids: Optional[List[str]] = None) -> None:
        pass

    def _next_request_id(self) -> str:
        request_id = str(self._last_request_id)
        self._last_request_id += 1
        return request_id

from .utils import *
class BlockAttentionRAG(RAG):
    def __init__(self, lm_name: str, max_concurrency: int = 10) -> None:
        self._docs: Dict[int, str] = {}
        self._last_request_id: int = 0

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=lm_name, use_fast=False)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            
        self._token_eos = self._tokenizer.eos_token_id
        self._max_tokens = 200
        self._document_max_len = 512
        
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=lm_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        self._model.eval()
        
        config = transformers.AutoConfig.from_pretrained(pretrained_model_name_or_path=lm_name)
        # Init RoPE
        self._emb = LlamaRotaryEmbedding(config=config).to(device=self._model.device, dtype=torch.float32)
        self._emb.eval()
        
        # Cache state
        self.prepared_doc_cache: List[DynamicCache] = []
        self.cached_tokens_len: int = 0
        self._local_attention_suffix = ""

        self._gpu_semaphore = asyncio.Semaphore(max_concurrency)

    def add_cache(self, docs: List[str]) -> List[int]:
        doc_ids = []
        for doc in docs:
            doc_id = len(self._docs)
            self._docs[doc_id] = doc
            doc_ids.append(doc_id)
        return doc_ids
    
    async def add_doc_async(self, request_id: str, docs_ids: List[int], num_local_attention_blocks: int = 10000) -> None:
        """异步处理文档，将其 KV Cache 转化为位置无关状态"""
        self.prepared_doc_cache = []
        self.cached_tokens_len = 0
        self.block_input_ids = None  # 【关键修复】：保存block的input_ids用于拼接
        
        blocks = [self._docs[doc_id] for doc_id in docs_ids]
        
        # 处理超出数量限制的块
        if len(blocks) > num_local_attention_blocks:
            self._local_attention_suffix = "".join(blocks[num_local_attention_blocks:])
            blocks = blocks[:num_local_attention_blocks]
        else:
            self._local_attention_suffix = ""
        
        if num_local_attention_blocks == 0:
            self._local_attention_suffix = "".join(blocks)
            blocks = []
        
        for b_idx, block in enumerate(blocks):
            with torch.no_grad():
                block_input_ids = torch.tensor(
                    data=[self._tokenizer.encode(block, add_special_tokens=False)],
                    dtype=torch.int64,
                    device=self._model.device
                )
                # 拼接所有block的input_ids（和block_generate_server.py一致）
                if b_idx == 0:
                    self.block_input_ids = block_input_ids
                else:
                    self.block_input_ids = torch.cat([self.block_input_ids, block_input_ids], dim=-1)
                
                # 1. 正常推理获取 KV Cache (此时位置从 0 开始)
                output = self._model(
                    input_ids=block_input_ids, 
                    use_cache=True, 
                    past_key_values=DynamicCache(), 
                    return_dict=True
                )
                # 2. 逆旋转处理：使其变为 Position-agnostic
                pkv = apply_pkv_rerotary_position_embeddings(pkv=output.past_key_values, emb=self._emb)
                
            self.prepared_doc_cache.append(pkv)
            self.cached_tokens_len += block_input_ids.shape[1]
        
        await asyncio.sleep(0) 
    
    async def iter_generate(
        self,
        doc_ids: List[int],
        query: str,
        sampling_params: SamplingParams,
        position_ids: Optional[List[int]] = None,
    ) -> AsyncGenerator[str, None]:
        
        async with self._gpu_semaphore:
            time_start = time.perf_counter()
            
            # 1. 构造 Prompt
            instruction = f"<|start_header_id|>user<|end_header_id|>\n\n" \
                          f"{getattr(self, '_local_attention_suffix', '')}{query}<|eot_id|>" \
                          f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            
            resp_ids = torch.tensor([self._tokenizer.encode(instruction, add_special_tokens=False)], 
                                    device=self._model.device)
            
            curr_len = resp_ids.shape[1]

            if self.prepared_doc_cache:
                # 合并并旋转 Cache (和 block_generate_server.py 一致)
                kv_cache = merge_and_rotary_past_key_values(self.prepared_doc_cache, self._emb)
                # 【关键修复】：拼接block_input_ids和resp_ids（和block_generate_server.py一致）
                input_ids = torch.cat([self.block_input_ids, resp_ids], dim=-1)
            else:
                kv_cache = None
                input_ids = resp_ids

            logger.info(f'Setup time: {time.perf_counter() - time_start:.4f}s')

            # 【关键修复】：使用同步生成（和 block_generate_server.py 完全一致）
            # 不使用 AsyncTextIteratorStreamer，直接同步调用 generate
            input_length = input_ids.size(-1)
            
            outputs = self._model.generate(
                input_ids=input_ids,
                generation_config=transformers.GenerationConfig(
                    do_sample=False,
                    max_new_tokens=128,
                    eos_token_id=self._tokenizer.eos_token_id,
                    pad_token_id=self._tokenizer.pad_token_id,
                    stop_strings=['<|im_end|>', "<|eot_id|>", "</s>"]
                ),
                past_key_values=kv_cache,
                use_cache=True,
                eos_token_id=[self._tokenizer.eos_token_id],
                tokenizer=self._tokenizer
            )
            
            # 解码生成的文本（跳过input部分）
            generated_text = self._tokenizer.decode(outputs[0][input_length:].tolist())
            yield generated_text

    def destroy_cache(self, doc_ids: Optional[List[str]] = None) -> None:
        """清理当前请求的缓存状态"""
        self.prepared_doc_cache = []
        self.cached_tokens_len = 0
        self._local_attention_suffix = ""
    
    def get_stats_dict(self) -> Dict[str, Any]:
        """Return statistics dictionary for benchmarking"""
        return {}

class BlockAttentionRAGvLLM(RAG):
    def __init__(self, lm_name: str) -> None:
        RAG.__init__(self)
        self._docs: Dict[DocumentId, str] = {}
        self._last_request_id: int = 0

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=lm_name, use_fast=False)
        self._token_eos = self._tokenizer.eos_token_id
        self._max_tokens = 200
        self._document_max_len = 512
        model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=lm_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            # attn_implementation="flash_attention_2"
        )
        model.eval()
        config = transformers.AutoConfig.from_pretrained(pretrained_model_name_or_path=lm_name)
        emb: LlamaRotaryEmbedding = LlamaRotaryEmbedding(config=config).to(device=model.device, dtype=torch.float32)
        emb.eval()
        self._model = model
        self._emb = emb
        self.prepared_doc_cache = []
        self.doc_length_cache = []
        self.input_ids = None
        self._local_attention_suffix = ""

    def add_cache(self, docs: List[str]) -> List[int]:
        doc_ids = []
        for doc in docs:
            doc_id = len(self._docs)
            self._docs[doc_id] = doc
            doc_ids.append(doc_id)
        return doc_ids
    
    async def add_doc_async(self, request_id: str, docs_ids: List[int], num_local_attention_blocks: int = 10000) -> None:
        """For BlockAttention, we need to add documents asynchronously.
        
        Args:
            request_id: Request identifier
            docs_ids: List of document IDs to process
            num_local_attention_blocks: Number of blocks to use block attention for.
                Blocks beyond this limit will use local attention (concatenated with instruction).
        """
        # process doc
        self.prepared_doc_cache = []
        self.doc_length_cache = []
        self._device = self._model.device
        
        # Get documents
        blocks = [self._docs[doc_id] for doc_id in docs_ids]
        
        # Handle num_local_attention_blocks logic (from block_generate_server.py)
        # Blocks beyond num_local_attention_blocks will be concatenated to instruction
        if len(blocks) > num_local_attention_blocks:
            self._local_attention_suffix = "".join(blocks[num_local_attention_blocks:])
            blocks = blocks[:num_local_attention_blocks]
        else:
            self._local_attention_suffix = ""
        
        if num_local_attention_blocks == 0:
            self._local_attention_suffix = "".join(blocks)
            blocks = []
        
        token_list = []
        for b_idx, block in enumerate(blocks):
            with torch.no_grad():
                block_input_ids = torch.tensor(
                    data=[self._tokenizer.encode(block, add_special_tokens=False)],
                    dtype=torch.int64,
                    device=self._model.device
                )
                output: CausalLMOutputWithPast = self._model(
                    input_ids=block_input_ids, use_cache=True, past_key_values=DynamicCache(), return_dict=True
                )
                pkv = apply_pkv_rerotary_position_embeddings(pkv=output.past_key_values, emb=self._emb)
            self.prepared_doc_cache.append(pkv)
            self.doc_length_cache.append(block_input_ids.shape[1])
            token_list.append(block_input_ids)
        
        if token_list:
            self.input_ids = torch.cat(token_list, dim=-1)
        else:
            self.input_ids = None
        
        async def async_iter(data):
            for item in data:
                yield item
                await asyncio.sleep(0) 
        async for _ in async_iter([1, 2, 3]):
            pass
    
    async def iter_generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: SamplingParams,
        position_ids: Optional[List[int]] = None,
    ) -> AsyncGenerator[str, None]:
        # rotate the cache
        time_start = time.perf_counter()
        
        # Prepend local attention suffix (blocks that use local attention instead of block attention)
        # This matches block_generate_server.py logic
        instruction = getattr(self, '_local_attention_suffix', '') + query
        instruction = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nPlease write a high-quality answer for the given question using only the provided search documents (some of which might be irrelevant)" + instruction
        
        response_input_ids = torch.tensor(
            data=[self._tokenizer.encode(instruction, add_special_tokens=False)],
            dtype=torch.int64,
            device=self._model.device
        )
        
        # Handle case when there are no block attention caches
        if self.prepared_doc_cache:
            kv_cache = merge_and_rotary_past_key_values(self.prepared_doc_cache, self._emb)
            input_ids = torch.cat(tensors=[self.input_ids, response_input_ids], dim=-1)
        else:
            kv_cache = None
            input_ids = response_input_ids
        # print('input_ids', input_ids)
        # print('kv_cache', kv_cache.key_cache[0].shape)
        logger.info(f'rotate time: {time.perf_counter() - time_start}')

        streamer = AsyncTextIteratorStreamer(self._tokenizer)
        generation_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "max_new_tokens": 128,
            "min_new_tokens": 128,
            "past_key_values": kv_cache,
            "generation_config": transformers.GenerationConfig(
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.0,
                num_beams=1,
                eos_token_id=self._tokenizer.eos_token_id,
                max_new_tokens=128,
                stop_strings=['<|im_end|>', "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>", "</s>", "Question:"]
            ),
            "use_cache": True,
            "eos_token_id": [self._token_eos],
            "tokenizer": self._tokenizer,
        }
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()
        # skip the first text
        first_text = True
        async for new_text in streamer:
            if first_text:
                first_text = False
                continue
            logger.info(f"new_text: {new_text}")
            yield new_text

    def destroy_cache(self, doc_ids: Optional[List[str]] = None) -> None:
        pass



class TransformerRAG(RAG):
    def __init__(self, lm_name: str, method: str) -> None:
        RAG.__init__(self)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(lm_name)
        self._token_eos = self._tokenizer.eos_token_id
        self._max_tokens = 128
        self._document_max_len = 512
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            lm_name, device_map="balanced", offload_folder="offload"
        )
        self._model.eval()

        self._method = method
        self._preamble = "Below we provide information and a related query. Answer the query as accurately as you can."

        self._docs: Dict[DocumentId, str] = {}
        self._last_request_id = 0

    def add_cache(self, docs: List[str]) -> List[int]:
        doc_ids = []
        for doc in docs:
            doc_id = len(self._docs)
            self._docs[doc_id] = doc
            doc_ids.append(doc_id)
        return doc_ids
    
    async def add_doc_async(self, request_id: str, docs_ids: List[int]) -> None:
        pass

    async def iter_generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: SamplingParams,
        position_ids: Optional[List[int]] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate answer for a given query."""
        documents = [self._docs[doc_id] for doc_id in doc_ids]

        # TODO: Yield from these generate methods.
        kv_cache = DynamicCache()
        if self._method == "r1":
            # regular generation
            generated_text = self._generate_r1(kv_cache=kv_cache, query=query, documents=documents)
        elif self._method == "r2":
            # regular generation with preamble
            generated_text = self._generate_r2(kv_cache=kv_cache, query=query, documents=documents)
        elif self._method == "m1":
            # masked generation
            generated_text = self._generate_m1(kv_cache=kv_cache, query=query, documents=documents)
            logger.info(f"generated_text: {generated_text}")
        elif self._method == "m2":
            # masked generation with preamble
            generated_text = self._generate_m2(kv_cache=kv_cache, query=query, documents=documents)
        elif self._method == "m2v2":
            # masked generation with preamble in an individual document
            generated_text = self._generate_m2v2(kv_cache=kv_cache, query=query, documents=documents)
        elif self._method == "m3":
            # masked generation with repeated query
            generated_text = self._generate_m3(kv_cache=kv_cache, query=query, documents=documents)
        else:
            raise ValueError(f"Invalid method: {self._method}")

        del kv_cache

        # release GPU memory
        torch.cuda.empty_cache()

        yield generated_text

    def destroy_cache(self, doc_ids: Optional[List[str]] = None) -> None:
        pass

    def _generate_r1(self, kv_cache: DynamicCache, query: str, documents: List[str]) -> str:
        """Regular generation."""
        output_tokens = torch.tensor([], dtype=torch.int64).to(self._device)
        for doc in documents:
            _, kv_cache = self.prefill(doc, kv_cache)
        next_token, kv_cache = self.prefill(query, kv_cache)  # next token [token_id]
        output_tokens = torch.cat([output_tokens, next_token])

        for i in range(self._max_tokens - 1):
            next_token, kv_cache = self.decode(next_token.unsqueeze(0), kv_cache)
            output_tokens = torch.cat([output_tokens, next_token])
            if next_token == self._token_eos:
                # logger.debug(f"EOS token found when {i + 1} tokens generated.")
                break
        # logger.debug(f"Generated {len(output_tokens)} tokens.\n {output_tokens}")
        return self._tokenizer.decode(output_tokens)

    def _generate_r2(self, kv_cache: DynamicCache, query: str, documents: List[str]) -> str:
        """Regular generation with preamble."""
        _, kv_cache = self.prefill(self._preamble, kv_cache)
        return self._generate_r1(kv_cache, query, documents)

    def _generate_m1(self, kv_cache: DynamicCache, query: str, documents: List[str]) -> str:
        """Masked generation."""
        output_tokens = torch.tensor([], dtype=torch.int64).to(self._device)
        for doc in documents:
            past_len = kv_cache.get_seq_length()
            # logger.debug(f"Masked - Past length: {past_len}")
            current_len = self._tokenizer(doc, return_tensors="pt").input_ids.shape[1]
            # logger.debug(f"Masked - Current length: {current_len}")
            attention_mask = torch.cat([torch.zeros(past_len), torch.ones(current_len)]).unsqueeze(0)
            _, kv_cache = self.prefill(doc, kv_cache, attention_mask)
            # logger.debug(f"Masked - Attention mask: {attention_mask}")
        next_token, kv_cache = self.prefill(query, kv_cache)
        output_tokens = torch.cat([output_tokens, next_token])

        for i in range(self._max_tokens - 1):
            next_token, kv_cache = self.decode(next_token.unsqueeze(0), kv_cache)
            output_tokens = torch.cat([output_tokens, next_token])
            if next_token == self._token_eos:
                # logger.debug(f"EOS token found when {i + 1} tokens generated.")
                break
        return self._tokenizer.decode(output_tokens)
    
    async def _iter_generate_m1(self, kv_cache: DynamicCache, query: str, documents: List[str]) -> str:
        """Masked generation - async version."""
        output_tokens = torch.tensor([], dtype=torch.int64).to(self._device)
        
        for doc in documents:
            past_len = kv_cache.get_seq_length()
            current_len = self._tokenizer(doc, return_tensors="pt").input_ids.shape[1]
            attention_mask = torch.cat([torch.zeros(past_len), torch.ones(current_len)]).unsqueeze(0)
            next_token, kv_cache = self.prefill(doc, kv_cache, attention_mask)
        
        next_token, kv_cache = self.prefill(query, kv_cache)
        output_tokens = torch.cat([output_tokens, next_token])

        for i in range(self._max_tokens - 1):
            next_token, kv_cache = await self.decode_async(next_token.unsqueeze(0), kv_cache)
            output_tokens = torch.cat([output_tokens, next_token])
            if next_token == self._token_eos:
                break
        
        return self._tokenizer.decode(output_tokens)  

    def _generate_m2(self, kv_cache: DynamicCache, query: str, documents: List[str]) -> str:
        """Masked generation with preamble."""
        output_tokens = torch.tensor([], dtype=torch.int64).to(self._device)
        _, kv_cache = self.prefill(self._preamble, kv_cache)
        preamble_len = kv_cache.get_seq_length()
        for doc in documents:
            past_len = kv_cache.get_seq_length()
            current_len = self._tokenizer(doc, return_tensors="pt").input_ids.shape[1]
            attention_mask = torch.cat(
                [torch.ones(preamble_len), torch.zeros(past_len - preamble_len), torch.ones(current_len)]
            ).unsqueeze(0)
            _, kv_cache = self.prefill(doc, kv_cache, attention_mask)
            # logger.debug(f"Masked - Attention mask: {attention_mask}")
        next_token, kv_cache = self.prefill(query, kv_cache)
        output_tokens = torch.cat([output_tokens, next_token])

        for i in range(self._max_tokens - 1):
            next_token, kv_cache = self.decode(next_token.unsqueeze(0), kv_cache)
            output_tokens = torch.cat([output_tokens, next_token])
            if next_token == self._token_eos:
                # logger.debug(f"EOS token found when {i + 1} tokens generated.")
                break
        return self._tokenizer.decode(output_tokens)

    def _generate_m2v2(self, kv_cache: DynamicCache, query: str, documents: List[str]) -> str:
        """Masked generation with preamble. Preamble in an individual document."""
        return self._generate_m1(kv_cache, query, [self._preamble] + documents)

    def _generate_m3(self, kv_cache: DynamicCache, query: str, documents: List[str]) -> str:
        """Masked generation with preamble and repeated query."""
        output_tokens = torch.tensor([], dtype=torch.int64).to(self._device)
        _, kv_cache = self.prefill(self._preamble, kv_cache)
        preamble_len = kv_cache.get_seq_length()

        next_token: Optional[torch.Tensor] = None
        assert len(documents) > 0, "What to do?"
        for doc in documents:
            dq = doc + " " + query
            past_len = kv_cache.get_seq_length()
            current_len = self._tokenizer(dq, return_tensors="pt").input_ids.shape[1]
            attention_mask = torch.cat(
                [torch.ones(preamble_len), torch.zeros(past_len - preamble_len), torch.ones(current_len)]
            ).unsqueeze(0)
            # logger.debug(f"Masked - Attention mask: {attention_mask}")
            next_token, kv_cache = self.prefill(dq, kv_cache, attention_mask)
        assert next_token is not None
        output_tokens = torch.cat([output_tokens, next_token])

        for i in range(self._max_tokens - 1):
            assert next_token is not None
            next_token, kv_cache = self.decode(next_token.unsqueeze(0), kv_cache)
            output_tokens = torch.cat([output_tokens, next_token])
            if next_token == self._token_eos:
                # logger.debug(f"EOS token found when {i + 1} tokens generated.")
                break
        return self._tokenizer.decode(output_tokens)

    def prefill(
        self, prompt: str, kv_cache: DynamicCache, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, DynamicCache]:
        """Prefill the key-value cache with the prompt."""
        with torch.no_grad():
            tokens = self._tokenizer(prompt, return_tensors="pt").input_ids
            for i in range(0, tokens.shape[1], self._document_max_len):
                chunk = tokens[:, i : i + self._document_max_len]
                outputs = self._model(
                    chunk.to(self._device),
                    past_key_values=kv_cache,
                    use_cache=True,
                    attention_mask=attention_mask.to(self._device) if attention_mask is not None else None,
                )
                logits = outputs.logits
                kv_cache = outputs.past_key_values
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)[0]
            return next_token, kv_cache

    def decode(self, in_tokens: torch.Tensor, kv_cache: DynamicCache) -> Tuple[torch.Tensor, DynamicCache]:
        """Decoding phase. Get a new token and update the key-value cache."""
        with torch.no_grad():
            outputs = self._model(in_tokens.to(self._device), past_key_values=kv_cache, use_cache=True)
            logits = outputs.logits
            kv_cache = outputs.past_key_values
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)[0]
            return next_token, kv_cache


PROMPT_CACHE_SCHEMA_TEMPLATE = r"""
<schema name="{schema_name}">
<system/>
<user>
{documents}
</user>
</schema>
"""

PROMPT_CACHE_SCHEMA_DOCUMENT_TEMPLATE = r"""
<module name="{document_name}">
{document_text}
</module>
"""

PROMPT_CACHE_PROMPT_TEMPLATE = r"""
<prompt schema="{schema_name}">
{document_tags}
<user>
{prompt_text}
</user>
</prompt>
"""

PROMPT_CACHE_DOCUMENT_TAG_TEMPLATE = r"""<{document_name}/>"""


class PromptCacheRAG(RAG):

    def __init__(self, lm_name: str, max_ctx_length: int, enable_cpu_inference: bool, cache_max_token: int) -> None:
        RAG.__init__(self)
        # overwrite the lm_name to "meta-llama/Llama-3.1-8B-Instruct"
        # lm_name = "meta-llama/Llama-3.1-8B-Instruct"
        self._lm = PromptCacheRAG._load_lm(lm_name)
        self._cache_engine = promptcache.CacheEngine(
            max_ctx_length=max_ctx_length,
            lm=self._lm,
            target_device="cpu" if enable_cpu_inference else None,
        )
        self._gen_engine = promptcache.GenerationEngine(self._lm)
        self._cache_max_token = cache_max_token
        self._parameter = promptcache.GenerationParameters(
            temperature=1.0,
            repetition_penalty=1.0,
            top_p=0.95,
            top_k=-1,
            max_new_tokens=512,
            stop_token_ids=self._lm.stop_token_ids,
            stop_str=self._lm.stop_str,
        )
        self._docs: Dict[DocumentId, str] = {}
        self._cached_schemas: Dict[frozenset[DocumentId], str] = {}
        self._sync_lock = asyncio.Lock()
        self._last_request_id: int = 0

    @staticmethod
    def _load_lm(lm_name: str) -> promptcache.model.LanguageModel:
        # return promptcache.model.AutoModel(lm_name)
        if lm_name == "meta-llama/Llama-3.1-8B-Instruct" or "tulu" in lm_name.lower():
            return promptcache.model.AutoModel(lm_name)
        elif "llama" in lm_name.lower():
            return promptcache.model.CodeLlama(lm_name, load_in_8bit=True, device_map="auto")
        else:
            raise ValueError(f"Invalid language model name {lm_name}")

    # From promptcache::benchmark/longbench.py
    @staticmethod
    def _escape_tags(input_str):
        # pattern = r'<(?P<content>.*?)>'

        # # The lambda function ensures only the first letter is capitalized
        # def repl(match):
        #     return '(' + match.group("content").capitalize() + ')'
        #
        # return re.sub(pattern, repl, input_str)
        return input_str.replace("<", "(").replace(">", ")")

    def add_cache(self, docs: List[str]) -> List[int]:
        doc_ids = []
        for doc in docs:
            doc_id = len(self._docs)
            self._docs[doc_id] = PromptCacheRAG._escape_tags(doc)
            doc_ids.append(doc_id)
        return doc_ids
    
    async def add_doc_async(self, request_id: str, docs_ids: List[int]) -> None:
        schema_start_time = time.time()
        schema_name, schema = await self._load_schema_if_not_cached(frozenset(docs_ids))
        schema_end_time = time.time()
        schema_time = float((schema_end_time - schema_start_time)*1000)
        logger.info(f"Schema generation time: {schema_time:.2f} ms")
        self._schema_name = schema_name
        self._schema = schema

    async def _load_schema_if_not_cached(self, doc_set: FrozenSet[DocumentId]) -> str:
        # Synchronously check cache and allocate new schema if needed.
        async with self._sync_lock:
            if doc_set in self._cached_schemas:
                return self._cached_schemas[doc_set]
            schema_name = f"schema_{len(self._cached_schemas)}"
            self._cached_schemas[doc_set] = schema_name

        # Compile all documents into XML schema.
        documents: List[str] = []
        for doc_id in doc_set:
            doc = self._docs[doc_id]
            documents.append(PROMPT_CACHE_SCHEMA_DOCUMENT_TEMPLATE.format(document_name=f"doc_{doc_id}", document_text=doc))
        schema_text = PROMPT_CACHE_SCHEMA_TEMPLATE.format(schema_name=schema_name, documents="\n".join(documents))
        preprocessed_schema_text = self._lm.get_formatter()(schema_text)
        schema = promptcache.Schema(
            preprocessed_schema_text,
            lm=self._lm,
            max_tokens=self._cache_max_token,
        )

        # Add to cache engine.
        self._cache_engine.add_schema(schema, max_tokens=self._cache_max_token)
        self._cached_schemas[doc_set] = schema_name
        logger.info(f"Generated and added PromptCache schema {schema_name} of length {len(schema)}")
        return schema_name, schema

    async def iter_generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: SamplingParams,
        position_ids: Optional[List[int]] = None,
    ) -> AsyncGenerator[str, None]:
        # Compile XML prompt.
        document_tags = [PROMPT_CACHE_DOCUMENT_TAG_TEMPLATE.format(document_name=f"doc_{doc_id}") for doc_id in doc_ids]
        prompt_text = PROMPT_CACHE_PROMPT_TEMPLATE.format(
            schema_name=self._schema_name, document_tags="\n".join(document_tags), prompt_text=query
        )
        prompt = promptcache.Prompt(spec=prompt_text, preproc=[self._lm.get_formatter()])  # type: ignore

        # Process cache.
        token_ids, position_ids, cache_time, cache = self._cache_engine.process(
            prompt=prompt,
            return_full_position_ids=self._lm.use_full_position_ids,
        )

        prompt_inference_start_time = time.time()
        # Generate response.
        output_stream = self._gen_engine.generate(
            token_ids=token_ids,
            position_ids=position_ids,
            params=self._parameter,
            cache=cache,
            stream_interval=1,
            use_full_position_ids=self._lm.use_full_position_ids,
        )
        prompt_inference_end_time = time.time()
        prompt_inference_time = float((prompt_inference_end_time - prompt_inference_start_time)*1000)
        logger.info(f"Prompt inference time: {prompt_inference_time:.2f} ms")

        # Parse response from output stream. Copied from promptcache::eval.py.
        pre = 0
        for outputs in output_stream:
            output_text = outputs.new_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                tt = " ".join(output_text[pre:now])
                yield tt + " "
                pre = now
        tt = " ".join(output_text[pre:])
        yield tt

    def destroy_cache(self, doc_ids: Optional[List[str]] = None) -> None:
        for _, schema_name in self._cached_schemas:
            if self._cache_engine.get_schema(schema_name) is not None:
                self._cache_engine.remove_schema(schema_name)

# simple adapted from LLMRAG          
class LazyRAG(RAG):
    def __init__(self, llm: "AsyncLLM") -> None:
        RAG.__init__(self)
        self._llm = llm
        self._docs: Dict[DocumentId, str] = {}
        self._last_request_id: int = 0
        self.doc_ids = set()
        self._prepared_doc_ids: Set[DocumentId] = set()
        self._prepare_lock = asyncio.Lock()
        self._log_document_seq = (
            os.environ.get("LAZY_RAG_LOG_DOCUMENT_SEQ", "0") == "1"
        )

    def add_cache(self, docs: List[str]) -> List[int]:
        doc_ids = []
        for doc in docs:
            doc_id = len(self._docs)
            self._docs[doc_id] = doc
            doc_ids.append(doc_id)
        return doc_ids
    
    async def add_doc_async(self, request_id: str, docs_ids: List[int]) -> None:
        if not docs_ids:
            return

        async with self._prepare_lock:
            missing_doc_ids = [
                doc_id for doc_id in docs_ids
                if doc_id not in self._prepared_doc_ids
            ]
            if not missing_doc_ids:
                return

            for idx, doc_id in enumerate(missing_doc_ids):
                doc = self._docs[doc_id]
                async for _ in self._llm.generate(
                    prompt=doc,
                    sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
                    request_id=f"cache_{request_id}_{idx}_{uuid.uuid4().hex}",
                ):
                    pass
                self._prepared_doc_ids.add(doc_id)

    async def iter_generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: SamplingParams,
        position_ids: Optional[List[int]] = None,
    ) -> AsyncGenerator[str, None]:
        self.doc_ids.update(doc_ids)

        request_id = self._next_request_id()
        latest_idx = 0
        document = [self._docs[doc_id] for doc_id in doc_ids]
        if isinstance(document, str):
            document_seq = [document]
        else:
            document_seq = document

        # adjust doc seq
        # preamble = document_seq[0]
        # docs = document_seq[1:]
        # # reserve the order of docs
        # docs = docs[::-1]
        # document_seq = [preamble] + docs
        if self._log_document_seq:
            print("document_seq", document_seq)
            print("query", query)
        async for generate_output in self._llm.generate(
            prompt=query,
            sampling_params=sampling_params,
            request_id=request_id,
            document_seq=document_seq,
        ):
            if len(generate_output.outputs) > 1:
                logger.warning(f"Found {len(generate_output.outputs)} outputs, yielding first one.")
            prev_latest_idx = latest_idx
            latest_idx = len(generate_output.outputs[0].text)
            if prev_latest_idx < latest_idx:
                yield generate_output.outputs[0].text[prev_latest_idx:]

    def destroy_cache(self, doc_ids: Optional[List[str]] = None) -> None:
        # Automatically handled by the LLM engine.
        pass

    def _next_request_id(self) -> str:
        request_id = str(self._last_request_id)
        self._last_request_id += 1
        return request_id

class BaselineLazyRAG(RAG):
    def __init__(self, llm: "AsyncLLM") -> None:
        RAG.__init__(self)
        self._llm = llm
        self._docs: Dict[DocumentId, str] = {}
        self._last_request_id: int = 0
        self._prepare_prefix_cache = (
            os.environ.get("BASELINE_PREPARE_PREFIX_CACHE", "0") == "1"
        )
        self._prepared_prefixes: Set[Tuple[DocumentId, ...]] = set()
        self._prepare_lock = asyncio.Lock()

    def add_cache(self, docs: List[str]) -> List[int]:
        doc_ids = []
        for doc in docs:
            doc_id = len(self._docs)
            self._docs[doc_id] = doc
            doc_ids.append(doc_id)
        return doc_ids

    async def add_doc_async(self, request_id: str, docs_ids: List[int]) -> None:
        if (not self._prepare_prefix_cache) or (not docs_ids):
            return

        prefix_keys = [tuple(docs_ids[: idx + 1]) for idx in range(len(docs_ids))]
        async with self._prepare_lock:
            missing_prefixes = [
                prefix_key for prefix_key in prefix_keys
                if prefix_key not in self._prepared_prefixes
            ]
            if not missing_prefixes:
                return

            for prefix_idx, prefix_key in enumerate(missing_prefixes):
                prompt = "".join(self._docs[doc_id] for doc_id in prefix_key)
                async for _ in self._llm.generate(
                    prompt=prompt,
                    sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
                    request_id=f"cache_{request_id}_{prefix_idx}_{uuid.uuid4().hex}",
                    document_seq=None,
                ):
                    pass
                self._prepared_prefixes.add(prefix_key)

    async def iter_generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: SamplingParams,
        position_ids: Optional[List[int]] = None,
    ) -> AsyncGenerator[str, None]:
        request_id = self._next_request_id()
        document = [self._docs[doc_id] for doc_id in doc_ids]
        if isinstance(document, str):
            context = document
        else:
            context = "".join(document)
        prompt = context + query
        latest_idx = 0
        async for generate_output in self._llm.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            document_seq=None,
        ):
            if len(generate_output.outputs) > 1:
                logger.warning(f"Found {len(generate_output.outputs)} outputs, yielding first one.")
            prev_latest_idx = latest_idx
            latest_idx = len(generate_output.outputs[0].text)
            if prev_latest_idx < latest_idx:
                yield generate_output.outputs[0].text[prev_latest_idx:]

    def destroy_cache(self, doc_ids: Optional[List[str]] = None) -> None:
        # Automatically handled by the LLM engine.
        pass

    def _next_request_id(self) -> str:
        request_id = str(self._last_request_id)
        self._last_request_id += 1
        return request_id


@dataclasses.dataclass
class RAGArgs:
    rag_type: str = "parrot"  # RAG model name.

    cachep_type: str = "lru"  # [CacheParrotRAG] Cache policy (e.g., no, seq, ptree, lru)
    cachep_capacity: int = 615_000  # [CacheParrotRAG] Cache capacity in tokens
    cachep_tokenizer: str = "ldsjmdy/Tulu3-Block-FT"  # [CacheParrotRAG] Tokenizer name

    trrag_method: str = "r1"  # [TransformerRAG] Prompting method [r1, r2, m1, m2, m3].
    trrag_lm_name: str = "ldsjmdy/Tulu3-Block-FT"  # [TransformerRAG] Language model name.

    pc_lm_name: str = "codellama/CodeLlama-7b-Instruct-hf"  # [PromptCacheRAG] Language model name.
    pc_max_ctx_length: int = 90000  # [PromptCacheRAG] Max context length.
    pc_enable_cpu_inference: bool = False  # [PromptCacheRAG] Inference on CPU.
    pc_cache_max_token: int = 800  # [PromptCacheRAG] Max tokens for document cache.


def make_cache_manager(args: RAGArgs) -> CacheManager:
    if args.cachep_type == "no":
        return NoCacheManager()
    elif args.cachep_type == "seq":
        return SequenceCacheManager(capacity=args.cachep_capacity)
    elif args.cachep_type == "ptree":
        return PrefixTreeCacheManager(capacity=args.cachep_capacity)
    elif args.cachep_type == "lru":
        return LRUCacheManager(capacity=args.cachep_capacity)
    logger.error(f"Invalid CacheManager type {args.cachep_type}")
    sys.exit(1)
    
def prepare_lmcache(async_engine_args: AsyncEngineArgs)-> None:
    import os
    # LMCache-related environment variables
    # Use experimental features in LMCache
    os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
    # LMCache is set to use 256 tokens per chunk
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    # Enable local CPU backend in LMCache
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    # Set local CPU memory limit to 5.0 GB
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "30.0"
    
    from vllm.config import KVTransferConfig
    lmcache_connector = "LMCacheConnectorV1"
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )
    
    async_engine_args.kv_transfer_config = ktc
    return async_engine_args


def make_rag(args: RAGArgs, engine_args: EngineArgs = EngineArgs()) -> RAG:
    # Avoid touching CUDA in the parent process for plain AsyncLLM/vLLM
    # baselines, since they may start worker processes via fork.
    if os.environ.get("VLLM_USE_LAZY_ATTENTION", "0") == "1":
        torch.cuda.empty_cache()
    engine_args.max_model_len = 8192 * 8 * 2
    # engine_args.dtype = "float32"
    # engine_args.compilation_config = None
    if args.rag_type == "parrot":
        return ParrotRAG()
    elif args.rag_type == "cachep":
        cache_manager = make_cache_manager(args)
        return CacheParrotRAG(tokenizer_id=args.cachep_tokenizer, cache_manager=cache_manager)
    elif args.rag_type == "llmrag":
        from lazy.ctxmgr import LazyAttentionContextManager
        LazyAttentionContextManager.apply_triton_backend()
        async_engine_args = AsyncEngineArgs(**dataclasses.asdict(engine_args))
        logger.info(f"[llmrag] Using async engine args: {async_engine_args}")
        return LLMRAG(llm=AsyncLLM.from_engine_args(async_engine_args))
    elif args.rag_type == "recllmrag": # full recomputation
        from lazy.ctxmgr import LazyAttentionContextManager
        LazyAttentionContextManager.apply_triton_backend()
        async_engine_args = AsyncEngineArgs(**dataclasses.asdict(engine_args))
        async_engine_args.enable_prefix_caching=False
        logger.info(f"[recllmrag] Using async engine args: {async_engine_args}")
        return RecomLLMRAG(llm=AsyncLLM.from_engine_args(async_engine_args))
    elif args.rag_type == "reullmrag": # full reuse
        from lazy.ctxmgr import LazyAttentionContextManager
        LazyAttentionContextManager.apply_triton_backend()
        async_engine_args = AsyncEngineArgs(**dataclasses.asdict(engine_args))
        logger.info(f"[reullmrag] Using async engine args: {async_engine_args}")
        return ReuseLLMRAG(llm=AsyncLLM.from_engine_args(async_engine_args))
    elif args.rag_type == "blockattnrag":
        return BlockAttentionRAG(lm_name=args.trrag_lm_name)
    elif args.rag_type == "cacheblend":
        engine_args.max_model_len = int(engine_args.max_model_len/10)
        engine_args = EngineArgs(**dataclasses.asdict(engine_args))
        logger.info(f"[cacheblend] Using engine args: {engine_args}")
        return CacheBlendRAG(llm=LLM(**dataclasses.asdict(engine_args)))
    elif args.rag_type == "trrag":
        return TransformerRAG(lm_name=args.trrag_lm_name, method=args.trrag_method)
    elif args.rag_type == "pcrag":
        return PromptCacheRAG(
            lm_name=args.pc_lm_name,
            max_ctx_length=args.pc_max_ctx_length,
            enable_cpu_inference=args.pc_enable_cpu_inference,
            cache_max_token=args.pc_cache_max_token,
        )
    elif args.rag_type == "lazyrag":
        async_engine_args = AsyncEngineArgs(**dataclasses.asdict(engine_args))
        logger.info(f"[lazyrag] Using async engine args: {async_engine_args}")
        return LazyRAG(llm=AsyncLLM.from_engine_args(async_engine_args))
    elif args.rag_type == "baseline":
        async_engine_args = AsyncEngineArgs(**dataclasses.asdict(engine_args))
        # prepare_lmcache(async_engine_args)
        logger.info(f"[baseline] Using async engine args: {async_engine_args}")
        return BaselineLazyRAG(llm=AsyncLLM.from_engine_args(async_engine_args))
    logger.error(f"Invalid RAG type {args.rag_type}")
    sys.exit(1)
    
