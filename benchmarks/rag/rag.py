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
    logger.warning("Missing vllm.v1.engine.async_llm.AsyncLLM")

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

    trrag_lm_name: str = "ldsjmdy/Tulu3-Block-FT"  # [BlockAttentionRAGvLLM] Language model name.


def make_rag(args: RAGArgs, engine_args: EngineArgs = EngineArgs()) -> RAG:
    # Avoid touching CUDA in the parent process for plain AsyncLLM/vLLM
    # baselines, since they may start worker processes via fork.
    if os.environ.get("VLLM_USE_LAZY_ATTENTION", "0") == "1":
        torch.cuda.empty_cache()
    if engine_args.max_model_len is None:
        engine_args.max_model_len = 8192 * 8 * 2
    # engine_args.dtype = "float32"
    # engine_args.compilation_config = None
    if args.rag_type == "parrot":
        return ParrotRAG()
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
        return BlockAttentionRAGvLLM(lm_name=args.trrag_lm_name)
    elif args.rag_type == "lazyrag":
        async_engine_args = AsyncEngineArgs(**dataclasses.asdict(engine_args))
        logger.info(f"[lazyrag] Using async engine args: {async_engine_args}")
        return LazyRAG(llm=AsyncLLM.from_engine_args(async_engine_args))
    elif args.rag_type == "baseline":
        async_engine_args = AsyncEngineArgs(**dataclasses.asdict(engine_args))
        logger.info(f"[baseline] Using async engine args: {async_engine_args}")
        return BaselineLazyRAG(llm=AsyncLLM.from_engine_args(async_engine_args))
    logger.error(f"Invalid RAG type {args.rag_type}")
    sys.exit(1)
    
