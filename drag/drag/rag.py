"""RAGs"""

import asyncio
import dataclasses
import sys
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import promptcache
import promptcache.model
import torch
import transformers
from transformers.cache_utils import DynamicCache

from drag.document import Document
from drag.logging import logger
from drag.utils import get_block_size, get_gpu_cache, get_model_runner, get_tokenizer
from vllm import LLM
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.model_executor.sampling_metadata import SamplingMetadata, SequenceGroupToSample
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SequenceData

try:
    from vllm.attention.backends.xformers import XFormersMetadata
except Exception as e:
    logger.error(f"Failed to import XFormersMetadata: {e}")

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

    def __init__(self, llm: AsyncLLMEngine) -> None:
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

    async def iter_generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: SamplingParams,
        position_ids: Optional[List[int]] = None,
    ) -> AsyncGenerator[str, None]:
        context = "\n\n".join([self._docs[doc_id] for doc_id in doc_ids])
        prompt = context + "\n\n" + query
        request_id = self._next_request_id()
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
            yield generate_output.outputs[0].text[prev_latest_idx:]

    def destroy_cache(self, doc_ids: Optional[List[str]] = None) -> None:
        pass

    def _next_request_id(self) -> str:
        request_id = str(self._last_request_id)
        self._last_request_id += 1
        return request_id


class TransformerRAG(RAG):
    def __init__(self, lm_name: str, method: str) -> None:
        RAG.__init__(self)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(lm_name)
        self._token_eos = self._tokenizer.eos_token_id
        self._max_tokens = 200
        self._document_max_len = 512
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            lm_name, device_map="balanced", offload_folder="offload"
        )
        self._model.eval()

        self._method = method
        self._preamble = "Below we provide information and a related query. Answer the query as accurately as you can."

        self._docs: Dict[DocumentId, str] = {}

    def add_cache(self, docs: List[str]) -> List[int]:
        doc_ids = []
        for doc in docs:
            doc_id = len(self._docs)
            self._docs[doc_id] = doc
            doc_ids.append(doc_id)
        return doc_ids

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
        elif self._method == "m2":
            # masked generation with preamble
            generated_text = self._generate_m2(kv_cache=kv_cache, query=query, documents=documents)
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
        output_tokens = torch.tensor([], dtype=torch.int64)
        for doc in documents:
            _, kv_cache = self.prefill(doc, kv_cache)
        next_token, kv_cache = self.prefill(query, kv_cache)  # next token [token_id]
        output_tokens = torch.cat([output_tokens, next_token])

        for i in range(self._max_tokens - 1):
            next_token, kv_cache = self.decode(next_token.unsqueeze(0), kv_cache)
            output_tokens = torch.cat([output_tokens, next_token])
            if next_token == self._token_eos:
                logger.debug(f"EOS token found when {i + 1} tokens generated.")
                break
        logger.debug(f"Generated {len(output_tokens)} tokens.\n {output_tokens}")
        return self._tokenizer.decode(output_tokens)

    def _generate_r2(self, kv_cache: DynamicCache, query: str, documents: List[str]) -> str:
        """Regular generation with preamble."""
        _, kv_cache = self.prefill(self._preamble, kv_cache)
        return self._generate_r1(kv_cache, query, documents)

    def _generate_m1(self, kv_cache: DynamicCache, query: str, documents: List[str]) -> str:
        """Masked generation."""
        output_tokens = torch.tensor([], dtype=torch.int64)
        for doc in documents:
            past_len = kv_cache.get_seq_length()
            logger.debug(f"Masked - Past length: {past_len}")
            current_len = self._tokenizer(doc, return_tensors="pt").input_ids.shape[1]
            logger.debug(f"Masked - Current length: {current_len}")
            attention_mask = torch.cat([torch.zeros(past_len), torch.ones(current_len)]).unsqueeze(0)
            _, kv_cache = self.prefill(doc, kv_cache, attention_mask)
            logger.debug(f"Masked - Attention mask: {attention_mask}")
        next_token, kv_cache = self.prefill(query, kv_cache)
        output_tokens = torch.cat([output_tokens, next_token])

        for i in range(self._max_tokens - 1):
            next_token, kv_cache = self.decode(next_token.unsqueeze(0), kv_cache)
            output_tokens = torch.cat([output_tokens, next_token])
            if next_token == self._token_eos:
                logger.debug(f"EOS token found when {i + 1} tokens generated.")
                break
        return self._tokenizer.decode(output_tokens)

    def _generate_m2(self, kv_cache: DynamicCache, query: str, documents: List[str]) -> str:
        """Masked generation with preamble."""
        output_tokens = torch.tensor([], dtype=torch.int64)
        _, kv_cache = self.prefill(self._preamble, kv_cache)
        preamble_len = kv_cache.get_seq_length()
        for doc in documents:
            past_len = kv_cache.get_seq_length()
            current_len = self._tokenizer(doc, return_tensors="pt").input_ids.shape[1]
            attention_mask = torch.cat(
                [torch.ones(preamble_len), torch.zeros(past_len - preamble_len), torch.ones(current_len)]
            ).unsqueeze(0)
            _, kv_cache = self.prefill(doc, kv_cache, attention_mask)
            logger.debug(f"Masked - Attention mask: {attention_mask}")
        next_token, kv_cache = self.prefill(query, kv_cache)
        output_tokens = torch.cat([output_tokens, next_token])

        for i in range(self._max_tokens - 1):
            next_token, kv_cache = self.decode(next_token.unsqueeze(0), kv_cache)
            output_tokens = torch.cat([output_tokens, next_token])
            if next_token == self._token_eos:
                logger.debug(f"EOS token found when {i + 1} tokens generated.")
                break
        return self._tokenizer.decode(output_tokens)

    def _generate_m3(self, kv_cache: DynamicCache, query: str, documents: List[str]) -> str:
        """Masked generation with preamble and repeated query."""
        output_tokens = torch.tensor([], dtype=torch.int64)
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
            logger.debug(f"Masked - Attention mask: {attention_mask}")
            next_token, kv_cache = self.prefill(dq, kv_cache, attention_mask)
        assert next_token is not None
        output_tokens = torch.cat([output_tokens, next_token])

        for i in range(self._max_tokens - 1):
            assert next_token is not None
            next_token, kv_cache = self.decode(next_token.unsqueeze(0), kv_cache)
            output_tokens = torch.cat([output_tokens, next_token])
            if next_token == self._token_eos:
                logger.debug(f"EOS token found when {i + 1} tokens generated.")
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
                outputs = self._model(chunk, past_key_values=kv_cache, use_cache=True, attention_mask=attention_mask)
                logits = outputs.logits
                kv_cache = outputs.past_key_values
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)[0]
            return next_token, kv_cache

    def decode(self, in_tokens: torch.Tensor, kv_cache: DynamicCache) -> Tuple[torch.Tensor, DynamicCache]:
        """Decoding phase. Get a new token and update the key-value cache."""
        with torch.no_grad():
            outputs = self._model(in_tokens, past_key_values=kv_cache, use_cache=True)
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

    @staticmethod
    def _load_lm(lm_name: str) -> promptcache.model.LanguageModel:
        if lm_name == "meta-llama/Llama-3.1-8B-Instruct":
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
        return schema_name

    async def iter_generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: SamplingParams,
        position_ids: Optional[List[int]] = None,
    ) -> AsyncGenerator[str, None]:
        schema_name = await self._load_schema_if_not_cached(frozenset(doc_ids))

        # Compile XML prompt.
        document_tags = [PROMPT_CACHE_DOCUMENT_TAG_TEMPLATE.format(document_name=f"doc_{doc_id}") for doc_id in doc_ids]
        prompt_text = PROMPT_CACHE_PROMPT_TEMPLATE.format(
            schema_name=schema_name, document_tags="\n".join(document_tags), prompt_text=query
        )
        prompt = promptcache.Prompt(spec=prompt_text, preproc=[self._lm.get_formatter()])  # type: ignore

        # Process cache.
        token_ids, position_ids, cache_time, cache = self._cache_engine.process(
            prompt=prompt,
            return_full_position_ids=self._lm.use_full_position_ids,
        )

        # Generate response.
        output_stream = self._gen_engine.generate(
            token_ids=token_ids,
            position_ids=position_ids,
            params=self._parameter,
            cache=cache,
            stream_interval=1,
            use_full_position_ids=self._lm.use_full_position_ids,
        )

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


@dataclasses.dataclass
class DynamicOutput:
    prompt_token_ids: List[int]
    block_table: List[int]
    next_token_id: int


class DynamicRAG(RAG):
    _count: int = -1

    def __init__(self, llm: LLM):
        RAG.__init__(self)
        self.llm = llm  # LLM instance frmo vllm
        self.model = get_model_runner(self.llm).model
        self.tokenizer = get_tokenizer(self.llm)
        self.generator = torch.Generator(device="cuda:0").manual_seed(2024)

        self.block_size = get_block_size(self.llm)
        self.kv_cache = get_gpu_cache(self.llm)[0]
        self.set_used_blocks = set()

        self.stage = None
        self.cached_documents: Dict[int, Document] = {}

    def add_cache(self, docs: List[str]) -> List[DocumentId]:
        """
        Add the given documents to the cache for reusing. Specifically, each
        document will generate kv cache for further query. Only prefill
        operation is involved during this process.
        :param docs: each doc is typically a long promt.
        :return: a list of document ids
        """
        doc_ids = []
        for doc in docs:
            document = Document(doc, self.llm)
            self.cached_documents[document.doc_id] = document
            # TODO: unify forbid block free
            self.set_used_blocks.update(document.block_table)
            doc_ids.append(document.doc_id)
        return doc_ids

    def prefill(
        self,
        seq_id: int,
        prompt_token_ids: List[int],
        query_token_ids: List[int],
        block_table: List[int],
        output_token_ids: List[int],  # not used
        sampling_params: SamplingParams,
    ) -> DynamicOutput:
        query_len = len(query_token_ids)
        context_len = len(prompt_token_ids)
        seq_len = context_len + query_len

        # allocate new blocks for query tokens
        query_block_ids, slot_mapping = DynamicRAG._allocate_block_and_slot(query_len, self.set_used_blocks, self.block_size)

        input_ids = torch.tensor(query_token_ids).cuda()  # dtype=torch.int32).cuda()
        position_ids = (torch.arange(query_len) + context_len).cuda()

        seq_lens = [seq_len]
        ctx_lens = [context_len]

        # get hidden status (complete prefill)
        attn_metadata = DynamicRAG._build_attn_metadata(
            num_prefill_tokens=query_len,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            ctx_lens=ctx_lens,
            seq_lens=seq_lens,
            block_table=block_table,
        )

        hidden_states = self.model(
            input_ids=input_ids,
            positions=position_ids,
            kv_caches=self.kv_cache,
            attn_metadata=attn_metadata,
        )

        # get a new token after prefill
        prompt_token_ids.extend(query_token_ids)
        seq_data = SequenceData.from_seqs(prompt_token_ids=prompt_token_ids)

        sampling_metadata = DynamicRAG._build_sampling_metadata(
            seq_ids=[seq_id],
            sampling_params=sampling_params,
            seq_data={seq_id: seq_data},
            seq_len=seq_lens[0],
            query_len=query_len,
            generator=self.generator,
            is_prompt=True,
        )

        # TODO(haocheng): return logits instead of token
        logits = self.model.compute_logits(hidden_states, sampling_metadata)
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        next_token_id = int(torch.argmax(logprobs, dim=-1).cpu())

        block_table.extend(query_block_ids)
        self.set_used_blocks.update(query_block_ids)
        logger.debug(f"Block table after prefill {block_table}")
        return DynamicOutput(prompt_token_ids=prompt_token_ids, block_table=block_table, next_token_id=next_token_id)

    def decode(
        self,
        seq_id: int,
        prompt_token_ids: List[int],
        block_table: List[int],
        output_token_ids: List[int],  # not used
        sampling_params: SamplingParams,
    ) -> DynamicOutput:
        seq_len = len(prompt_token_ids) + len(output_token_ids)
        context_len = seq_len - 1
        seq_lens = [seq_len]
        ctx_lens = [context_len]

        new_block_ids = None
        # check if we need new block
        if int(np.ceil(seq_len / self.block_size)) > len(block_table):
            # allocate new block table
            new_block_ids, slot_mapping = DynamicRAG._allocate_block_and_slot(
                1, self.set_used_blocks, self.block_size  # query is the last token
            )
        else:
            # do not need new block
            tail_block = block_table[-1]
            slot_mapping = [tail_block * self.block_size + ((seq_len - 1) % self.block_size)]

        input_ids = torch.tensor(output_token_ids[-1:]).cuda()
        position_ids = torch.tensor([seq_len - 1]).cuda()

        # get hidden status (complete prefill)
        attn_metadata = DynamicRAG._build_attn_metadata(
            num_prefill_tokens=0,
            num_decode_tokens=1,
            slot_mapping=slot_mapping,
            ctx_lens=ctx_lens,
            seq_lens=seq_lens,
            block_table=block_table,
        )

        hidden_states = self.model(
            input_ids=input_ids,
            positions=position_ids,
            kv_caches=self.kv_cache,
            attn_metadata=attn_metadata,
        )

        seq_data = SequenceData.from_seqs(prompt_token_ids=prompt_token_ids, output_token_ids=output_token_ids)

        seq_data._num_computed_tokens = seq_len - 1

        sampling_metadata = DynamicRAG._build_sampling_metadata(
            seq_ids=[seq_id],
            sampling_params=sampling_params,
            seq_data={seq_id: seq_data},
            seq_len=seq_lens[0],
            query_len=1,
            generator=self.generator,
            is_prompt=False,
        )

        # TODO(haocheng): return logits instead of token
        logits = self.model.compute_logits(hidden_states, sampling_metadata)
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        next_token_id = int(torch.argmax(logprobs, dim=-1).cpu())

        if new_block_ids is not None:
            block_table.extend(new_block_ids)
            self.set_used_blocks.update(new_block_ids)

        return DynamicOutput(prompt_token_ids=prompt_token_ids, block_table=block_table, next_token_id=next_token_id)

    def step(
        self,
        stage: str,
        seq_id: int,
        prompt_token_ids: List[int],
        query_token_ids: Optional[List[int]],
        block_table: List[int],
        output_token_ids: List[int],
        sampling_params: SamplingParams,
    ) -> DynamicOutput:
        """
        Generate one token each step
        """

        if stage == "prefill":
            outputs = self.prefill(seq_id, prompt_token_ids, query_token_ids, block_table, output_token_ids, sampling_params)
        elif stage == "decode":
            outputs = self.decode(seq_id, prompt_token_ids, block_table, output_token_ids, sampling_params)
        else:
            raise ValueError("Invalid generate stage.")
        return outputs

    async def iter_generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: Optional[SamplingParams] = None,
        position_ids: Optional[List[int]] = None,  # not used
    ) -> AsyncGenerator[str, None]:
        """
        Perform inference with the given documents and query.
        :param doc_ids: context documents
        :param query: query text
        :param sampling_params: vLLM SamplingParams
        :param position_ids: not used yet
        :return:
        """
        assert sampling_params.max_tokens > 0

        # init generate
        seq_id = Document.next()
        block_table = []
        prompt_token_ids = []
        query_token_ids = []
        output_token_ids = []

        for doc_id in doc_ids:
            block_table.extend(self.cached_documents[doc_id].block_table)
            prompt_token_ids.extend(self.cached_documents[doc_id].token_ids)
        query_token_ids = self.tokenizer.encode(query)

        for i in range(sampling_params.max_tokens):
            if i == 0:
                stage = "prefill"
            else:
                stage = "decode"
            outputs = self.step(
                stage, seq_id, prompt_token_ids, query_token_ids, block_table, output_token_ids, sampling_params
            )

            # update for next step
            prompt_token_ids = outputs.prompt_token_ids
            block_table = outputs.block_table
            next_token_id = outputs.next_token_id
            output_token_ids.append(next_token_id)

            next_token = self.tokenizer.batch_decode([next_token_id])
            yield next_token

    def destroy_cache(self, doc_ids: Optional[List[str]] = None) -> None:
        if doc_ids is None:
            doc_ids = list(self.cached_documents.keys())
        for doc_id in doc_ids:
            self.cached_documents[doc_id].destroy()

    @classmethod
    def next(cls):
        cls._count += 1
        return cls._count + 1

    @staticmethod
    def _allocate_block_and_slot(num_tokens: int, set_used_blocks: set, block_size: int = 16) -> Tuple[List[int], List[int]]:
        num_needed_blocks = int(np.ceil(num_tokens / block_size))
        num_tail_tokens = num_tokens % block_size if num_tokens % block_size != 0 else block_size
        num_used_blocks = len(set_used_blocks)
        set_candidate_blocks = set(np.arange(num_needed_blocks + num_used_blocks, dtype=np.int32))
        logger.debug(f"original candidate blocks {set_candidate_blocks}")
        set_candidate_blocks.difference_update(set_used_blocks)
        logger.debug(f"filtered candidate blocks {set_candidate_blocks}")
        allocated_block_ids = list(set_candidate_blocks)[:num_needed_blocks]
        # convert to slot mapping
        slot_mapping = []
        for i in range(num_needed_blocks):
            start_slot_idx = allocated_block_ids[i] * block_size
            if i != num_needed_blocks - 1:  # not reach tail
                slot_mapping.extend(list(np.arange(start_slot_idx, start_slot_idx + block_size)))
            else:  # reach tail
                slot_mapping.extend(list(np.arange(start_slot_idx, start_slot_idx + num_tail_tokens)))

        return allocated_block_ids, slot_mapping

    @staticmethod
    def _build_attn_metadata(
        num_prefill_tokens: int,
        num_decode_tokens: int,
        slot_mapping: List[int],
        ctx_lens: List[int],
        seq_lens: List[int],
        block_table: List[int],
    ):
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64).cuda()
        logger.debug(f"Slot mapping is {slot_mapping}")
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32).cuda()
        ctx_lens_tensor = torch.tensor(ctx_lens, dtype=torch.int32).cuda()
        block_tables = torch.tensor([block_table], dtype=torch.int32).cuda()

        if num_prefill_tokens > 0:
            attn_metadata = XFormersMetadata(
                num_prefills=1,
                num_prefill_tokens=num_prefill_tokens,
                num_decode_tokens=0,
                slot_mapping=slot_mapping,
                seq_lens=seq_lens,
                seq_lens_tensor=seq_lens_tensor,
                max_query_len=num_prefill_tokens,
                max_prefill_seq_len=seq_lens[0],
                max_decode_seq_len=0,
                query_start_loc=torch.tensor([0, num_prefill_tokens]).cuda(),
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
                max_query_len=1,
                max_prefill_seq_len=0,
                max_decode_seq_len=seq_lens[0],
                query_start_loc=torch.tensor([0, 1]).cuda(),
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

    @staticmethod
    def _build_sampling_metadata(
        seq_ids: List[int],
        sampling_params: SamplingParams,
        seq_data: Dict[int, SequenceData],
        seq_len: int,
        query_len: int,
        generator: torch.Generator,
        is_prompt: bool,
    ):

        logger.debug(f"query len is {query_len}")
        return SamplingMetadata(
            seq_groups=[
                SequenceGroupToSample(
                    seq_ids=seq_ids,
                    sampling_params=sampling_params,
                    seq_data=seq_data,
                    seq_len=seq_len,
                    query_len=query_len,
                    generator=generator,
                    is_prompt=is_prompt,
                    prompt_logprob_indices=[],
                    sample_indices=[0],
                )
            ],
            selected_token_indices=torch.tensor([query_len - 1], dtype=torch.int32).cuda(),
            categorized_sample_indices={
                SamplingType.GREEDY: torch.tensor([], dtype=torch.int32).cuda(),
                SamplingType.RANDOM: torch.tensor([], dtype=torch.int32).cuda(),
                SamplingType.RANDOM_SEED: torch.tensor([], dtype=torch.int32).cuda(),
            },
            num_prompts=1,
        )


@dataclasses.dataclass
class RAGArgs:
    rag_type: str = "parrot"  # RAG model name.

    trrag_method: str = "r1"  # [TransformerRAG] Prompting method [r1, r2, m1, m2, m3].
    trrag_lm_name: str = "meta-llama/Llama-3.1-8B-Instruct"  # [TransformerRAG] Language model name.

    pc_lm_name: str = "codellama/CodeLlama-7b-Instruct-hf"  # [PromptCacheRAG] Language model name.
    pc_max_ctx_length: int = 5000  # [PromptCacheRAG] Max context length.
    pc_enable_cpu_inference: bool = False  # [PromptCacheRAG] Inference on CPU.
    pc_cache_max_token: int = 800  # [PromptCacheRAG] Max tokens for document cache.


def make_rag(args: RAGArgs, engine_args: EngineArgs = EngineArgs()) -> RAG:
    if args.rag_type == "parrot":
        return ParrotRAG()
    elif args.rag_type == "llmrag":
        async_engine_args = AsyncEngineArgs(**dataclasses.asdict(engine_args))
        return LLMRAG(llm=AsyncLLMEngine.from_engine_args(async_engine_args))
    elif args.rag_type == "trrag":
        return TransformerRAG(lm_name=args.trrag_lm_name, method=args.trrag_method)
    elif args.rag_type == "pcrag":
        return PromptCacheRAG(
            lm_name=args.pc_lm_name,
            max_ctx_length=args.pc_max_ctx_length,
            enable_cpu_inference=args.pc_enable_cpu_inference,
            cache_max_token=args.pc_cache_max_token,
        )
    elif args.rag_type == "drag":
        return DynamicRAG(llm=LLM(**dataclasses.asdict(engine_args)))
    logger.error(f"Invalid RAG type {args.rag_type}")
    sys.exit(1)
