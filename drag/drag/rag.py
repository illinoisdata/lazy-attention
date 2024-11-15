"""RAGs"""

import dataclasses
import sys
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import torch

from drag.document import Document
from drag.logging import logger
from drag.utils import get_block_size, get_gpu_cache, get_model_runner, get_tokenizer
from vllm import LLM
from vllm.attention.backends.xformers import XFormersMetadata
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.sampling_metadata import SamplingMetadata, SequenceGroupToSample
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SequenceData

DocumentId = int


class RAG:

    def add_cache(self, docs: List[str]) -> List[DocumentId]:
        raise NotImplementedError("Abstract method")

    # TODO: Make this method async to stream output.
    def iter_generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: SamplingParams,
        position_ids: Optional[List[int]] = None,
    ) -> Generator[str, None, None]:
        raise NotImplementedError("Abstract method")

    def generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: SamplingParams,
        position_ids: Optional[List[int]] = None,
    ) -> List[str]:
        return list(
            self.iter_generate(
                doc_ids=doc_ids,
                query=query,
                sampling_params=sampling_params,
                position_ids=position_ids,
            )
        )

    def destroy_cache(self, doc_ids: Optional[List[str]] = None) -> None:
        raise NotImplementedError("Abstract method")


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

    def iter_generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: SamplingParams,
        position_ids: Optional[List[int]] = None,
    ) -> Generator[str, None, None]:
        yield query

    def destroy_cache(self, doc_ids: Optional[List[str]] = None) -> None:
        pass


class LLMRAG(RAG):

    def __init__(self, llm: LLM) -> None:
        RAG.__init__(self)
        self._llm = llm
        self._docs: Dict[DocumentId, str] = {}

    def add_cache(self, docs: List[str]) -> List[int]:
        doc_ids = []
        for doc in docs:
            doc_id = len(self._docs)
            self._docs[doc_id] = doc
            doc_ids.append(doc_id)
        return doc_ids

    def iter_generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: SamplingParams,
        position_ids: Optional[List[int]] = None,
    ) -> Generator[str, None, None]:
        # TODO: Use AsyncLLMEngine to stream output.
        context = "\n\n".join([self._docs[doc_id] for doc_id in doc_ids])
        prompt = context + "\n\n" + query
        generate_outputs = self._llm.generate(
            prompt,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        for generate_output in generate_outputs:
            for output in generate_output.outputs:
                yield output.text

    def destroy_cache(self, doc_ids: Optional[List[str]] = None) -> None:
        pass


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
        query_block_ids, slot_mapping = DynamicRAG._allocate_block_and_slot(
            query_len, 
            self.set_used_blocks,
            self.block_size
        )
                
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
        return DynamicOutput(prompt_token_ids=prompt_token_ids,
                             block_table=block_table,
                             next_token_id=next_token_id)

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

    def iter_generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: Optional[SamplingParams] = None,
        position_ids: Optional[List[int]] = None,  # not used
    ) -> Generator[str, None, None]:
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
    def _allocate_block_and_slot(num_tokens: int, 
                        set_used_blocks: set, 
                        block_size: int = 16) -> Tuple[List[int], List[int]]:
        num_needed_blocks = int(np.ceil(num_tokens / block_size))
        num_tail_tokens = (num_tokens % block_size 
                           if num_tokens % block_size != 0 
                           else block_size)
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
    def _build_attn_metadata(num_prefill_tokens: int,
                             num_decode_tokens: int,
                             slot_mapping: List[int],
                             ctx_lens: List[int],
                             seq_lens: List[int],
                             block_table: List[int],):
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64).cuda()
        logger.debug(f"Slot mapping is {slot_mapping}")
        seq_lens_tensor=torch.tensor(seq_lens, dtype=torch.int32).cuda()
        ctx_lens_tensor=torch.tensor(ctx_lens, dtype=torch.int32).cuda()
        block_tables=torch.tensor([block_table], dtype=torch.int32).cuda()

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


def make_rag(args: RAGArgs, engine_args: EngineArgs = EngineArgs()) -> RAG:
    if args.rag_type == "parrot":
        return ParrotRAG()
    elif args.rag_type == "llmrag":
        return LLMRAG(llm=LLM(**dataclasses.asdict(engine_args)))
    elif args.rag_type == "drag":
        return DynamicRAG(llm=LLM(**dataclasses.asdict(engine_args)))
    logger.error(f"Invalid RAG type {args.rag_type}")
    sys.exit(1)
