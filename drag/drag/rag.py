"""RAGs"""

import dataclasses
import sys
from typing import Dict, Generator, List, Optional

import torch

from drag.document import Document
from drag.logging import logger
from drag.utils import get_gpu_cache, get_model_runner
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
        max_tokens=3,
    ) -> Generator[str, None, None]:
        raise NotImplementedError("Abstract method")

    def generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: SamplingParams,
        position_ids: Optional[List[int]] = None,
        max_tokens=3,
    ) -> List[str]:
        return list(
            self.iter_generate(
                doc_ids=doc_ids,
                query=query,
                sampling_params=sampling_params,
                position_ids=position_ids,
                max_tokens=max_tokens,
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
        max_tokens=3,
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
        max_tokens=3,
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


class DynamicRAG(RAG):
    _count: int = -1

    def __init__(self, llm: LLM):
        RAG.__init__(self)
        self.llm = llm  # LLM instance frmo vllm
        self.block_size = llm.llm_engine.cache_config.block_size

        self.cached_documents: Dict[int, Document] = {}

    def add_cache(self, docs: List[str]) -> List[DocumentId]:
        """
        Add the given documents to the cache for reusing.
        Specifically, each document will generate kv cache for further query.
        Only prefill operation is involved during this process.
        :param docs: each doc is typically a long promt.
        :return: a list of document ids
        """
        doc_ids = []
        for doc in docs:
            document = Document(doc, self.llm)
            self.cached_documents[document.doc_id] = document
            doc_ids.append(document.doc_id)
        return doc_ids

    def iter_generate(
        self,
        doc_ids: List[DocumentId],
        query: str,
        sampling_params: SamplingParams,
        position_ids: Optional[List[int]] = None,
        max_tokens=3,
    ) -> Generator[str, None, None]:
        """
        Perform inference with the given documents and query.
        :param doc_ids:
        :param query:
        :param position_ids:
        :return: retrieved documents
        """
        assert max_tokens > 0

        output_token_ids = []
        merged_block_table = []
        merged_token_ids = []
        num_reuse_blocks = 0
        for doc_id in doc_ids:
            merged_block_table.extend(self.cached_documents[doc_id].block_table)
            merged_token_ids.extend(self.cached_documents[doc_id].token_ids)
            num_reuse_blocks += self.cached_documents[doc_id].num_blocks

        tokenizer = self.llm.llm_engine.tokenizer
        query_token_ids = tokenizer.encode(query)
        input_ids = torch.tensor(query_token_ids).cuda()

        position_ids = (torch.arange(len(input_ids)) + num_reuse_blocks * self.block_size).cuda()

        kv_cache = get_gpu_cache(self.llm)[0]

        # TODO: adaptive
        slot_mapping = torch.clone(position_ids)
        seq_lens = [len(input_ids) + num_reuse_blocks * self.block_size]
        seq_lens_tensor = []
        seq_lens_tensor.extend(seq_lens)
        seq_lens_tensor = torch.tensor(seq_lens_tensor).cuda()
        # construct xformer metadata
        attn_metadata = XFormersMetadata(
            num_prefills=1,
            num_prefill_tokens=len(query_token_ids),
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor.to(torch.int32),
            max_query_len=len(query_token_ids),
            max_prefill_seq_len=seq_lens[0],
            max_decode_seq_len=0,
            query_start_loc=torch.tensor([0, len(query_token_ids)]).cuda(),
            context_lens_tensor=torch.tensor([num_reuse_blocks * self.block_size]).cuda(),
            block_tables=torch.tensor([merged_block_table], dtype=torch.int32).cuda(),
            use_cuda_graph=False,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=None,
            encoder_seq_lens_tensor=None,
            max_encoder_seq_len=None,
            cross_slot_mapping=None,
            cross_block_tables=None,
        )

        # get model
        model_executable = get_model_runner(self.llm).model  # LlamaForCausalLM
        hidden_states = model_executable(
            input_ids=input_ids,
            positions=position_ids,
            kv_caches=kv_cache,
            attn_metadata=attn_metadata,
        )

        seq_id = Document.next()
        merged_token_ids.extend(query_token_ids)
        seq_data = SequenceData.from_seqs(prompt_token_ids=merged_token_ids)
        sampling_metadata = SamplingMetadata(
            seq_groups=[
                SequenceGroupToSample(
                    seq_ids=[seq_id],
                    sampling_params=sampling_params,
                    seq_data={seq_id: seq_data},
                    seq_len=seq_lens[0],
                    query_len=len(query_token_ids),
                    generator=torch.Generator(device="cuda:0").manual_seed(2024),
                    is_prompt=True,
                    prompt_logprob_indices=[],
                    sample_indices=[0],
                )
            ],
            selected_token_indices=torch.tensor([len(query_token_ids) - 1], dtype=torch.int32).cuda(),
            categorized_sample_indices={
                SamplingType.GREEDY: torch.tensor([], dtype=torch.int32).cuda(),
                SamplingType.RANDOM: torch.tensor([], dtype=torch.int32).cuda(),
                SamplingType.RANDOM_SEED: torch.tensor([], dtype=torch.int32).cuda(),
            },
            num_prompts=1,
        )
        logits = model_executable.compute_logits(hidden_states, sampling_metadata)

        # probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        greedy_result = int(torch.argmax(logprobs, dim=-1).cpu())
        print(f"greedy results {greedy_result}")

        # next_tokens = model_executable.sample(logits, sampling_metadata)

        next_token_id = greedy_result  # next_tokens.outputs[0].samples[0].output_token
        output_token_ids.append(next_token_id)
        output = tokenizer.tokenizer.batch_decode([next_token_id])
        print("otk", output)
        yield output

        # ---------------------------------------------------------------------
        # repeated docode
        # ---------------------------------------------------------------------

        extended = False
        for _ in range(max_tokens - 1):
            seq_lens[0] += 1  # inc
            seq_lens_tensor[0] += 1
            input_ids = torch.tensor([next_token_id]).cuda()
            position_ids = torch.tensor([seq_lens[0] - 1]).cuda()

            if not extended:
                merged_block_table.extend([len(merged_block_table)])  # apply one more block
                extended = True

            slot_mapping = torch.clone(position_ids)
            attn_metadata = XFormersMetadata(
                num_prefills=0,  # now we only decode
                num_prefill_tokens=0,
                num_decode_tokens=1,
                slot_mapping=slot_mapping,
                seq_lens=seq_lens,
                seq_lens_tensor=seq_lens_tensor.to(torch.int32),
                max_query_len=1,
                max_prefill_seq_len=0,
                max_decode_seq_len=seq_lens[0],
                query_start_loc=torch.tensor([0, 1]).cuda(),
                context_lens_tensor=torch.tensor([seq_lens[0] - 1]).cuda(),
                block_tables=torch.tensor([merged_block_table], dtype=torch.int32).cuda(),
                use_cuda_graph=False,
                # Begin encoder & cross attn fields below...
                encoder_seq_lens=None,
                encoder_seq_lens_tensor=None,
                max_encoder_seq_len=None,
                cross_slot_mapping=None,
                cross_block_tables=None,
            )

            hidden_states = model_executable(
                input_ids=input_ids,
                positions=position_ids,
                kv_caches=kv_cache,
                attn_metadata=attn_metadata,
            )

            seq_data = SequenceData.from_seqs(prompt_token_ids=merged_token_ids, output_token_ids=output_token_ids)
            seq_data._num_computed_tokens = seq_lens[0] - 1
            sampling_metadata = SamplingMetadata(
                seq_groups=[
                    SequenceGroupToSample(
                        seq_ids=[seq_id],
                        sampling_params=sampling_params,
                        seq_data={seq_id: seq_data},
                        seq_len=None,
                        query_len=1,
                        generator=torch.Generator(device="cuda:0").manual_seed(2024),
                        is_prompt=False,
                        prompt_logprob_indices=[],
                        sample_indices=[0],
                    )
                ],
                selected_token_indices=torch.tensor([0], dtype=torch.int32).cuda(),
                categorized_sample_indices={
                    SamplingType.GREEDY: torch.tensor([], dtype=torch.int32).cuda(),
                    SamplingType.RANDOM: torch.tensor([], dtype=torch.int32).cuda(),
                    SamplingType.RANDOM_SEED: torch.tensor([], dtype=torch.int32).cuda(),
                },
                num_prompts=1,
            )

            logits = model_executable.compute_logits(hidden_states, sampling_metadata)
            # next_tokens = model_executable.sample(logits, sampling_metadata)

            # next_token_id = next_tokens.outputs[0].samples[0].output_token
            # probs = torch.softmax(logits, dim=-1, dtype=torch.float)
            # Compute the log probabilities.
            logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
            greedy_result = torch.argmax(logprobs, dim=-1).cpu()

            # next_tokens = model_executable.sample(logits, sampling_metadata)

            next_token_id = int(greedy_result)  # next_tokens.outputs[0].samples[0].output_token
            output_token_ids.append(next_token_id)
            output = tokenizer.tokenizer.batch_decode([next_token_id])
            print("otk", output)
            yield output

    def destroy_cache(self, doc_ids: Optional[List[str]] = None) -> None:
        if doc_ids is None:
            doc_ids = list(self.cached_documents.keys())
        for doc_id in doc_ids:
            self.cached_documents[doc_id].destroy()

    @classmethod
    def next(cls):
        cls._count += 1
        return cls._count + 1


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
