from typing import List

import torch
from drag.document import Document
from drag.RAGRequest import RAGRequest
from drag.KV_cache_manager import KV_Cache_Manager
from drag.RAGScheduler import PrefillRAGSchedulerOutput, RAGScheduler, RAGSchedulerOutput
from drag.RAGSequence import RAGSequence
from drag.rag import DynamicRAG
from drag.utils import get_model_runner
from vllm import LLM
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer


class RAGEngine(object):
    def __init__(self, docDB: dict[int, Document], llm: LLM, sampling_params: SamplingParams):
        self.sample_params = sampling_params
        self.documents = docDB
        self.llm = llm
        self.scheduler = RAGScheduler(512,docDB)
        self.seq_id_counter = 0
        self.tokenizer = get_tokenizer(self.llm)
        self.model_runner = get_model_runner(self.llm)
        self.generator = torch.Generator(device="cuda:0").manual_seed(2024)

    def generate(self, queries: List[str], query_doc_ids:List[List[int]]) -> List[Output]:
        query_token_ids: List[List[int]] = [self.tokenizer.encode(query) for query in queries]
        
        outputs = []
        seqs:RAGSequence = []
        for doc_ids, query in zip(query_doc_ids, query_token_ids):
            doc_token_ids = [self.documents[doc_id].token_ids for doc_id in doc_ids]
            seqs.append(RAGSequence(self.seq_id_counter, doc_ids, query, doc_token_ids,[]))
        
        self.scheduler.add_sequence(seqs)
        while self.scheduler.has_unfinished_seqs():
            output = self._step()


    
    def _step(self):
        prefill_reqs, decode_reqs = self.scheduler.schedule()
        # execute the prefill reqs
        prefill_model_input = self._build_batch_model_input(
            prefill_reqs
        )
        prefill_output: List[SamplerOutput] = self.model_runner.execute_model(
            model_input=prefill_model_input,
            kv_caches=self.kv_cache,
            intermediate_tensors=None,
            num_steps=1,
        )
        # decode tokens to text

        # update the seq token ids info
        
        # execute the decode reqs
        decode_model_input = self._build_batch_model_input(
            decode_reqs
        )
        decode_output: List[SamplerOutput] = self.model_runner.execute_model(
            model_input=decode_model_input,
            kv_caches=self.kv_cache,
            intermediate_tensors=None,
            num_steps=1,
        )
        # TODO:
        # update the seq token ids info

        # decode tokens to text
        

        # tell scheduler to end seqs that have reached the exit tokens

        # output the generated text

 
    
    def _build_batch_model_input(self, schedulerOutput: RAGSchedulerOutput) -> dict:
        model_input = DynamicRAG._build_batch_model_input(
            batch_size = schedulerOutput.batch_size,
            batch_query_lens = schedulerOutput.batch_query_lens,
            batch_context_lens = schedulerOutput.batch_context_lens,
            batch_seq_lens = schedulerOutput.batch_seq_lens,    
            batch_prompt_token_ids = schedulerOutput.batch_ctx_token_ids,
            batch_query_token_ids = schedulerOutput.batch_query_token_ids,
            batch_num_prefill_tokens = schedulerOutput.batch_num_prefill_tokens,
            batch_block_tables = schedulerOutput.batch_block_tables,
            batch_slot_mapping = schedulerOutput.batch_slot_mapping,
            batch_output_token_ids = schedulerOutput.batch_output_token_ids,
            req_ids = schedulerOutput.req_ids,
            seq_ids = schedulerOutput.req_ids,
            generator = self.generator,
            sampling_params = self.sample_params
        )
        return model_input


