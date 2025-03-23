from dataclasses import dataclass
from typing import AsyncGenerator, List, Tuple

import torch
from drag.document import Document
from drag.RAGRequest import RAGRequest
from drag.RAGScheduler import  RAGScheduler, RAGSchedulerOutput
from drag.RAGSequence import RAGSequence
from drag.KV_cache_manager import KV_Cache_Manager
from drag.utils import get_block_size, get_cpu_cache, get_gpu_cache, get_llm_engine, get_model_runner, get_num_cpu_blocks, get_num_gpu_blocks, get_tokenizer
from drag.VllmMetadataBuilder import VllmMetadataBuilder
from vllm import LLM
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sampling_params import SamplingParams

@dataclass
class RAGRequestOutput:
    # the output for one step execution of the RAGRequest
    req: RAGRequest
    origin_seq: RAGSequence
    req_finished: bool

@dataclass
class RAGgenerateOutput:
    prompt:str
    doc_ids:List[int]
    generated_text:str


class RAGEngine(object):
    def __init__(self, docDB: dict[int, Document], llm: LLM, sampling_params: SamplingParams):
        self.sample_params = sampling_params
        self.documents = docDB
        self.llm = llm
        self.scheduler = self._init_scheduler()
        self.seq_id_counter = 0
        self.tokenizer = get_tokenizer(self.llm)
        self.model_runner = get_model_runner(self.llm)
        self.generator = torch.Generator(device="cuda:0").manual_seed(2024)

    def generate(self, queries: List[str], query_doc_ids:List[List[int]]) -> List[RAGgenerateOutput]:
        query_token_ids: List[List[int]] = [self.tokenizer.encode(query) for query in queries]
        
        gen_output:Tuple[str,list[int],str] = []
        seqs:RAGSequence = []
        for doc_ids, query_token, query in zip(query_doc_ids, query_token_ids, queries):
            doc_token_ids = [self.documents[doc_id].token_ids for doc_id in doc_ids]
            seqs.append(RAGSequence(self.seq_id_counter, doc_ids, query, query_token, doc_token_ids,[])) 
        
        self.scheduler.add_sequence(seqs)
        finished_seqs: List[RAGSequence] = []
        while self.scheduler.has_unfinished_seqs():
            output = self._step()
            for out in output:
                if out.req_finished and out.req.get_type() == "QUERY":
                    finished_seqs.append(out.origin_seq)

        for seq in finished_seqs:
            generated_text = self.tokenizer.decode(seq.generated_token_ids)
            gen_output.append(RAGgenerateOutput(seq.query_text, seq.doc_ids, generated_text))
        return gen_output
        

    # async def iter_generate(self, queries: List[str], query_doc_ids:List[List[int]]) -> AsyncGenerator[List[RAGgenerateOutput]]:
    #     #todo: implement async version of generate
    #     pass
    def _init_scheduler(self) -> RAGScheduler:
        block_size = get_block_size(self.llm)
        gpu_kv_cache = get_gpu_cache(self.llm)[0]
        cpu_kv_cache = get_cpu_cache(self.llm)[0]
        num_gpu_blocks = get_num_gpu_blocks(self.llm)
        num_cpu_blocks = get_num_cpu_blocks(self.llm)
        kv_cache_manager = KV_Cache_Manager(block_size, gpu_kv_cache, cpu_kv_cache, num_gpu_blocks, num_cpu_blocks)
        return RAGScheduler(512,self.documents,kv_cache_manager)
    
    def _step(self) -> List[RAGRequestOutput]:
        output:List[RAGRequestOutput] = []

        prefill_reqs, decode_reqs = self.scheduler.schedule()

        # execute the prefill reqs
        prefill_model_input = self._scheduler_output_to_batch_model_input(
            prefill_reqs
        )
        prefill_output: List[SamplerOutput] = self.model_runner.execute_model(
            model_input=prefill_model_input,
            kv_caches=self.kv_cache,
            intermediate_tensors=None,
            num_steps=1,
        )

        # decode tokens to text
        batch_next_token_ids = [output.sampled_token_ids[0] for output in prefill_output]

        # update the seq generated token ids info (prefilling phase's generated token won't be the exit token)
        for i in range(len(prefill_reqs)):
            seq_id = prefill_reqs.rag_req[i].original_seq_id
            seq = self.scheduler.seq_id_to_seqs[seq_id]
            if prefill_reqs.rag_req[i].get_type() == "QUERY":
                if batch_next_token_ids[i] == self.sample_params.stop:
                    self.scheduler.finish_seq(seq_id)
                else:
                    seq.generated_token_ids.append(batch_next_token_ids[i])
                output.append(RAGRequestOutput(prefill_reqs.rag_req[i], seq, batch_next_token_ids[i] == self.sample_params.stop))

            else:
                output.append(RAGRequestOutput(prefill_reqs.rag_req[i], seq, True)) #we're not supporting chunked prefilling, so all prefilling doc_cache requests are finished
                


        
        # execute the decode reqs
        decode_model_input = self._scheduler_output_to_batch_model_input(
            decode_reqs
        )
        decode_output: List[SamplerOutput] = self.model_runner.execute_model(
            model_input=decode_model_input,
            kv_caches=self.kv_cache,
            intermediate_tensors=None,
            num_steps=1,
        )


        # decode tokens to text
        batch_next_token_ids = [output.sampled_token_ids[0] for output in decode_output]
        

        # tell scheduler to end seqs that have reached the exit tokens
        for i in range(len(decode_reqs)):
            assert decode_reqs.rag_req[i].get_type() == "QUERY"
            seq_id = decode_reqs.rag_req[i].original_seq_id
            seq = self.scheduler.seq_id_to_seqs[seq_id]
            if batch_next_token_ids[i] == self.sample_params.stop:
                self.scheduler.finish_seq(seq_id)
            else:
                seq.generated_token_ids.append(batch_next_token_ids[i])
            output.append(RAGRequestOutput(decode_reqs.rag_req[i], seq, batch_next_token_ids[i] == self.sample_params.stop))

        # output the generated text 
        return output

    
    def _scheduler_output_to_batch_model_input(self, schedulerOutput: RAGSchedulerOutput) -> dict:
        model_input = VllmMetadataBuilder.build_batch_model_input(
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


