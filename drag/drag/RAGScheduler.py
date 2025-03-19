from dataclasses import dataclass
import math
from typing import List, Union, cast

from drag.RAGSequence import RAGSequence
from drag.RAGRequest import CacheDocRequest, QueryRequest, RAGRequest, RAGRequestType
from drag.RAGRequest import RAGRequestPhase
from drag.KV_cache_manager import KV_Cache_Manager
from drag.document import Document


@dataclass
class RAGSchedulerOutput:
    batch_size: int
    batch_query_lens: List[int] # length of kv_cache to be computed (i.e., length of tokens to be processed) in this step
    batch_context_lens: List[int] # length of already-computed kv_cache
    batch_seq_lens: List[int] # for encoding, query_lens + documents' length. # for decoding, query_lens + documents' length + generated tokens length
    batch_ctx_token_ids: List[List[int]]
    batch_query_token_ids: List[List[int]]
    batch_num_prefill_tokens: List[int] # for decode, all 0. for prefill, the number of tokens to be prefilled.
    batch_block_tables: List[List[int]] # block ids of the already computed kv_cache
    batch_slot_mapping: List[List[int]] # slot ids of the new kv_cache to be computed
    batch_output_token_ids: List[List[int]] # the generated output token ids of the requests in decoding phase
    rag_req: List[RAGRequest] # the RAGRequest id, an original sequence may generate multiple CacheDocRequest and one QueryRequest

@dataclass
class PrefillRAGSchedulerOutput(RAGSchedulerOutput):
    pass  # Inherits all fields from RAGSchedulerOutput


@dataclass
class DecodeRAGSchedulerOutput(RAGSchedulerOutput):
    pass


class RAGScheduler:
    def __init__(self, max_token_per_seq:int, docDB:dict[int, Document]):
        self.docDB = docDB
        self.kv_cache_manager = KV_Cache_Manager()
        self.RequestCounter = 0
        self.MAX_TOKEN_PER_SEQ = max_token_per_seq # number of maximum token allowed to generated
        self.NUM_DECODE_TOKEN_PER_STEP = 1 # number of tokens to generate in each step (decode)

        
        self.preempt_seq_list:List[RAGSequence] = []
        # update when new sequences are added
        self.new_seq_list:List[RAGSequence] = []
        self.seq_id_to_seqs:dict[int, RAGSequence] = {}

        # update when trying to add new sequences to running list
        self.seq_id_to_request:dict[int, List[RAGRequest]] = {}

        self.running_list:List[RAGRequest] = []
        self.prio_wait_list:List[QueryRequest] = []

        # update before adding to running list
        self.req_id_to_slot_mapping:dict[int, List[int]] = {}
        self.req_id_to_new_block_ids:dict[int, List[int]] = {}

        # update before and after running
        self.docs_to_be_filled_next:set[int] = set()

        # update after running
        self.req_id_to_phase:dict[int, RAGRequestPhase] = {}
        self.req_id_to_block_table:dict[int, List[int]] = {}
        self.req_id_to_num_cached_token:dict[int, int] = {}




    def schedule(self) -> tuple[PrefillRAGSchedulerOutput, DecodeRAGSchedulerOutput]:
        for req in self.running_list:
            # all prefill requests (including doc cache requests) from the previous step should have already finished
            assert req.get_type() == RAGRequestType.QUERY
            assert self.req_id_to_phase[req.request_id] == RAGRequestPhase.DECODE
            
            if self.req_id_to_generated_token_num[req.request_id] >= self.MAX_TOKEN_PER_SEQ:
                self.running_list.remove(req)
                #TODO: raise warning if the request is added to the error list

            # reserve more space for the request to run in this step
            slot_mapping,new_block_ids = self.kv_cache_manager.append_slots(req, self.NUM_DECODE_TOKEN_PER_STEP)
            # update req slot mapping for the new tokens to be generated
            if slot_mapping is not None:
                self.req_id_to_slot_mapping[req.request_id] = slot_mapping
                self.req_id_to_new_block_ids[req.request_id] = new_block_ids
            else:
                seq_id = req.sequence_id
                seq = self.seq_id_to_seqs[seq_id]
                self.preempt_seq_list.append(seq)
                for req_in_seq in self.seq_id_to_request[seq_id]:
                    self.kv_cache_manager.free(req_in_seq)
                    self.running_list.remove(req_in_seq)

        for req in self.prio_wait_list:
            # the only reason for a request to enter prio_wait_list is that in last step, there's pending dependant doc_cache requests
            # the slots for the requests in prio_wait_list is already preserved
            assert self.req_id_to_phase[req.request_id] == RAGRequestPhase.PREFILL
            assert req.type == RAGRequestType.QUERY
            self.running_list.append(req)

        for seq in self.preempt_seq_list:
            if self._try_add_seq_to_running_list(self.seq_id_to_request[seq.sequence_id]):
                self.preempt_seq_list.remove(seq)
                
                
        for seq in self.new_seq_list:
            if not seq.sequence_id in self.seq_id_to_request:
                self._seq_to_reqs(seq)
            if self._try_add_seq_to_running_list(self.seq_id_to_request[seq.sequence_id]):
                self.new_seq_list.remove(seq)

        scheduler_output = self._gen_scheduler_output()
        self._update_request_state()
        return scheduler_output

                
    def add_sequence(self, seqs: List[RAGSequence]) -> None:
        self.new_seq_list.extend(seqs)
        self.seq_id_to_seqs.update({seq.sequence_id: seq for seq in seqs})


    def finish_seq(self, seq_id: int) -> None:
        for req in self.seq_id_to_request[seq_id]:
            self.kv_cache_manager.free(req)


    def has_unfinished_seqs(self) -> bool:
        return len(self.running_list) != 0 or len(self.prio_wait_list) != 0 \
            or len(self.preempt_seq_list) != 0 or len(self.new_seq_list) != 0


    def _try_add_seq_to_running_list(self, seq_id) -> bool:
        reqs = self.seq_id_to_request[seq_id]
        self._init_request_id_to_block_table(reqs, False)
        
     
        # step 1:check if gpu memory is enough for doing all the prefilling, and there's at least one block left after prefilling
        # otherwise, return False
        num_computed_blocks_list = [len(self.req_id_to_block_table[req.request_id]) for req in reqs]
        total_num_computed_blocks = sum(num_computed_blocks_list)
        num_needed_blocks_list = []
        total_num_needed_blocks = 0 # total num of blocks needed for prefilling
        for req in reqs:
            if req.get_type() == RAGRequestType.CACHE_DOC:
                num_needed_blocks = math.ceil(cast(CacheDocRequest, req).doc_length/self.kv_cache_manager.BLOCK_SIZE)
            else:
                num_needed_blocks = math.ceil(len(cast(QueryRequest, req).prompt)/self.kv_cache_manager.BLOCK_SIZE)
            num_needed_blocks_list.append(num_needed_blocks)
            total_num_needed_blocks += num_needed_blocks
        total_num_needed_blocks += 1 

        if total_num_needed_blocks - total_num_computed_blocks > self.kv_cache_manager.num_free_blocks:
            return False
        
        # step 2: swap in the cpu blocks if needed
        if self.kv_cache_manager.enable_swap_in_cpu_blocks:
            for req in reqs:
                self._init_request_id_to_block_table(req, True) # the swap must be successful because we've checked the gpu memory is enough
            num_computed_blocks_list = [len(self.req_id_to_block_table[req.request_id]) for req in reqs]


        
        # step 3: allocate slots for all doc_cache requests and add them to the running list if not already cached/added to running list
        all_doc_cached = True
        for i, req in enumerate(reqs[:-1]):
            assert req.get_type() == RAGRequestType.CACHE_DOC
            if num_computed_blocks_list[i] == num_needed_blocks_list[i]:
                self.kv_cache_manager.allocate_slots(req,0)
                if cast(CacheDocRequest, req).doc_id in self.docs_to_be_filled_next:
                    all_doc_cached = False # the doc is added to running list by another sequence, but it's not cached yet.
            else:
                all_doc_cached = False
                slot_mapping, new_block_ids = self.kv_cache_manager.allocate_slots(req, cast(CacheDocRequest, req).doc_length - num_computed_blocks_list[i] * self.kv_cache_manager.BLOCK_SIZE)
                assert slot_mapping is not None # since we have checked the gpu memory is enough
                self.req_id_to_slot_mapping[req.request_id] = slot_mapping
                self.req_id_to_new_block_ids[req.request_id] = new_block_ids
                self.docs_to_be_filled_next.add(cast(CacheDocRequest, req).doc_id)
                self.running_list.append(req)

        # step 4: allocate slots for the query request and add it to the running list
        req = reqs[-1]
        assert req.get_type() == RAGRequestType.QUERY
        if num_computed_blocks_list[-1] >= num_needed_blocks_list[-1]:
            assert all_doc_cached #if the query's prefill is cached, all docs it needs must already be cached
            #if already prefilled, add to running list as decode phase
            self.kv_cache_manager.allocate_slots(req, 0)
            # TODO: If we also consider prefix caching for query, it's possible the query is already in the running list, but not prefilled yet. In this case, we should not add the query of decoding phase into the running list directly. we need to add it to the prio_wait_list
            # reserve more space for the request to run in this step
            slot_mapping, new_block_ids = self.kv_cache_manager.append_slots(req, self.NUM_DECODE_TOKEN_PER_STEP) # TODO: this may generate error because we only checked if the gpu memory is enough for encoding. But since #computed blocks >= #needed blocks inherently means that previously the query was preempted due to lack of gpu memory in decoding phase, the same error may happen again.
            self.req_id_to_slot_mapping[req.request_id] = slot_mapping
            self.req_id_to_new_block_ids[req.request_id] = new_block_ids
            self.req_id_to_phase[req.request_id] == RAGRequestPhase.DECODE
            self.running_list.append(req)
        
        else:
            slot_mapping, new_block_ids = self.kv_cache_manager.allocate_slots(req, len(cast(QueryRequest, req).prompt) - num_computed_blocks_list[-1] * self.kv_cache_manager.BLOCK_SIZE)
            assert slot_mapping is not None
            self.req_id_to_slot_mapping[req.request_id] = slot_mapping
            self.req_id_to_new_block_ids[req.request_id] = new_block_ids
            self.req_id_to_phase[req.request_id] = RAGRequestPhase.PREFILL  
            if all_doc_cached:
                self.running_list.append(req)
            else:
                self.prio_wait_list.append(req)

        return True


    def _init_request_id_to_block_table(self, reqs:List[RAGRequest], swap_in_cpu_blocks=False) -> None:
        for req in reqs:
            self.req_id_to_block_table[req.request_id] = self.kv_cache_manager.get_computed_gpu_blocks(req, swap_in_cpu_blocks)
            self.req_id_to_num_cached_token = len(self.req_id_to_block_table[req.request_id]) * self.kv_cache_manager.BLOCK_SIZE
        

    def _seq_to_reqs(self, seq: RAGSequence) -> None:
        for doc_id in seq.doc_ids:
            self.seq_id_to_request[seq.sequence_id].append(CacheDocRequest(self.RequestCounter, doc_id, self.docDB[doc_id], seq.sequence_id))
            self.RequestCounter += 1
        self.seq_id_to_request[seq.sequence_id].append(QueryRequest(self.RequestCounter, seq.prompt, seq.doc_ids, seq.sequence_id))
        self.RequestCounter += 1


    def _gen_scheduler_output(self) -> tuple[PrefillRAGSchedulerOutput, DecodeRAGSchedulerOutput]:
        def initialize_batch():
            return {
                "batch_size": 0,
                "batch_query_lens": [],
                "batch_context_lens": [],
                "batch_seq_lens": [],
                "batch_ctx_token_ids": [],
                "batch_query_token_ids": [],
                "batch_num_prefill_tokens": [],
                "batch_block_tables": [],
                "batch_slot_mapping": [],
                "batch_output_token_ids": [],
                "rag_req": []
            }
        
        prefill_data = initialize_batch()
        decode_data = initialize_batch()

        for req in self.running_list:
            batch = prefill_data if req.get_type() == RAGRequestType.CACHE_DOC or self.req_id_to_phase[req.request_id] == RAGRequestPhase.PREFILL else decode_data
            batch["batch_size"] += 1
            batch["batch_query_lens"].append(len(self.req_id_to_slot_mapping[req.request_id]))
            batch["batch_context_lens"].append(self.req_id_to_num_cached_token[req.request_id])
            batch["batch_seq_lens"].append(
                cast(CacheDocRequest, req).doc_length if req.get_type() == RAGRequestType.CACHE_DOC
                else batch["batch_query_lens"][-1] + batch["batch_context_lens"][-1]
            )
            batch["batch_num_prefill_tokens"].append(batch["batch_query_lens"][-1] if batch is prefill_data else 0)
            batch["batch_block_tables"].append(self.req_id_to_block_table[req.request_id])
            batch["batch_slot_mapping"].append(self.req_id_to_slot_mapping[req.request_id])
            batch["rag_req"].append(req)

            if req.get_type() == RAGRequestType.CACHE_DOC:
                token_ids = cast(CacheDocRequest, req).doc_token_ids
                batch["batch_ctx_token_ids"].append(token_ids[:self.req_id_to_num_cached_token[req.request_id]])
                batch["batch_query_token_ids"].append(token_ids[self.req_id_to_num_cached_token[req.request_id]:])
                batch["batch_output_token_ids"].append([])
            else:
                seq = self.seq_id_to_seqs[req.original_seq_id]
                if self.req_id_to_phase[req.request_id] == RAGRequestPhase.DECODE:
                    batch["batch_output_token_ids"].append(seq.generated_token_ids)
                    batch["batch_ctx_token_ids"].append(seq.seq_token_ids[:-1])
                    batch["batch_query_token_ids"].append([seq.query_token_ids[-1]])
                else:
                    batch["batch_output_token_ids"].append([])
                    batch["batch_ctx_token_ids"].append(seq.seq_token_ids[:-batch["batch_query_lens"][-1]])
                    batch["batch_query_token_ids"].append(seq.seq_token_ids[-batch["batch_query_lens"][-1]:])

        return PrefillRAGSchedulerOutput(**prefill_data), DecodeRAGSchedulerOutput(**decode_data)

    
    def _update_request_state(self):
        for req in self.running_list:
            # change the prefill queryRequests to decode phase, and remove the cache_doc requests from running_list because they are done after this round.
            self.req_id_to_block_table[req.request_id].extend(self.req_id_to_new_block_ids)
            self.req_id_to_num_cached_token[req.request_id] += len(self.req_id_to_slot_mapping)
            if req.get_type() == RAGRequestType.QUERY and self.req_id_to_phase[req.request_id] == RAGRequestPhase.PREFILL:
                self.req_id_to_phase[req.request_id] == RAGRequestPhase.DECODE
            if req.get_type() == RAGRequestType.CACHE_DOC:
                self.running_list.remove(req)
                self.docs_to_be_filled_next.remove(cast(CacheDocRequest, req).doc_id)


        
