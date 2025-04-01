# unit test of the RAGScheduler class.
import math
from typing import List, Tuple, cast
import pytest
from unittest.mock import MagicMock
from drag.RAGScheduler import RAGScheduler, RAGSequence
from drag.KV_cache_manager import KV_Cache_Manager
from drag.document import Document
from drag.RAGRequest import CacheDocRequest, RAGRequest, RAGRequestType
from drag.utils import get_tokenizer
from vllm.entrypoints.llm import LLM
from transformers import AutoTokenizer

class MockKVCacheManager:
    def __init__(self, num_free_blocks=65535, computed_blocks={}):
        self.block_size = 16
        self.enable_swap_in_cpu_blocks = False
        self.num_free_gpu_blocks = num_free_blocks
        self.computed_blocks = computed_blocks
        self.doc_to_blocks = {}
        self.request_id_to_blocks = {}

    def num_free_blocks(self) -> int:
        return self.num_free_gpu_blocks

    def get_computed_gpu_blocks(self, request: RAGRequest, swap_in_cpu_blocks=False) -> List[int]:
        if request.get_type() == RAGRequestType.CACHE_DOC and cast(CacheDocRequest, request).doc_id in self.doc_to_blocks:
            return self.doc_to_blocks[cast(CacheDocRequest, request).doc_id]
        return self.computed_blocks.get(request.request_id, [])

    def append_slots(self, request: RAGRequest, num_new_tokens: int) -> Tuple[List[int], List[int]]:
        # Simulate slot append logic(for decoding phase)
        if request.get_type() == RAGRequestType.CACHE_DOC:
            doc_id = cast(CacheDocRequest, request).doc_id
            self.doc_to_blocks[doc_id].extend(list(range(1))) 
        return list(range(num_new_tokens)),[]  # Mock allocated slots

    def allocate_slots(self, request: RAGRequest, num_new_tokens: int) -> Tuple[List[int], List[int]]:
        # Simulate slot initialization logic
        self.num_free_gpu_blocks -= 1  # all requests for testing will only occupy one block.
        if request.get_type() == RAGRequestType.CACHE_DOC:
            doc_id = cast(CacheDocRequest, request).doc_id
            self.doc_to_blocks[doc_id] = list(range(1))
        return list(range(num_new_tokens)),list(range(1))  # Mock allocated slots

    def free(self, request: RAGRequest) -> None:
        self.num_free_gpu_blocks += 1 # all requests for testing will only occupy one block.

@pytest.fixture
def kv_cache_manager_infinite_blocks():
    return MockKVCacheManager(num_free_blocks=float("inf"))

@pytest.fixture
def kv_cache_manager_limited_blocks():
    return MockKVCacheManager(num_free_blocks=10)

@pytest.fixture
def kv_cache_manager_with_computed_blocks():
    return MockKVCacheManager(computed_blocks={1: [0, 1], 2: [2, 3]})

@pytest.fixture
def kv_cache_manager_empty_computed_blocks():
    return MockKVCacheManager(computed_blocks={})

@pytest.fixture
def mocked_llm():
    llm = MagicMock(spec=LLM)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    llm.llm_engine = MagicMock()
    llm.llm_engine.tokenizer = MagicMock()
    llm.llm_engine.tokenizer.tokenizer = tokenizer
    llm.llm_engine.cache_config = MagicMock()
    llm.llm_engine.cache_config.block_size = 16
    return llm

@pytest.fixture
def docDB(mocked_llm):
    docs = [
        "Lionel Messi scored 13 goals at FIFA World Cups.",
        "Cristiano Ronaldo scored 8 goals at FIFA World Cups.", 
        "Neymar scored 2 goals at FIFA World Cups."
    ]
    docDB = {}
    for i, doc_text in enumerate(docs):
        doc = Document(doc_text, mocked_llm, prefill=False)
        doc.doc_id = i
        docDB[doc.doc_id] = doc
    return docDB

@pytest.fixture
def single_seq(docDB, mocked_llm):
    return RAGSequence.make_RAGSequence(1, "test prompt", [1], docDB, get_tokenizer(mocked_llm))

@pytest.fixture
def multiple_seq(docDB, mocked_llm):
    return [RAGSequence.make_RAGSequence(1, "test prompt", [1], docDB, get_tokenizer(mocked_llm)),
            RAGSequence.make_RAGSequence(2, "test prompt", [2,1], docDB, get_tokenizer(mocked_llm)),
            RAGSequence.make_RAGSequence(3, "test prompt", [0], docDB, get_tokenizer(mocked_llm))]


def test_add_seq(single_seq: RAGSequence, docDB: dict[int,Document], kv_cache_manager_infinite_blocks: KV_Cache_Manager):
    scheduler = RAGScheduler(10, docDB, kv_cache_manager_infinite_blocks)
    scheduler.add_sequence([single_seq])
    assert len(scheduler.new_seq_list) == 1
    assert scheduler.seq_id_to_seqs[1] == single_seq
    assert scheduler.new_seq_list[0] == single_seq


def test_finish_seq(single_seq: RAGSequence, docDB: dict[int,Document], kv_cache_manager_infinite_blocks):
    scheduler = RAGScheduler(10, docDB, kv_cache_manager_infinite_blocks)
    scheduler.add_sequence([single_seq])
    assert scheduler.has_unfinished_seqs() == True
    scheduler.schedule()
    scheduler.finish_seq(single_seq.sequence_id)
    assert scheduler.has_unfinished_seqs() == False


def test_schedule_single_seq(single_seq: RAGSequence, docDB: dict[int,Document], kv_cache_manager_infinite_blocks):
    # the first step will only schedule CacheDocRequest.
    scheduler = RAGScheduler(10, docDB, kv_cache_manager_infinite_blocks)
    scheduler.add_sequence([single_seq])
    prefill_plan, decode_plan = scheduler.schedule()

    assert prefill_plan.batch_size == 1
    assert prefill_plan.batch_query_lens[0] == len(docDB[1].token_ids)
    assert prefill_plan.batch_seq_lens[0] == len(docDB[1].token_ids)
    assert prefill_plan.batch_num_prefill_tokens[0] == len(docDB[1].token_ids)

    assert prefill_plan.batch_context_lens[0] == 0
    assert len(prefill_plan.batch_block_tables[0]) == 1

    assert prefill_plan.batch_rag_reqs[0].get_type() == RAGRequestType.CACHE_DOC

    assert decode_plan.batch_size == 0

    # the second step will only schedule QueryRequest in the prefill phase.
    prefill_plan, decode_plan = scheduler.schedule()
    assert prefill_plan.batch_size == 1

    assert prefill_plan.batch_query_lens[0] == len(single_seq.query_token_ids)
    assert prefill_plan.batch_seq_lens[0] == len(single_seq.query_token_ids) + len(docDB[1].token_ids)
    assert prefill_plan.batch_num_prefill_tokens[0] == len(single_seq.query_token_ids)

    assert prefill_plan.batch_context_lens[0] == len(docDB[1].token_ids) # the same as the cached doc length
    assert len(prefill_plan.batch_block_tables[0]) == len(docDB[1].token_ids) / kv_cache_manager_infinite_blocks.block_size + \
        math.ceil(len(single_seq.query_token_ids)/kv_cache_manager_infinite_blocks.block_size)

    assert prefill_plan.batch_rag_reqs[0].get_type() == RAGRequestType.QUERY

    assert decode_plan.batch_size == 0

    # mock the generated token id of the QueryRequest prefill phase
    single_seq.append_generated_token(123, 0.0)

    # the third step will schedule the QueryRequest in the decode phase.
    prefill_plan, decode_plan = scheduler.schedule()
    
    assert decode_plan.batch_size == 1

    assert decode_plan.batch_query_lens[0] == 1
    assert decode_plan.batch_seq_lens[0] == len(single_seq.query_token_ids) + len(docDB[1].token_ids) + 1
    assert decode_plan.batch_num_prefill_tokens[0] == 0

    assert decode_plan.batch_context_lens[0] == len(docDB[1].token_ids) + len(single_seq.query_token_ids)
    assert len(decode_plan.batch_block_tables[0]) == len(docDB[1].token_ids) / kv_cache_manager_infinite_blocks.block_size + math.ceil(len(single_seq.query_token_ids) / kv_cache_manager_infinite_blocks.block_size)

    assert decode_plan.batch_rag_reqs[0].get_type() == RAGRequestType.QUERY
    
    assert prefill_plan.batch_size == 0

    scheduler.finish_seq(single_seq.sequence_id)
    assert scheduler.has_unfinished_seqs() == False


def test_schedule_multiple_seqs(multiple_seq: List[RAGSequence], docDB: dict[int,Document], kv_cache_manager_infinite_blocks: KV_Cache_Manager):
    # test scheduling multiple requests sharing partially the same documents

    # the first step will only schedule CacheDocRequest.
    scheduler = RAGScheduler(10, docDB, kv_cache_manager_infinite_blocks)
    scheduler.add_sequence(multiple_seq)
    prefill_plan, decode_plan = scheduler.schedule()

    # Verify CacheDocRequest prefill phase
    assert prefill_plan.batch_size == 3
    assert len(prefill_plan.batch_rag_reqs) == 3
    
    # Check all requests are CacheDocRequests
    for req in prefill_plan.batch_rag_reqs:
        assert req.get_type() == RAGRequestType.CACHE_DOC
    
    # Verify doc_ids are loaded correctly (1, 2, 0)
    assert prefill_plan.batch_rag_reqs[0].doc_id == 1
    assert prefill_plan.batch_rag_reqs[1].doc_id == 2
    assert prefill_plan.batch_rag_reqs[2].doc_id == 0
    
    # Check batch context setup
    assert all(ctx_len == 0 for ctx_len in prefill_plan.batch_context_lens)

    assert all(len(block_table) == 1 for block_table in prefill_plan.batch_block_tables)
    
    # Check query lengths match document lengths
    assert all(query_len == 16 for query_len in prefill_plan.batch_query_lens)
    assert all(seq_len == 16 for seq_len in prefill_plan.batch_seq_lens)
    assert all(prefill_tokens == 16 for prefill_tokens in prefill_plan.batch_num_prefill_tokens)
    
    # No decode batch in first step
    assert decode_plan.batch_size == 0

    # the second step will schedule QueryRequest in the prefill phase.
    prefill_plan, decode_plan = scheduler.schedule()
    
    # Verify QueryRequest prefill phase
    assert prefill_plan.batch_size == 3
    assert len(prefill_plan.batch_rag_reqs) == 3
    
    # Check all requests are QueryRequests
    for req in prefill_plan.batch_rag_reqs:
        assert req.get_type() == RAGRequestType.QUERY
    
    # Check prompt lengths and sequence-to-original mappings
    assert all(query_len == 3 for query_len in prefill_plan.batch_query_lens)
    assert prefill_plan.batch_rag_reqs[0].original_seq_id == 1
    assert prefill_plan.batch_rag_reqs[1].original_seq_id == 2
    assert prefill_plan.batch_rag_reqs[2].original_seq_id == 3
    
    # Check context lengths (first has doc1, second has doc2+doc1, third has doc0)
    assert prefill_plan.batch_context_lens[0] == 16  # doc1
    assert prefill_plan.batch_context_lens[1] == 32  # doc2 + doc1
    assert prefill_plan.batch_context_lens[2] == 16  # doc0
    
    # Check sequence lengths (context + prompt)
    assert prefill_plan.batch_seq_lens[0] == 19  # 16 + 3
    assert prefill_plan.batch_seq_lens[1] == 35  # 32 + 3
    assert prefill_plan.batch_seq_lens[2] == 19  # 16 + 3
    
    # Check block tables based on document count
    assert len(prefill_plan.batch_block_tables[0]) == 2  # One document + prompt
    assert len(prefill_plan.batch_block_tables[1]) == 3  # Two documents + prompt
    assert len(prefill_plan.batch_block_tables[2]) == 2  # One document + prompt
    
    # Check document IDs for each sequence
    assert prefill_plan.batch_rag_reqs[0].doc_ids == [1]
    assert prefill_plan.batch_rag_reqs[1].doc_ids == [2, 1]
    assert prefill_plan.batch_rag_reqs[2].doc_ids == [0]
    
    # No decode batch in second step
    assert decode_plan.batch_size == 0

    # mock the generated token id of the QueryRequest prefill phase
    for seq in multiple_seq:
        seq.append_generated_token(123, 0.0)

    # the third step will schedule the QueryRequest in the decode phase.
    prefill_plan, decode_plan = scheduler.schedule()
    
    # No prefill batch in third step
    assert prefill_plan.batch_size == 0
    
    # Verify QueryRequest decode phase
    assert decode_plan.batch_size == 3
    assert len(decode_plan.batch_rag_reqs) == 3
    
    # Check all requests are QueryRequests  
    for req in decode_plan.batch_rag_reqs:
        assert req.get_type() == RAGRequestType.QUERY
    
    # Check query lengths are all 1 (one token each)
    assert all(query_len == 1 for query_len in decode_plan.batch_query_lens)
    
    # Check context lengths include documents + prompt
    assert decode_plan.batch_context_lens[0] == 19  # doc1 + prompt
    assert decode_plan.batch_context_lens[1] == 35  # doc2 + doc1 + prompt
    assert decode_plan.batch_context_lens[2] == 19  # doc0 + prompt
    
    # Check sequence lengths (context + generated token)
    assert decode_plan.batch_seq_lens[0] == 20  # 19 + 1
    assert decode_plan.batch_seq_lens[1] == 36  # 35 + 1
    assert decode_plan.batch_seq_lens[2] == 20  # 19 + 1
    
    # Check block tables based on documents + prompt
    assert len(decode_plan.batch_block_tables[0]) == 2  # One document + prompt
    assert len(decode_plan.batch_block_tables[1]) == 3  # Two documents + prompt
    assert len(decode_plan.batch_block_tables[2]) == 2  # One document + prompt
    
    # Verify output tokens are included
    assert all(output_tokens == [123] for output_tokens in decode_plan.batch_output_token_ids)


def test_schedule_multiple_seqs_memory_restricted(multiple_seq: List[RAGSequence], docDB: dict[int,Document], mocked_llm):
    # Test scheduling with memory restrictions (limited blocks force sequential execution)
    
    # Create a KV cache manager with only 3 free blocks
    # This is just enough for one sequence at a time, forcing sequential scheduling
    kv_cache_manager = MockKVCacheManager(num_free_blocks=3)
    
    scheduler = RAGScheduler(10, docDB, kv_cache_manager)
    scheduler.add_sequence(multiple_seq)
    
    # PHASE 1: First Document Cache Request Round
    prefill_plan, decode_plan = scheduler.schedule()
    
    # Should schedule only one document (first doc from first sequence) due to memory constraints
    assert prefill_plan.batch_size == 1
    assert len(prefill_plan.batch_rag_reqs) == 1
    assert prefill_plan.batch_rag_reqs[0].get_type() == RAGRequestType.CACHE_DOC
    assert prefill_plan.batch_rag_reqs[0].doc_id == 1  # First sequence needs doc1
    
    # Verify there's no decode plan yet
    assert decode_plan.batch_size == 0
    
    # PHASE 2: First Query Prefill
    prefill_plan, decode_plan = scheduler.schedule()
    
    # Should schedule query for first sequence
    assert prefill_plan.batch_size == 1
    assert prefill_plan.batch_rag_reqs[0].get_type() == RAGRequestType.QUERY
    assert prefill_plan.batch_rag_reqs[0].original_seq_id == 1
    assert prefill_plan.batch_context_lens[0] == 16  # doc1
    
    # Mock generated token
    multiple_seq[0].append_generated_token(123, 0.0)
    
    # PHASE 3: First Sequence Decode
    prefill_plan, decode_plan = scheduler.schedule()
    
    assert prefill_plan.batch_size == 0
    assert decode_plan.batch_size == 1
    assert decode_plan.batch_rag_reqs[0].original_seq_id == 1
    
    # Finish sequence 1 to free up memory
    scheduler.finish_seq(1)
    assert kv_cache_manager.num_free_blocks() == 3  # Should have 3 free blocks now
    
    # PHASE 4: Second Document Cache Request Round (should start handling sequence 2)
    prefill_plan, decode_plan = scheduler.schedule()
    
    # Should now schedule second sequence's first document (doc2, because doc1 is already cached)
    assert prefill_plan.batch_size == 1
    assert prefill_plan.batch_rag_reqs[0].get_type() == RAGRequestType.CACHE_DOC
    assert prefill_plan.batch_rag_reqs[0].doc_id == 2
    
    # PHASE 5: Should now schedule second sequence's query
    prefill_plan, decode_plan = scheduler.schedule()
    
    assert prefill_plan.batch_size == 1
    assert prefill_plan.batch_rag_reqs[0].get_type() == RAGRequestType.QUERY
    assert prefill_plan.batch_rag_reqs[0].original_seq_id == 2
    assert prefill_plan.batch_rag_reqs[0].doc_ids == [2, 1]  # Checks that correct docs are referenced
    
    # Mock generated token
    multiple_seq[1].append_generated_token(123, 0.0)
    
    # PHASE 6: Second Sequence Decode
    prefill_plan, decode_plan = scheduler.schedule()
    
    assert prefill_plan.batch_size == 0
    assert decode_plan.batch_size == 1
    assert decode_plan.batch_rag_reqs[0].original_seq_id == 2
    
    # Finish sequence 2 to free up memory
    scheduler.finish_seq(2)

    assert kv_cache_manager.num_free_blocks() == 3  # Should have 3 free blocks now
    
    # PHASE 7: Third Document Cache Request
    prefill_plan, decode_plan = scheduler.schedule()
    
    # Should now schedule third sequence's document (doc0)
    assert prefill_plan.batch_size == 1
    assert prefill_plan.batch_rag_reqs[0].get_type() == RAGRequestType.CACHE_DOC
    assert prefill_plan.batch_rag_reqs[0].doc_id == 0
    
    # PHASE 8: Third Sequence Query
    prefill_plan, decode_plan = scheduler.schedule()
    
    assert prefill_plan.batch_size == 1
    assert prefill_plan.batch_rag_reqs[0].get_type() == RAGRequestType.QUERY
    assert prefill_plan.batch_rag_reqs[0].original_seq_id == 3
    assert prefill_plan.batch_rag_reqs[0].doc_ids == [0]
    
    # Mock generated token
    multiple_seq[2].append_generated_token(123, 0.0)
    
    # PHASE 9: Third Sequence Decode
    prefill_plan, decode_plan = scheduler.schedule()
    
    assert prefill_plan.batch_size == 0
    assert decode_plan.batch_size == 1
    assert decode_plan.batch_rag_reqs[0].original_seq_id == 3
    
    # Finish sequence 3
    scheduler.finish_seq(3)

    assert kv_cache_manager.num_free_blocks() == 3  # Should have 3 free blocks now
    
    # All sequences should be completed now
    assert not scheduler.has_unfinished_seqs()
    
    # PHASE 10: No more sequences
    prefill_plan, decode_plan = scheduler.schedule()
    assert prefill_plan.batch_size == 0
    assert decode_plan.batch_size == 0

def test_schedule_single_seq_with_pre_computed_prefix_caches():
    # Test scheduling with pre-computed prefix caches for queries
    pass
    # TOOD: test this feature

def test_async_schedule():
    # Instead of put all requests in the same batch, we can put them in different batches and schedule them asynchronously.
    pass
    # TOOD: test this feature


# TODO: test preemption and resumption. And two scenerios of resumption:
    # 1. we still have the docs, prompt and part of the generated text in the memory
    # 2. The prefilling part is partially lost.
# TODO: test prefix matching for a new QueryRequest
    # one potential bug: find exact prefix matching with another sequence, the sequence will enter decoding phase directly.
    # but the generated tokens of the new sequence is still empty. There's no starting token for the decoding phase.