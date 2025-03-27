# unit test of the RAGScheduler class.
import math
from typing import List, Tuple
import pytest
from unittest.mock import MagicMock
from drag.RAGScheduler import RAGScheduler, RAGSequence
from drag.KV_cache_manager import KV_Cache_Manager
from drag.document import Document
from drag.RAGRequest import RAGRequest, RAGRequestType
from drag.utils import get_tokenizer
from vllm.entrypoints.llm import LLM
from transformers import AutoTokenizer

class MockKVCacheManager:
    def __init__(self, num_free_blocks=float("inf"), computed_blocks={}):
        self.BLOCK_SIZE = 16
        self.enable_swap_in_cpu_blocks = False
        self.num_free_blocks_value = num_free_blocks
        self.computed_blocks = computed_blocks

    def num_free_blocks(self) -> int:
        return self.num_free_blocks_value

    def get_computed_gpu_blocks(self, request: RAGRequest, swap_in_cpu_blocks=False) -> List[int]:
        return self.computed_blocks.get(request.request_id, [])

    def append_slots(self, request: RAGRequest, num_new_tokens: int) -> Tuple[List[int], List[int]]:
        # Simulate slot allocation logic
        num_required_blocks = math.ceil(num_new_tokens / self.BLOCK_SIZE)
        return list(range(num_new_tokens)),list(range(num_required_blocks))  # Mock allocated slots

    def allocate_slots(self, request: RAGRequest, num_new_tokens: int) -> Tuple[List[int], List[int]]:
        return self.append_slots(request, num_new_tokens)

    def free(self, request: RAGRequest) -> None:
        pass  # Simulate freeing resources

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
    assert prefill_plan.batch_ctx_token_ids[0] == []
    assert len(prefill_plan.batch_block_tables[0]) == 0

    assert prefill_plan.rag_req[0].get_type() == RAGRequestType.CACHE_DOC

    assert decode_plan.batch_size == 0

    # the second step will only schedule QueryRequest in the prefill phase.
    prefill_plan, decode_plan = scheduler.schedule()
    assert prefill_plan.batch_size == 1

    assert prefill_plan.batch_query_lens[0] == len(single_seq.query_token_ids)
    assert prefill_plan.batch_seq_lens[0] == len(single_seq.query_token_ids) + len(docDB[1].token_ids)
    assert prefill_plan.batch_num_prefill_tokens[0] == len(single_seq.query_token_ids)

    assert prefill_plan.batch_context_lens[0] == len(docDB[1].token_ids) # the same as the cached doc length
    assert prefill_plan.batch_ctx_token_ids[0] == docDB[1].token_ids
    assert len(prefill_plan.batch_block_tables[0]) == len(docDB[1].token_ids) / kv_cache_manager_infinite_blocks.BLOCK_SIZE

    assert prefill_plan.rag_req[0].get_type() == RAGRequestType.QUERY

    assert decode_plan.batch_size == 0

    # mock the generated token id of the QueryRequest prefill phase
    single_seq.generated_token_ids = [123]

    # the third step will schedule the QueryRequest in the decode phase.
    prefill_plan, decode_plan = scheduler.schedule()
    
    assert decode_plan.batch_size == 1

    assert decode_plan.batch_query_lens[0] == 1
    assert decode_plan.batch_seq_lens[0] == len(single_seq.query_token_ids) + len(docDB[1].token_ids) + 1
    assert decode_plan.batch_num_prefill_tokens[0] == 0

    assert decode_plan.batch_ctx_token_ids[0] == docDB[1].token_ids + single_seq.query_token_ids
    assert decode_plan.batch_context_lens[0] == len(docDB[1].token_ids) + len(single_seq.query_token_ids)
    assert len(decode_plan.batch_block_tables[0]) == len(docDB[1].token_ids) / kv_cache_manager_infinite_blocks.BLOCK_SIZE + math.ceil(len(single_seq.query_token_ids) / kv_cache_manager_infinite_blocks.BLOCK_SIZE)

    assert decode_plan.rag_req[0].get_type() == RAGRequestType.QUERY
    
    assert prefill_plan.batch_size == 0

    scheduler.finish_seq(single_seq.sequence_id)
    assert scheduler.has_unfinished_seqs() == False


