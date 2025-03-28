import pytest
from unittest.mock import MagicMock, patch
import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.layers.sampler import SamplerOutput

from drag.RAGEngine import RAGEngine, RAGgenerateOutput, RAGRequestOutput
from drag.RAGScheduler import RAGScheduler, RAGSequence
from drag.document import Document
from drag.KV_cache_manager import KV_Cache_Manager
from drag.RAGRequest import RAGRequest

# Reuse the docDB setup from testRAGScheduler
@pytest.fixture
def docDB():
    docs = [
        "Lionel Messi scored 13 goals at FIFA World Cups.",
        "Cristiano Ronaldo scored 8 goals at FIFA World Cups.",
        "Neymar scored 2 goals at FIFA World Cups."
    ]
    docDB = {}
    for i, doc_text in enumerate(docs):
        doc = Document(doc_text, None, prefill=False)
        doc.doc_id = i
        doc.token_ids = [i+100, i+200, i+300]  # Mock token IDs
        docDB[doc.doc_id] = doc
    return docDB

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda x: [101, 102, 103]
    tokenizer.decode.return_value = "Generated text"
    return tokenizer

@pytest.fixture
def mock_model_runner():
    model_runner = MagicMock()
    sampler_output = MagicMock(spec=SamplerOutput)
    sampler_output.sampled_token_ids = [[104]]  # Not a stop token
    model_runner.execute_model.return_value = [sampler_output]
    return model_runner

@pytest.fixture
def mock_llm(mock_tokenizer, mock_model_runner):
    llm = MagicMock(spec=LLM)
    return llm

@pytest.fixture
def rag_engine(docDB, mock_llm, mock_tokenizer, mock_model_runner):
    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=0,
        stop_token_ids=[128008, 128001]
    )
    
    with patch('drag.RAGEngine.get_tokenizer', return_value=mock_tokenizer):
        with patch('drag.RAGEngine.get_model_runner', return_value=mock_model_runner):
            with patch('drag.RAGEngine.get_block_size', return_value=16):
                with patch('drag.RAGEngine.get_gpu_cache', return_value=[{}]):
                    engine = RAGEngine(docDB, mock_llm, sampling_params)
                    
                    # Mock the _step method to simulate request completion
                    def mock_step():
                        seq = engine.scheduler.seq_id_to_seqs[0]
                        req = MagicMock(spec=RAGRequest)
                        req.get_type.return_value = "QUERY"
                        req.original_seq_id = 0
                        return [RAGRequestOutput(req=req, origin_seq=seq, req_finished=True)]
                    
                    engine._step = MagicMock(side_effect=[mock_step])
                    engine.scheduler.has_unfinished_seqs = MagicMock(side_effect=[True, False])
                    
                    return engine

def test_rag_engine_init(docDB, mock_llm):
    """Test RAGEngine initialization"""
    sampling_params = SamplingParams(max_tokens=10, temperature=0)
    
    with patch('drag.RAGEngine.get_tokenizer'):
        with patch('drag.RAGEngine.get_model_runner'):
            with patch('drag.RAGEngine.get_block_size'):
                with patch('drag.RAGEngine.get_gpu_cache', return_value=[{}]):
                    engine = RAGEngine(docDB, mock_llm, sampling_params)
                    
                    assert engine.documents == docDB
                    assert engine.llm == mock_llm
                    assert engine.sample_params == sampling_params
                    assert isinstance(engine.scheduler, RAGScheduler)

def test_generate(rag_engine, docDB):
    """Test the generate function similar to drag_engine_main.py"""
    # Test data from drag_engine_main.py
    queries = [
        "Who scored more goals at FIFA World Cups, Messi or Ronaldo?"
    ]
    query_doc_ids = [
        [0, 1]
    ]
    
    # Create a sequence that will be returned by the scheduler
    seq = RAGSequence(
        sequence_id=0,
        doc_ids=[0, 1],
        prompt=queries[0],
        query_token_ids=[101, 102, 103],
        doc_token_ids=[[100, 200, 300], [101, 201, 301]],
        generated_token_ids=[104, 105]
    )
    rag_engine.scheduler.seq_id_to_seqs = {0: seq}
    
    # Execute generate
    outputs = rag_engine.generate(queries, query_doc_ids)
    
    # Verify results
    assert len(outputs) == 1
    assert isinstance(outputs[0], RAGgenerateOutput)
    assert outputs[0].prompt == queries[0]
    assert outputs[0].doc_ids == query_doc_ids[0]
    assert outputs[0].generated_text == "Generated text"

    # Verify the tokenizer was called to encode queries
    rag_engine.tokenizer.encode.assert_called()

    # Verify scheduler was used correctly
    assert rag_engine.scheduler.add_sequence.called
    assert rag_engine.scheduler.has_unfinished_seqs.called

def test_generate_multiple_queries(rag_engine, mock_tokenizer):
    """Test generating responses for multiple queries"""
    queries = [
        "Who scored more goals at FIFA World Cups, Messi or Ronaldo?",
        "Who scored more goals at FIFA World Cups, Messi or Neymar?"
    ]
    query_doc_ids = [
        [0, 1],
        [0, 2]
    ]
    
    # Create sequences that will be returned by the scheduler
    seq1 = RAGSequence(
        sequence_id=0,
        doc_ids=[0, 1],
        prompt=queries[0],
        query_token_ids=[101, 102, 103],
        doc_token_ids=[[100, 200, 300], [101, 201, 301]],
        generated_token_ids=[104, 105]
    )
    
    seq2 = RAGSequence(
        sequence_id=1,
        doc_ids=[0, 2],
        prompt=queries[1],
        query_token_ids=[101, 102, 103],
        doc_token_ids=[[100, 200, 300], [102, 202, 302]],
        generated_token_ids=[106, 107]
    )
    
    # Update our mocks for multiple sequences
    rag_engine.scheduler.seq_id_to_seqs = {0: seq1, 1: seq2}
    
    def mock_step():
        req1 = MagicMock(spec=RAGRequest)
        req1.get_type.return_value = "QUERY"
        req1.original_seq_id = 0
        
        req2 = MagicMock(spec=RAGRequest)
        req2.get_type.return_value = "QUERY"
        req2.original_seq_id = 1
        
        return [
            RAGRequestOutput(req=req1, origin_seq=seq1, req_finished=True),
            RAGRequestOutput(req=req2, origin_seq=seq2, req_finished=True)
        ]
    
    rag_engine._step = MagicMock(return_value=mock_step())
    rag_engine.scheduler.has_unfinished_seqs = MagicMock(side_effect=[True, False])
    
    # Execute generate
    outputs = rag_engine.generate(queries, query_doc_ids)
    
    # Verify results
    assert len(outputs) == 2
    assert outputs[0].prompt == queries[0]
    assert outputs[0].doc_ids == query_doc_ids[0]
    assert outputs[1].prompt == queries[1]
    assert outputs[1].doc_ids == query_doc_ids[1]
