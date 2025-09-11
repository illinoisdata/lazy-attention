import pytest
import torch
import logging

from vllm.distributed import cleanup_dist_env_and_memory

import lazy.__vllm__

from utils import timeout

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pathlib import Path
import json

def test_load_data_reuse():
    data_path = Path(__file__).parent / 'data' / 'test_data_reuse.jsonl'
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data

class TestVLLM:
    @pytest.mark.gpu
    @pytest.mark.integration
    @timeout(3000, "Base test took too long (50 minutes). Interrupted!")
    def test_e2e_reuse_sync(self, 
                             mock_model_name,
                             mock_sampling_params,):
        torch.cuda.empty_cache()
        data = test_load_data_reuse()
        llm = None
        import vllm
        try:
            llm = vllm.LLM(model=mock_model_name,                      
                           gpu_memory_utilization=0.9,
                           enforce_eager=False,
                           enable_prefix_caching=True,
                           seed=42,
                           max_model_len=2048,)
            docs = [item["docs"] for item in data]
            prompts = [item["query"] for item in data]
            outputs = llm.generate(prompts=prompts,
                                   sampling_params=mock_sampling_params,
                                   document_seqs=docs,)
            for output in outputs:
                logger.info(output.outputs[0].text)
        finally:
            if llm is not None:
                del llm
            cleanup_dist_env_and_memory()
            torch.cuda.synchronize()