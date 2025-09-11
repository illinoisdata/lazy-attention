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

def test_load_data_acc():
    data_path = Path(__file__).parent / 'data' / 'test_data_acc.jsonl'
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data

class TestVLLM:
    @pytest.mark.gpu
    @pytest.mark.integration
    @timeout(3000, "Base test took too long (50 minutes). Interrupted!")
    def test_e2e_acc_sync(self, 
                             mock_model_name,
                             mock_sampling_params,):
        torch.cuda.empty_cache()
        data = test_load_data_acc()
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

    @pytest.mark.gpu
    @pytest.mark.integration
    def test_e2e_acc_async(self, 
                           mock_model_name,):
        torch.cuda.empty_cache()
        model = None
        try:    
            import asyncio
            import time
            from vllm import AsyncEngineArgs, SamplingParams
            from vllm.v1.engine.async_llm import AsyncLLM

            engine_args = AsyncEngineArgs(model=mock_model_name, 
                                          enforce_eager=False, 
                                          max_model_len=2048)
            model = AsyncLLM.from_engine_args(engine_args)
            
            load_data_acc = test_load_data_acc()
            docs = load_data_acc[0]["docs"]
            prompt = load_data_acc[0]["query"]

            async def generate_streaming(prompt):
                results_generator = model.generate(prompt, 
                                                   SamplingParams(seed=42, temperature=0), 
                                                   request_id='1',
                                                   document_seq=docs,)
                previous_text = ""
                async for request_output in results_generator:
                    text = request_output.outputs[0].text
                    print(text[len(previous_text):], end="", flush=True)
                    previous_text = text

            asyncio.run(generate_streaming(prompt))
        finally:
            if model is not None:
                del model
            cleanup_dist_env_and_memory()
            torch.cuda.synchronize()
