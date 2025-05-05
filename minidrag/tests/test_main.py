import pytest
import torch 

from vllm.distributed import cleanup_dist_env_and_memory

import minidrag.__vllm__


class TestMiniDRAG:
    # @pytest.mark.gpu
    # @pytest.mark.integration
    # def test_e2e_simple_sync(self, 
    #                     mock_prompts,
    #                     mock_model_name,
    #                     mock_sampling_params,):
    #     torch.cuda.empty_cache()
    #     llm = None
    #     try:
    #         import vllm
    #         llm = vllm.LLM(model=mock_model_name,                      
    #                        gpu_memory_utilization=0.9,
    #                        enforce_eager=False,
    #                        enable_prefix_caching=True,
    #                        seed=42,
    #                        max_model_len=2048,)
    #         outputs = llm.generate(prompts=mock_prompts,
    #                      sampling_params=mock_sampling_params,
    #                      document_seqs=[["doc1 "*50, "doc2 "*50], ["doc3 "*50, "doc4 "*50],
    #                                     ["doc5 "*50, "doc6 "*50], ["doc7 "*50, "doc8 "*50]],)
    #         for output in outputs:
    #             print(output.outputs[0].text)
                
    #         outputs = llm.generate(prompts=mock_prompts,
    #                     sampling_params=mock_sampling_params,
    #                     document_seqs=[["doc2 "*50, "doc1 "*50], ["doc4 "*50, "doc3 "*50],
    #                                    ["doc6 "*50, "doc5 "*50], ["doc8 "*50, "doc7 "*50]],)
    #         for output in outputs:
    #             print(output.outputs[0].text)
    #     finally:
    #         if llm is not None:
    #             del llm
    #         cleanup_dist_env_and_memory()
    #         torch.cuda.synchronize()

    @pytest.mark.gpu
    @pytest.mark.integration
    def test_e2e_simple_async(self, 
                        mock_model_name,):
        torch.cuda.empty_cache()
        model = None
        try:    
            import asyncio
            import time
            from vllm import AsyncEngineArgs, SamplingParams
            from vllm.v1.engine.async_llm import AsyncLLM

            engine_args = AsyncEngineArgs(model=mock_model_name, 
                                          enforce_eager=True, 
                                          max_model_len=2048)
            model = AsyncLLM.from_engine_args(engine_args)

            async def generate_streaming(prompt):
                results_generator = model.generate(prompt, 
                                                SamplingParams(), 
                                                request_id='1', # DO NOT USE arrival time-like id, will be blocked
                                                document_seq=["doc1 "*50, "doc2 "*50],)
                previous_text = ""
                async for request_output in results_generator:
                    text = request_output.outputs[0].text
                    print(text[len(previous_text):], end="")
                    previous_text = text

            asyncio.run(generate_streaming("Hello world! Jane is a student in"))
        finally:
            if model is not None:
                del model
            cleanup_dist_env_and_memory()
            torch.cuda.synchronize()