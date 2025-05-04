import pytest
import torch 

from vllm.distributed import cleanup_dist_env_and_memory

import minidrag.__vllm__


class TestMiniDRAG:
    @pytest.mark.gpu
    @pytest.mark.integration
    def test_e2e_simple(self, 
                        mock_prompts,
                        mock_model_name,
                        mock_sampling_params,):
        torch.cuda.empty_cache()
        llm = None
        try:
            import vllm
            llm = vllm.LLM(model=mock_model_name,                      
                           gpu_memory_utilization=0.9,
                           enforce_eager=False,
                           enable_prefix_caching=True,
                           seed=42,
                           max_model_len=2048,)
            # llm.generate(prompts=mock_prompts,
            #              sampling_params=mock_sampling_params,
            #              document_seqs=[["doc1", "doc2"], ["doc3", "doc4"],
            #                             ["doc5", "doc6"], ["doc7", "doc8"],],)
            llm.generate(prompts=mock_prompts[0],
                         sampling_params=mock_sampling_params,
                         document_seqs=[["doc1", "doc2"], ["doc3", "doc4"],
                                        ["doc5", "doc6"], ["doc7", "doc8"],][0],)
 
        finally:
            if llm is not None:
                del llm
            cleanup_dist_env_and_memory()
            torch.cuda.synchronize()