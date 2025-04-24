
# debug_cudagraph.py
import os
import sys
import torch
import multiprocessing

minidrag_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../minidrag"))
sys.path.insert(0, minidrag_root)


def setup_deterministic_env():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def test_cudagraph_determinism(model_name="meta-llama/Llama-3.2-1B"):
    setup_deterministic_env()
    
    from minidrag.ctxmgr import MiniDynamicRAG
    MiniDynamicRAG.apply_patches()
    torch.cuda.synchronize()
    MiniDynamicRAG.apply_patches_subprocess()
    torch.cuda.synchronize()
    
    import vllm
    llm = vllm.LLM(
        model=model_name,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
        enable_prefix_caching=True,
        seed=42,
    )
    
    prompts = ["Hello, my name is"]
    from vllm.sampling_params import SamplingParams
    params = SamplingParams(temperature=0.0, max_tokens=10)
    
    outputs1 = llm.generate(prompts, params)
    result1 = outputs1[0].outputs[0].text
    print(f"First result: {result1}")
    
    for i in range(5):
        outputs = llm.generate(prompts, params)
        result = outputs[0].outputs[0].text
        print(f"Run {i+1}: {result}")
        if result != result1:
            print(f"Mismatch: {result} vs {result1}")
        else:
            print("Matched ✓")

if __name__ == '__main__':
    multiprocessing.freeze_support()  # For Windows compatibility
    
    test_cudagraph_determinism()