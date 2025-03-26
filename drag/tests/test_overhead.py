from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from typing import Dict, List
import copy
from drag import DynamicRAG
import numpy as np

MAX_TOKENS = 50
MULTIPLE_RUN = True
NUM_RUNS = 10

docs = ["Lionel Messi scored 13 goals at FIFA World Cups.",
        "Cristiano Ronaldo scored 8 goals at FIFA World Cups.",]
query = "Who scored more goals at FIFA World Cups, Messi or Ronaldo?"

sampling_param = SamplingParams(max_tokens=MAX_TOKENS,
                                seed=2024,
                                temperature=0,
                                # stop=['<|im_end|>', "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"],
                                stop_token_ids=[128008, 128001])


def dynamic_attn() -> Dict:
    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
              gpu_memory_utilization=0.9,
              enforce_eager=True,
              enable_prefix_caching=True,
              use_dynamic_attn=True,)
              # enable_chunked_prefill=False)
    rag = DynamicRAG(llm)
    doc_ids = rag.add_cache(docs)  # validated
    outputs = rag.generate(doc_ids, query, sampling_param)
    res  = copy.deepcopy(rag.profiling_stat)
    rag.destroy_cache(doc_ids)
    print(f"dynamic attn output: {''.join(outputs)}")
    del llm
    del rag
    cleanup_dist_env_and_memory()
    return res


def static_attn() -> Dict:
    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
              gpu_memory_utilization=0.9,
              enforce_eager=True,
              enable_prefix_caching=True,)
              # enable_chunked_prefill=False)
    rag = DynamicRAG(llm)
    doc_ids = rag.add_cache(docs)  # validated
    outputs = rag.generate(doc_ids, query, sampling_param)
    res  = copy.deepcopy(rag.profiling_stat)
    rag.destroy_cache(doc_ids)
    print(f"static attn output: {''.join(outputs)}")
    del llm
    del rag
    cleanup_dist_env_and_memory()
    return res


def merge_stats(list_stats: List[Dict]) -> Dict:
    res = {
        "prefill": {
            "forward": [],
            "compute_logits": [],
            "sample_token": [],
            "step": []
        },
        "decode": {
            "forward": [],
            "compute_logits": [],
            "sample_token": [],
            "step": []
        }
    }
    for stat in list_stats:
        for phase in ["prefill", "decode"]:
            for metric in ["forward", "compute_logits", "sample_token", "step"]:
                res[phase][metric].append(stat[phase][metric])
    
    for phase in ["prefill", "decode"]:
        for metric in ["forward", "compute_logits", "sample_token", "step"]:
            res[phase][metric] = np.mean(res[phase][metric], axis=0).tolist()
    
    return res


def test_overhead():
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--use_dynamic_attn", type=int, default=1)
    # args = ap.parse_args()
    prefill_overhead = []
    decode_overhead = []
    # test for different number of blocks
    global docs
    docs_origin = copy.deepcopy(docs)
    num_blocks = [2, 16, 128, 1024] # , 4096, 8192]
    #num_blocks = [1024]
    for num_block in num_blocks:
        print('-' * 30, num_block)
        # docs = docs_origin * (num_block // 2)
        docs[0] = docs_origin[0] * (num_block // 2)
        docs[1] = docs_origin[1] * (num_block // 2)
        if not MULTIPLE_RUN:
            dynamic_profiling_res = dynamic_attn()
            static_profiling_res = static_attn()
        else:
            list_dynamic_profiling_res = []
            list_static_profiling_res = []
            for _ in range(NUM_RUNS):
                list_dynamic_profiling_res.append(dynamic_attn())
                list_static_profiling_res.append(static_attn())
            # each element is a dict, so we need to average the values
            dynamic_profiling_res = merge_stats(list_dynamic_profiling_res)
            static_profiling_res = merge_stats(list_static_profiling_res)
        print('_' * 30)
        print(dynamic_profiling_res)
        print('_' * 30)
        print(static_profiling_res)

        # compute overheads
        metrics = ['forward', 'compute_logits', 'sample_token', 'step']
        
        # prefill phase
        prefill_dynamic = dynamic_profiling_res['prefill']
        prefill_static = static_profiling_res['prefill']
        print("\nPrefill Phase Overheads:")
        for metric in metrics:
            overhead = prefill_dynamic[metric][0] / prefill_static[metric][0]
            print(f"overhead in {metric}: {overhead:.3f}")
            if metric == 'forward':
                prefill_overhead.append(overhead)
        # decode phase
        decode_dynamic = dynamic_profiling_res['decode']
        decode_static = static_profiling_res['decode']
        
        print("\nDecode Phase Overheads:")
        for metric in metrics:
            dynamic_array = np.array(decode_dynamic[metric])
            static_array = np.array(decode_static[metric])
            overhead = dynamic_array / static_array
            avg_overhead = np.average(overhead)
            print(f"overhead in {metric}: {overhead}")
            print(f"avg overhead in {metric}: {avg_overhead:.3f}")
            if metric == 'forward':
                decode_overhead.append(avg_overhead)
    # report
    print('-' * 30)
    print("\nOverhead Summary:")
    headers = ['Num Blocks', 'Prefill Overhead', 'Decode Overhead']
    row_format = "{:>15}" * len(headers)
    
    print(row_format.format(*headers))
    print('-' * 45)
    for blocks, prefill, decode in zip(num_blocks, prefill_overhead, decode_overhead):
        print(row_format.format(str(blocks), f"{prefill:.3f}", f"{decode:.3f}"))
    
    # assert all(np.array(prefill_overhead) < 1.2), "prefill overhead should be less than 10%"
    # assert all(np.array(decode_overhead) < 1.1), "decode overhead should be less than 10%"
    

if __name__ == "__main__":
    test_overhead()
