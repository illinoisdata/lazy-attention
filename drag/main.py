from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
import os
import argparse

os.environ['VLLM_LOGGING_LEVEL'] = 'DEBUG'

docs = ["Lionel Messi scored 13 goals at FIFA World Cups.",
        "Cristiano Ronaldo scored 8 goals at FIFA World Cups.",
        "Neymar scored 2 goals at FIFA World Cups."]
query = "Who scored more goals at FIFA World Cups, Messi or Ronaldo?"
# query = "Now, you are a helpful assistant. Please answer the following question: who scored more goals at FIFA World Cups, Messi or Ronaldo?<|eot_id|>"
sampling_param = SamplingParams(max_tokens=50,
                                seed=2024,
                                temperature=0,
                                # stop=['<|im_end|>', "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"],
                                stop_token_ids=[128008, 128001])
from drag import DynamicRAG

def dynamic_attn():
    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
              gpu_memory_utilization=0.9,
              enforce_eager=True,
              enable_prefix_caching=True,
              use_dynamic_attn=True,
              )

    rag = DynamicRAG(llm)
    doc_ids = rag.add_cache(docs)  # validated
    print('############################################# add cache done')
    outputs = rag.generate([doc_ids[0], doc_ids[1]], query, sampling_param)
    rag.destroy_cache(doc_ids)
    print(f"dynamic attn output: {''.join(outputs)}")


def static_attn():
    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
              gpu_memory_utilization=0.9,
              enforce_eager=True,
              enable_prefix_caching=True,
              use_dynamic_attn=False,
              )

    rag = DynamicRAG(llm)
    doc_ids = rag.add_cache(docs)  # validated
    print('############################################# add cache done')
    outputs = rag.generate([doc_ids[0], doc_ids[1]], query, sampling_param)
    rag.destroy_cache(doc_ids)
    print(f"static attn output: {''.join(outputs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_dynamic_attn", type=int, default=1)
    args = ap.parse_args()

    if args.use_dynamic_attn == 1:
        dynamic_attn()
    else:
        static_attn()


if __name__ == "__main__":
    main()
