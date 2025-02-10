import pytest
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

from drag import DynamicRAG


docs = ["Lionel Messi scored 13 goals at FIFA World Cups.",
        "Cristiano Ronaldo scored 8 goals at FIFA World Cups."]
query = "Who scored more goals at FIFA World Cups, Messi or Ronaldo?"

# only for prefilling -> generate one token
sampling_param = SamplingParams(max_tokens=1,
                                seed=2024,
                                temperature=0,
                                stop=['<|im_end|>', "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"],
                                stop_token_ids=[128008, 128001])

def prefill(docs, query):
    base_llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
                   gpu_memory_utilization=0.9,
                   enable_prefix_caching=True,  # TODO: remove the dependency on prefix caching
                   enforce_eager=True,)
    rag = DynamicRAG(base_llm)
    doc_ids = rag.add_cache(docs)
    outputs = rag.generate(doc_ids, query, sampling_param)
    base_answer = outputs[0]
    del base_llm
    del rag
    cleanup_dist_env_and_memory()

    drag_llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
                   gpu_memory_utilization=0.9,
                   enforce_eager=True,
                   enable_prefix_caching=True,
                   use_dynamic_attn=True,)
    rag = DynamicRAG(drag_llm)
    doc_ids = rag.add_cache(docs)
    outputs = rag.generate(doc_ids, query, sampling_param)
    drag_answer = outputs[0]
    del drag_llm
    cleanup_dist_env_and_memory()
    assert base_answer == drag_answer


def test_one_doc():
    prefill(docs[:1], query)


def test_multi_doc():
    # prefill(docs, query)
    pass


if __name__ == "__main__":
    test_one_doc()
    test_multi_doc()
