import argparse
from drag.document import Document
from drag.RAGEngine import RAGEngine
from vllm import LLM, SamplingParams

docs = ["Lionel Messi scored 13 goals at FIFA World Cups.",
        "Cristiano Ronaldo scored 8 goals at FIFA World Cups.",
        "Neymar scored 2 goals at FIFA World Cups."]
queries = [
    "Who scored more goals at FIFA World Cups, Messi or Ronaldo?",
    "Who scored more goals at FIFA World Cups, Messi or Neymar?",
    "Who scored more goals at FIFA World Cups, Ronaldo or Neymar?",
]
query_doc_ids = [
    [0,1],
    [0,2],
    [1,2]
]
# query = "Now, you are a helpful assistant. Please answer the following question: who scored more goals at FIFA World Cups, Messi or Ronaldo?<|eot_id|>"
sampling_param = SamplingParams(max_tokens=50,
                                seed=2024,
                                temperature=0,
                                # stop=['<|im_end|>', "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"],
                                stop_token_ids=[128008, 128001])

def attn(llm: LLM):
    # initialize the doc db, currently it's a simple dict structure
    # will change in the future to be a in memory db to handle large number of docs
    docDB: dict[int, Document] = {}
    for doc in docs:
        # register and encode all possible documents
        doc = Document(doc, llm, prefill=False) # no need to prefill in advance
        docDB[doc.doc_id] = doc

    ragEngine = RAGEngine(docDB, llm, sampling_param)
    outputs = ragEngine.generate(queries, query_doc_ids)

    for output in outputs:
        prompt = output.prompt
        doc_ids = output.doc_ids
        generated_text = output.generated_text
        print(f"Prompt: {prompt}")
        print(f"Doc IDs: {doc_ids}")
        print(f"Generated Text: {generated_text}")
        print("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_dynamic_attn", type=int, default=1)
    args = ap.parse_args()

    if args.use_dynamic_attn == 1:
        llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
              gpu_memory_utilization=0.9,
              enforce_eager=True,
              enable_prefix_caching=True,
              use_dynamic_attn=True,
              )
        attn(llm)
    else:
        llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
              gpu_memory_utilization=0.9,
              enforce_eager=True,
              enable_prefix_caching=True,
              use_dynamic_attn=False,
              )
        attn(llm)


if __name__ == "__main__":
    main()