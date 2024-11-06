import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from drag import DynamicRAG
from vllm import LLM, SamplingParams

# docs = [
#    ("You are an expert school principal, skilled in effectively managing "
#    "faculty and staff. Draft 10-15 questions for a potential first grade "
#    "Head Teacher for my K-12, all-girls', independent school that emphasizes "
#    "community, joyful discovery, and life-long learning. The candidate is "
#    "coming in for a first-round panel interview for a 8th grade Math "
#    "teaching role. They have 5 years of previous teaching experience "
#    "as an assistant teacher at a co-ed, public school with experience "
#    "in middle school math teaching. Based on these information, answer "
#    "the following question."),
#    (" ".join(["test"] * 15)),
#    ]

docs = [
        "As all we know Lionel Messi scored 13 goals at FIFA World Cups.",
        "As all we know Cristiano Ronaldo scored 8 goals at FIFA World Cups."
        ]

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
          gpu_memory_utilization=0.9,
          enforce_eager=True,
          enable_prefix_caching=True)

sample_params = SamplingParams(max_tokens=2,
                               seed=2024)

sample_params.stop_token_ids = [128008, 128001]

rag = DynamicRAG(llm)
doc_ids = rag.add_cache(docs)
print("doc_ids", doc_ids)
print("cached_doc", rag.cached_documents)
outputs = rag.inference([doc_ids[1], doc_ids[0]], query="Who scored more goals at FIFA World Cups, Messi or Ronaldo?", sample_params=sample_params)
# outputs = rag.inference(doc_ids, query="Who scored more goals at FIFA World Cups, Messi or Ronaldo?", sample_params=sample_params)
print(outputs)
rag.destroy_cache(doc_ids)
