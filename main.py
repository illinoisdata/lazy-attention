import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s '
                           '- %(levelname)s - %(message)s')

from vllm import LLM, SamplingParams

# vllm ------------------------------------------------------------------------
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
          gpu_memory_utilization=0.9,
          enforce_eager=True,
          enable_prefix_caching=True)
sampling_param = SamplingParams(max_tokens=4,
                                seed=2024,
                                stop_token_ids = [128008, 128001])
# -----------------------------------------------------------------------------

import sys
path = "./drag"
if path not in sys.path:
    sys.path.append(path)

from drag import DynamicRAG

docs = ["As all we know Lionel Messi scored 13 goals at FIFA World Cups",
        "As all we know Cristiano Ronaldo scored 8 goals at FIFA World Cups"]
query = "Who scored more goals at FIFA World Cups?" #, Messi or Ronaldo?"

rag = DynamicRAG(llm)
doc_ids = rag.add_cache(docs)
outputs = rag.generate([doc_ids[1], doc_ids[0]], query, sampling_param)
rag.destroy_cache(doc_ids)

print(f"Dynamic RAG's infernce output: {outputs}")
