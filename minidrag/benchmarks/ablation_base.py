# We use async llm
import asyncio
import time
from vllm import AsyncEngineArgs, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
import argparse

import os

os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1"


async def generate_streaming_pre(prompt, num_docs=5, context_size=64000, model=None):
    doc_len = context_size // num_docs // 2
    docs = []
    for i in range(num_docs):
        docs.append(' '.join([f'{str(i)} '*doc_len]))
    prompt = ' '.join(docs)
    print(len(prompt.split(' ')))
    results_generator = model.generate(prompt+'test', 
        SamplingParams(seed=42, temperature=0), 
        request_id=f'pre_stream_{i}', # DO NOT USE arrival time-like id, will be blocked
        )
    previous_text = ""
    async for request_output in results_generator:
        text = request_output.outputs[0].text
        print(text[len(previous_text):], end="")
        previous_text = text

async def generate_streaming(prompt, num_docs=5, context_size=64000, model=None):
    doc_len = context_size // num_docs // 2
    docs = []
    for i in range(num_docs):
        docs.append(' '.join([f'{str(i)} '*doc_len]))
    prompt = ' '.join(docs) + prompt
    print(len(prompt.split(' ')))
    await generate_streaming_pre(prompt, num_docs=num_docs, context_size=context_size, model=model)
    results_generator = model.generate(prompt, 
        SamplingParams(seed=42, temperature=0), 
        request_id=f'stream_{i}', # DO NOT USE arrival time-like id, will be blocked
        )
    previous_text = ""
    async for request_output in results_generator:
        text = request_output.outputs[0].text
        print(text[len(previous_text):], end="")
        previous_text = text

# for i in [1]:
#     print(f"num_docs: {i}")
#     asyncio.run(generate_streaming("Hello world! Jane is a student in", num_docs=i))

def main():
    engine_args = AsyncEngineArgs(model="ldsjmdy/Tulu3-Block-FT", 
                              enforce_eager=False, 
                              max_model_len=65000*2,)
    model = AsyncLLM.from_engine_args(engine_args)
    # get params
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_docs", type=int, default=1)
    parser.add_argument("--context_size", type=int, default=64000)
    args = parser.parse_args()

    # asyncio.run(generate_streaming_pre("Hello world! Jane is a student in", num_docs=args.num_docs, context_size=args.context_size, model=model))
    asyncio.run(generate_streaming("Hello world! Jane is a student in", num_docs=args.num_docs, context_size=args.context_size, model=model))

if __name__ == "__main__":
    main()

# PYTHONPATH=. python benchmarks/ablation_base.py --num_docs 1 --context_size 64000 > ablation_base_1.txt
# PYTHONPATH=. python benchmarks/ablation_base.py --num_docs 2 --context_size 64000 > ablation_base_2.txt
# PYTHONPATH=. python benchmarks/ablation_base.py --num_docs 4 --context_size 64000 > ablation_base_4.txt
# PYTHONPATH=. python benchmarks/ablation_base.py --num_docs 8 --context_size 64000 > ablation_base_8.txt
# PYTHONPATH=. python benchmarks/ablation.py --num_docs 1 --context_size 64000 > ablation_1.txt
# PYTHONPATH=. python benchmarks/ablation.py --num_docs 2 --context_size 64000 > ablation_2.txt
# PYTHONPATH=. python benchmarks/ablation.py --num_docs 4 --context_size 64000 > ablation_4.txt
# PYTHONPATH=. python benchmarks/ablation.py --num_docs 8 --context_size 64000 > ablation_8.txt