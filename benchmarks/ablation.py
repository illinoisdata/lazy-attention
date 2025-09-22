# We use async llm
import asyncio
import time
from vllm import AsyncEngineArgs, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
import lazy.__vllm__
import argparse


async def generate_streaming_pre(prompt, num_docs=5, context_size=64000, model=None):
    doc_len = context_size // num_docs // 2
    docs = []
    for i in range(num_docs):
        docs.append(' '.join([f'{str(i)} '*doc_len]))
    results_generator = model.generate('test', 
        SamplingParams(seed=42, temperature=0), 
        request_id=f'pre_stream_{i}', # DO NOT USE arrival time-like id, will be blocked
        document_seq=docs,)
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
    await generate_streaming_pre(prompt, num_docs=num_docs, context_size=context_size, model=model)
    results_generator = model.generate(prompt, 
        SamplingParams(seed=42, temperature=0), 
        request_id=f'stream_{i}', # DO NOT USE arrival time-like id, will be blocked
        document_seq=docs,)
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

    asyncio.run(generate_streaming("Hello world! Jane is a student in", num_docs=args.num_docs, context_size=args.context_size, model=model))

if __name__ == "__main__":
    main()

# PYTHONPATH=. python benchmarks/ablation.py --num_docs 1 --context_size 64000 > ablation_1.txt
# PYTHONPATH=. python benchmarks/ablation.py --num_docs 2 --context_size 64000 > ablation_2.txt
# PYTHONPATH=. python benchmarks/ablation.py --num_docs 4 --context_size 64000 > ablation_4.txt
# PYTHONPATH=. python benchmarks/ablation.py --num_docs 8 --context_size 64000 > ablation_8.txt



def sample_longbench_requests(
    args: longbench.LongBenchArgs,
    rag: RAG,
    tokenizer: PreTrainedTokenizerBase,
) -> List[RAGRequest]:
    # Get prompt_list
    longbench_dataset = longbench.load_dataset(args.longbench_dataset_name)
    logger.info(f"Loaded {len(longbench_dataset.rows)} LongBench prompts")

    # Fill document cache and collect prompt document IDs.
    doc_hash_to_id: Dict[int, DocumentId] = {}
    doc_ids_by_prompt: List[List[DocumentId]] = []
    document_len_by_prompt: List[int] = []
    sum_tokens: int = 0
    for row in longbench_dataset.rows:
        document = row.context  # One document per LongBench prompt.
        # make sure is list
        if isinstance(document, str):
            document = [document]
            
        # print(f"Document: {document}")
        document_token = 0
        context_doc_ids = []
        for doc in document:
            document_token += len(tokenizer.encode(doc))
            doc_hash = hash(doc)
            if doc_hash not in doc_hash_to_id:
                doc_ids = rag.add_cache([doc])
                doc_id = doc_ids[0]
                doc_hash_to_id[doc_hash] = doc_id
            else:
                doc_id = doc_hash_to_id[doc_hash]
            context_doc_ids.append(doc_id)
        assert len(context_doc_ids) == len(document), f"doc_ids: {len(doc_ids)}, document: {len(document)}"
        sum_tokens += document_token
        doc_ids_by_prompt.append(context_doc_ids)
        document_len_by_prompt.append(document_token)
    logger.info(f"{len(doc_hash_to_id)} unique documents, sum tokens= {sum_tokens}")

    # Generate input requests.
    input_requests = []
    max_len = 0
    for row, prompt_doc_ids, document_len in zip(longbench_dataset.rows, doc_ids_by_prompt, document_len_by_prompt):
        prompt = row.input
        prompt_len = len(tokenizer.encode(prompt))
        output_len = args.longbench_out_seq_len
        input_requests.append(
            RAGRequest(
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=output_len,
                document_len=document_len,
                documents=prompt_doc_ids,
                sampling_params=SamplingParams(max_tokens=output_len, ignore_eos=True, 
                                               temperature=0, seed=42, min_tokens=output_len),
                ground_truth=row.answers,
            )
        )
        max_len = max(max_len, prompt_len + document_len)
    logger.info(f"max_len= {max_len} tokens")
    return input_requests