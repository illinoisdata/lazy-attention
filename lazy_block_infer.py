"""Lazy-style inference with hxia7/Llama-3.2-1B-block-FT.

Uses the Block-Attention prompt template (Llama-3 chat format) matching the
training data from Block-Attention/data_process/rag/2wiki.py.

Supports two modes:
  - Vanilla: full prompt passed as a single input (no document caching)
  - Lazy:    documents split into document_seqs for per-block caching

Reference: block_infer.py, benchmarks/blockbench.py
"""
import argparse
import os
import sys

os.environ["VLLM_USE_LAZY_ATTENTION"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lazy_attn"))

import lazy.__vllm__  # noqa: F401

import vllm.transformers_utils.tokenizer as _vtok
_orig = _vtok.get_cached_tokenizer
_vtok.get_cached_tokenizer = lambda t: (
    setattr(t, "all_special_tokens_extended", t.all_special_tokens)
    or _orig(t)
) if not hasattr(t, "all_special_tokens_extended") else _orig(t)

from vllm import SamplingParams
from lazy.entrypoints.llm import LazyLLM

MODEL = "hxia7/Llama-3.2-1B-block-FT"


def build_full_prompt(documents: list[dict[str, str]], question: str) -> str:
    """Build the full Llama-3 chat-formatted prompt with Block-Attention RAG template.

    Format (matches blockbench.py / 2wiki.py):
      <|begin_of_text|><|start_header_id|>system
      [system preamble + doc titles/texts]
      <|eot_id|><|start_header_id|>user
      Please write a high-quality answer... Question: {question}
      <|eot_id|><|start_header_id|>assistant
    """
    system = ""
    for doc in documents:
        system += f"- Title: {doc['title']}\n{doc['text'].strip()}\n"
    system = system.strip()

    return (
        f"<|begin_of_text|><|start_header_id|>system\n\n"
        f"You are an intelligent AI assistant. Please answer questions based on "
        f"the user's instructions. Below are some reference documents that may "
        f"help you in answering the user's question.\n\n"
        f"{system}"
        f"\n\n"
        f"<|eot_id|><|start_header_id|>user\n\n"
        f"Please write a high-quality answer for the given question using "
        f"only the provided search documents (some of which might be irrelevant).\n"
        f"Question: {question}"
        f"\n\n\n"
        f"<|eot_id|><|start_header_id|>assistant\n\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--mode", choices=["vanilla", "lazy"], default="vanilla",
                        help="vanilla=full prompt, lazy=split docs for caching")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    print(f"Loading model: {args.model} (mode={args.mode})")
    llm = LazyLLM(
        model=args.model,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        trust_remote_code=True,
        enforce_eager=True,
    )
    print("Model loaded.\n")

    documents = [
        {"title": "Polish-Russian War (film)",
         "text": "Polish-Russian War (Wojna polsko-ruska) is a 2009 Polish film directed by Xawery Zulawski based on the novel Polish-Russian War under the white-red flag by Dorota Maslowska."},
        {"title": "Xawery Zulawski",
         "text": "Xawery Zulawski (born 22 December 1971 in Warsaw) is a Polish film director. He is the son of actress Malgorzata Braunek and director Andrzej Zulawski."},
        {"title": "Viktor Yeliseyev",
         "text": "Viktor Petrovich Yeliseyev (born June 9, 1950) is a Russian general."},
    ]

    questions = [
        "Who is the mother of the director of film Polish-Russian War (Film)?",
        "Who directed the film Polish-Russian War?",
    ]

    prompts = [build_full_prompt(documents, q) for q in questions]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    if args.mode == "vanilla":
        # Full prompt as single input — no document caching
        outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    else:
        # Lazy mode: each document as a separate block for per-block KV caching.
        # The prompt contains the full template, but document_seqs provides the
        # same documents separately so the lazy scheduler can cache them independently.
        doc_list = [f"- Title: {d['title']}\n{d['text'].strip()}" for d in documents]
        outputs = llm.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            document_seqs=[doc_list] * len(questions),
        )

    for q, output in zip(questions, outputs):
        text = output.outputs[0].text
        print(f"Q: {q}")
        print(f"A: {text}\n")

    print("Done.")


if __name__ == "__main__":
    main()
