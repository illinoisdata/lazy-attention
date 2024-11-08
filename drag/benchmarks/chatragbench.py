"""Extracted from ChatRAG-Bench (https://huggingface.co/datasets/nvidia/ChatRAG-Bench)."""

import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from simple_parsing import ArgumentParser
from transformers import AutoTokenizer

from drag.logging import logger
from drag.rag import DocumentId, RAGArgs, make_rag
from vllm import SamplingParams


@dataclasses.dataclass
class ChatRAGBenchArgs:
    """ChatRAG-Bench arguments"""

    # Dataset path
    data_folder: Path = dataclasses.field(default_factory=lambda: Path("."))  # path to the data folder of ChatRAG Bench
    output_folder: Path = dataclasses.field(default_factory=lambda: Path("."))  # path to the output folder of ChatRAG Bench
    eval_dataset: str = ""
    doc2dial_path: Path = dataclasses.field(default_factory=lambda: Path("doc2dial/test.json"))
    convfinqa_path: Path = dataclasses.field(default_factory=lambda: Path("convfinqa/dev.json"))
    quac_path: Path = dataclasses.field(default_factory=lambda: Path("quac/test.json"))
    qrecc_path: Path = dataclasses.field(default_factory=lambda: Path("qrecc/test.json"))
    doqa_cooking_path: Path = dataclasses.field(default_factory=lambda: Path("doqa/test_cooking.json"))
    doqa_travel_path: Path = dataclasses.field(default_factory=lambda: Path("doqa/test_travel.json"))
    doqa_movies_path: Path = dataclasses.field(default_factory=lambda: Path("doqa/test_movies.json"))
    coqa_path: Path = dataclasses.field(default_factory=lambda: Path("coqa/dev.json"))
    hybridial_path: Path = dataclasses.field(default_factory=lambda: Path("hybridial/test.json"))
    sqa_path: Path = dataclasses.field(default_factory=lambda: Path("sqa/test.json"))
    topiocqa_path: Path = dataclasses.field(default_factory=lambda: Path("topiocqa/dev.json"))
    inscit_path: Path = dataclasses.field(default_factory=lambda: Path("inscit/dev.json"))

    # Others
    tokenizer_model: str = "nvidia/ChatQA-1.5-8B"
    out_seq_len: int = 64
    num_ctx: int = 5
    max_tokens: int = 64


def load_data(datapath: Path):
    logger.info(f"loading data from {datapath}")
    with open(datapath, "r") as f:
        data_list = json.load(f)

    return data_list


def reformat_question(turn_list, dataset_name: str):

    # Only take the lastest 7 turns
    turn_list = turn_list[-7:]
    assert turn_list[-1]["role"] == "user"

    long_answer_dataset_list = [
        "doc2dial",
        "quac",
        "qrecc",
        "inscit",
        "doqa_movies",
        "doqa_travel",
        "doqa_cooking",
        "hybridial",
        "convfinqa",
    ]
    long_and_short_dataset_list = ["topiocqa"]
    entity_dataset_list = ["sqa"]
    short_dataset_list = ["coqa"]

    if dataset_name in long_answer_dataset_list:
        for item in turn_list:
            if item["role"] == "user":
                # Only needs to add it on the first user turn
                item["content"] = "Please give a full and complete answer for the question. " + item["content"]
                break

    elif dataset_name in long_and_short_dataset_list:
        turn_list[-1]["content"] = (
            "Answer the following question with a short span, or a full and complete answer. " + turn_list[-1]["content"]
        )

    elif dataset_name in entity_dataset_list:
        turn_list[-1]["content"] = "Answer the following question with one or a list of items. " + turn_list[-1]["content"]

    elif dataset_name in short_dataset_list:
        turn_list[-1]["content"] = (
            "Answer the following question with a short span. The answer needs to be just in a few words. "
            + turn_list[-1]["content"]
        )

    else:
        raise Exception("please input a correct dataset name!")

    question = ""
    for item in turn_list:
        if item["role"] == "user":
            question += "User: " + item["content"] + "\n\n"
        else:
            assert item["role"] == "assistant"
            question += "Assistant: " + item["content"] + "\n\n"

    question += "Assistant:"

    return question


def get_inputs(
    data_list: dict, dataset_name: str, tokenizer, num_ctx: int, max_output_len: int, max_seq_length: int = 4096
) -> Tuple[List[str], List[str]]:

    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."  # noqa: E501

    prompt_list = []
    prompt_without_context_list = []
    len_context_tokens: List[int] = []
    len_doc_tokens: List[int] = []
    for item in data_list:
        turn_list = item["messages"]
        question_formatted = reformat_question(turn_list, dataset_name)

        ctx_list = ["title: " + ctx["title"] + ", source: " + ctx["text"] for ctx in item["ctxs"][:num_ctx]]
        context = "\n\n".join(ctx_list)

        context_tokens = tokenizer.encode(context)
        question_tokens = tokenizer.encode(question_formatted)
        system_tokens = tokenizer.encode(system)
        len_context_tokens.append(len(context_tokens))
        for doc in ctx_list:
            doc_tokens = tokenizer.encode(doc)
            len_doc_tokens.append(len(doc_tokens))
        # logger.info(
        #     f"{len(context_tokens)} context_tokens + "
        #     f"{len(question_tokens)} question_tokens + "
        #     f"{len(system_tokens)} system_tokens"
        # )

        if len(context_tokens) + len(question_tokens) + len(system_tokens) + max_output_len >= max_seq_length:
            context_tokens = context_tokens[: max_seq_length - max_output_len - len(question_tokens) - len(system_tokens)]
            context = tokenizer.decode(context_tokens, skip_special_tokens=True)

        model_input = system + "\n\n" + context + "\n\n" + question_formatted
        model_input_without_context = system + "\n\n" + question_formatted

        prompt_list.append(model_input)
        prompt_without_context_list.append(model_input_without_context)
    logger.info(
        f"Context tokens, avg= {np.mean(len_context_tokens):.0f}, "
        f"stddev= {np.std(len_context_tokens):.0f}, "
        f"max= {np.max(len_context_tokens)}, "
        f"count= {len(len_context_tokens)}"
    )
    logger.info(
        f"Document tokens, avg= {np.mean(len_doc_tokens):.0f}, "
        f"stddev= {np.std(len_doc_tokens):.0f}, "
        f"max= {np.max(len_doc_tokens)}, "
        f"count= {len(len_doc_tokens)}"
    )

    return prompt_list, prompt_without_context_list


def get_prompt_list(args: ChatRAGBenchArgs) -> Tuple[List[str], dict, List[str]]:

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)

    # Get input data
    if args.eval_dataset == "doc2dial":
        input_datapath = args.data_folder / args.doc2dial_path
    elif args.eval_dataset == "convfinqa":
        input_datapath = args.data_folder / args.convfinqa_path
    elif args.eval_dataset == "quac":
        input_datapath = args.data_folder / args.quac_path
    elif args.eval_dataset == "qrecc":
        input_datapath = args.data_folder / args.qrecc_path
    elif args.eval_dataset == "doqa_cooking":
        input_datapath = args.data_folder / args.doqa_cooking_path
    elif args.eval_dataset == "doqa_travel":
        input_datapath = args.data_folder / args.doqa_travel_path
    elif args.eval_dataset == "doqa_movies":
        input_datapath = args.data_folder / args.doqa_movies_path
    elif args.eval_dataset == "coqa":
        input_datapath = args.data_folder / args.coqa_path
    elif args.eval_dataset == "sqa":
        input_datapath = args.data_folder / args.sqa_path
    elif args.eval_dataset == "topiocqa":
        input_datapath = args.data_folder / args.topiocqa_path
    elif args.eval_dataset == "inscit":
        input_datapath = args.data_folder / args.inscit_path
    elif args.eval_dataset == "hybridial":
        input_datapath = args.data_folder / args.hybridial_path
    else:
        raise Exception(f"Invalid eval_dataset name ({args.eval_dataset})!")

    data_list = load_data(input_datapath)
    logger.info(f"number of samples in the dataset: {len(data_list)}")
    prompt_list, prompt_without_context_list = get_inputs(
        data_list, args.eval_dataset, tokenizer, num_ctx=args.num_ctx, max_output_len=args.out_seq_len
    )

    return prompt_list, data_list, prompt_without_context_list


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(RAGArgs, "rag")
    parser.add_arguments(ChatRAGBenchArgs, "chatragbench")
    args = parser.parse_args()
    logger.info(args)

    # Bos token for llama-3
    bos_token = "<|begin_of_text|>"

    # Get prompt_list
    prompt_list, data_list, prompt_without_context_list = get_prompt_list(args.chatragbench)

    # Get output_datapath
    output_datapath = args.chatragbench.output_folder / f"{args.chatragbench.eval_dataset}_output.txt"
    output_datapath.parent.mkdir(parents=True, exist_ok=True)

    # Run inference
    sampling_params = SamplingParams(temperature=0, top_k=1, max_tokens=args.chatragbench.max_tokens)

    # Make RAG
    rag = make_rag(args.rag)

    # Fill document cache
    doc_hash_to_id: Dict[int, DocumentId] = {}
    prompt_doc_ids: List[List[DocumentId]] = []
    for item in data_list:
        doc_ids: List[DocumentId] = []
        for ctx in item["ctxs"][: args.chatragbench.num_ctx]:
            document = ctx["text"]
            doc_hash = hash(document)
            if doc_hash not in doc_hash_to_id:
                doc_ids = rag.add_cache([document])
                assert len(doc_ids) == 1
                doc_hash_to_id[doc_hash] = doc_ids[0]
            doc_ids.append(doc_hash_to_id[doc_hash])
        prompt_doc_ids.append(doc_ids)
    logger.info(f"{len(doc_hash_to_id)} unique documents")

    # Generate output texts
    output_list = []
    for prompt, doc_ids in zip(prompt_without_context_list, prompt_doc_ids):
        prompt = bos_token + prompt
        generated_strs = rag.generate(doc_ids, prompt, sampling_params)
        generated_text = " ".join(generated_strs).strip().replace("\n", " ")

        # logger.info(f"generated_text: {generated_text}")
        output_list.append(generated_text)

    logger.info(f"writing to {output_datapath}")
    with open(output_datapath, "w") as f:
        for output in output_list:
            f.write(output + "\n")


if __name__ == "__main__":
    """
    Example:
        python benchmarks/chatragbench.py --eval_dataset doc2dial \
            --data_folder ChatRAG-Bench/data --output_folder results/chatragbench \
            --rag_type=parrot
    """
    main()
