"""Utilize BlockAttention Dataset."""

from __future__ import annotations  # isort:skip

import copy
import dataclasses
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from transformers import AutoTokenizer

from rag.logging import logger


@dataclasses.dataclass
class ChatRAGBenchArgs:
    """ChatRAG-Bench arguments"""

    # Dataset path
    data_folder: Path = dataclasses.field(
        default_factory=lambda: Path("./Hotpot-Bench")
    )  # path to the data folder of ChatRAG Bench
    output_folder: Path = dataclasses.field(default_factory=lambda: Path(".")) 
    eval_dataset: str = ""
    eval_path: Path = dataclasses.field(default_factory=lambda: Path("hotpot_test_v1.json"))
    out_seq_len: int = 512
    num_ctx: int = 1000  # Original default was 5.
    max_tokens: int = 512


def load_data(datapath: Path):
    logger.info(f"loading data from {datapath}")
    with open(datapath, "r") as f:
        data_list = json.load(f)
    return data_list


def get_inputs(
    data_list: dict, dataset_name: str, tokenizer, num_ctx: int, max_output_len: int, max_seq_length: int = 4096
) -> List[str]:

    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."  # noqa: E501

    prompt_without_context_list = []
    # len_context_tokens: List[int] = []
    len_doc_tokens: List[int] = []
    for item in data_list:
        turn_list = item["messages"]
        question_formatted = reformat_question(turn_list, dataset_name)

        ctx_list = ["title: " + ctx["title"] + ", source: " + ctx["text"] for ctx in item["ctxs"][:num_ctx]]

        # context_tokens = tokenizer.encode("\n\n".join(ctx_list))
        # question_tokens = tokenizer.encode(question_formatted)
        # system_tokens = tokenizer.encode(system)
        # logger.info(
        #     f"{len(context_tokens)} context_tokens + "
        #     f"{len(question_tokens)} question_tokens + "
        #     f"{len(system_tokens)} system_tokens"
        # )
        # len_context_tokens.append(len(context_tokens))

        for doc in ctx_list:
            doc_tokens = tokenizer.encode(doc)
            len_doc_tokens.append(len(doc_tokens))

        model_input_without_context = system + "\n\n" + question_formatted
        prompt_without_context_list.append(model_input_without_context)
    # logger.info(
    #     f"Context tokens, avg= {np.mean(len_context_tokens):.0f}, "
    #     f"stddev= {np.std(len_context_tokens):.0f}, "
    #     f"max= {np.max(len_context_tokens)}, "
    #     f"count= {len(len_context_tokens)}"
    # )
    logger.info(
        f"Document tokens, avg= {np.mean(len_doc_tokens):.0f}, "
        f"stddev= {np.std(len_doc_tokens):.0f}, "
        f"max= {np.max(len_doc_tokens)}, "
        f"count= {len(len_doc_tokens)}"
    )

    return prompt_without_context_list


def get_datapath(args: ChatRAGBenchArgs) -> Path:
    if args.eval_dataset == "doc2dial":
        return args.data_folder / args.doc2dial_path
    elif args.eval_dataset == "convfinqa":
        return args.data_folder / args.convfinqa_path
    elif args.eval_dataset == "quac":
        return args.data_folder / args.quac_path
    elif args.eval_dataset == "qrecc":
        return args.data_folder / args.qrecc_path
    elif args.eval_dataset == "doqa_cooking":
        return args.data_folder / args.doqa_cooking_path
    elif args.eval_dataset == "doqa_travel":
        return args.data_folder / args.doqa_travel_path
    elif args.eval_dataset == "doqa_movies":
        return args.data_folder / args.doqa_movies_path
    elif args.eval_dataset == "coqa":
        return args.data_folder / args.coqa_path
    elif args.eval_dataset == "sqa":
        return args.data_folder / args.sqa_path
    elif args.eval_dataset == "topiocqa":
        return args.data_folder / args.topiocqa_path
    elif args.eval_dataset == "inscit":
        return args.data_folder / args.inscit_path
    elif args.eval_dataset == "hybridial":
        return args.data_folder / args.hybridial_path
    else:
        raise Exception(f"Invalid eval_dataset name ({args.eval_dataset})!")


def get_prompt_list(args: ChatRAGBenchArgs) -> Tuple[dict, List[str]]:

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)

    # Get input data
    input_datapath = get_datapath(args)
    data_list = load_data(input_datapath)
    logger.info(f"number of samples in the dataset: {len(data_list)}")
    prompt_without_context_list = get_inputs(
        data_list, args.eval_dataset, tokenizer, num_ctx=args.num_ctx, max_output_len=args.out_seq_len
    )

    return data_list, prompt_without_context_list