r"""Benchmark online serving throughput.

Extracted from VLLM/benchmark/benchmark_serving.py

Example
    python3 benchmarks/benchmark_rag_serving.py \
        --exp parrot \
        --rag_type=parrot \
        --dataset-name random \
        --tokenizer facebook/opt-125m
"""

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import chatragbench
import longbench
from simple_parsing import ArgumentParser

from rag.logging import logger

MILLISECONDS_TO_SECONDS_CONVERSION = 1000


@dataclasses.dataclass
class GradeAccuracyArgs:
    result_paths: List[Path]  # Paths to result JSON files.


def read_results(args: GradeAccuracyArgs) -> Dict[Path, Dict[str, Any]]:
    result_jsons = {}
    for result_path in args.result_paths:
        if not result_path.exists():
            logger.error(f"No such file: '{result_path}'")
            continue
        with open(result_path, "r", encoding="utf-8") as f:
            result_json = json.load(f)
            result_jsons[result_path] = result_json
        logger.info(f"Read results from {result_path}")
    return result_jsons


def grade_longbench(
    args: longbench.LongBenchArgs,
    result_jsons: Dict[Path, Dict[str, Any]],
) -> Dict[Path, float]:
    # Get prompt_list once.
    longbench_dataset = longbench.load_dataset(args.longbench_dataset_name)
    logger.info(f"Loaded {len(longbench_dataset.rows)} LongBench prompts")

    # Parse answer for each request ID.
    answers_by_id: List[List[str]] = []
    all_classes: Optional[List[str]] = None
    for idx, row in enumerate(longbench_dataset.rows):
        answers_by_id.append(row.answers)
        all_classes = row.all_classes

    # Score one by one.
    result_scores = {}
    for result_path, result_json in result_jsons.items():
        # Get answer and prediction pairs.
        answers: List[List[str]] = []
        predictions: List[str] = []
        for request_id, prediction in zip(result_json["input_request_ids"], result_json["generated_texts"]):
            answers.append(answers_by_id[request_id])
            predictions.append(prediction)

        # Grade.
        total_score = longbench.scorer(
            dataset=args.longbench_dataset_name,
            predictions=predictions,
            answers=answers,
            all_classes=all_classes,
        )
        result_scores[result_path] = total_score
        logger.info(
            f"Longbench[{args.longbench_dataset_name},{result_path.stem}]: "
            f"{len(predictions)} predictions, score= {total_score}"
        )

    return result_scores


def grade_chatragbench(
    args: chatragbench.ChatRAGBenchArgs,
    result_jsons: Dict[Path, Dict[str, Any]],
) -> Dict[Path, chatragbench.AccurracyResult]:
    # Get prompt_list once.
    groundtruth_answers = chatragbench.load_groundtruth_answers(args)
    logger.info(f"Loaded {len(groundtruth_answers)} ChatRAGBench prompts")

    # Score one by one.
    result_scores = {}
    for result_path, result_json in result_jsons.items():
        # Get answer and prediction pairs.
        selected_groundtruth_answers: List[List[str]] = []
        prediction_answers: List[str] = []
        for request_id, prediction in zip(result_json["input_request_ids"], result_json["generated_texts"]):
            selected_groundtruth_answers.append(groundtruth_answers[request_id])
            prediction_answers.append(prediction)

        # Grade.
        total_score = chatragbench.scorer(
            eval_dataset=args.eval_dataset,
            groundtruth_answers=selected_groundtruth_answers,
            prediction_answers=prediction_answers,
        )
        result_scores[result_path] = total_score
        logger.info(
            f"ChatRAGBench[{args.eval_dataset},{result_path.stem}]: "
            f"{len(prediction_answers)} predictions, score= {total_score}"
        )

    return result_scores

def grade_blockbench_cacheblendbench(
    args: longbench.LongBenchArgs,
    result_jsons: Dict[Path, Dict[str, Any]],
) -> Dict[Path, float]:

    longbench_dataset = longbench.load_dataset(args.longbench_dataset_name)
    logger.info(f"Loaded {len(longbench_dataset.rows)} LongBench prompts")

    all_classes = longbench_dataset.rows[0].all_classes

    result_scores: Dict[Path, float] = {}
    for result_path, result_json in result_jsons.items():
        predictions: List[str] = result_json["generated_texts"]
        answers: List[List[str]] = result_json["ground_truths"]

        if len(predictions) != len(answers):
            raise ValueError(
                f"Mismatch in counts for {result_path.stem}: "
                f"{len(predictions)} predictions vs {len(answers)} ground_truths"
            )

        # Compute the LongBench score
        total_score = longbench.scorer(
            dataset=args.longbench_dataset_name,
            predictions=predictions,
            answers=answers,
            all_classes=all_classes,
        )
        result_scores[result_path] = total_score

        logger.info(
            f"Block Bench[{args.longbench_dataset_name},{result_path.stem}]: "
            f"{len(predictions)} predictions, score= {total_score:.4f}"
        )

    return result_scores

block_bench_dataset_names = ["2wikimqa_block", "hotpotqa_block", "nq_block", "triviaqa_block"]
cache_blendbench_dataset_names = ["2wikimqa_cacheblend", "samsum_cacheblend", "musique_cacheblend"]
def grade(args: argparse.Namespace, result_jsons: Dict[Path, Dict[str, Any]]):
    if args.dataset_name == "longbench":
        grade_longbench(args.longbench, result_jsons)
    elif args.dataset_name == "chatragbench":
        grade_chatragbench(args.chatragbench, result_jsons)
    elif args.dataset_name in block_bench_dataset_names or args.dataset_name in cache_blendbench_dataset_names:
        grade_blockbench_cacheblendbench(args.longbench, result_jsons)
    else:
        raise ValueError(f"Unknown dataset for grading: {args.dataset_name}")


if __name__ == "__main__":
    # TODO: Move arguments into dataclasses.
    parser = ArgumentParser(description="Grade answers in benchmark_rag_serving.py results.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        choices=["random", "chatragbench", "longbench", "2wikimqa_block", "hotpotqa_block","2wikimqa_cacheblend", "samsum_cacheblend", "musique_cacheblend"],
        help="Name of the dataset to benchmark on.",
    )
    random_group = parser.add_argument_group("random dataset options")
    parser.add_argument(
        "--random-num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to generate.",
    )
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-document-len",
        type=int,
        default=64,
        help="Number of tokens per document, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-num-documents",
        type=int,
        default=4,
        help="Number of documents to generate, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-num-documents-per-prompt",
        type=int,
        default=4,
        help="Number of documents included in each prompt, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=1.0,
        help="Range of sampled ratio of input/output length, " "used only for random sampling.",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help="Number of fixed prefix tokens before random "
        " context. The length range of context in a random "
        " request is [random-prefix-len, "
        " random-prefix-len + random-prefix-len * random-range-ratio).",
    )
    parser.add_arguments(GradeAccuracyArgs, "grade")
    parser.add_arguments(chatragbench.ChatRAGBenchArgs, "chatragbench")
    parser.add_arguments(longbench.LongBenchArgs, "longbench")
    args, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        logger.warning(f"Unrecognized arguments: {unknown}")
    logger.info(args)

    result_jsons = read_results(args.grade)
    grade(args, result_jsons)
