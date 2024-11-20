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
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import chatragbench
import longbench
from simple_parsing import ArgumentParser

from drag.logging import logger

MILLISECONDS_TO_SECONDS_CONVERSION = 1000


def grade_longbench(
    args: longbench.LongBenchArgs,
    result_json: Dict[str, Any],
) -> float:
    # Get prompt_list
    longbench_dataset = longbench.load_dataset(args.longbench_dataset_name)
    logger.info(f"Loaded {len(longbench_dataset.rows)} LongBench prompts")

    # Parse answer for each request ID.
    answers_by_id: List[List[str]] = []
    all_classes: Optional[List[str]] = None
    for idx, row in enumerate(longbench_dataset.rows):
        answers_by_id.append(row.answers)
        all_classes = row.all_classes

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
    logger.info(f"Longbench[{args.longbench_dataset_name}]: {len(predictions)} predictions, score= {total_score}")

    return total_score


def main(args: argparse.Namespace):
    logger.info(args)
    result_path = Path(args.result_dir) / f"{args.exp}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "r", encoding="utf-8") as f:
        result_json = json.load(f)
    logger.info(f"Read results from {result_path}")

    if args.dataset_name == "longbench":
        grade_longbench(args.longbench, result_json)
    else:
        raise ValueError(f"Unknown dataset for grading: {args.dataset_name}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Grade answers in benchmark_rag_serving.py results.")
    parser.add_argument(
        "--exp",
        type=str,
        help="Experiment name to be graded.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        choices=["random", "chatragbench", "longbench"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="results/",
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_arguments(chatragbench.ChatRAGBenchArgs, "chatragbench")
    parser.add_arguments(longbench.LongBenchArgs, "longbench")
    args = parser.parse_args()
    main(args)
