import dataclasses
from typing import List, Optional

import datasets

from drag.logging import logger

DATASET_NAMES = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
]


@dataclasses.dataclass
class LongBenchArgs:
    longbench_dataset_name: str  # LongBench dataset name.
    longbench_out_seq_len: int = 64


@dataclasses.dataclass
class LongBenchRow:
    input: str
    context: str
    answers: List[str]
    length: int
    dataset: str
    language: str
    all_classes: Optional[List[str]]
    _id: str


@dataclasses.dataclass
class LongBenchDataset:
    rows: List[LongBenchRow]


def load_dataset(dataset_name: str) -> LongBenchDataset:
    if dataset_name not in DATASET_NAMES:
        logger.error(f"Invalid LongBench dataset_name {dataset_name}; options are {DATASET_NAMES}")
        raise ValueError(f"Invalid LongBench dataset_name {dataset_name}")
    return LongBenchDataset(
        rows=[
            LongBenchRow(
                input=row["input"],
                context=row["context"],
                answers=row["answers"],
                length=row["length"],
                dataset=row["dataset"],
                language=row["language"],
                all_classes=row["all_classes"],
                _id=row["_id"],
            )
            for row in datasets.load_dataset("THUDM/LongBench", dataset_name, split="test")
        ]
    )
