import dataclasses
import re
import string
from collections import Counter
from typing import List, Optional, Union, TypedDict, Tuple
from pathlib import Path
import datasets
import jieba
from fuzzywuzzy import fuzz
from rouge import Rouge
import json

from rag.logging import logger

DATASET_NAMES = [
    "2wiki",
    "tqa",
    "hqa",
    "nq"
]

@dataclasses.dataclass
class BlockBenchArgs:
    blockbench_dataset_name: str = "nq"  # BlockBench dataset name.
    # We set to 200 to align the BlockAttention paper setting.
    blockbench_out_seq_len: int = 10  # Output sequence length.

Document = TypedDict("Document", {"title": str, "text": str, "score": float})

@dataclasses.dataclass
class BlockBenchRow:
    prompt: str
    question: str
    answers: List[str]
    generated: str
    documents: List[Document]
    blocks: List[str]
    instruction: str
    # length: int
    # dataset: str
    # language: str
    # all_classes: Optional[List[str]]
    _id: str


@dataclasses.dataclass
class BlockBenchDataset:
    rows: List[BlockBenchRow]
    name: str = "BlockBench"

def load_dataset(dataset_name: str) -> BlockBenchDataset:
    LIMITATION = None  # 2000
    if dataset_name not in DATASET_NAMES:
        logger.error(f"Invalid BlockBench dataset_name {dataset_name}; options are {DATASET_NAMES}")
        raise ValueError(f"Invalid BlockBench dataset_name {dataset_name}")
    # Find the path
    dataset_path = Path(__file__).parent.parent / "block_attn" / "Block-Attention" / "datahub" / "rag" /f"{dataset_name}_eval"
    data = []
    with open(dataset_path / "dataset", "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    if LIMITATION is not None:
        data = data[:LIMITATION]
        logger.info(f"Limiting to {LIMITATION} rows for efficiency")

    logger.info(f"Loaded {len(data)} BlockBench rows")

    return BlockBenchDataset(
        name=dataset_name,
        rows=[
            BlockBenchRow(
                prompt=row["prompt"],
                question=row["question"],
                answers=row["answers"],
                generated=row["generated"],
                documents=row["documents"],
                blocks=make_blocks(row["prompt"])[0],
                instruction=make_blocks(row["prompt"])[1],
                # length=row["length"],
                # dataset=row["dataset"],
                # language=row["language"],
                # all_classes=row["all_classes"],
                _id=i,
            )
            for i, row in enumerate(data)
        ]
    )
    

# --------------------------------------------------------------------------
# Functions from Block Attention repo
# --------------------------------------------------------------------------
def make_blocks(prompt: str) -> Tuple[List[str], str]:
    blocks: List[str] = [
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference documents that may help you in answering the user's question.\n\n"
    ]
    assert prompt.startswith(blocks[0]), json.dumps({
        "prompt": prompt, "blocks": blocks[0]}, ensure_ascii=False, indent=4
    )
    content = prompt[len(blocks[0]):]

    pos = content.find("<|eot_id|>") + len("<|eot_id|>")
    documents = content[:pos]
    instruction_ans_response = content[pos:]

    pos = documents.find("\n- Title:")
    while pos != -1:
        doc = documents[:pos + 1]
        blocks.append(doc)
        documents = documents[pos + 1:]
        pos = documents.find("\n- Title:")
    assert documents.startswith("- Title:") and documents.endswith("<|eot_id|>"), documents
    blocks.append(documents[:-len("<|eot_id|>")])
    instruction_ans_response = "<|eot_id|>" + instruction_ans_response

    assert instruction_ans_response.startswith(
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "Please write a high-quality answer for the given question using only the provided search documents"
    )
    blocks = [b for b in blocks if b != ""]

    # In order to align with Tulu's chat_template, do the following processing:
    assert "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" in blocks[0], blocks[0]
    blocks[0] = blocks[0].replace(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
        "<|user|>\n"
    )
    assert "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" in instruction_ans_response, instruction_ans_response
    instruction_ans_response = instruction_ans_response.replace(
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n", "\n\n"
    )
    assert "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" in instruction_ans_response, instruction_ans_response
    instruction_ans_response = instruction_ans_response.replace(
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "\n<|assistant|>\n"
    )
    return blocks, instruction_ans_response



"""Accuracy Measurments"""


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip("\n").split("\n")
    prediction = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            prediction = line
            break
    return fuzz.ratio(prediction, ground_truth) / 100


def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = 1.0 / len(em_match_list)
    else:
        score = 0.0
    return score

def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    score = rouge_score(prediction, ground_truth)
    return score


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

def qa_em_score(prediction, ground_truth, **kwargs):
    if isinstance(ground_truth, list) and isinstance(ground_truth[0], list):
        assert len(ground_truth) == 1, "Only support single ground truth list"
        ground_truth = ground_truth[0]
    assert isinstance(prediction, str), f"Expected string, got {type(prediction)}, content: {prediction}"
    assert isinstance(ground_truth, list), f"Expected list, got {type(ground_truth)}, content: {ground_truth}"
    for gt in ground_truth:
        if normalize_answer(gt) in normalize_answer(prediction):
            # Appears in any of the ground truths
            return 1.0
    return 0.0

dataset2metric = {
    "2wiki": qa_em_score,
    "tqa": qa_em_score,
    "hqa": qa_em_score,
    "nq": qa_em_score,
}

for dataset in DATASET_NAMES:
    assert dataset in dataset2metric, f"Missing {dataset} metrics"

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)