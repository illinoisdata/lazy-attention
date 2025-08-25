import json
import logging
import argparse
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507"
    max_seq_length: int = 10000

    # LoRA
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    full_finetuning: bool = False  # if True, disables LoRA

    # Training
    output_dir: str = "./qwen3-bf16-ft"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    seed: int = 42
    wandb_project: str = "qwen3-finetune"
    wandb_run_name: str = "qwen3-4b-lora-bf16-epoch1-len10k"

    # Dtypes
    torch_dtype: str = "bfloat16"  # we’ll enforce bf16
    use_tf32: bool = True         

def load_jsonl(path: str, seed: int = 42, test_path: str = "test.jsonl"):
    """
    Load JSONL file, split into train/val/test (70/15/15).
    Saves test.jsonl, returns dict with train + val only (lists of dicts).
    """
    path = Path(path)
    assert path.exists(), f"File not found: {path}"

    # Load records
    with path.open("r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = [json.loads(line) for line in f if line.strip()]

    # Shuffle
    random.seed(seed)
    random.shuffle(data)

    n_total = len(data)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    # Save test set
    with open(test_path, "w", encoding="utf-8") as f:
        for ex in test_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Dataset sizes → train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")
    print(f"Test set written to {test_path}")

    return {
        "train": train_data,
        "validation": val_data
    }


class ChatDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], tokenizer, max_length: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Ensure pad token
        if getattr(self.tokenizer, "pad_token", None) is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        msgs = item.get("messages", []) or []
        if not isinstance(msgs, list):
            msgs = []

        input_ids = None
        labels = None
        prompt_len = 0

        # Try to use tokenizer.apply_chat_template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                tokenized = self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=True,
                    add_generation_prompt=False,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                # tokenized may be a dict or tensor-like
                if isinstance(tokenized, dict):
                    input_ids = tokenized.get("input_ids")
                else:
                    input_ids = tokenized
                if input_ids is not None:
                    input_ids = input_ids.squeeze(0)
                    labels = input_ids.clone()
                # compute prompt length using the non-tokenized template string
                try:
                    prompt_str = self.tokenizer.apply_chat_template(
                        msgs[:-1], tokenize=False, add_generation_prompt=True
                    )
                    p_tok = self.tokenizer(
                        prompt_str, return_tensors="pt", truncation=True, max_length=self.max_length
                    ).input_ids.squeeze(0)
                    prompt_len = int(p_tok.size(0))
                except Exception:
                    prompt_len = 0
            except Exception as e:
                print(f"apply_chat_template failed or returned unexpected type: {e}")
                input_ids = None  # force fallback

        # Fallback: simple ChatML-ish join and standard tokenizer
        if input_ids is None:
            logging.debug("Falling back to simple join fallback for tokenization.")
            def join_messages(messages):
                parts = []
                for m in messages:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    parts.append(f"<|{role}|>:\n{content}\n")
                return "\n".join(parts)

            full = join_messages(msgs)
            enc = self.tokenizer(full, truncation=True, max_length=self.max_length, return_tensors="pt")
            input_ids = enc.input_ids.squeeze(0)
            labels = input_ids.clone()

            if len(msgs) >= 1:
                prompt_text = join_messages(msgs[:-1]).strip()
                logging.debug(f"Prompt text (fallback) length {len(prompt_text)}: {prompt_text!r}")
                if prompt_text:
                    p_tok = self.tokenizer(
                        prompt_text, truncation=True, max_length=self.max_length, return_tensors="pt"
                    ).input_ids.squeeze(0)
                    prompt_len = int(p_tok.size(0))
                else:
                    prompt_len = 0
            else:
                prompt_len = 0

        # Safety: ensure prompt_len isn't longer than sequence length
        seq_len = int(labels.size(0))
        if prompt_len >= seq_len:
            # If prompt covers full sequence, mask everything (no target tokens)
            print("WARN: prompt_len >= seq_len", {"seq_len": seq_len, "prompt_len": prompt_len, "sample": item})
            labels[:] = -100
        elif prompt_len > 0:
            labels[:prompt_len] = -100
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_id is not None and prompt_len - 1 >= 0 and labels[prompt_len - 1] == eos_id:
                labels[prompt_len - 1] = -100

        return {"input_ids": input_ids.long(), "labels": labels.long()}


def setup_model_and_tokenizer(cfg: TrainConfig):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Model (pure bf16)
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
        device_map=None,  # put whole model on default device; set "auto" if you want HF device mapping
    )

    # If we added tokens (e.g., PAD), resize embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Optional perf/memory knobs
    if cfg.use_tf32 and torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

    # Enable gradient checkpointing to reduce memory (optional)
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    # LoRA (unless full finetune)
    if cfg.use_lora and not cfg.full_finetuning:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        # Optionally let embeddings / lm_head train too
        for n, p in model.named_parameters():
            if "embed_tokens" in n or "lm_head" in n:
                p.requires_grad = True
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.output_dir:
        cfg.output_dir = args.output_dir

    logging.basicConfig(level=logging.INFO)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=vars(cfg))

    # Load and split JSONL (returns dict with lists)
    splits = load_jsonl(args.train_file)
    train_data = splits["train"]
    eval_data = splits["validation"]

    logging.info(f"Train / Eval sizes: {len(train_data)} / {len(eval_data)}")

    model, tokenizer = setup_model_and_tokenizer(cfg)
    train_ds = ChatDataset(train_data, tokenizer, cfg.max_seq_length)
    eval_ds = ChatDataset(eval_data, tokenizer, cfg.max_seq_length)

    def collate_fn(batch):
        input_ids = [b["input_ids"] for b in batch]
        labels = [b["labels"] for b in batch]
        pad_id = tokenizer.pad_token_id
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = input_ids.ne(pad_id)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    # Base arguments (constructor) — canonical attributes
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        eval_steps=cfg.eval_steps,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,

        eval_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",

        bf16=True,
        report_to=["wandb"],
        run_name=cfg.wandb_run_name,
        logging_first_step=True,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    logging.info(f"Model saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
