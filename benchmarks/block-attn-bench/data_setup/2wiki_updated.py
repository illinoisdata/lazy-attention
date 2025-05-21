import os
import json
import argparse
import random
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Set

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel

Document = Dict[str, Any]
SFTInstance = Dict[str, Any]

@dataclass
class RetrievalConfig:
    top_k: int = 10                      # number of docs to return per prompt
    candidate_pool: int = 20             # number of top relevance docs to consider
    alpha: float = 1.0                   # relevance weight
    beta: float = 0.8                    # persistence weight
    window_size: int = 50                # how many past prompts to recall
    decay: float = 0.9                   # decay factor for persistence
    skew_lambda: float = 2.0             # skewness parameter for sampling (1=softmax)
    gen_max_length: int = 32            # max tokens to generate for ground truth

# --- History Buffer ---
class History:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.buffer: deque[Set[int]] = deque(maxlen=window_size)

    def append(self, ids: Set[int]):
        self.buffer.append(ids)

    def persistence_score(self, candidate_ids: List[int], decay: float, device: torch.device) -> torch.Tensor:
        scores = torch.zeros(len(candidate_ids), device=device)
        id_to_idx = {gid: i for i, gid in enumerate(candidate_ids)}
        for age, s in enumerate(reversed(self.buffer)):
            weight = decay ** age
            for gid in s:
                if gid in id_to_idx:
                    scores[id_to_idx[gid]] += weight
        return scores

# --- Embedding Utility ---
@torch.no_grad()
def compute_embeddings(texts: List[str], model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(model.device)
    output = model(**inputs, return_dict=True)
    last_hidden = output.last_hidden_state
    mask = inputs.attention_mask.unsqueeze(-1).expand_as(last_hidden)
    masked = last_hidden * mask
    pooled = masked.sum(1) / mask.sum(1).clamp(min=1e-9)
    return pooled

# --- Skewed Sampling Retrieval ---
def retrieve_with_skew(
    question: str,
    docs: List[Document],
    retriever: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    history: History,
    config: RetrievalConfig
) -> List[Document]:
    texts = [question] + [d['text'] for d in docs]
    emb = compute_embeddings(texts, retriever, tokenizer)
    q_emb = emb[0].unsqueeze(0)
    d_emb = emb[1:]
    rel = torch.matmul(q_emb, d_emb.T).squeeze(0)
    device = rel.device
    M = min(config.candidate_pool, rel.size(0))
    vals, idxs = torch.topk(rel, k=M)
    pool_docs = [docs[i] for i in idxs.tolist()]
    pool_scores = vals.clone().detach()
    pool_ids = [d['global_id'] for d in pool_docs]
    pers = history.persistence_score(pool_ids, config.decay, device)
    combined = config.alpha * pool_scores + config.beta * pers
    probs = torch.softmax(config.skew_lambda * combined, dim=0)
    k = min(config.top_k, len(pool_docs))
    chosen = torch.multinomial(probs, num_samples=k, replacement=False)
    selected = [pool_docs[i] for i in chosen.tolist()]
    history.append({d['global_id'] for d in selected})
    return selected

# --- File Processing ---
def process_split(
    input_fp: str,
    output_fp: str,
    retriever: PreTrainedModel,
    retr_tok: PreTrainedTokenizer,
    generator: AutoModelForCausalLM,
    gen_tok: PreTrainedTokenizer,
    llama_tok: PreTrainedTokenizer,
    config: RetrievalConfig,
    state: Dict[str, Any],
    num_samples: int = -1
):
    history: History = state['history']
    usage: Counter = state['usage']
    doc_log: List[List[int]] = state['log']
    next_gid: int = state['next_gid']

    df = pd.read_parquet(input_fp)
    records = df.to_dict(orient='records')
    if num_samples > 0:
        records = random.sample(records, k=min(num_samples, len(records)))

    os.makedirs(os.path.dirname(output_fp), exist_ok=True)
    with open(output_fp, 'w', encoding='utf-8') as fout:
        for rec in tqdm(records, desc=f"Process {os.path.basename(input_fp)}"):
            ctx = rec.get('context', [])
            if isinstance(ctx, str): ctx = json.loads(ctx)
            docs = []
            for local_i, (title, txt_list) in enumerate(ctx):
                txt = ''.join(txt_list)
                key = (title, txt)
                if key not in state['global_map']:
                    state['global_map'][key] = next_gid
                    next_gid += 1
                gid = state['global_map'][key]
                docs.append({'title': title, 'text': txt, 'id': local_i, 'global_id': gid, 'score':0.0})

            selected = retrieve_with_skew(
                rec['question'], docs, retriever, retr_tok, history, config
            )
            gid_list = []
            system = "You are a helpful assistant. Use the following documents to answer the question.\n"
            for d in selected:
                usage[d['global_id']] += 1
                gid_list.append(d['global_id'])
                system += f"- {d['title']}: {d['text']}\n"
            doc_log.append(gid_list)
            user = f"Question: {rec['question']}"
            conv = [{'role':'system','content':system}, {'role':'user','content':user}]
            if llama_tok.chat_template is None:
                llama_tok.chat_template = (
                    "{% set loop_messages = messages %}{% for message in loop_messages %}"
                    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
                    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"  
                    "{{ content }}{% endfor %}{% if add_generation_prompt %}"  
                    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
                )
            prompt = llama_tok.apply_chat_template(
                conversation=conv, tokenize=False, add_generation_prompt=True
            )
            gen_inputs = gen_tok(prompt, return_tensors='pt').to(generator.device)
            gen_outputs = generator.generate(
                **gen_inputs,
                max_length=gen_inputs['input_ids'].size(1) + config.gen_max_length,
                pad_token_id=gen_tok.eos_token_id,
                do_sample=False
            )
            gen_text = gen_tok.decode(
                gen_outputs[0][gen_inputs['input_ids'].size(1):], skip_special_tokens=True
            ).strip()

            out = {
                'prompt': prompt,
                'question': rec['question'],
                'answers': [gen_text],
                'documents': selected
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    state['next_gid'] = next_gid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fp', required=True)
    parser.add_argument('--dev_fp', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--skew_lambda', type=float, default=2.0)
    parser.add_argument('--num_dev_samples', type=int, default=200, help="Number of dev prompts to sample.")
    args = parser.parse_args()

    cfg = RetrievalConfig(skew_lambda=args.skew_lambda)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    retr_tok = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
    retr_model = AutoModel.from_pretrained('facebook/contriever-msmarco').to(device).eval()

    llama_tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct', use_fast=False)
    if llama_tok.pad_token is None and llama_tok.eos_token is not None:
        llama_tok.pad_token = llama_tok.eos_token
    gen_tok = llama_tok
    generator = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct').to(device).eval()

    state = {
        'history': History(cfg.window_size),
        'usage': Counter(),
        'log': [],
        'global_map': {},
        'next_gid': 0
    }

    state['history'] = History(cfg.window_size)
    dev_out = os.path.join(args.output_dir, 'dev.jsonl')
    process_split(
        args.dev_fp, dev_out,
        retr_model, retr_tok,
        generator, gen_tok,
        llama_tok,
        cfg, state,
        num_samples=args.num_dev_samples
    )

    stats = {'usage_counts': dict(state['usage']), 'doc_log': state['log']}
    with open(os.path.join(args.output_dir, 'skew_stats.json'), 'w') as fs:
        json.dump(stats, fs, indent=2)
    print("Done.")

if __name__ == '__main__':
    main()
