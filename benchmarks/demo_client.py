#!/usr/bin/env python3
"""Lazy-vs-baseline A/B demo driver.

Pushes one shared doc corpus into two RAG servers (Lazy on one port, a baseline
on another), then fires a sequence of queries that each retrieve the SAME hot
docs but in a DIFFERENT order. That ordering is the whole point:

  * Lazy-Attn caches one position-agnostic copy per doc, so every retrieved doc
    is a cache hit regardless of slot -> fast, flat TTFT.
  * Prefix caching (stock vLLM) only reuses a contiguous prefix, so as soon as
    the doc order changes the prefix diverges and the rest is recomputed -> TTFT
    grows with context.

Prints a per-query side-by-side TTFT table and an aggregate speedup.
"""
from __future__ import annotations

import argparse
import random
import sys
import time

import requests

GREEN, YEL, CYAN, BOLD, DIM, RST = "\033[32m", "\033[33m", "\033[36m", "\033[1m", "\033[2m", "\033[0m"


def _wait_healthy(url: str, timeout: float = 600.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.ok and r.json().get("status") == "ok":
                return r.json()
        except requests.RequestException:
            pass
        print(f"  waiting for {url} ...", flush=True)
        time.sleep(3)
    sys.exit(f"server at {url} never became healthy")


def _make_corpus(n: int, doc_len: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    vocab = [f"tok{i:04d}" for i in range(4000)]
    docs = []
    for d in range(n):
        words = [f"[DOC{d}]"] + [rng.choice(vocab) for _ in range(doc_len)]
        docs.append(" ".join(words) + ".")
    return docs


def _post_docs(url: str, docs: list[str]) -> list[int]:
    r = requests.post(f"{url}/docs", json={"docs": docs, "prewarm": True}, timeout=1800)
    r.raise_for_status()
    body = r.json()
    print(f"  {url}: cached {len(body['doc_ids'])} docs (prewarm {body['prewarm_ms']:.0f} ms)", flush=True)
    return body["doc_ids"]


def _chat(question: str) -> str:
    """Wrap a bare question in the Llama-3 chat format so the instruction-tuned
    model actually answers instead of emitting EOS on raw text."""
    return ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "Answer the question using the documents above.\n"
            f"Question: {question}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n")


def _query(url: str, doc_ids: list[int], q: str, max_tokens: int) -> dict:
    r = requests.post(f"{url}/query",
                      json={"doc_ids": doc_ids, "query": q, "max_tokens": max_tokens, "prewarm": False},
                      timeout=600)
    r.raise_for_status()
    return r.json()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lazy-url", default="http://127.0.0.1:8001")
    ap.add_argument("--base-url", default="http://127.0.0.1:8002")
    ap.add_argument("--lazy-name", default="Lazy-Attn (ours)")
    ap.add_argument("--base-name", default="Prefix Caching")
    ap.add_argument("--num-docs", type=int, default=16, help="size of the shared hot corpus")
    ap.add_argument("--k", type=int, default=8, help="docs retrieved per query")
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--doc-len", type=int, default=384, help="words per doc")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"{BOLD}== Lazy-vs-baseline RAG demo =={RST}")
    h_lazy = _wait_healthy(args.lazy_url)
    h_base = _wait_healthy(args.base_url)
    print(f"  {GREEN}{args.lazy_name}{RST}: {h_lazy['rag_type']} @ {h_lazy['model']}")
    print(f"  {YEL}{args.base_name}{RST}: {h_base['rag_type']} @ {h_base['model']}\n")

    corpus = _make_corpus(args.num_docs, args.doc_len, args.seed)
    print(f"Caching shared corpus ({args.num_docs} docs x ~{args.doc_len} words):", flush=True)
    ids_lazy = _post_docs(args.lazy_url, corpus)
    ids_base = _post_docs(args.base_url, corpus)

    rng = random.Random(args.seed + 1)
    questions = [
        "Summarize the key facts from the documents above.",
        "What entities appear most often across these passages?",
        "List the main topics covered by the retrieved documents.",
        "Which document is most relevant and why?",
    ]

    # Untimed warmup: the very first request to each engine pays one-off CUDA
    # graph / allocator costs that would otherwise skew query 0.
    warm = list(range(args.k))
    _query(args.lazy_url, [ids_lazy[j] for j in warm], _chat("warmup"), 4)
    _query(args.base_url, [ids_base[j] for j in warm], _chat("warmup"), 4)

    print(f"\n{BOLD}Each query retrieves the same {args.k} hot docs in a NEW order "
          f"(prefix cache can't reuse; lazy can):{RST}")
    hdr = f"{'#':>2}  {'order (doc ids)':<26}  {args.lazy_name:>18}  {args.base_name:>16}  {'speedup':>8}"
    print(DIM + hdr + RST)
    print(DIM + "-" * len(hdr) + RST)

    sum_lazy = sum_base = 0.0
    for i in range(args.queries):
        order = rng.sample(range(args.num_docs), args.k)
        q = _chat(questions[i % len(questions)])
        dl = _query(args.lazy_url, [ids_lazy[j] for j in order], q, args.max_tokens)
        db = _query(args.base_url, [ids_base[j] for j in order], q, args.max_tokens)
        sum_lazy += dl["ttft_ms"]
        sum_base += db["ttft_ms"]
        sp = db["ttft_ms"] / max(dl["ttft_ms"], 1e-6)
        order_str = ",".join(map(str, order))
        if len(order_str) > 24:
            order_str = order_str[:23] + "…"
        print(f"{i:>2}  {order_str:<26}  {GREEN}{dl['ttft_ms']:>15.1f}ms{RST}  "
              f"{YEL}{db['ttft_ms']:>13.1f}ms{RST}  {BOLD}{sp:>7.2f}x{RST}")

    print(DIM + "-" * len(hdr) + RST)
    mean_lazy = sum_lazy / args.queries
    mean_base = sum_base / args.queries
    print(f"{BOLD}mean TTFT{RST}  {GREEN}{mean_lazy:>33.1f}ms{RST}  "
          f"{YEL}{mean_base:>13.1f}ms{RST}  {BOLD}{mean_base / max(mean_lazy, 1e-6):>7.2f}x{RST}")
    print(f"\n{BOLD}{GREEN}Lazy-Attn is {mean_base / max(mean_lazy, 1e-6):.2f}x faster "
          f"to first token on reordered RAG retrieval.{RST}")


if __name__ == "__main__":
    main()
