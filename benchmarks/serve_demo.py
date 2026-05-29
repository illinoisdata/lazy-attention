#!/usr/bin/env python3
"""Single-SUT RAG demo server (one model per process).

Wraps the existing `rag.rag.RAG` abstraction in a tiny HTTP API so a doc corpus
can be pushed into the KV cache once, then queried repeatedly. Run one process
per method (e.g. lazyrag on :8001, llmrag on :8002) and drive both from
demo_client.py for a live Lazy-vs-baseline TTFT comparison.

Endpoints:
  GET  /health                      -> {status, rag_type, model, n_docs}
  POST /docs   {docs:[str], prewarm} -> {doc_ids:[int], prewarm_ms}
  POST /query  {doc_ids, query, ...} -> {answer, ttft_ms, total_ms, output_tokens, tok_per_s}
  POST /reset                        -> {ok:true}   (drops cached doc KV)

Two engine-env facts shape the file layout:
  * Lazy variants install their Triton kernel via `import lazy.__vllm__`, which
    rag.rag runs at IMPORT time iff VLLM_USE_LAZY_ATTENTION=1. So the env must be
    configured at module top, BEFORE `from rag.rag import ...`.
  * The stock path (llmrag) spawns a vLLM worker. multiprocessing's spawn child
    re-imports this module (as "__mp_main__"); the engine must therefore be built
    only under `if __name__ == "__main__"`, or the worker recursively rebuilds it
    and dies in the spawn-bootstrap guard.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid


def _early_rag_type() -> str:
    """Resolve --rag-type before argparse so we can set the lazy/spawn env prior
    to importing rag.rag. Env wins (it is inherited across the spawn boundary);
    otherwise scan argv WITHOUT argparse so a spawned worker re-importing this
    module under vLLM's own argv does not choke on unknown flags."""
    rt = os.environ.get("DEMO_RAG_TYPE")
    if rt:
        return rt
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--rag-type" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--rag-type="):
            return a.split("=", 1)[1]
    return "lazyrag"


def _configure_engine_env(rag_type: str) -> None:
    lazy_kernel = rag_type in ("lazyrag", "baseline")
    if lazy_kernel:
        os.environ["VLLM_USE_LAZY_ATTENTION"] = "1"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
    else:
        os.environ.pop("VLLM_USE_LAZY_ATTENTION", None)
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN_VLLM_V1")


_RAG_TYPE = _early_rag_type()
os.environ["DEMO_RAG_TYPE"] = _RAG_TYPE   # so spawned workers inherit the choice
_configure_engine_env(_RAG_TYPE)

# rag.rag lives in benchmarks/rag/; make the script dir importable explicitly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataclasses  # noqa: E402
from typing import List, Optional  # noqa: E402

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from vllm import SamplingParams  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from rag.rag import make_rag, RAGArgs  # noqa: E402


def _install_tokenizer_shim() -> None:
    """Newer transformers' tokenizer backend lacks `all_special_tokens_extended`,
    which vLLM's get_cached_tokenizer reads. Mirror the shim the benchmark
    harness applies so the demo server loads the same tokenizers."""
    import vllm.transformers_utils.tokenizer as _vtok

    def _ensure_extended(tok):
        if tok is not None and not hasattr(tok, "all_special_tokens_extended"):
            try:
                tok.all_special_tokens_extended = tok.all_special_tokens
            except Exception:
                pass
        return tok

    for _name in ("get_tokenizer", "get_cached_tokenizer"):
        _orig = getattr(_vtok, _name, None)
        if _orig is None or getattr(_orig, "_lazy_shim", False):
            continue

        def _wrap(*a, __orig=_orig, **k):
            if a:
                _ensure_extended(a[0])
            return _ensure_extended(__orig(*a, **k))

        _wrap._lazy_shim = True
        setattr(_vtok, _name, _wrap)


_install_tokenizer_shim()

# Module-level state set by main(); handlers read these at request time.
RAG = None          # type: ignore[assignment]
LABEL = _RAG_TYPE
MODEL = ""

app = FastAPI(title="lazy-attn demo")


class DocsBody(BaseModel):
    docs: List[str]
    prewarm: bool = True


class QueryBody(BaseModel):
    doc_ids: List[int]
    query: str
    max_tokens: int = 64
    min_tokens: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    prewarm: bool = True


async def _prewarm(doc_ids: List[int]) -> float:
    """Prefill the given docs into the KV cache (cost excluded from query TTFT).
    Not every RAG implements add_doc_async; ParrotRAG etc. simply skip."""
    fn = getattr(RAG, "add_doc_async", None)
    if fn is None or not doc_ids:
        return 0.0
    t0 = time.perf_counter()
    await fn(f"warm_{uuid.uuid4().hex}", list(doc_ids))
    return (time.perf_counter() - t0) * 1e3


def _sampling(body: QueryBody) -> SamplingParams:
    return SamplingParams(temperature=body.temperature, top_p=body.top_p,
                          repetition_penalty=body.repetition_penalty,
                          max_tokens=body.max_tokens,
                          min_tokens=min(body.min_tokens, body.max_tokens))


@app.get("/health")
async def health():
    return {"status": "ok", "label": LABEL, "rag_type": _RAG_TYPE,
            "model": MODEL, "n_docs": len(getattr(RAG, "_docs", {}))}


@app.post("/docs")
async def add_docs(body: DocsBody):
    doc_ids = RAG.add_cache(body.docs)
    prewarm_ms = await _prewarm(doc_ids) if body.prewarm else 0.0
    return {"doc_ids": doc_ids, "prewarm_ms": round(prewarm_ms, 2)}


@app.post("/query")
async def query(body: QueryBody):
    """Run the query against already-cached docs; report TTFT + throughput."""
    if body.prewarm:
        await _prewarm(body.doc_ids)
    sp = _sampling(body)
    t0 = time.perf_counter()
    ttft: Optional[float] = None
    text_parts: List[str] = []
    async for delta in RAG.iter_generate(doc_ids=body.doc_ids, query=body.query, sampling_params=sp):
        if ttft is None:
            ttft = (time.perf_counter() - t0) * 1e3
        text_parts.append(delta)
    total_ms = (time.perf_counter() - t0) * 1e3
    answer = "".join(text_parts)
    out_toks = max(1, len(answer.split()))
    decode_ms = max(total_ms - (ttft or total_ms), 1e-6)
    return {
        "label": LABEL,
        "answer": answer,
        "ttft_ms": round(ttft or total_ms, 2),
        "total_ms": round(total_ms, 2),
        "output_tokens": out_toks,
        "tok_per_s": round(out_toks / (decode_ms / 1e3), 2),
    }


@app.post("/query/stream")
async def query_stream(body: QueryBody):
    if body.prewarm:
        await _prewarm(body.doc_ids)
    sp = _sampling(body)

    async def gen():
        async for delta in RAG.iter_generate(doc_ids=body.doc_ids, query=body.query, sampling_params=sp):
            yield delta

    return StreamingResponse(gen(), media_type="text/plain")


@app.post("/reset")
async def reset():
    RAG.destroy_cache()
    return {"ok": True}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-SUT RAG demo server")
    p.add_argument("--rag-type", default=_RAG_TYPE,
                   help="lazyrag | baseline | llmrag | recllmrag | blockattnrag ...")
    p.add_argument("--model", default=os.environ.get("DEMO_MODEL", "hxia7/Llama-3.2-1B-Block-FT"))
    p.add_argument("--tokenizer", default=os.environ.get("DEMO_TOKENIZER", ""), help="defaults to --model")
    p.add_argument("--host", default=os.environ.get("DEMO_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("DEMO_PORT", "8001")))
    p.add_argument("--gpu-memory-utilization", type=float,
                   default=float(os.environ.get("DEMO_GPU_MEM_UTIL", "0.45")))
    p.add_argument("--max-model-len", type=int, default=int(os.environ.get("DEMO_MAX_MODEL_LEN", "16384")))
    p.add_argument("--enforce-eager", action="store_true",
                   default=os.environ.get("DEMO_ENFORCE_EAGER", "1") == "1")
    p.add_argument("--label", default=os.environ.get("DEMO_LABEL", ""), help="display name (defaults to rag-type)")
    return p.parse_args()


def _build_rag(args: argparse.Namespace):
    engine_args = EngineArgs(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=True,
    )
    rag_args = RAGArgs(rag_type=args.rag_type)
    rag_args.trrag_lm_name = args.model
    return make_rag(rag_args, engine_args=engine_args)


def main() -> None:
    global RAG, LABEL, MODEL
    args = _parse_args()
    LABEL = args.label or args.rag_type
    MODEL = args.model
    print(f"[serve_demo] label={LABEL} rag_type={args.rag_type} model={args.model} "
          f"port={args.port} lazy={os.environ.get('VLLM_USE_LAZY_ATTENTION','0')}", flush=True)
    RAG = _build_rag(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
