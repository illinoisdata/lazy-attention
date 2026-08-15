"""Smoke-test LazyAttention end to end.

Runs the same questions three ways -- documents concatenated into the prompt,
documents through LazyAttention's `document_seqs` path, and the same documents
in a different order (the case prefix caching cannot reuse but LazyAttention
can).

Every answer must contain the fact its documents support. That is the check
that fails when attention or the cache regresses: a broken rotation or a
mis-addressed block yields fluent text that no longer answers the question.
The inlined run is held to the same standard -- the patches are global, so it
is not an independent control, and letting it off would hide the failures that
break everything at once.

The lazy answer is *not* required to match the inlined one token for token.
They run different attention patterns by construction -- one causal sequence
versus per-document position-agnostic encoding -- so they routinely agree on
the fact and differ in phrasing or length. Differences are printed.

    python scripts/validate_lazy.py [--model MODEL]

Runs greedily (temperature=0) so the comparison is deterministic.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import NamedTuple

# Patch vLLM before anything imports it.
import lazy.__vllm__  # noqa: F401  isort:skip

from vllm import LLM, SamplingParams  # noqa: E402


class Case(NamedTuple):
    prompt: str
    documents: list[str]
    expected: str  # the fact the documents support; must appear in the answer


CASES = [
    Case(
        "Question: Which city is the capital of France? Answer:",
        [
            "Paris is the capital and largest city of France.",
            "Berlin is the capital of Germany.",
            "Rome is the capital of Italy.",
        ],
        "Paris",
    ),
    Case(
        "Question: Which city is the capital of Italy? Answer:",
        [
            "Madrid is the capital of Spain.",
            "Rome is the capital of Italy, on the river Tiber.",
            "Lisbon is the capital of Portugal.",
        ],
        "Rome",
    ),
]

LABELS = ("baseline", "lazy", "reordered")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        default=os.environ.get("LAZY_TEST_MODEL",
                                               "hxia7/Llama-3.2-1B-Block-FT"))
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--max-model-len", type=int, default=2048)
    args = parser.parse_args()

    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0)
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )

    def answers(prompts, **kwargs) -> list[str]:
        outputs = llm.generate(prompts=prompts, sampling_params=sampling,
                               **kwargs)
        return [o.outputs[0].text for o in outputs]

    prompts = [case.prompt for case in CASES]
    docs = [case.documents for case in CASES]

    baselines = answers(  # documents inlined into the prompt, no lazy path
        ["\n".join(case.documents) + "\n" + case.prompt for case in CASES])
    lazy = answers(prompts, document_seqs=docs)
    # Same documents, new order -- the reordering case prefix caching misses.
    reordered = answers(prompts,
                        document_seqs=[list(reversed(d)) for d in docs])

    failures = 0
    for i, (case, texts) in enumerate(
            zip(CASES, zip(baselines, lazy, reordered))):
        print(f"\n--- case {i}: {case.prompt}")
        for label, text in zip(LABELS, texts):
            print(f"  {label:<10}: {text!r}")

        # The baseline is checked too, and just as strictly. It does not run
        # stock vLLM -- the patches are global, so it goes through the same
        # patched engine with the documents inlined. A broken patch garbles it
        # as well, and skipping the case when the baseline looks wrong would
        # turn the loudest possible failure into a pass.
        for label, text in zip(LABELS, texts):
            if not text.strip():
                print(f"  FAIL: {label} produced no output")
                failures += 1
            elif case.expected.lower() not in text.lower():
                print(f"  FAIL: {label} answer does not contain "
                      f"{case.expected!r} -- the documents did not reach "
                      "attention intact")
                failures += 1

        # Expected: a different attention pattern reaches the same fact by a
        # different route. Shown so a sudden change in wording is visible.
        baseline_text, lazy_text, reordered_text = texts
        if lazy_text != baseline_text:
            print("  note: lazy wording differs from the baseline")
        if lazy_text != reordered_text:
            print("  note: reordering the documents changed the wording")

    print()
    if failures:
        print(f"FAILED ({failures} check(s))")
        return 1
    print(f"OK: {len(CASES)} cases answered correctly through the lazy path, "
          "in both document orders")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    sys.exit(main())
