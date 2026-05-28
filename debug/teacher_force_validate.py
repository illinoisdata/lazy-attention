"""Teacher-forced per-position agreement between block reference and lazy.

Greedy exact-match is brittle on this 1B model because the top candidates are
often within ~1 logit (near-ties around EOS), and the lazy path's extra bf16
Q-rotation can flip them. To validate the lazy attention over MANY tokens
without that brittleness, we teacher-force both paths on the SAME fixed
sequence and compare the argmax prediction at every position:

  1. Block (Transformers, block-attention mask) greedily generates a reference
     answer. By construction its per-position argmax == the generated tokens.
  2. The lazy path (LazyLLM) is fed `query_block + reference_answer` as the
     prompt with prompt_logprobs, so we read its argmax prediction at every
     position of the answer region given identical preceding context.
  3. Report how often lazy's argmax matches the block reference's choice.

High agreement => the lazy mechanism faithfully reproduces block attention
over a long context; the only greedy divergences are numerical near-ties.
"""

from __future__ import annotations

import argparse

import torch

import compare_lazy_block_infer as C


DOC_BLOCKS = [
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are an intelligent AI assistant. Answer the user's question using "
    "only the provided documents. Write a detailed, multi-sentence answer.\n\n",
    "- Title: Polish-Russian War (film)\n"
    "Polish-Russian War (Wojna polsko-ruska) is a 2009 Polish film directed by "
    "Xawery Zulawski based on the novel Polish-Russian War under the white-red "
    "flag by Dorota Maslowska.\n",
    "- Title: Xawery Zulawski\n"
    "Xawery Zulawski (born 22 December 1971 in Warsaw) is a Polish film "
    "director. He is the son of actress Malgorzata Braunek and director Andrzej "
    "Zulawski.\n",
    "- Title: Andrzej Zulawski\n"
    "Andrzej Zulawski (1940-2016) was a Polish film director known for "
    "challenging, avant-garde films. He was married to Malgorzata Braunek.\n",
]

QUERY_BLOCK = (
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "Please write a high-quality, detailed answer for the question using only "
    "the provided documents.\n"
    "Question: Who directed the film Polish-Russian War, and who are that "
    "director's parents?"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


def lazy_prompt_argmax(model_name, query_text, ref_ids, tokenizer):
    """Teacher-force `query_text + ref` through lazy; return per-position argmax."""
    import lazy.__vllm__  # noqa: F401
    import vllm.transformers_utils.tokenizer as vtok
    from lazy.entrypoints.llm import LazyLLM
    from vllm import SamplingParams

    orig = vtok.get_cached_tokenizer
    vtok.get_cached_tokenizer = lambda t: (
        setattr(t, "all_special_tokens_extended", t.all_special_tokens)
        or orig(t)) if not hasattr(t, "all_special_tokens_extended") else orig(t)

    llm = LazyLLM(model=model_name, gpu_memory_utilization=0.9,
                  enable_prefix_caching=True, trust_remote_code=True,
                  enforce_eager=True)

    ref_text = tokenizer.decode(ref_ids)
    full_prompt = query_text + ref_text
    out = llm.generate(
        prompts=[full_prompt],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1,
                                       prompt_logprobs=1),
        document_seqs=[DOC_BLOCKS],
    )[0]

    prompt_ids = out.prompt_token_ids
    plps = out.prompt_logprobs  # list aligned with prompt_ids; [0] is None
    # argmax prediction at each position = the rank-0 entry of plps[i]
    argmax = [None]
    for d in plps[1:]:
        top = min(d.items(), key=lambda kv: kv[1].rank)
        argmax.append(top[0])
    return prompt_ids, argmax


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=C.MODEL)
    parser.add_argument("--max-tokens", type=int, default=80)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    C.DOC_BLOCKS = DOC_BLOCKS
    C.QUERY_BLOCK = QUERY_BLOCK

    # Block reference: greedy generation; its per-position argmax == ref ids.
    ref_ids, ref_text, _ = C.run_block(args.model, args.max_tokens)
    if ref_ids and ref_ids[-1] == tok.eos_token_id:
        ref_ids = ref_ids[:-1]
    torch.cuda.empty_cache()
    print(f"reference ({len(ref_ids)} tokens): {ref_text!r}")

    # Lazy: teacher-force query+ref, read per-position argmax.
    # NOTE: in the lazy path, prompt_logprobs is aligned with the FULL internal
    # sequence (prepended padded documents + query + ref), which is longer than
    # prompt_token_ids (query + ref only). The reference answer is appended last,
    # so its predictions are the final len(ref_ids) entries of lazy_argmax.
    _, lazy_argmax = lazy_prompt_argmax(args.model, QUERY_BLOCK, ref_ids, tok)
    n = len(ref_ids)
    answer_argmax = lazy_argmax[-n:]

    agree = 0
    first_div = None
    for j, ref_tok in enumerate(ref_ids):
        if answer_argmax[j] == ref_tok:
            agree += 1
        elif first_div is None:
            first_div = j
    print(f"teacher-forced positions compared: {n}")
    print(f"lazy argmax == block argmax: {agree}/{n} "
          f"({100.0 * agree / max(n, 1):.1f}%)")
    if first_div is not None:
        print(f"first disagreement at answer position {first_div}: "
              f"block={ref_ids[first_div]} {tok.decode([ref_ids[first_div]])!r} "
              f"lazy={answer_argmax[first_div]} "
              f"{tok.decode([answer_argmax[first_div]])!r}")


if __name__ == "__main__":
    main()
