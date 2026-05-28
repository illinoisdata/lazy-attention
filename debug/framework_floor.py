"""Show the vLLM-vs-HF framework floor (no lazy attention involved).

Runs the SAME plain prompt with FULL causal attention through:
  - HF Transformers eager (the reference framework), and
  - vanilla vLLM (no document_seqs => normal, non-lazy attention path),
both greedy. The whole model (RMSNorm, MLP, attention, LM head) is computed by
two different kernel stacks. If even this diverges, then comparing the lazy
path against HF can never be bit-identical regardless of the lazy attention's
correctness -- the floor is the framework, not the rotation.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import compare_lazy_block_infer as C


PROMPT = (
    "<|start_header_id|>system\n"
    "You are an intelligent AI assistant. Answer the question using the "
    "documents below. Write a detailed answer.\n\n"
    "- Title: Polish-Russian War (film)\n"
    "Polish-Russian War is a 2009 Polish film directed by Xawery Zulawski "
    "based on the novel by Dorota Maslowska.\n"
    "- Title: Xawery Zulawski\n"
    "Xawery Zulawski is a Polish film director, the son of actress Malgorzata "
    "Braunek and director Andrzej Zulawski.\n\n"
    "Question: Who directed the film Polish-Russian War, and who are his "
    "parents?\n\n\n"
)


@torch.inference_mode()
def hf_greedy(model_name, max_tokens):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        use_cache=True, attn_implementation="eager")
    model.eval()
    # Match vLLM's default tokenization, which prepends BOS.
    ids = tok.encode(PROMPT, add_special_tokens=True)
    inp = torch.tensor([ids], device=model.device)
    out = model.generate(inp, max_new_tokens=max_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    gen = out[0][len(ids):].tolist()
    del model
    torch.cuda.empty_cache()
    return gen, tok.decode(gen, skip_special_tokens=True)


def vllm_greedy(model_name, max_tokens):
    import lazy.__vllm__  # noqa: F401
    import vllm.transformers_utils.tokenizer as vtok
    from lazy.entrypoints.llm import LazyLLM
    from vllm import SamplingParams
    o = vtok.get_cached_tokenizer
    vtok.get_cached_tokenizer = lambda t: (
        setattr(t, "all_special_tokens_extended", t.all_special_tokens)
        or o(t)) if not hasattr(t, "all_special_tokens_extended") else o(t)
    llm = LazyLLM(model=model_name, gpu_memory_utilization=0.9,
                  enable_prefix_caching=False, trust_remote_code=True,
                  enforce_eager=True)
    # No document_seqs => non-lazy, ordinary full-attention path.
    out = llm.generate(prompts=[PROMPT],
                       sampling_params=SamplingParams(temperature=0.0,
                                                      max_tokens=max_tokens))[0]
    s = out.outputs[0]
    return list(s.token_ids), s.text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=C.MODEL)
    p.add_argument("--max-tokens", type=int, default=80)
    args = p.parse_args()

    hf_ids, hf_txt = hf_greedy(args.model, args.max_tokens)
    vllm_ids, vllm_txt = vllm_greedy(args.model, args.max_tokens)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if hf_ids and hf_ids[-1] == tok.eos_token_id:
        hf_ids = hf_ids[:-1]
    if vllm_ids and vllm_ids[-1] == tok.eos_token_id:
        vllm_ids = vllm_ids[:-1]

    print(f"HF   ({len(hf_ids)}): {hf_txt!r}")
    print(f"vLLM ({len(vllm_ids)}): {vllm_txt!r}")
    if hf_ids == vllm_ids:
        print(f"HF == vanilla-vLLM : MATCH over {len(hf_ids)} tokens")
    else:
        first = next((i for i, (a, b) in enumerate(zip(hf_ids, vllm_ids))
                      if a != b), min(len(hf_ids), len(vllm_ids)))
        print(f"HF vs vanilla-vLLM FIRST DIVERGE at index {first} "
              f"(no lazy attention involved)")


if __name__ == "__main__":
    main()
