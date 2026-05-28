"""Does HF's attention backend explain the vLLM divergence?

Runs the SAME full-causal prompt through HF-eager, HF-sdpa, and vanilla vLLM
(Triton). If HF-eager == HF-sdpa but both still differ from vLLM, then the
attention *backend* is NOT the source of divergence -- it's the whole-model
framework difference (RMSNorm, MLP, RoPE, LM head, matmul order).
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import compare_lazy_block_infer as C
import framework_floor as F

MAXT = 80


@torch.inference_mode()
def hf(model_name, attn_impl):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation=attn_impl).eval()
    ids = tok.encode(F.PROMPT, add_special_tokens=True)
    out = model.generate(torch.tensor([ids], device=model.device),
                         max_new_tokens=MAXT, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    gen = out[0][len(ids):].tolist()
    del model
    torch.cuda.empty_cache()
    return [t for t in gen if t != tok.eos_token_id]


def vllm(model_name):
    import lazy.__vllm__  # noqa
    import vllm.transformers_utils.tokenizer as vtok
    o = vtok.get_cached_tokenizer
    vtok.get_cached_tokenizer = lambda t: (
        setattr(t, "all_special_tokens_extended", t.all_special_tokens)
        or o(t)) if not hasattr(t, "all_special_tokens_extended") else o(t)
    from lazy.entrypoints.llm import LazyLLM
    from vllm import SamplingParams
    llm = LazyLLM(model=model_name, gpu_memory_utilization=0.9,
                  enable_prefix_caching=False, trust_remote_code=True,
                  enforce_eager=True)
    s = llm.generate(prompts=[F.PROMPT],
                     sampling_params=SamplingParams(temperature=0.0,
                                                    max_tokens=MAXT))[0].outputs[0]
    tok = AutoTokenizer.from_pretrained(model_name)
    return [t for t in s.token_ids if t != tok.eos_token_id]


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    eager = hf(C.MODEL, "eager")
    sdpa = hf(C.MODEL, "sdpa")
    vl = vllm(C.MODEL)
    print(f"HF-eager len={len(eager)}")
    print(f"HF-sdpa  len={len(sdpa)}")
    print(f"vLLM     len={len(vl)}")
    print(f"HF-eager vs HF-sdpa : {'MATCH' if eager == sdpa else f'diverge @ {first_div(eager, sdpa)}'}")
    print(f"HF-eager vs vLLM    : {'MATCH' if eager == vl else f'diverge @ {first_div(eager, vl)}'}")
    print(f"HF-sdpa  vs vLLM    : {'MATCH' if sdpa == vl else f'diverge @ {first_div(sdpa, vl)}'}")


if __name__ == "__main__":
    main()
