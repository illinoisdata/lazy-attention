"""Minimal end-to-end smoke test for the promptcache baseline.

Loads a small causal LM, registers a one-document schema, processes a prompt
that references the cached document, and generates a few tokens. Success =
runs end-to-end on this machine's GPU and emits text.

Run: promptcache/.venv/bin/python promptcache/smoke_test.py
"""

import torch

import promptcache
from promptcache.model import AutoModel

# promptcache is a Llama-2-era baseline; it imports transformers.file_utils
# (removed ~4.40) so it must run on transformers < 4.40, which predates llama3
# RoPE scaling. Use a plain Llama-2-architecture model (ungated) so the stock
# rotary embedding pre-builds its cos/sin cache to max_position_embeddings.
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

SCHEMA = """
<schema name="s0">
<system/>
<user>
<module name="doc_0">
The Eiffel Tower is located in Paris, France. It was completed in 1889.
</module>
</user>
</schema>
"""

PROMPT = """
<prompt schema="s0">
<doc_0/>
<user>
Question: In which city is the Eiffel Tower located?
</user>
</prompt>
"""


def _expand_rotary_cache(model, max_len):
    """Make every RoPE cos/sin cache full-size and stop slicing it to kv length.

    PromptCache assigns NON-contiguous position ids (each schema module gets a
    reserved position range), so a decode token's position id can exceed the
    number of cached KV entries. Stock transformers sizes the rotary cos/sin
    cache to kv_seq_len and then does cos[position_ids], which goes out of
    bounds. We pre-build the cache to max_len and return it un-sliced.
    """
    patched = 0
    for m in model.modules():
        if hasattr(m, "cos_cached") and hasattr(m, "_set_cos_sin_cache"):
            m._set_cos_sin_cache(max_len, m.inv_freq.device,
                                 torch.get_default_dtype())
            m.forward = (lambda x, seq_len=None, _m=m:
                         (_m.cos_cached.to(x.dtype), _m.sin_cached.to(x.dtype)))
            patched += 1
    return patched


def main():
    lm = AutoModel(MODEL)
    # The cache engine allocates fp32 KV buffers, so keep the model fp32.
    lm.hf_model.to(device="cuda")
    n = _expand_rotary_cache(lm.hf_model,
                             lm.hf_model.config.max_position_embeddings)
    print("model on", lm.device, lm.hf_model.dtype, "| rotary patched:", n)

    cache_engine = promptcache.CacheEngine(max_ctx_length=2048, lm=lm)
    gen_engine = promptcache.GenerationEngine(lm)
    params = promptcache.GenerationParameters(
        temperature=0.0, repetition_penalty=1.0, top_p=1.0, top_k=-1,
        max_new_tokens=16, stop_token_ids=lm.stop_token_ids,
        stop_str=lm.stop_str)

    schema = promptcache.Schema(lm.get_formatter()(SCHEMA), lm=lm,
                                max_tokens=2048)
    cache_engine.add_schema(schema, max_tokens=2048)
    print("schema added, length", len(schema))

    prompt = promptcache.Prompt(spec=PROMPT, preproc=[lm.get_formatter()])
    token_ids, position_ids, cache_time, cache = cache_engine.process(
        prompt=prompt, return_full_position_ids=lm.use_full_position_ids)
    print(f"cache.process OK ({len(token_ids)} tokens, {cache_time:.1f} ms)")

    text = ""
    for out in gen_engine.generate(
            token_ids=token_ids, position_ids=position_ids, params=params,
            cache=cache, stream_interval=1,
            use_full_position_ids=lm.use_full_position_ids):
        text = out.new_text
    print("GENERATED:", repr(text.strip()))
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
