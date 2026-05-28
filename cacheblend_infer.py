"""Standalone CacheBlend reference (knowledge fusion of cached chunk KV).

Extracted from the bundled vLLM-0.4.1 fork (cacheblend/vllm_blend) into a clean
transformers-only reference, so the algorithm can be validated independently and
serve as a correctness oracle for a vLLM-0.8.5 port.

CacheBlend serves RAG by reusing per-chunk KV caches that were each computed in
ISOLATION (no cross-chunk attention), then "fusing" them: it recomputes the KV
of only a small fraction of tokens -- the ones whose value vector deviates most
from a fresh recompute at an early "check" layer -- and reuses the cached KV for
everyone else. This restores most of the cross-chunk attention at a fraction of
the prefill cost.

Algorithm (per the fork's model_executor/models/llama.py + xformers backend):
  1. Collect: run each chunk alone (here at its GLOBAL contiguous positions, so
     RoPE is already correct) and store per-layer K,V -> `old_kv`.
  2. Fuse over the concatenated sequence, layer by layer, with a per-layer status:
       layer  < check_layer : full prefill (fresh KV for all tokens)
       layer == check_layer : compute fresh KV for all tokens; per non-suffix
            token, deviation = ||V_fresh - V_cached||^2; pick the top
            `recomp_ratio` as `imp` (always include the suffix/query tokens);
            from here on only `imp` tokens propagate.
       layer  > check_layer : recompute KV for `imp` only, merge into the cached
            KV (old_kv[imp] = fresh), attend `imp` queries over the merged KV.
  3. The merged per-layer KV is the fused cache; decode proceeds normally.

Run: .venv/bin/python cacheblend_infer.py --model hxia7/Llama-3.2-1B-Block-FT
"""

from __future__ import annotations

import argparse
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        use_cache=True)
    model.eval()
    return model, tok


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class CacheBlend:
    """Manual Llama forward implementing CacheBlend fusion (batch size 1)."""

    def __init__(self, model, check_layer: int = 1, recomp_ratio: float = 0.16):
        self.model = model
        cfg = model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim",
                                cfg.hidden_size // cfg.num_attention_heads)
        self.rep = self.n_heads // self.n_kv
        self.scale = self.head_dim ** -0.5
        self.check_layer = check_layer
        self.recomp_ratio = recomp_ratio
        self.device = model.device

    def _cos_sin(self, positions: torch.Tensor):
        """cos/sin for `positions` (1D) -> each [N, head_dim]."""
        dummy = torch.zeros(1, 1, 1, device=self.device, dtype=self.model.dtype)
        cos, sin = self.model.model.rotary_emb(dummy, positions.unsqueeze(0))
        return cos[0], sin[0]

    def _qkv(self, layer, h, positions):
        """Project + RoPE. h:[N,d] -> q:[N,nH,hd], k/v:[N,nKV,hd]."""
        attn = layer.self_attn
        n = h.shape[0]
        q = attn.q_proj(h).view(n, self.n_heads, self.head_dim)
        k = attn.k_proj(h).view(n, self.n_kv, self.head_dim)
        v = attn.v_proj(h).view(n, self.n_kv, self.head_dim)
        cos, sin = self._cos_sin(positions)              # [N, hd]
        cos = cos.unsqueeze(1).to(h.dtype)               # [N,1,hd]
        sin = sin.unsqueeze(1).to(h.dtype)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        return q, k, v

    def _attn(self, layer, q, k, v, pos_q, pos_k):
        """Causal attention by GLOBAL position. q:[Nq,nH,hd] k/v:[Nk,nKV,hd]."""
        k = k.repeat_interleave(self.rep, dim=1)         # [Nk,nH,hd]
        v = v.repeat_interleave(self.rep, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * self.scale
        mask = pos_k[None, :] <= pos_q[:, None]          # [Nq,Nk] causal
        scores = scores.masked_fill(~mask[None], float("-inf"))
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", probs, v)     # [Nq,nH,hd]
        out = out.reshape(q.shape[0], self.n_heads * self.head_dim)
        return layer.self_attn.o_proj(out)

    def _mlp_block(self, layer, h):
        resid = h
        h = layer.post_attention_layernorm(h)
        return resid + layer.mlp(h)

    @torch.no_grad()
    def collect(self, chunks_ids: List[List[int]]):
        """Per-layer cached K,V for each chunk at its GLOBAL positions."""
        old_k = [[] for _ in range(self.n_layers)]
        old_v = [[] for _ in range(self.n_layers)]
        offset = 0
        for ids in chunks_ids:
            t = torch.tensor(ids, device=self.device)
            pos = torch.arange(offset, offset + len(ids), device=self.device)
            h = self.model.model.embed_tokens(t)         # [L,d]
            for li in range(self.n_layers):
                layer = self.model.model.layers[li]
                resid = h
                hn = layer.input_layernorm(h)
                q, k, v = self._qkv(layer, hn, pos)
                attn = self._attn(layer, q, k, v, pos, pos)  # chunk attends self
                h = self._mlp_block(layer, resid + attn)
                old_k[li].append(k)
                old_v[li].append(v)
            offset += len(ids)
        old_k = [torch.cat(x, dim=0) for x in old_k]      # [T,nKV,hd] per layer
        old_v = [torch.cat(x, dim=0) for x in old_v]
        return old_k, old_v

    @torch.no_grad()
    def fuse(self, full_ids: List[int], suffix_len: int, old_k, old_v):
        """Fused prefill. Returns (next_token_logits, fused_k, fused_v)."""
        T = len(full_ids)
        pos_all = torch.arange(T, device=self.device)
        t = torch.tensor(full_ids, device=self.device)
        h = self.model.model.embed_tokens(t)             # [T,d]
        fused_k = [kk.clone() for kk in old_k]
        fused_v = [vv.clone() for vv in old_v]

        imp = None
        pos_act = pos_all
        for li in range(self.n_layers):
            layer = self.model.model.layers[li]
            resid = h
            hn = layer.input_layernorm(h)

            if li < self.check_layer:                    # full prefill
                q, k, v = self._qkv(layer, hn, pos_all)
                attn = self._attn(layer, q, k, v, pos_all, pos_all)
                fused_k[li], fused_v[li] = k, v
                h = self._mlp_block(layer, resid + attn)

            elif li == self.check_layer:                 # select imp tokens
                q, k, v = self._qkv(layer, hn, pos_all)
                dev = ((v[:T - suffix_len].float()
                        - fused_v[li][:T - suffix_len].float()) ** 2).sum((1, 2))
                topk = int((T - suffix_len) * self.recomp_ratio)
                top = torch.sort(torch.topk(dev, topk).indices).values
                suffix = torch.arange(T - suffix_len, T, device=self.device)
                imp = torch.cat([top, suffix])
                fused_k[li], fused_v[li] = k, v           # fresh full at check
                attn = self._attn(layer, q[imp], k, v, pos_all[imp], pos_all)
                h = self._mlp_block(layer, resid[imp] + attn)
                pos_act = pos_all[imp]

            else:                                         # recompute imp only
                q, k, v = self._qkv(layer, hn, pos_act)
                fused_k[li][imp] = k
                fused_v[li][imp] = v
                attn = self._attn(layer, q, fused_k[li], fused_v[li],
                                  pos_act, pos_all)
                h = self._mlp_block(layer, resid + attn)

        h = self.model.model.norm(h)                      # [len(imp),d]
        logits = self.model.lm_head(h[-1])                # last = suffix end
        return logits, fused_k, fused_v

    @torch.no_grad()
    def generate(self, chunks_ids, suffix_len, max_new_tokens, eos_id):
        old_k, old_v = self.collect(chunks_ids)
        full_ids = [tid for ids in chunks_ids for tid in ids]
        logits, fk, fv = self.fuse(full_ids, suffix_len, old_k, old_v)

        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        for li in range(self.n_layers):
            cache.update(fk[li].transpose(0, 1).unsqueeze(0),   # [1,nKV,T,hd]
                         fv[li].transpose(0, 1).unsqueeze(0), li)

        out_ids, cur = [], len(full_ids)
        nxt = int(logits.argmax())
        for _ in range(max_new_tokens):
            out_ids.append(nxt)
            if nxt == eos_id:
                break
            inp = torch.tensor([[nxt]], device=self.device)
            pos = torch.tensor([[cur]], device=self.device)
            o = self.model(input_ids=inp, position_ids=pos,
                           past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            nxt = int(o.logits[0, -1].argmax())
            cur += 1
        return out_ids


# ---- Toy multi-document RAG example (same shape as block_infer.py) ----
SYSTEM = ("<|start_header_id|>system\nYou are an intelligent AI assistant. "
          "Answer the question using only the reference documents.\n\n")
DOCS = [
    "- Title: Polish-Russian War (film)\nPolish-Russian War is a 2009 Polish "
    "film directed by Xawery Zulawski based on the novel by Dorota Maslowska.\n",
    "- Title: Xawery Zulawski\nXawery Zulawski (born 22 December 1971 in Warsaw) "
    "is a Polish film director. He is the son of actress Malgorzata Braunek.\n",
    "- Title: Viktor Yeliseyev\nViktor Petrovich Yeliseyev (born June 9, 1950) "
    "is a Russian general.\n",
]
QUERY = ("Question: Who is the mother of the director of film "
         "Polish-Russian War?\n\n\n")


@torch.no_grad()
def full_prefill_generate(model, tok, full_ids, max_new_tokens):
    inp = torch.tensor([full_ids], device=model.device)
    out = model.generate(input_ids=inp,
                         attention_mask=torch.ones_like(inp),
                         max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return out[0, len(full_ids):].tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="hxia7/Llama-3.2-1B-Block-FT")
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--recomp-ratio", type=float, default=0.16)
    p.add_argument("--check-layer", type=int, default=1)
    args = p.parse_args()

    model, tok = load_model(args.model)

    chunks = [SYSTEM] + DOCS + [QUERY]
    chunks_ids = [tok.encode(c, add_special_tokens=False) for c in chunks]
    if tok.bos_token_id is not None:
        chunks_ids[0] = [tok.bos_token_id] + chunks_ids[0]
    suffix_len = len(chunks_ids[-1])
    full_ids_in = [tid for ids in chunks_ids for tid in ids]
    eos = tok.eos_token_id

    def show(name, ids):
        print(f"[{name:26}] {tok.decode(ids, skip_special_tokens=True)!r}")

    # Oracle: full cross-attention prefill over the SAME tokens.
    oracle_ids = full_prefill_generate(model, tok, full_ids_in, args.max_new_tokens)
    show("full prefill (oracle)", oracle_ids)

    # CacheBlend fusion.
    cb = CacheBlend(model, check_layer=args.check_layer,
                    recomp_ratio=args.recomp_ratio)
    cb_ids = cb.generate(chunks_ids, suffix_len, args.max_new_tokens, eos)
    show(f"cacheblend r={args.recomp_ratio}", cb_ids)

    # Naive reuse: no recompute (recomp_ratio 0) -> shows fusion matters.
    cb0 = CacheBlend(model, check_layer=args.check_layer, recomp_ratio=0.0)
    naive_ids = cb0.generate(chunks_ids, suffix_len, args.max_new_tokens, eos)
    show("naive reuse (r=0)", naive_ids)

    n = min(len(cb_ids), len(oracle_ids))
    agree = sum(a == b for a, b in zip(cb_ids[:n], oracle_ids[:n]))
    print(f"\ncacheblend vs full prefill: {agree}/{n} tokens agree")


if __name__ == "__main__":
    main()
