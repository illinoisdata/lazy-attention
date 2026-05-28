"""Bandwidth-bound RoPE-apply microbenchmark: load cos_sin_cache vs compute.

Unlike the fused lazy-decode kernel (which amortizes cos/sin to ~15 L2-resident
events behind a huge KV stream, so it's NOT bandwidth-bound), this is a pure
RoPE-apply: one program per token-row, N rows, each rotated by its own position.
At large N this saturates memory bandwidth, and the per-row cos_sin load becomes
a real fraction of traffic -- the regime where "avoid the extra IO by computing"
should win.

  LOAD    per row: read x[D] + read cos_sin_cache[pos][D] + write out[D]
  COMPUTE per row: read x[D]                              + write out[D]  (+ SFU)

So LOAD moves ~1.5-2x the bytes of COMPUTE. If the kernel is BW-bound, COMPUTE
should be ~that factor faster. POS_MODE controls locality of the cos_sin load:
  scatter -> random positions over MAXPOS (cache-missing, DRAM, worst for LOAD)
  seq     -> sequential positions (coalesced/L2-friendly, best for LOAD)

Pure torch+triton (NO vllm import, so triton is the real backend).
Env: N (rows, default 4194304), D (head_dim, 64), POS_MODE (scatter|seq), DT (bf16|fp16).
"""

from __future__ import annotations

import os
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

N = int(os.environ.get("N", str(4 * 1024 * 1024)))
D = int(os.environ.get("D", "64"))
HALF = D // 2
BASE = 10000.0
MAXPOS = int(os.environ.get("MAXPOS", "131072"))
POS_MODE = os.environ.get("POS_MODE", "scatter")
DT = {"bf16": torch.bfloat16, "fp16": torch.float16}[os.environ.get("DT", "bf16")]


@triton.jit
def rope_load(x_ptr, out_ptr, cs_ptr, pos_ptr, N, D: tl.constexpr):
    n = tl.program_id(0)
    if n >= N:
        return
    cols = tl.arange(0, D)
    half = D // 2
    x = tl.load(x_ptr + n * D + cols).to(tl.float32)
    rev = (cols + half) % D
    xr = tl.load(x_ptr + n * D + rev).to(tl.float32)
    pos = tl.load(pos_ptr + n)
    cc = cols % half
    cos = tl.load(cs_ptr + pos * D + cc)
    sin = tl.load(cs_ptr + pos * D + half + cc)
    rh = tl.where(cols < half, -xr, xr)
    out = x * cos + rh * sin
    tl.store(out_ptr + n * D + cols, out.to(out_ptr.dtype.element_ty))


@triton.jit
def rope_compute(x_ptr, out_ptr, pos_ptr, N, D: tl.constexpr, BASE: tl.constexpr):
    n = tl.program_id(0)
    if n >= N:
        return
    cols = tl.arange(0, D)
    half = D // 2
    x = tl.load(x_ptr + n * D + cols).to(tl.float32)
    rev = (cols + half) % D
    xr = tl.load(x_ptr + n * D + rev).to(tl.float32)
    pos = tl.load(pos_ptr + n)
    inv = 1.0 / libdevice.pow(BASE, 2 * (cols % half) / D)
    theta = pos * inv
    cos = libdevice.cos(theta)
    sin = libdevice.sin(theta)
    rh = tl.where(cols < half, -xr, xr)
    out = x * cos + rh * sin
    tl.store(out_ptr + n * D + cols, out.to(out_ptr.dtype.element_ty))


@triton.jit
def rope_compute_precomp(x_ptr, out_ptr, inv_ptr, pos_ptr, N, D: tl.constexpr):
    # Fair compute: load the tiny position-INDEPENDENT inv_freq[D] (reused every
    # row -> L1-resident, ~free), do only the irreducible cos/sin. No pow.
    n = tl.program_id(0)
    if n >= N:
        return
    cols = tl.arange(0, D)
    half = D // 2
    x = tl.load(x_ptr + n * D + cols).to(tl.float32)
    rev = (cols + half) % D
    xr = tl.load(x_ptr + n * D + rev).to(tl.float32)
    pos = tl.load(pos_ptr + n)
    inv = tl.load(inv_ptr + cols)
    theta = pos * inv
    cos = libdevice.cos(theta)
    sin = libdevice.sin(theta)
    rh = tl.where(cols < half, -xr, xr)
    out = x * cos + rh * sin
    tl.store(out_ptr + n * D + cols, out.to(out_ptr.dtype.element_ty))


def build_cache(device):
    half = HALF
    p = torch.arange(MAXPOS, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(half, device=device, dtype=torch.float32)[None, :]
    inv = 1.0 / (BASE ** (2 * i / D))
    ang = p * inv
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=1).contiguous()  # [MAXPOS, D]


def timed(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    dev = "cuda"
    torch.manual_seed(0)
    x = torch.randn(N, D, device=dev, dtype=DT)
    out = torch.empty_like(x)
    cache = build_cache(dev)
    if POS_MODE == "scatter":
        pos = torch.randint(0, MAXPOS, (N,), device=dev, dtype=torch.int32)
    else:
        pos = (torch.arange(N, device=dev, dtype=torch.int32) % MAXPOS)
    grid = (N,)
    i = (torch.arange(D, device=dev) % HALF).float()
    inv_vec = (1.0 / (BASE ** (2 * i / D))).to(torch.float32).contiguous()  # [D]

    # correctness: all three must match
    rope_load[grid](x, out, cache, pos, N, D)
    o_load = out.clone()
    rope_compute[grid](x, out, pos, N, D, BASE)
    diff_c = (o_load.float() - out.float()).abs().max().item()
    rope_compute_precomp[grid](x, out, inv_vec, pos, N, D)
    diff_p = (o_load.float() - out.float()).abs().max().item()

    ms_load = timed(lambda: rope_load[grid](x, out, cache, pos, N, D))
    ms_comp = timed(lambda: rope_compute[grid](x, out, pos, N, D, BASE))
    ms_prec = timed(lambda: rope_compute_precomp[grid](x, out, inv_vec, pos, N, D))

    bytes_x = N * D * x.element_size() * 2  # read x (+rev reuses cache) + write out
    bytes_cs = N * D * 4  # cos_sin fp32 row per token
    bw_load = (bytes_x + bytes_cs) / (ms_load * 1e-3) / 1e9
    bw_comp = bytes_x / (ms_comp * 1e-3) / 1e9
    bw_prec = bytes_x / (ms_prec * 1e-3) / 1e9
    print(f"N={N} D={D} dt={DT} pos={POS_MODE}  diff(compute)={diff_c:.1e} diff(precomp)={diff_p:.1e}")
    print(f"  LOAD          : {ms_load:.3f} ms   eff_BW~{bw_load:.0f} GB/s")
    print(f"  COMPUTE(pow)  : {ms_comp:.3f} ms   eff_BW~{bw_comp:.0f} GB/s")
    print(f"  COMPUTE(prec) : {ms_prec:.3f} ms   eff_BW~{bw_prec:.0f} GB/s  <- no pow, tiny inv load")
    best = min((ms_load, "LOAD"), (ms_comp, "COMPUTE(pow)"), (ms_prec, "COMPUTE(prec)"))
    print(f"  winner = {best[1]}   precomp/load = {ms_prec/ms_load:.2f}x")


if __name__ == "__main__":
    main()
