#!/usr/bin/env python3
"""Capture a live Lazy-vs-Prefix-Caching token-stream race and render it as a GIF.

Two RAG servers (left = Prefix Caching / stock vLLM, right = Lazy-Attn / ours)
are given the SAME docs and the SAME query, with the retrieved docs in an order
that does NOT match how prefix caching saw them. Both /query/stream endpoints
are hit at the SAME instant (threading.Barrier); every streamed token is
timestamped. The result is rendered into an animated GIF that shows the right
panel (Lazy) producing its first token while the left panel (Prefix) is still
recomputing -- the real, measured TTFT gap.

    python benchmarks/demo_race.py --out analysis/lazy_vs_prefix_demo.gif
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import requests

LAZY = "lazy"
BASE = "base"


# Exact Block-Attention RAG template (github.com/TemporaryLoRA/Block-Attention).
# The model (Tulu3-Block-FT / Llama-3.2-1B-Block-FT) was fine-tuned on this
# format: a system block (preamble + "- Title:" doc blocks) followed by a user
# block with the instruction + question. Each block is cached independently.
BA_SYSTEM_PREAMBLE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are an intelligent AI assistant. Please answer questions based on the "
    "user's instructions. Below are some reference documents that may help you "
    "in answering the user's question.\n\n"
)
BA_USER_SUFFIX = (
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "Please write a high-quality answer for the given question using only the "
    "provided search documents (some of which might be irrelevant).\n"
    "Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


def _doc_block(doc: dict) -> str:
    return f"- Title: {doc['title']}\n{doc['text'].strip()}\n"


def _load_record(path: str, index: int) -> dict:
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i == index:
                return json.loads(line)
    raise SystemExit(f"record {index} not found in {path}")


# Short factual passages so the 1B model produces coherent output on screen
# (pure-random tokens make it emit EOS immediately). Each is padded to a
# realistic retrieval length so prefix-cache recompute is visibly expensive.
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
    raise SystemExit(f"server at {url} never became healthy")


def _stream(url: str, payload: dict) -> dict:
    """Stream one query, timestamping every token relative to the request start.
    Measured in ISOLATION (one engine at a time) so each method gets the whole
    GPU -- running both at once on a single GPU would make them contend and
    mask the real TTFT gap."""
    t0 = time.perf_counter()
    events: list[list] = []
    with requests.post(f"{url}/query/stream", json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                events.append([(time.perf_counter() - t0) * 1e3, chunk])
    return {"events": events, "end_ms": (time.perf_counter() - t0) * 1e3,
            "ttft_ms": events[0][0] if events else None}


def _reorder(k: int, seed: int) -> list[int]:
    """Deterministic doc reordering (same for both engines, even when each is
    measured in a separate process)."""
    natural = list(range(k))
    order = natural[:]
    rng = random.Random(seed + 7)
    while order == natural:
        rng.shuffle(order)
    return order


def measure_side(url: str, role_name: str, args) -> dict:
    """Cache the Block-Attention blocks into ONE engine, warm them in natural
    order, then stream the SAME docs reordered and timestamp every token."""
    health = _wait_healthy(url)
    rec = _load_record(args.dataset, args.record_index)
    question = rec["question"]
    docs = rec["documents"][:args.k]
    # block 0 = system preamble; blocks 1..N = "- Title: ..." passages. The user
    # turn (instruction + question) is the never-cached per-query suffix.
    blocks = [BA_SYSTEM_PREAMBLE] + [_doc_block(d) for d in docs]
    query = BA_USER_SUFFIX.format(question=question)
    ctx_words = sum(len(b.split()) for b in blocks)
    print(f"[{role_name}] question: {question}", flush=True)
    print(f"[{role_name}] caching {len(blocks)} blocks (preamble + {len(docs)} docs, ~{ctx_words} words) ...", flush=True)
    ids = requests.post(f"{url}/docs", json={"docs": blocks, "prewarm": True}, timeout=1800).json()["doc_ids"]

    gen_kw = {"max_tokens": args.max_tokens, "min_tokens": args.min_tokens,
              "temperature": args.temperature, "top_p": args.top_p,
              "repetition_penalty": args.repetition_penalty, "prewarm": False}
    pre, doc_ids = ids[0], ids[1:]
    # Warmup in natural retrieval order: this is what the engine caches. Prefix
    # caching stores that exact token sequence; lazy stores per-block KV.
    requests.post(f"{url}/query", json={"doc_ids": [pre] + doc_ids, "query": query, **gen_kw}, timeout=600)

    # Timed query: SAME docs, NEW order -> prefix caching must recompute past the
    # first reordered block; lazy reuses every block's KV regardless of slot.
    order = _reorder(len(doc_ids), args.seed)
    print(f"[{role_name}] measuring (doc order {order}) ...", flush=True)
    tl = _stream(url, {"doc_ids": [pre] + [doc_ids[j] for j in order], "query": query, **gen_kw})
    tl["meta"] = {"model": health["model"], "question": question, "answers": rec.get("answers"),
                  "order": order, "k": len(docs), "ctx_words": ctx_words}
    print(f"[{role_name}] TTFT {tl['ttft_ms']:.0f} ms, {len(tl['events'])} tokens", flush=True)
    return tl


def _pack(lazy_tl: dict, base_tl: dict, args) -> dict:
    m = lazy_tl.get("meta", {})
    data = {LAZY: lazy_tl, BASE: base_tl,
            "meta": {"lazy_name": args.lazy_name, "base_name": args.base_name,
                     "lazy_model": lazy_tl.get("meta", {}).get("model"),
                     "base_model": base_tl.get("meta", {}).get("model"),
                     "question": m.get("question"), "answers": m.get("answers"),
                     "order": m.get("order"), "k": m.get("k"), "ctx_words": m.get("ctx_words")}}
    lt, bt = lazy_tl["ttft_ms"], base_tl["ttft_ms"]
    if lt and bt:
        print(f"  {args.base_name:22}: TTFT {bt:.0f} ms")
        print(f"  {args.lazy_name:22}: TTFT {lt:.0f} ms")
        print(f"  -> Lazy first token {bt / lt:.1f}x sooner")
    return data


# ----------------------------- rendering -----------------------------------

def _find_fonts() -> tuple[str, str]:
    """Locate a monospace TTF (regular, bold), trying system DejaVu first and
    falling back to the copy matplotlib bundles -- so rendering works on the
    server too, where the system font may be absent."""
    import glob
    cands = [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
    ]
    try:
        import matplotlib
        d = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
        cands.append((os.path.join(d, "DejaVuSansMono.ttf"), os.path.join(d, "DejaVuSansMono-Bold.ttf")))
    except Exception:
        pass
    for reg, bold in cands:
        if os.path.exists(reg):
            return reg, (bold if os.path.exists(bold) else reg)
    hits = glob.glob("/usr/**/DejaVuSansMono.ttf", recursive=True)
    if hits:
        return hits[0], hits[0]
    raise SystemExit("no DejaVuSansMono.ttf found; install fonts-dejavu or matplotlib")


def _cumulative(events: list[list], t_ms: float) -> str:
    return "".join(c for (te, c) in events if te <= t_ms)


def _wrap(text: str, max_chars: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        cur = ""
        for word in para.split(" "):
            while len(word) > max_chars:
                if cur:
                    lines.append(cur)
                    cur = ""
                lines.append(word[:max_chars])
                word = word[max_chars:]
            cand = word if not cur else f"{cur} {word}"
            if len(cand) <= max_chars:
                cur = cand
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


def render_gif(data: dict, out_path: str, speed: float = 4.0, frames: int = 110) -> None:
    from PIL import Image, ImageDraw, ImageFont

    FONT, FONT_B = _find_fonts()
    f_title = ImageFont.truetype(FONT_B, 22)
    f_head = ImageFont.truetype(FONT_B, 19)
    f_body = ImageFont.truetype(FONT, 16)
    f_badge = ImageFont.truetype(FONT_B, 17)
    f_clock = ImageFont.truetype(FONT_B, 18)

    BG, PANEL, BORD = (13, 17, 23), (22, 27, 34), (48, 54, 61)
    TEXT, DIM, WHITE = (201, 209, 217), (110, 118, 129), (240, 246, 252)
    GREEN, AMBER = (63, 185, 80), (219, 154, 4)

    W, H, M = 1060, 560, 20
    pw = (W - 3 * M) // 2
    p_top, p_h = 96, 410
    body_x_pad, body_y0 = 16, 96
    char_w = f_body.getlength("M")
    max_chars = int((pw - 2 * body_x_pad) / char_w)
    line_h = 22
    body_top = p_top + body_y0
    max_lines = (p_h - body_y0 - 12) // line_h

    meta = data["meta"]
    t_end = max(data[LAZY]["end_ms"], data[BASE]["end_ms"])
    tail = max(250.0, 0.08 * t_end)
    span = t_end + tail
    # Non-linear playback: the whole story is the time-to-first-token gap, but
    # decode (many tokens) takes far longer and would visually drown it. So spend
    # most frames in slow-motion over the prefill race [0, ~base TTFT], where the
    # left panel sits on "prefilling" while the right is already answering, then
    # fast-forward through the parallel decode.
    bt = data[BASE]["ttft_ms"] or t_end
    t1 = min(span, bt * 1.35 + 150.0)            # end of the emphasized race window
    n1 = max(1, int(frames * 0.6))               # dense, slow frames for the race
    n2 = max(1, frames - n1)                     # sparse, fast frames for decode
    times = [i / n1 * t1 for i in range(n1)] + [t1 + (i + 1) / n2 * (span - t1) for i in range(n2)]
    disp_ms = max(45, int(round(70 / max(speed, 1e-6) * 4)))  # ~70ms/frame at speed=4

    panels = [
        (M, BASE, meta["base_name"], AMBER, data[BASE]),
        (2 * M + pw, LAZY, meta["lazy_name"], GREEN, data[LAZY]),
    ]

    def draw_frame(t_ms: float, fi: int, final: bool) -> Image.Image:
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((M, 16), "Lazy-Attention vs Prefix Caching", font=f_title, fill=WHITE)
        d.text((M, 46), "same documents, retrieved in a new order  →  prefix cache misses, lazy reuses every doc",
                font=f_body, fill=DIM)

        for px, key, name, accent, side in panels:
            d.rounded_rectangle([px, p_top, px + pw, p_top + p_h], radius=10, fill=PANEL, outline=BORD, width=1)
            d.rectangle([px, p_top, px + pw, p_top + 4], fill=accent)
            d.text((px + body_x_pad, p_top + 14), name, font=f_head, fill=accent)

            ttft = side["ttft_ms"]
            got = ttft is not None and t_ms >= ttft
            if got:
                badge = f"first token @ {ttft:.0f} ms"
                bcol = accent
            else:
                badge = "prefilling …"
                bcol = DIM
            d.text((px + body_x_pad, p_top + 44), badge, font=f_badge, fill=bcol)

            text = _cumulative(side["events"], t_ms)
            lines = _wrap(text, max_chars)[-max_lines:]
            ty = body_top + p_top - p_top  # = body_top
            for i, ln in enumerate(lines):
                d.text((px + body_x_pad, body_top + i * line_h), ln, font=f_body, fill=TEXT)
            done = t_ms >= side["end_ms"]
            if got and not done and (fi // 3) % 2 == 0:
                last = lines[-1] if lines else ""
                cx = px + body_x_pad + f_body.getlength(last)
                cy = body_top + (len(lines) - 1 if lines else 0) * line_h
                d.rectangle([cx + 1, cy + 2, cx + 10, cy + 18], fill=accent)

        # status bar
        d.text((M, H - 32), f"t = {min(t_ms, t_end):6.0f} ms", font=f_clock, fill=WHITE)
        lt, bt = data[LAZY]["ttft_ms"], data[BASE]["ttft_ms"]
        if final and lt:
            msg = f"Lazy-Attn reaches first token {bt / lt:.1f}x sooner   ({lt:.0f} ms vs {bt:.0f} ms)"
            tw = f_clock.getlength(msg)
            d.text((W - M - tw, H - 32), msg, font=f_clock, fill=GREEN)
        return img

    imgs, durs = [], []
    for fi, t in enumerate(times):
        imgs.append(draw_frame(t, fi, final=False))
        durs.append(disp_ms)
    # hold the finished frame
    imgs.append(draw_frame(span, len(times), final=True))
    durs.append(2600)

    pal = imgs[-1].convert("P", palette=Image.ADAPTIVE, colors=128)
    imgs = [im.quantize(palette=pal, dither=Image.NONE) for im in imgs]
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=durs,
                 loop=0, optimize=False, disposal=2)
    print(f"wrote {out_path}  ({len(imgs)} frames, {sum(durs)/1000:.1f}s, {disp_ms}ms/frame, playback {speed}x slower)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["both", "lazy", "base", "render-pair"], default="both",
                    help="measure both engines (need both up), one engine, or render two saved timelines")
    ap.add_argument("--url", default="", help="endpoint for single-engine --role lazy|base")
    ap.add_argument("--lazy-url", default="http://127.0.0.1:8001")
    ap.add_argument("--base-url", default="http://127.0.0.1:8002")
    ap.add_argument("--lazy-name", default="Lazy-Attn (ours)")
    ap.add_argument("--base-name", default="Prefix Caching (vLLM)")
    ap.add_argument("--k", type=int, default=10, help="number of retrieved docs to cache/reorder")
    ap.add_argument("--dataset", default="benchmarks/demo_data/2wiki_demo.jsonl",
                    help="RAG eval jsonl (real docs + question), Block-Attention 2WikiMultihopQA format")
    ap.add_argument("--record-index", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--min-tokens", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy, matches eval")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="analysis/lazy_vs_prefix_demo.gif")
    ap.add_argument("--out-json", default="", help="for --role lazy|base: write this side's timeline")
    ap.add_argument("--lazy-json", default="", help="for --role render-pair")
    ap.add_argument("--base-json", default="", help="for --role render-pair")
    ap.add_argument("--speed", type=float, default=4.0, help="playback slowdown factor")
    args = ap.parse_args()

    if args.role == "lazy" or args.role == "base":
        url = args.url or (args.lazy_url if args.role == "lazy" else args.base_url)
        tl = measure_side(url, args.role, args)
        out_json = args.out_json or f"{args.out.rsplit('.', 1)[0]}_{args.role}.json"
        with open(out_json, "w") as fh:
            json.dump(tl, fh)
        print(f"wrote {out_json}")
        return

    if args.role == "render-pair":
        with open(args.lazy_json) as fh:
            lazy_tl = json.load(fh)
        with open(args.base_json) as fh:
            base_tl = json.load(fh)
        data = _pack(lazy_tl, base_tl, args)
        render_gif(data, args.out, speed=args.speed)
        return

    # role == both: both engines up; measure base then lazy (local convenience).
    base_tl = measure_side(args.base_url, "base", args)
    lazy_tl = measure_side(args.lazy_url, "lazy", args)
    data = _pack(lazy_tl, base_tl, args)
    render_gif(data, args.out, speed=args.speed)


if __name__ == "__main__":
    main()
