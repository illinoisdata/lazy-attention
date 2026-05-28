#!/bin/bash
# Parse a hit-ratio sweep job log into a (gpu_util, hits, queries, hit_ratio)
# table. The last [LAZY_DOC_KV_HIT] line within each gpu_util segment is the
# run-aggregate for that budget.
# Usage: bash scripts/benchmark/extract_hit_ratio.sh slurm/hit_ratio_scale_<jobid>.out
python - "${1:?usage: extract_hit_ratio.sh <logfile>}" <<'PY'
import sys, re
util = None
rows = {}
for ln in open(sys.argv[1], errors="ignore"):
    m = re.search(r'HIT_RATIO_SWEEP gpu_util=([0-9.]+)', ln)
    if m:
        util = m.group(1)
        continue
    m = re.search(r'\[LAZY_DOC_KV_HIT\] hits=(\d+) queries=(\d+) ratio=([0-9.]+)', ln)
    if m and util is not None:
        rows[util] = (int(m.group(1)), int(m.group(2)), float(m.group(3)))
print(f"{'gpu_util':>8} {'hits':>9} {'queries':>9} {'hit_ratio':>9}")
for u in sorted(rows, key=float):
    h, q, r = rows[u]
    print(f"{u:>8} {h:>9} {q:>9} {r:>9.4f}")
if not rows:
    print("(no [LAZY_DOC_KV_HIT] lines found - run may not be lazy, or <200 doc-block queries)")
PY
