#!/usr/bin/env bash
#
# LazyAttention installer.
#
#   bash scripts/install.sh                 # prebuilt vLLM wheel + lazy_attn (~2 min)
#   bash scripts/install.sh --venv .venv    # ... into a fresh virtualenv
#   bash scripts/install.sh --bench         # ... plus the benchmark dependencies
#   bash scripts/install.sh --check         # only report on the environment
#   bash scripts/install.sh --source        # build vLLM from source (see vllm_proj/)
#
# LazyAttention and BlockAttention are pure-Python monkey patches over vLLM:
# no .cu/.cpp of their own, and the hot path is Triton, which is JIT-compiled
# at runtime. So the stock vLLM wheel already contains every compiled kernel
# they need -- there is nothing to build. Use --source only if you are changing
# vLLM's own C++/CUDA.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# vLLM 0.9.2 is the oldest release whose wheels ship sm_120 (Blackwell) kernels
# while still exposing the internals LazyAttention patches.
VLLM_VERSION="0.9.2"
TORCH_VERSION="2.7.0"
# torch 2.7.0 on PyPI is a cu126 build with no sm_120 kernels; take cu128.
TORCH_CUDA="cu128"
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"
# vLLM 0.9.2 predates the transformers 5.x config registry changes.
TRANSFORMERS_VERSION="4.53.2"
# Triton 3.3 (torch 2.7's pin) aborts compiling tl.dot for sm_120; 3.4 works
# and is what torch 2.8+ ships anyway.
TRITON_SM120_VERSION="3.4.0"

FROM_SOURCE=0
WITH_BENCH=0
CHECK_ONLY=0
VERIFY=1
VENV_PATH=""

# ---------------------------------------------------------------- args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)     FROM_SOURCE=1; shift ;;
        --bench)      WITH_BENCH=1; shift ;;
        --check)      CHECK_ONLY=1; shift ;;
        --no-verify)  VERIFY=0; shift ;;
        --venv)       VENV_PATH="${2:?--venv needs a path}"; shift 2 ;;
        -h|--help)    sed -n '3,17p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *)            echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# -------------------------------------------------------------- venv -----
if [[ -n "${VENV_PATH}" ]]; then
    say "Creating virtualenv at ${VENV_PATH}"
    python3 -m venv "${VENV_PATH}"
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
    pip install --quiet --upgrade pip
fi

command -v python3 >/dev/null || die "python3 not found on PATH"

# ------------------------------------------------------------- report ----
say "Environment"
python3 - <<'EOF'
import platform, shutil, subprocess, sys

print(f"  python   {platform.python_version()}  ({sys.executable})")
if not (3, 9) <= sys.version_info[:2] < (3, 13):
    print("  ^ vLLM 0.9.2 supports Python 3.9-3.12", file=sys.stderr)

if nvidia_smi := shutil.which("nvidia-smi"):
    out = subprocess.run(
        [nvidia_smi, "--query-gpu=name,memory.total,compute_cap",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    for line in out.splitlines():
        print(f"  gpu      {line}")
else:
    print("  gpu      no nvidia-smi found")

print(f"  nvcc     {shutil.which('nvcc') or 'not found (only needed for --source)'}")
EOF

# Blackwell consumer cards need a newer Triton than torch 2.7 pins.
NEEDS_SM120=0
if command -v nvidia-smi >/dev/null 2>&1; then
    while read -r cap; do
        [[ "${cap}" == "12.0" ]] && NEEDS_SM120=1
    done < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)
fi

if [[ ${CHECK_ONLY} -eq 1 ]]; then
    [[ ${NEEDS_SM120} -eq 1 ]] && \
        say "Detected sm_120: would install Triton ${TRITON_SM120_VERSION}"
    say "Check only; nothing installed."
    exit 0
fi

# -------------------------------------------------------------- vllm ----
# The pinned torch lives on PyTorch's own index; everything else on PyPI.
pip_pinned() {
    pip install \
        --index-url "${TORCH_INDEX}" \
        --extra-index-url https://pypi.org/simple \
        "torch==${TORCH_VERSION}+${TORCH_CUDA}" \
        "transformers==${TRANSFORMERS_VERSION}" \
        "$@"
}

if [[ ${FROM_SOURCE} -eq 1 ]]; then
    # vllm_proj/install.sh needs torch importable and cmake/ninja on PATH
    # before it starts -- vLLM's setup.py reads torch to configure the CUDA
    # build. Without this a fresh virtualenv dies at that prerequisite check,
    # before cloning anything.
    say "Installing build prerequisites (torch ${TORCH_VERSION}+${TORCH_CUDA})"
    pip_pinned cmake ninja

    say "Building vLLM from source (20-60 min)"
    bash "${REPO_ROOT}/vllm_proj/install.sh"
else
    say "Installing vLLM ${VLLM_VERSION} with torch ${TORCH_VERSION}+${TORCH_CUDA}"
    pip_pinned "vllm==${VLLM_VERSION}"
fi

if [[ ${NEEDS_SM120} -eq 1 ]]; then
    say "Blackwell (sm_120) detected -- installing Triton ${TRITON_SM120_VERSION}"
    echo "  Triton 3.3 cannot compile tl.dot for sm_120."
    pip install "triton==${TRITON_SM120_VERSION}"
fi

# ---------------------------------------------------------- lazy_attn ----
say "Installing lazy_attn (editable)"
pip install -e "${REPO_ROOT}/lazy_attn"

if [[ ${WITH_BENCH} -eq 1 ]]; then
    say "Installing benchmark dependencies"
    pip install -r "${REPO_ROOT}/benchmarks/requirements.txt"
fi

# ------------------------------------------------------------ verify ----
if [[ ${VERIFY} -eq 0 ]]; then
    say "Done (verification skipped)."
    exit 0
fi

say "Verifying"
python3 - <<'EOF'
import sys

problems = []

import torch
print(f"  torch    {torch.__version__} (cuda {torch.version.cuda})")

import triton
print(f"  triton   {triton.__version__}")

import vllm
print(f"  vllm     {vllm.__version__}")

try:
    import vllm._C  # noqa: F401
    print("  vllm._C  loaded")
except ImportError as exc:
    problems.append(
        f"vllm._C failed to import ({exc}). The vLLM install has no compiled "
        "kernels -- a source build that silently failed leaves it in this state.")

import lazy.__vllm__  # noqa: F401  applies the LazyAttention patches
print("  lazy     patches applied")

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    name = torch.cuda.get_device_name(0)
    supported = torch.cuda.get_arch_list()
    print(f"  device   {name} ({arch})")
    if arch not in supported:
        problems.append(
            f"{name} is {arch}, but this torch only ships {', '.join(supported)}. "
            "Every CUDA kernel will fail with 'no kernel image is available'.")
    if (major, minor) >= (12, 0):
        major_t, minor_t = (int(p) for p in triton.__version__.split(".")[:2])
        if (major_t, minor_t) < (3, 4):
            problems.append(
                f"Triton {triton.__version__} cannot compile tl.dot for {arch}; "
                "install triton>=3.4.0.")
else:
    problems.append("torch.cuda.is_available() is False -- no usable GPU.")

if problems:
    print("", file=sys.stderr)
    for p in problems:
        print(f"\033[31mFAIL:\033[0m {p}", file=sys.stderr)
    sys.exit(1)
EOF

say "Done. Try: python scripts/validate_lazy.py"
