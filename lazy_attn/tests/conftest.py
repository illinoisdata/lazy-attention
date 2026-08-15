# lazyattn/tests/conftest.py
import os
import sys
import pytest

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECK_MODULES = True

# ================================================
# os.environ['PYTORCH_CUDA_GRAPH_DEBUG'] = '1'
# ================================================

# Add lazyattn directory to python path
lazyattn_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../lazy_attn"))
sys.path.insert(0, lazyattn_root)
logger.info(f"Added lazyattn to Python path: {lazyattn_root}")

# Check the availability of vllm and lazy modules
if CHECK_MODULES:
    try:
        import vllm
        import lazy
    except ImportError as e:
        logger.warning(f"Failed to import module: {e}")

@pytest.fixture(scope="session")
def mock_prompts():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    return prompts


@pytest.fixture(scope="session")
def mock_sampling_params():
    from vllm import SamplingParams
    return SamplingParams(temperature=0, max_tokens=20, min_tokens=20, seed=42)


# Default to the 1B block-fine-tuned model so the suite fits on a single
# consumer GPU. Point LAZY_TEST_MODEL at ldsjmdy/Tulu3-Block-FT (8B) to
# reproduce the paper's accuracy numbers.
DEFAULT_TEST_MODEL = "hxia7/Llama-3.2-1B-Block-FT"


@pytest.fixture(scope="session")
def mock_model_name():
    return os.environ.get("LAZY_TEST_MODEL", DEFAULT_TEST_MODEL)


@pytest.fixture(scope="session")
def mock_gpu_memory_utilization():
    return float(os.environ.get("LAZY_TEST_GPU_MEM_UTIL", "0.6"))
