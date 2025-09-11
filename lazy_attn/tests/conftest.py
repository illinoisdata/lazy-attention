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
    return SamplingParams(temperature=0, max_tokens=10, min_tokens=5, seed=42)


@pytest.fixture(scope="session")
def mock_model_name():
    return "ldsjmdy/Tulu3-Block-FT"
