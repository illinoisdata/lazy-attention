# minidrag/tests/conftest.py
import os
import sys
import pytest


# add root directory to python path
vllm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, vllm_root)
minidrag_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../minidrag"))
sys.path.insert(0, minidrag_root)


# inject custom ops for all tests (not affect any functionality)
# specifically, we just inject two functions:
# 1. vllm._custom_ops.rotary_embedding_q = rotary_embedding_q
# 2. vllm._custom_ops.batched_rotary_embedding_q = batched_rotary_embedding_q
from minidrag._custom_ops import apply_patch
apply_patch()


@pytest.fixture(scope="session")
def mock_sampling_params():
    from vllm import SamplingParams
    return SamplingParams(temperature=0.0, max_tokens=100)


@pytest.fixture(scope="session")
def mock_model_name():
    return "meta-llama/Llama-3.2-1B"